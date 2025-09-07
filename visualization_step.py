import math
import os
import sys

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from PyQt5.QtCore import QEvent, QPoint, QRect, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

matplotlib.use("Agg")  # for PDF export without GUI backend

# ==========================
# Helpers
# ==========================


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    imin, imax = float(np.nanmin(img)), float(np.nanmax(img))
    if not np.isfinite([imin, imax]).all() or imax <= imin:
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img.astype(np.float32) - imin) * (255.0 / (imax - imin))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _to_rgb_like(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img8 = _ensure_uint8(img)
        return np.stack([img8, img8, img8], axis=-1)
    elif img.ndim == 3:
        h, w, c = img.shape
        img8 = _ensure_uint8(img)
        if c == 1:
            img8 = img8.reshape(h, w)
            return np.stack([img8, img8, img8], axis=-1)
        elif c in (3, 4):
            return img8[:, :, :3]
        else:
            if c > 3:
                return img8[:, :, :3]
            gray = _ensure_uint8(np.mean(img8, axis=-1))
            return np.stack([gray, gray, gray], axis=-1)
    else:
        raise ValueError("Unsupported image array shape")


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w, c = arr.shape
    if c == 3:
        qimg = QImage(bytes(arr.data), w, h, 3 * w, QImage.Format_RGB888)
    elif c == 4:
        qimg = QImage(bytes(arr.data), w, h, 4 * w, QImage.Format_RGBA8888)
    else:
        rgb = _to_rgb_like(arr)
        return numpy_to_qimage(rgb)
    return qimg.copy()


def _apply_cmap_to_map(
    arr: np.ndarray,
    cmap_name: str,
    gain: float = 1.0,
    bias: float = 0.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Apply a Matplotlib colormap to a 2D array and return uint8 RGB.

    gain  >1 brightens (multiplier on normalized values) bias  shifts
    brightness (adds after gain) gamma <1 brightens midtones, >1 darkens
    midtones
    """
    if arr.ndim != 2:
        return _to_rgb_like(arr)

    data = arr.astype(np.float32)
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
        norm = np.zeros_like(data, dtype=np.float32)
    else:
        norm = (data - vmin) / (vmax - vmin + 1e-12)
        # apply simple brightness/contrast controls
        norm = np.clip(norm * max(gain, 1e-6) + bias, 0.0, 1.0)
        if gamma <= 0:
            gamma = 1.0
        norm = np.power(norm, 1.0 / gamma)

    if cmap_name is None or cmap_name.lower() in ("gray", "greys"):
        img8 = _ensure_uint8(norm * 255.0)
        return np.stack([img8, img8, img8], axis=-1)

    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm, bytes=True)  # returns uint8 RGBA
    rgb = rgba[..., :3]
    return rgb


# ==========================
# Data Loader
# ==========================


class DataLoader(QThread):
    data_loaded = pyqtSignal(list, list, dict, str, str)

    def __init__(self, data_path=None):
        super().__init__()
        self.data_path = data_path or "data.h5"

    def run(self):
        try:
            file_path = self.data_path or "data.h5"
            data_list, image_list = [], []
            maps_dict = {}

            with h5py.File(file_path, "r") as f:
                if "X-ray Maps" in f:
                    for group_name, group in f["X-ray Maps"].items():
                        if "data" in group:
                            data = np.array(group["data"][()])
                            data_list.append((group_name, data))
                            maps_dict[group_name] = data

                if "Images" in f:
                    for image_name, dataset in f["Images"].items():
                        image = np.array(dataset[()])
                        image_list.append((image_name, image))

            msg = f"Loaded {len(data_list)} maps and {len(image_list)} images from: {os.path.abspath(file_path)}"
            self.data_loaded.emit(data_list, image_list, maps_dict, msg, "")
        except Exception as e:
            self.data_loaded.emit([], [], {}, "", f"{type(e).__name__}: {e}")


# ==========================
# Image Canvas (ROI drawing overlay) with correct pixmap offset math
# ==========================


class ImageCanvas(QLabel):
    roi_drawn = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._mark_mode = False
        self._dragging = False
        self._start_px = None  # in pixmap coords
        self._current_px = None  # in pixmap coords
        self._overlay_rect_px = None  # QRect in pixmap coords

        # Make the label expand without enforcing giant minimums
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(100, 100)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)  # we draw our own scaled pixmap upstream

    def pixmap_rect_in_label(self) -> QRect:
        pm = self.pixmap()
        if not pm:
            return QRect(0, 0, 0, 0)
        lw, lh = self.width(), self.height()
        pw, ph = pm.width(), pm.height()
        x = max(0, (lw - pw) // 2)
        y = max(0, (lh - ph) // 2)
        return QRect(x, y, pw, ph)

    def label_to_pixmap_pos(self, pt: QPoint) -> QPoint:
        r = self.pixmap_rect_in_label()
        x = pt.x() - r.x()
        y = pt.y() - r.y()
        return QPoint(x, y)

    def clamp_to_pixmap(self, pt: QPoint) -> QPoint:
        pm = self.pixmap()
        if not pm:
            return QPoint(0, 0)
        x = max(0, min(pm.width(), pt.x()))
        y = max(0, min(pm.height(), pt.y()))
        return QPoint(x, y)

    def set_mark_mode(self, enabled: bool):
        self._mark_mode = enabled
        self._dragging = False
        self._overlay_rect_px = None
        self.update()

    def mousePressEvent(self, event):
        if self._mark_mode and event.button() == Qt.LeftButton and self.pixmap() is not None:
            pt_px = self.clamp_to_pixmap(self.label_to_pixmap_pos(event.pos()))
            self._dragging = True
            self._start_px = pt_px
            self._current_px = pt_px
            self._overlay_rect_px = QRect(self._start_px, self._current_px)
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._mark_mode and self._dragging and self.pixmap() is not None:
            pt_px = self.clamp_to_pixmap(self.label_to_pixmap_pos(event.pos()))
            self._current_px = pt_px
            self._overlay_rect_px = QRect(self._start_px, self._current_px)
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._mark_mode and self._dragging and event.button() == Qt.LeftButton and self.pixmap() is not None:
            self._dragging = False
            rect = self._overlay_rect_px.normalized() if self._overlay_rect_px is not None else None
            self._overlay_rect_px = None
            if rect is not None and rect.width() > 1 and rect.height() > 1:
                self.roi_drawn.emit(rect.left(), rect.top(), rect.right(), rect.bottom())
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._overlay_rect_px is not None:
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing, True)
            pen = QPen(QColor(255, 200, 0), 2, Qt.DashLine)
            p.setPen(pen)
            # draw at correct offset inside the label
            r = self.pixmap_rect_in_label()
            rr = self._overlay_rect_px.normalized()
            draw_rect = QRect(r.x() + rr.x(), r.y() + rr.y(), rr.width(), rr.height())
            p.drawRect(draw_rect)
            p.end()


# ==========================
# Main Widget
# ==========================


class VisualizationStep(QWidget):

    def _default_folder(self):
        """Prefer an explicit working_dir if set; else use the folder of
        data_path; else CWD."""
        folder = getattr(self, "working_dir", None)
        if not folder and getattr(self, "data_path", None):
            try:
                folder = os.path.dirname(os.path.abspath(self.data_path))
            except Exception:
                folder = None
        return folder or os.getcwd()

    def set_working_dir(self, folder: str):
        """Allow main.py to inject the working folder."""
        try:
            self.working_dir = os.path.abspath(folder) if folder else ""
        except Exception:
            self.working_dir = folder or ""

    def __init__(self, data_path=None):
        super().__init__()
        self.data_path = data_path

        self.data_list = []
        self.image_list = []
        self.maps = {}
        self.roi_coordinates = pd.DataFrame(columns=["X1", "Y1", "X2", "Y2", "ROI Name"])
        self.roi_source_path = None

        self.roi_color = Qt.red

        # Background display state
        self._current_kind = None  # 'image' | 'map'
        self._current_name = None
        self._current_qimage = None  # QImage in *source* pixels
        self._current_map_array = None  # 2D numpy if kind == 'map'

        # Brightness controls for maps
        self._gain = 1.0  # 0.5 .. 2.0
        self._bias = 0.0  # not exposed in UI for now
        self._gamma = 1.0  # not exposed in UI for now

        # Scale mode
        self._scale_mode = "Fit (keep AR)"  # other: Fill (crop, keep AR), Stretch (ignore AR)

        # Available colormaps
        self.available_cmaps = [
            "gray",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
            "cubehelix",
            "terrain",
            "hot",
            "cool",
            "spring",
            "summer",
            "autumn",
            "winter",
        ]

        # Scale bar state
        self._show_scale_bar = False
        self._scale_bar_thickness = 6
        self._scale_bar_pos = "Bottom-Left"  # Bottom-Left/Bottom-Right/Top-Left/Top-Right
        # Physical calibration for scale bar
        self._pixel_size_value = 1.0  # default 1 µm per pixel
        self._pixel_size_unit = "µm"  # one of ['mm','µm','nm']
        self._scale_target_um = 100.0  # fixed physical length to represent

        self._init_ui()

        self.loader = DataLoader(data_path=self.data_path)
        self.loader.data_loaded.connect(self.on_data_loaded)
        self.loader.start()

    # ---------- UI ----------
    def _init_ui(self):
        self.setWindowTitle("Visualization & Reporting")
        self.resize(1300, 880)
        self.setMinimumSize(900, 600)

        root = QHBoxLayout(self)

        # LEFT
        # LEFT (fixed width; only internal vertical split is resizable)
        left_container = QWidget(self)
        left_container.setObjectName("leftPanel")
        left_container.setFixedWidth(320)  # tweak (e.g., 300–360) to taste

        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Vertical splitter: top = log, bottom = controls
        splitter_left = QSplitter(Qt.Vertical, self)
        splitter_left.setHandleWidth(6)
        try:
            splitter_left.setChildrenCollapsible(False)
            splitter_left.setCollapsible(0, False)
            splitter_left.setCollapsible(1, False)
        except Exception:
            pass

        # --- Log panel ---
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumWidth(0)  # remove 420px constraint
        self.log_text.setMinimumHeight(120)  # give it some height

        # --- Controls group ---
        controls = QGroupBox("Plot & ROI Controls", self)
        form = QFormLayout()

        self.btn_load_roi = QPushButton("Load ROI File", self)
        self.btn_load_roi.clicked.connect(self.browse_and_load_roi_excel)
        form.addRow(self.btn_load_roi)

        self.btn_save_roi = QPushButton("Save ROI File", self)
        self.btn_save_roi.clicked.connect(self.save_roi_excel)
        form.addRow(self.btn_save_roi)

        self.dd_background = QComboBox(self)
        self.dd_background.currentIndexChanged.connect(self.on_background_changed)
        form.addRow("Background:", self.dd_background)

        # NEW: Scale mode selector
        self.dd_scale = QComboBox(self)
        self.dd_scale.addItems(["Fit (keep AR)", "Fill (crop, keep AR)", "Stretch (ignore AR)"])
        self.dd_scale.setCurrentText(self._scale_mode)
        self.dd_scale.currentIndexChanged.connect(self.on_scale_mode_changed)
        form.addRow("Scale mode:", self.dd_scale)

        # Map colormap
        self.dd_cmap = QComboBox(self)
        self.dd_cmap.addItems(self.available_cmaps)
        self.dd_cmap.setCurrentText("gray")
        self.dd_cmap.currentIndexChanged.connect(lambda _: self.show_background_selection())
        form.addRow("Map colormap:", self.dd_cmap)

        # Brightness slider (for maps)
        self.slider_bright = QSlider(Qt.Horizontal, self)
        self.slider_bright.setRange(-100, 100)
        self.slider_bright.setValue(0)
        self.slider_bright.valueChanged.connect(self.on_brightness_changed)
        self.lbl_bright_val = QLabel("0%", self)
        hb_bright = QHBoxLayout()
        hb_bright.addWidget(self.slider_bright)
        hb_bright.addWidget(self.lbl_bright_val)
        bright_row = QWidget(self)
        bright_row.setLayout(hb_bright)
        form.addRow("Brightness:", bright_row)

        # # Scale bar controls
        # self.cb_scale_bar = QPushButton("Show Scale Bar", self)
        # self.cb_scale_bar.setCheckable(True)
        # self.cb_scale_bar.setChecked(self._show_scale_bar)
        # self.cb_scale_bar.clicked.connect(self._toggle_scale_bar)

        # self.spin_pixel_size = QDoubleSpinBox(self)
        # self.spin_pixel_size.setDecimals(6)
        # self.spin_pixel_size.setRange(1e-9, 1e6)
        # self.spin_pixel_size.setSingleStep(0.1)
        # self.spin_pixel_size.setValue(self._pixel_size_value)
        # self.spin_pixel_size.valueChanged.connect(self._set_pixel_size_value)

        # self.dd_pixel_unit = QComboBox(self)
        # self.dd_pixel_unit.addItems(["mm","µm","nm"])
        # self.dd_pixel_unit.setCurrentText(self._pixel_size_unit)
        # self.dd_pixel_unit.currentIndexChanged.connect(
        #     lambda _: self._set_pixel_size_unit(self.dd_pixel_unit.currentText())
        # )

        # self.dd_scale_pos = QComboBox(self)
        # self.dd_scale_pos.addItems(["Bottom-Left","Bottom-Right","Top-Left","Top-Right"])
        # self.dd_scale_pos.setCurrentText(self._scale_bar_pos)
        # self.dd_scale_pos.currentIndexChanged.connect(
        #     lambda _: self._set_scale_pos(self.dd_scale_pos.currentText())
        # )

        # sb_row = QHBoxLayout()
        # sb_row.addWidget(self.cb_scale_bar)
        # sb_row.addWidget(QLabel("Pixel size:"))
        # sb_row.addWidget(self.spin_pixel_size)
        # sb_row.addWidget(self.dd_pixel_unit)
        # sb_row.addWidget(QLabel("Pos:"))
        # sb_row.addWidget(self.dd_scale_pos)
        # sbw = QWidget(self); sbw.setLayout(sb_row)
        # form.addRow("Scale bar (100 µm):", sbw)

        # Plot type
        self.dd_plot_type = QComboBox(self)
        self.dd_plot_type.addItems(["XY plot", "Dual axis plot", "Triangular plot"])
        self.dd_plot_type.currentIndexChanged.connect(self.on_plot_type_changed)
        form.addRow("Type of plot:", self.dd_plot_type)

        # correlation coefficient dropdown
        self.dd_corr = QComboBox(self)
        self.dd_corr.addItems(["Pearson", "Spearman", "None"])
        self.dd_corr.setCurrentText("Pearson")
        form.addRow("Correlation:", self.dd_corr)

        self.dd_style = QComboBox(self)
        self.dd_style.addItems(["Scatter", "Density"])
        self.dd_style.setCurrentText("Scatter")
        self.dd_style.currentIndexChanged.connect(lambda _: setattr(self, "plot_style", self.dd_style.currentText()))
        form.addRow("Plot style:", self.dd_style)
        self.plot_style = "Scatter"

        self.dd_container = QGroupBox("Plot Inputs", self)
        self.dd_form = QFormLayout()
        self.dd_container.setLayout(self.dd_form)
        form.addRow(self.dd_container)

        self.btn_report = QPushButton("Generate Report (PDF)", self)
        self.btn_report.clicked.connect(self.generate_report_pdf)
        form.addRow(self.btn_report)

        self.btn_mark_roi = QPushButton("Mark ROI (Draw)", self)
        self.btn_mark_roi.setCheckable(True)
        self.btn_mark_roi.toggled.connect(self.on_mark_roi_toggled)
        form.addRow(self.btn_mark_roi)

        self.dd_delete_roi = QComboBox(self)
        self.dd_delete_roi.addItem("-- Select ROI to delete --")
        self.btn_delete_roi = QPushButton("Delete Selected ROI", self)
        self.btn_delete_roi.clicked.connect(self.delete_selected_roi)
        form.addRow(self.dd_delete_roi, self.btn_delete_roi)

        self.btn_clear_rois = QPushButton("Clear All ROIs", self)
        self.btn_clear_rois.clicked.connect(self.clear_all_rois)
        form.addRow(self.btn_clear_rois)

        controls.setLayout(form)

        # Put log + controls into the vertical splitter
        splitter_left.addWidget(self.log_text)
        splitter_left.addWidget(controls)
        splitter_left.setSizes([220, 360])  # initial heights (log, controls)

        # Mount the splitter into the fixed-width left container
        left_layout.addWidget(splitter_left)

        # Finally add the left container to root; keep stretch=0 so it stays
        # narrow
        root.addWidget(left_container, stretch=0)

        # RIGHT
        right = QVBoxLayout()
        self.image_label = ImageCanvas(self)
        self.image_label.setStyleSheet("background: #111; color: white;")
        self.image_label.roi_drawn.connect(self.on_roi_drawn_from_canvas)
        right.addWidget(self.image_label, stretch=1)

        bottom = QHBoxLayout()
        self.color_dropdown = QComboBox(self)
        self.color_dropdown.addItems(["Red", "Green", "Blue", "Black", "Custom"])
        self.color_dropdown.currentIndexChanged.connect(self.on_color_changed)
        bottom.addWidget(QLabel("ROI Color:"))
        bottom.addWidget(self.color_dropdown)

        self.prev_button = QPushButton("Previous Image", self)
        self.prev_button.clicked.connect(self.show_previous_image)
        bottom.addWidget(self.prev_button)

        self.next_button = QPushButton("Next Image", self)
        self.next_button.clicked.connect(self.show_next_image)
        bottom.addWidget(self.next_button)

        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.upload_image)
        bottom.addWidget(self.upload_button)

        self.save_button = QPushButton("Save Image + ROI", self)
        self.save_button.clicked.connect(self.save_current_image_with_roi)
        bottom.addWidget(self.save_button)

        right.addLayout(bottom)
        root.addLayout(right, stretch=1)

        # Resize handling: keep image centered and scaled
        self.image_label.installEventFilter(self)

        self.plot_inputs = {}
        self.on_plot_type_changed(0)

    # ---------- Event filter to catch label resize ----------
    def eventFilter(self, obj, event):
        if obj is self.image_label and event.type() == QEvent.Resize:
            self._update_scaled_pixmap()
        return super().eventFilter(obj, event)

    # ---------- Logging ----------
    def log(self, msg: str):
        self.log_text.append(msg)

    # ---------- Loader ----------
    def on_data_loaded(self, data_list, image_list, maps_dict, msg, err):
        if err:
            self.log(f"[ERROR] {err}")
        if msg:
            self.log(msg)

        self.data_list = data_list or []
        self.maps = maps_dict or {}
        self.image_list = []

        for name, arr in image_list or []:
            try:
                arr_rgb = _to_rgb_like(arr)
                self.image_list.append((name, arr_rgb))
            except Exception as e:
                self.log(f"[WARN] Skipped image '{name}': {e}")

        self.refresh_background_dropdown_items()
        self.refresh_map_dropdown_items()

        # Preferred initial background
        if self.image_list:
            name, img = self.image_list[0]
            self._set_current_image(name, img)
        elif self.maps:
            name = next(iter(self.maps.keys()))
            self._set_current_map(name, self.maps[name])
        else:
            return

        if self.data_path and os.path.isfile(self.data_path):
            self.load_roi_from_folder(os.path.dirname(self.data_path))
        elif not self.data_path:
            self.log("Tip: Use 'Load ROI File' to add ROIs.")

    # ---------- Background helpers ----------
    def refresh_background_dropdown_items(self):
        self.dd_background.blockSignals(True)
        self.dd_background.clear()
        for name, _ in self.image_list:
            self.dd_background.addItem(f"Image: {name}")
        for name in sorted(self.maps.keys()):
            self.dd_background.addItem(f"Map: {name}")
        self.dd_background.blockSignals(False)

    def on_background_changed(self, _index: int):
        self.show_background_selection()

    def on_scale_mode_changed(self, _index: int):
        self._scale_mode = self.dd_scale.currentText()
        self._update_scaled_pixmap()

    def show_background_selection(self):
        text = self.dd_background.currentText() if self.dd_background.count() > 0 else ""

        if text.startswith("Image: "):
            name = text.replace("Image: ", "", 1)
            for nm, arr in self.image_list:
                if nm == name:
                    self._set_current_image(nm, arr)
                    break
        elif text.startswith("Map: "):
            name = text.replace("Map: ", "", 1)
            if name in self.maps:
                self._set_current_map(name, self.maps[name])
        else:
            if self.image_list:
                nm, arr = self.image_list[0]
                self._set_current_image(nm, arr)
            elif self.maps:
                nm = next(iter(self.maps.keys()))
                self._set_current_map(nm, self.maps[nm])

    def _set_current_image(self, name: str, rgb_arr: np.ndarray):
        self._current_kind = "image"
        self._current_name = name
        self._current_map_array = None
        self.slider_bright.setEnabled(False)
        qimg = numpy_to_qimage(_to_rgb_like(rgb_arr))
        self._current_qimage = qimg
        self._update_scaled_pixmap()

    def _set_current_map(self, name: str, map_arr: np.ndarray):
        self._current_kind = "map"
        self._current_name = name
        self._current_map_array = map_arr
        self.slider_bright.setEnabled(True)
        self._rebuild_qimage_from_map()
        self._update_scaled_pixmap()

    def _rebuild_qimage_from_map(self):
        if self._current_kind != "map" or self._current_map_array is None:
            return
        cmap_name = self.dd_cmap.currentText()
        # Cache min/max for colorbar
        try:
            data = self._current_map_array.astype(np.float32)
            vmin = float(np.nanmin(data))
            vmax = float(np.nanmax(data))
            if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
                vmin = 0.0
                vmax = 1.0
        except Exception:
            vmin, vmax = 0.0, 1.0
        self._last_map_min = vmin
        self._last_map_max = vmax
        self._last_cmap_name = cmap_name
        rgb = _apply_cmap_to_map(
            self._current_map_array,
            cmap_name,
            gain=self._gain,
            bias=self._bias,
            gamma=self._gamma,
        )
        self._current_qimage = numpy_to_qimage(rgb)

    def _compose_pixmap_with_rois(self, base_pm: QPixmap) -> QPixmap:
        pm = QPixmap(base_pm)  # copy
        # Draw colorbar if map background
        if getattr(self, "_current_kind", None) == "map" and pm and not pm.isNull():
            pm = self._draw_colorbar(pm)
        # Draw scale bar next
        if getattr(self, "_show_scale_bar", False) and pm and not pm.isNull():
            pm = self._draw_scale_bar(pm)
        # Draw ROI rectangles on top
        if self.roi_coordinates is not None and not self.roi_coordinates.empty:
            pm = self._draw_rectangles(pm)
        return pm

    def _toggle_scale_bar(self):
        self._show_scale_bar = not self._show_scale_bar
        self._update_scaled_pixmap()

    def _set_pixel_size_value(self, v: float):
        try:
            self._pixel_size_value = float(v)
        except Exception:
            self._pixel_size_value = 0.0
        if self.cb_scale_bar.isChecked():
            self._update_scaled_pixmap()

    def _set_pixel_size_unit(self, unit: str):
        self._pixel_size_unit = str(unit)
        if self.cb_scale_bar.isChecked():
            self._update_scaled_pixmap()

    def _draw_colorbar(self, pm: QPixmap) -> QPixmap:
        """Overlay a vertical colorbar on the right side when a map is
        displayed."""
        vmin = getattr(self, "_last_map_min", None)
        vmax = getattr(self, "_last_map_max", None)
        try:
            cmap_name = self.dd_cmap.currentText()
        except Exception:
            cmap_name = "viridis"
        if vmin is None or vmax is None or not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
            return pm

        bar_w = max(20, pm.width() // 40)
        margin = max(6, min(pm.width(), pm.height()) // 50)
        x0 = pm.width() - margin - bar_w
        y0 = margin
        y1 = pm.height() - margin

        # Use matplotlib colormap sampled at 256 levels
        try:
            cmap = plt.get_cmap(cmap_name)
        except Exception:
            cmap = plt.get_cmap("viridis")
        n = 256
        grad = np.linspace(1.0, 0.0, n, dtype=np.float32)
        rgba = cmap(grad, bytes=True)
        rgb = rgba[:, :3]
        bar_img = np.repeat(rgb[np.newaxis, :, :], bar_w, axis=0).transpose(1, 0, 2)
        qbar = numpy_to_qimage(bar_img)
        qpm_bar = QPixmap.fromImage(qbar).scaled(bar_w, y1 - y0, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(
            QRect(x0 - 4, y0 - 4, bar_w + 8, (y1 - y0) + 8),
            QColor(0, 0, 0, 120),
        )
        p.drawPixmap(x0, y0, qpm_bar)

        f = QFont()
        f.setPointSize(max(7, int(pm.height() * 0.025)))
        p.setFont(f)
        p.setPen(QColor(255, 255, 255))
        p.drawText(x0 - 6, y0 + 10, f"{vmax:.3g}")
        p.drawText(x0 - 6, y1, f"{vmin:.3g}")
        p.end()
        return pm

    def _draw_scale_bar(self, pm: QPixmap) -> QPixmap:
        from PyQt5.QtGui import QFont

        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = pm.width(), pm.height()

        # Geometry & conversion
        margin = max(6, min(w, h) // 50)
        # Convert pixel size to micrometers per pixel
        unit = getattr(self, "_pixel_size_unit", "µm")
        val = max(1e-12, float(getattr(self, "_pixel_size_value", 0.0) or 0.0))
        if unit == "mm":
            um_per_px = val * 1000.0
        elif unit in ("µm", "um"):
            um_per_px = val
        else:  # "nm"
            um_per_px = val * 1e-3

        target_um = float(getattr(self, "_scale_target_um", 100.0))
        px_len = max(1.0, target_um / max(um_per_px, 1e-12))
        bar_len = int(min(px_len, w - 2 * margin))
        bar_th = int(max(2, min(self._scale_bar_thickness, max(2, h // 60))))

        # Placement
        pos = getattr(self, "_scale_bar_pos", "Bottom-Left")
        if pos == "Bottom-Right":
            x = w - margin - bar_len
            y = h - margin - bar_th
            text_x = x + bar_len
            text_y = y - 6
        elif pos == "Top-Left":
            x = margin
            y = margin
            text_x = x
            text_y = y - 6
        elif pos == "Top-Right":
            x = w - margin - bar_len
            y = margin
            text_x = x + bar_len
            text_y = y - 6
        else:  # Bottom-Left
            x = margin
            y = h - margin - bar_th
            text_x = x
            text_y = y - 6

        # Draw bar (white with black border)
        pen = QPen(QColor(0, 0, 0))
        pen.setWidth(max(1, bar_th + 2))
        p.setPen(pen)
        p.drawLine(x, y + bar_th // 2, x + bar_len, y + bar_th // 2)
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(max(1, bar_th))
        p.setPen(pen)
        p.drawLine(x, y + bar_th // 2, x + bar_len, y + bar_th // 2)

        # Label (fixed 100 µm)
        label = f"{int(target_um) if abs(target_um - int(target_um)) < 1e-6 else target_um:g} µm"
        f = QFont()
        f.setPointSize(max(7, int(h * 0.025)))
        p.setFont(f)
        # text shadow for legibility
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            p.setPen(QColor(0, 0, 0))
            p.drawText(text_x + dx, text_y + dy, label)
        p.setPen(QColor(255, 255, 255))
        p.drawText(text_x, text_y, label)

        p.end()
        return pm

    def _update_scaled_pixmap(self):
        """Scale the current QImage according to the chosen scale mode
        and center it.

        - Fit (keep AR): image fits inside the panel (letterbox if needed)
        - Fill (crop, keep AR): image fills the panel by cropping
        - Stretch (ignore AR): image is stretched to panel
        """
        if self._current_qimage is None:
            return
        area_size = self.image_label.size()
        if area_size.width() <= 2 or area_size.height() <= 2:
            return

        if self._scale_mode.startswith("Fit"):
            mode = Qt.KeepAspectRatio
        elif self._scale_mode.startswith("Fill"):
            mode = Qt.KeepAspectRatioByExpanding
        else:
            mode = Qt.IgnoreAspectRatio

        scaled = self._current_qimage.scaled(area_size, mode, Qt.SmoothTransformation)
        # If we used Fill/Expanding, scaled might be bigger than label; crop to
        # label size
        if mode == Qt.KeepAspectRatioByExpanding:
            # center crop to label size
            x = max(0, (scaled.width() - area_size.width()) // 2)
            y = max(0, (scaled.height() - area_size.height()) // 2)
            cropped = scaled.copy(x, y, area_size.width(), area_size.height())
            pm = QPixmap.fromImage(cropped)
        else:
            pm = QPixmap.fromImage(scaled)
        pm = self._compose_pixmap_with_rois(pm)
        self.image_label.setPixmap(pm)
        self.image_label.update()

    # ---------- Plot inputs ----------
    def _rebuild_plot_inputs_ui(self, keys):
        while self.dd_form.rowCount() > 0:
            self.dd_form.removeRow(0)
        self.plot_inputs.clear()
        for key in keys:
            dd = QComboBox(self)
            self.plot_inputs[key] = dd
            self.dd_form.addRow(f"{key}:", dd)

    def refresh_map_dropdown_items(self):
        names = sorted(list(self.maps.keys()))
        plot_type = self.dd_plot_type.currentText()

        if plot_type == "XY plot":
            self._rebuild_plot_inputs_ui(["X Map", "Y Map"])
        elif plot_type == "Dual axis plot":
            self._rebuild_plot_inputs_ui(["Bottom Axis Map", "Left Axis Map", "Right Axis Map"])
        else:
            self._rebuild_plot_inputs_ui(["Right Apex", "Left Apex", "Top Apex"])

        for dd in self.plot_inputs.values():
            dd.blockSignals(True)
            dd.clear()
            if names:
                dd.addItems(names)
            else:
                dd.addItem("-- No maps loaded --")
            dd.blockSignals(False)

    def on_plot_type_changed(self, _index: int):
        self.refresh_map_dropdown_items()

    # ---------- ROI load/save ----------
    def load_roi_from_folder(self, folder: str):
        tried = []
        if self.data_path:
            base = os.path.splitext(os.path.basename(self.data_path))[0]
            candidate = os.path.join(folder, f"{base}_roi.xlsx")
            tried.append(candidate)
            if os.path.exists(candidate):
                self._load_roi_excel(candidate)
                return
        default = os.path.join(folder, "roi_coordinates.xlsx")
        tried.append(default)
        if os.path.exists(default):
            self._load_roi_excel(default)
        else:
            self.log("ROI file not found in: " + " | ".join(tried))

    def browse_and_load_roi_excel(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open ROI Excel", "", "Excel Files (*.xlsx *.xls)")
        if file_path:
            self._load_roi_excel(file_path)

    def _load_roi_excel(self, path: str):
        try:
            df = pd.read_excel(path)
            df = df.rename(columns={c: c.strip().upper() for c in df.columns})
            required = ["X1", "Y1", "X2", "Y2"]
            if not all(col in df.columns for col in required):
                raise ValueError("ROI Excel must contain columns: X1, Y1, X2, Y2")
            out = pd.DataFrame(
                {
                    "X1": df["X1"].astype(int),
                    "Y1": df["Y1"].astype(int),
                    "X2": df["X2"].astype(int),
                    "Y2": df["Y2"].astype(int),
                }
            )
            roi_name_series = None
            if "ROI NAME" in df.columns:
                roi_name_series = df["ROI NAME"].astype(str)
            elif "NAME" in df.columns:
                roi_name_series = df["NAME"].astype(str)

            if roi_name_series is None:
                names = [f"ROI{i+1}" for i in range(len(out))]
            else:
                names = []
                for i, s in enumerate(roi_name_series):
                    s = str(s).strip()
                    if s == "" or s.lower() in ("nan", "none"):
                        names.append(f"ROI{i+1}")
                    else:
                        names.append(s)
            out["ROI Name"] = names

            self.roi_coordinates = self._clip_and_save_roi(out, save_copy=False)
            self.roi_source_path = path
            self.log(f"Loaded ROI coordinates from: {os.path.abspath(path)}")
            self._refresh_delete_roi_dropdown()
            if self._current_kind == "map":
                self._rebuild_qimage_from_map()
            self._update_scaled_pixmap()
            try:
                self.roi_coordinates.to_excel(self.roi_source_path, index=False)
                self.log("(Auto) Added missing ROI names and saved back.")
            except Exception as e:
                self.log(f"[WARN] Could not save auto-named ROIs: {e}")
        except Exception as e:
            self.log(f"[ERROR] Failed to load ROI Excel: {e}")

    def save_roi_excel(self):
        if self.roi_coordinates is None or self.roi_coordinates.empty:
            QMessageBox.information(self, "Save ROI", "No ROI to save.")
            return
        if self.roi_source_path:
            path = self.roi_source_path
        else:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save ROI Excel",
                os.path.join(self._default_folder(), "roi_coordinates.xlsx"),
                "Excel Files (*.xlsx)",
            )
            if not path:
                return
            self.roi_source_path = path
        try:
            self.roi_coordinates.to_excel(path, index=False)
            self.log(f"Saved ROI to: {path}")
            QMessageBox.information(self, "Save ROI", f"Saved ROI to:\n{path}")
        except Exception as e:
            self.log(f"[ERROR] Failed to save ROI: {e}")
            QMessageBox.critical(self, "Save ROI", f"Failed to save ROI:\n{e}")

    def _clip_and_save_roi(self, roi_df: pd.DataFrame, save_copy=True) -> pd.DataFrame:
        if roi_df is None or roi_df.empty:
            return roi_df
        pm = self.image_label.pixmap()
        if pm is not None:
            W, H = pm.width(), pm.height()
        else:
            if self._current_kind == "map" and self._current_map_array is not None:
                H, W = self._current_map_array.shape[:2]
            elif self._current_qimage is not None:
                W, H = (
                    self._current_qimage.width(),
                    self._current_qimage.height(),
                )
            else:
                return roi_df
        xs1 = np.clip(roi_df["X1"].to_numpy(), 0, W)
        ys1 = np.clip(roi_df["Y1"].to_numpy(), 0, H)
        xs2 = np.clip(roi_df["X2"].to_numpy(), 0, W)
        ys2 = np.clip(roi_df["Y2"].to_numpy(), 0, H)
        out = pd.DataFrame(
            {
                "X1": xs1,
                "Y1": ys1,
                "X2": xs2,
                "Y2": ys2,
                "ROI Name": roi_df["ROI Name"],
            }
        )
        if save_copy:
            updated_path = os.path.join(os.getcwd(), "roi_coordinates_updated.xlsx")
            try:
                out.to_excel(updated_path, index=False)
                self.log(f"Saved clipped ROI to: {updated_path}")
            except Exception as e:
                self.log(f"[WARN] Could not save updated ROI Excel: {e}")
        return out

    # ---------- Draw ROI rects ----------
    def _draw_rectangles(self, pixmap: QPixmap) -> QPixmap:
        out = pixmap.copy()
        painter = QPainter(out)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor(self.roi_color), 2, Qt.SolidLine)
        painter.setPen(pen)
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        for idx, row in self.roi_coordinates.iterrows():
            try:
                x1, y1, x2, y2 = (
                    int(row["X1"]),
                    int(row["Y1"]),
                    int(row["X2"]),
                    int(row["Y2"]),
                )
                name = str(row.get("ROI Name", f"ROI{idx+1}"))
                painter.drawRect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
                painter.drawText(x1, max(12, y1 - 5), name)
            except Exception:
                pass
        painter.end()
        return out

    # ---------- ROI events ----------
    def on_mark_roi_toggled(self, enabled: bool):
        self.image_label.set_mark_mode(enabled)
        self.log("Mark ROI mode: " + ("ON" if enabled else "OFF"))

    def on_roi_drawn_from_canvas(self, x1, y1, x2, y2):
        base = "ROI"
        idx = 1
        existing = set(self.roi_coordinates["ROI Name"].astype(str)) if not self.roi_coordinates.empty else set()
        while f"{base}{idx}" in existing or f"{base} {idx}" in existing:
            idx += 1
        name = f"{base}{idx}"
        new_row = {"X1": x1, "Y1": y1, "X2": x2, "Y2": y2, "ROI Name": name}
        self.roi_coordinates = pd.concat([self.roi_coordinates, pd.DataFrame([new_row])], ignore_index=True)
        self._refresh_delete_roi_dropdown()
        self._update_scaled_pixmap()
        self.log(f"Added {name}: ({x1},{y1})–({x2},{y2})")
        if self.roi_source_path:
            try:
                self.roi_coordinates.to_excel(self.roi_source_path, index=False)
                self.log(f"(Auto) Saved ROI to: {self.roi_source_path}")
            except Exception as e:
                self.log(f"[WARN] Auto-save ROI failed: {e}")

    def _refresh_delete_roi_dropdown(self):
        names = ["-- Select ROI to delete --"]
        if self.roi_coordinates is not None and not self.roi_coordinates.empty:
            names += list(self.roi_coordinates["ROI Name"].astype(str))
        self.dd_delete_roi.blockSignals(True)
        self.dd_delete_roi.clear()
        self.dd_delete_roi.addItems(names)
        self.dd_delete_roi.blockSignals(False)

    def delete_selected_roi(self):
        name = self.dd_delete_roi.currentText()
        if not name or name.startswith("--"):
            QMessageBox.information(self, "Delete ROI", "Please select an ROI to delete.")
            return
        mask = self.roi_coordinates["ROI Name"].astype(str) != name
        before = len(self.roi_coordinates)
        self.roi_coordinates = self.roi_coordinates[mask].reset_index(drop=True)
        after = len(self.roi_coordinates)
        self._refresh_delete_roi_dropdown()
        self._update_scaled_pixmap()
        self.log(f"Deleted ROI '{name}'. ({before}->{after})")
        if self.roi_source_path:
            try:
                self.roi_coordinates.to_excel(self.roi_source_path, index=False)
                self.log(f"(Auto) Saved ROI to: {self.roi_source_path}")
            except Exception as e:
                self.log(f"[WARN] Auto-save ROI failed: {e}")

    def clear_all_rois(self):
        n = len(self.roi_coordinates)
        self.roi_coordinates = pd.DataFrame(columns=["X1", "Y1", "X2", "Y2", "ROI Name"])
        self._refresh_delete_roi_dropdown()
        self._update_scaled_pixmap()
        self.log(f"Cleared all ROIs ({n}).")
        if self.roi_source_path:
            try:
                self.roi_coordinates.to_excel(self.roi_source_path, index=False)
                self.log(f"(Auto) Saved ROI to: {self.roi_source_path}")
            except Exception as e:
                self.log(f"[WARN] Auto-save ROI failed: {e}")

    # ---------- Color ----------
    def on_color_changed(self, _index: int):
        text = self.color_dropdown.currentText()
        if text == "Red":
            self.roi_color = Qt.red
        elif text == "Green":
            self.roi_color = Qt.green
        elif text == "Blue":
            self.roi_color = Qt.blue
        elif text == "Black":
            self.roi_color = Qt.black
        else:
            color = QColorDialog.getColor()
            if color.isValid():
                self.roi_color = color
        self._update_scaled_pixmap()

    # ---------- Brightness ----------
    def on_brightness_changed(self, val: int):
        self._gain = 1.0 + (val / 100.0)
        self._gain = max(0.5, min(2.0, self._gain))
        self.lbl_bright_val.setText(f"{val:+d}%")
        if self._current_kind == "map":
            self._rebuild_qimage_from_map()
            self._update_scaled_pixmap()

    # ---------- Save image ----------
    def save_current_image_with_roi(self):
        pm = self.image_label.pixmap()
        if not pm:
            self.log("No background to save.")
            return
        default_name = "background_with_roi.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image with ROI",
            os.path.join(self._default_folder(), default_name),
            "PNG Files (*.png)",
        )
        if not save_path:
            return
        if not pm.save(save_path):
            self.log(f"[ERROR] Failed to save image: {save_path}")
        else:
            self.log(f"Saved image with ROI: {save_path}")

    # ---------- Navigation ----------
    def show_next_image(self):
        image_names = [f"Image: {n}" for n, _ in self.image_list]
        if not image_names:
            return
        cur = self.dd_background.currentText()
        try:
            idx = image_names.index(cur)
        except ValueError:
            idx = -1
        idx = (idx + 1) % len(image_names)
        self.dd_background.setCurrentText(image_names[idx])
        self.show_background_selection()

    def show_previous_image(self):
        image_names = [f"Image: {n}" for n, _ in self.image_list]
        if not image_names:
            return
        cur = self.dd_background.currentText()
        try:
            idx = image_names.index(cur)
        except ValueError:
            idx = 0
        idx = (idx - 1) % len(image_names)
        self.dd_background.setCurrentText(image_names[idx])
        self.show_background_selection()

    # ---------- Upload ----------
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.bmp);;All Files (*)",
        )
        if not file_path:
            return
        qimg = QImage(file_path)
        if qimg.isNull():
            self.log("[ERROR] Failed to load the selected image.")
            return
        width, height = qimg.width(), qimg.height()
        qimg_conv = qimg.convertToFormat(QImage.Format_RGBA8888)
        ptr = qimg_conv.bits().asstring(width * height * 4)
        np_img = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
        np_img = np_img[:, :, :3][:, :, ::-1].copy()  # BGRA->RGB
        self.image_list.append((os.path.basename(file_path), np_img))
        self.refresh_background_dropdown_items()
        self.dd_background.setCurrentText(f"Image: {os.path.basename(file_path)}")
        self._set_current_image(os.path.basename(file_path), np_img)
        self.log(f"Uploaded image: {file_path}")

    # ==========================
    # Plotting utilities (PDF report) — unchanged logic
    # ==========================

    def _roi_mask(self, roi, target_shape=None):
        """Return a boolean mask for the ROI.

        If target_shape is None, the mask is in *pixmap* coordinates
        (for display). If target_shape=(H,W) is given, the mask is
        mapped into the *data array* coordinates accounting for the
        current pixmap scaling/cropping mode.
        """
        pm = self.image_label.pixmap()
        if not pm:
            return None
        pw, ph = pm.width(), pm.height()
        # ROI rectangle in pixmap coordinates
        x1, y1, x2, y2 = (
            int(roi.get("X1", 0)),
            int(roi.get("Y1", 0)),
            int(roi.get("X2", 0)),
            int(roi.get("Y2", 0)),
        )
        x1, x2 = sorted((max(0, x1), min(pw, x2)))
        y1, y2 = sorted((max(0, y1), min(ph, y2)))

        # If no target_shape requested, return mask in pixmap coords (old
        # behaviour)
        if target_shape is None:
            mask = np.zeros((ph, pw), dtype=bool)
            mask[y1:y2, x1:x2] = True
            return mask

        # Data (source) array shape
        Ht, Wt = int(target_shape[0]), int(target_shape[1])

        # Source QImage size (built from the map/image before scaling)
        qw = qh = None
        try:
            if getattr(self, "_current_qimage", None) is not None:
                qw, qh = (
                    self._current_qimage.width(),
                    self._current_qimage.height(),
                )
        except Exception:
            qw = qh = None

        # Fallback: if we don't know the source size, map proportionally from
        # pixmap->target
        if not qw or not qh:
            xs = x1 * (Wt / float(pw or 1))
            xe = x2 * (Wt / float(pw or 1))
            ys = y1 * (Ht / float(ph or 1))
            ye = y2 * (Ht / float(ph or 1))
        else:
            # Compute mapping from pixmap -> source coords depending on scale
            # mode
            scale_mode = getattr(self, "_scale_mode", "Fit (keep AR)") or "Fit (keep AR)"
            if scale_mode.startswith("Fit"):
                # pm is scaled version of source with uniform scale s = pw/qw =
                # ph/qh
                s = pw / float(qw)
                xs_src, xe_src = x1 / s, x2 / s
                ys_src, ye_src = y1 / s, y2 / s
            elif scale_mode.startswith("Fill"):
                # pm is a center-cropped version of the scaled source
                s = max(pw / float(qw), ph / float(qh))
                sw, sh = int(round(qw * s)), int(round(qh * s))
                x_crop = max(0, (sw - pw) // 2)
                y_crop = max(0, (sh - ph) // 2)
                xs_src, xe_src = (x1 + x_crop) / s, (x2 + x_crop) / s
                ys_src, ye_src = (y1 + y_crop) / s, (y2 + y_crop) / s
            else:
                # Stretch (ignore AR)
                sx = pw / float(qw)
                sy = ph / float(qh)
                xs_src, xe_src = x1 / sx, x2 / sx
                ys_src, ye_src = y1 / sy, y2 / sy

            # If target array shape differs from source QImage, scale
            # accordingly
            if (Ht, Wt) != (qh, qw):
                xs = xs_src * (Wt / float(qw))
                xe = xe_src * (Wt / float(qw))
                ys = ys_src * (Ht / float(qh))
                ye = ye_src * (Ht / float(qh))
            else:
                xs, xe, ys, ye = xs_src, xe_src, ys_src, ye_src

        # Clamp to integer pixel bounds in target
        xi1 = max(0, min(Wt, int(math.floor(min(xs, xe)))))
        xi2 = max(0, min(Wt, int(math.ceil(max(xs, xe)))))
        yi1 = max(0, min(Ht, int(math.floor(min(ys, ye)))))
        yi2 = max(0, min(Ht, int(math.ceil(max(ys, ye)))))

        mask = np.zeros((Ht, Wt), dtype=bool)
        if xi2 > xi1 and yi2 > yi1:
            mask[yi1:yi2, xi1:xi2] = True
        return mask

    def _flatten_roi(self, arr, mask):
        # Ensure boolean mask matches array shape
        try:
            if mask.shape != arr.shape[:2]:
                # Attempt proportional remap from mask->arr using simple
                # scaling
                mh, mw = mask.shape[:2]
                ah, aw = arr.shape[:2]
                # Compute bounding box of mask and map to arr
                ys, xs = np.where(mask)
                if ys.size and xs.size:
                    y1, y2 = ys.min(), ys.max() + 1
                    x1, x2 = xs.min(), xs.max() + 1
                    # Map coordinates proportionally
                    y1a = int(math.floor(y1 * (ah / float(mh))))
                    y2a = int(math.ceil(y2 * (ah / float(mh))))
                    x1a = int(math.floor(x1 * (aw / float(mw))))
                    x2a = int(math.ceil(x2 * (aw / float(mw))))
                    y1a = max(0, min(ah, y1a))
                    y2a = max(0, min(ah, y2a))
                    x1a = max(0, min(aw, x1a))
                    x2a = max(0, min(aw, x2a))
                    new_mask = np.zeros((ah, aw), dtype=bool)
                    if y2a > y1a and x2a > x1a:
                        new_mask[y1a:y2a, x1a:x2a] = True
                    mask = new_mask
                else:
                    mask = np.zeros((arr.shape[0], arr.shape[1]), dtype=bool)
            vals = arr[mask].astype(float).ravel()
        except Exception:
            vals = np.asarray(arr, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        return vals

    def _pearson_r(self, a, b):
        if a.size < 2 or b.size < 2:
            return np.nan
        try:
            r = np.corrcoef(a, b)[0, 1]
        except Exception:
            r = np.nan
        return float(r)

    def _spearman_r(self, a, b):
        if a.size < 2 or b.size < 2:
            return np.nan
        try:
            ra = pd.Series(a).rank(method="average").to_numpy()
            rb = pd.Series(b).rank(method="average").to_numpy()
            r = np.corrcoef(ra, rb)[0, 1]
        except Exception:
            r = np.nan
        return float(r)

    def _compute_shared_limits_xy(self, X, Y, rois):
        xmin = ymin = np.inf
        xmax = ymax = -np.inf
        for roi in rois:
            mask = self._roi_mask(roi, X.shape)
            if mask is None:
                continue
            xv = self._flatten_roi(X, mask)
            yv = self._flatten_roi(Y, mask)
            if xv.size and yv.size:
                xmin = min(xmin, float(np.nanmin(xv)))
                xmax = max(xmax, float(np.nanmax(xv)))
                ymin = min(ymin, float(np.nanmin(yv)))
                ymax = max(ymax, float(np.nanmax(yv)))
        if not np.isfinite([xmin, xmax, ymin, ymax]).all():
            return None
        dx = 0.02 * (xmax - xmin + 1e-12)
        dy = 0.02 * (ymax - ymin + 1e-12)
        return (xmin - dx, xmax + dx, ymin - dy, ymax + dy)

    def _compute_shared_limits_dual(self, XB, XL, XR, rois):
        xmin = lmin = rmin = np.inf
        xmax = lmax = rmax = -np.inf
        for roi in rois:
            mask = self._roi_mask(roi, XB.shape)
            if mask is None:
                continue
            xv = self._flatten_roi(XB, mask)
            lv = self._flatten_roi(XL, mask)
            rv = self._flatten_roi(XR, mask)
            if xv.size:
                xmin = min(xmin, float(np.nanmin(xv)))
                xmax = max(xmax, float(np.nanmax(xv)))
            if lv.size:
                lmin = min(lmin, float(np.nanmin(lv)))
                lmax = max(lmax, float(np.nanmax(lv)))
            if rv.size:
                rmin = min(rmin, float(np.nanmin(rv)))
                rmax = max(rmax, float(np.nanmax(rv)))
        if not np.isfinite([xmin, xmax, lmin, lmax, rmin, rmax]).all():
            return None
        dx = 0.02 * (xmax - xmin + 1e-12)
        dl = 0.02 * (lmax - lmin + 1e-12)
        dr = 0.02 * (rmax - rmin + 1e-12)
        return (
            xmin - dx,
            xmax + dx,
            lmin - dl,
            lmax + dl,
            rmin - dr,
            rmax + dr,
        )

    # ---- Ternary helpers ----
    def _normalize_composition(self, A, B, C):
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        C = np.asarray(C, dtype=np.float64)
        for arr in (A, B, C):
            arr[~np.isfinite(arr)] = 0.0
        A = np.maximum(A, 0.0)
        B = np.maximum(B, 0.0)
        C = np.maximum(C, 0.0)
        S = A + B + C
        mask = S > 0.0
        A2 = np.zeros_like(A)
        B2 = np.zeros_like(B)
        C2 = np.zeros_like(C)
        A2[mask] = A[mask] / S[mask]
        B2[mask] = B[mask] / S[mask]
        C2[mask] = C[mask] / S[mask]
        return A2, B2, C2, mask

    def _barycentric_to_cartesian(self, A, B, C):
        x = B + 0.5 * C
        y = (math.sqrt(3) / 2.0) * C
        return x, y

    def _in_triangle(self, A, B, C, eps=1e-12):
        return (A >= -eps) & (B >= -eps) & (C >= -eps) & (np.abs(A + B + C - 1.0) <= 1e-6)

    def _draw_triangle_frame(
        self,
        ax,
        a_label,
        b_label,
        c_label,
        ticks=(0.2, 0.4, 0.6, 0.8),
        tick_len=0.030,
        label_offset=0.038,
        show_grid=False,
    ):
        import math as _math

        import numpy as _np

        h = _math.sqrt(3.0) / 2.0
        B = _np.array([0.0, 0.0])
        A = _np.array([1.0, 0.0])
        C = _np.array([0.5, h])
        centroid = (A + B + C) / 3.0
        pad = max(tick_len + label_offset + 0.02, 0.07)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlim(-pad, 1.0 + pad)
        ax.set_ylim(-pad, h + pad)
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot([B[0], A[0]], [B[1], A[1]], lw=1.1, color="black")
        ax.plot([A[0], C[0]], [A[1], C[1]], lw=1.1, color="black")
        ax.plot([B[0], C[0]], [B[1], C[1]], lw=1.1, color="black")
        if show_grid:
            for t in ticks:
                ax.plot(
                    [1 - t, 0.5 * (1 - t)],
                    [0, h * (1 - t)],
                    lw=0.3,
                    alpha=0.5,
                    color="0.6",
                )
                ax.plot(
                    [0.5 * t, 0.5 + 0.5 * (1 - t)],
                    [h * t, h * (1 - t)],
                    lw=0.3,
                    alpha=0.5,
                    color="0.6",
                )
                ax.plot(
                    [t, 1 - 0.5 * t],
                    [0, h * t],
                    lw=0.3,
                    alpha=0.5,
                    color="0.6",
                )

        def _unit(v):
            n = _np.linalg.norm(v)
            return v / n if n else v

        def _outward_normal(p, edge_vec):
            n = _np.array([-edge_vec[1], edge_vec[0]])
            n = _unit(n)
            if _np.dot(n, centroid - p) > 0:
                n = -n
            return n

        def _draw_edge_ticks(P0, P1):
            d = P1 - P0
            for t in ticks:
                lbl = f"{round(t*100):.0f}"
                p = P0 + t * d
                n = _outward_normal(p, d)
                p2 = p + n * tick_len
                ax.plot([p[0], p2[0]], [p[1], p2[1]], color="black", lw=1.0)
                lp = p + n * label_offset
                ax.text(lp[0], lp[1], lbl, ha="center", va="center", fontsize=8)

        _draw_edge_ticks(B, A)
        _draw_edge_ticks(A, C)
        _draw_edge_ticks(B, C)
        ax.text(1.02, -0.01, a_label, ha="left", va="top", fontsize=9)
        ax.text(-0.02, -0.01, b_label, ha="right", va="top", fontsize=9)
        ax.text(0.5, h + 0.03, c_label, ha="center", va="bottom", fontsize=9)

    def _robust_contour_levels(self, H, n=6):
        import numpy as _np

        if H is None or H.size == 0:
            return None
        pos = H[H > 0]
        if pos.size == 0:
            return None
        vmin = float(pos.min())
        vmax = float(pos.max())
        if not _np.isfinite([vmin, vmax]).all() or vmax <= vmin:
            return None
        qs = _np.linspace(0.50, 0.98, n)
        levels = _np.quantile(pos, qs)
        eps = (vmax - vmin) * 1e-6
        levels = _np.clip(levels, vmin + eps, vmax - eps)
        levels = _np.unique(levels)
        if levels.size < 2:
            levels = _np.linspace(vmin + 0.2 * (vmax - vmin), vmin + 0.95 * (vmax - vmin), n)
            levels = _np.unique(levels)
        return levels if levels.size >= 2 else None

    # ==========================
    # Report generation (unchanged)
    # ==========================

    def generate_report_pdf(self):

        if self.roi_coordinates is None or self.roi_coordinates.empty:
            QMessageBox.warning(self, "Generate Report", "Please load/add at least one ROI.")
            return

        plot_type = self.dd_plot_type.currentText()
        style = getattr(self, "plot_style", None)
        if style is None and hasattr(self, "dd_style"):
            style = self.dd_style.currentText()
        if not style:
            style = "Scatter"
        density_mode = str(style).strip().lower() == "density"

        try:
            if plot_type == "XY plot":
                xmap_name = self.plot_inputs["X Map"].currentText()
                ymap_name = self.plot_inputs["Y Map"].currentText()
                X = self.maps[xmap_name]
                Y = self.maps[ymap_name]
            elif plot_type == "Dual axis plot":
                b_name = self.plot_inputs["Bottom Axis Map"].currentText()
                l_name = self.plot_inputs["Left Axis Map"].currentText()
                r_name = self.plot_inputs["Right Axis Map"].currentText()
                XB = self.maps[b_name]
                XL = self.maps[l_name]
                XR = self.maps[r_name]
            else:
                a_name = self.plot_inputs["Right Apex"].currentText()
                b_name = self.plot_inputs["Left Apex"].currentText()
                c_name = self.plot_inputs["Top Apex"].currentText()
                A = self.maps[a_name]
                B = self.maps[b_name]
                C = self.maps[c_name]
        except KeyError:
            QMessageBox.warning(
                self,
                "Generate Report",
                "Please select valid map names for the chosen plot.",
            )
            return
        except Exception as e:
            QMessageBox.critical(self, "Generate Report", f"Unexpected error: {e}")
            return

        def _strip_ext(s):
            base = os.path.basename(str(s))
            return os.path.splitext(base)[0]

        if plot_type == "XY plot":
            x_label = _strip_ext(xmap_name)
            y_label = _strip_ext(ymap_name)
        elif plot_type == "Dual axis plot":
            b_label = _strip_ext(b_name)
            l_label = _strip_ext(l_name)
            r_label = _strip_ext(r_name)
        else:
            a_label = _strip_ext(a_name)
            b2_label = _strip_ext(b_name)
            c_label = _strip_ext(c_name)

        default_name = f"report_{plot_type.replace(' ', '_').lower()}_{style.lower()}.pdf"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Report PDF",
            os.path.join(self._default_folder(), default_name),
            "PDF Files (*.pdf)",
        )
        if not save_path:
            return

        rois = list(self.roi_coordinates.to_dict(orient="records"))
        chosen_corr = self.dd_corr.currentText()

        shared_xy = None
        shared_dual = None
        if plot_type == "XY plot":
            shared_xy = self._compute_shared_limits_xy(X, Y, rois)
        elif plot_type == "Dual axis plot":
            shared_dual = self._compute_shared_limits_dual(XB, XL, XR, rois)

        ncols, nrows = 3, 4
        per_page = ncols * nrows

        with PdfPages(save_path) as pdf:
            for page_start in range(0, len(rois), per_page):
                page_rois = rois[page_start : page_start + per_page]
                try:
                    fig = plt.figure(figsize=(8.27, 11.69), constrained_layout=True)
                except TypeError:
                    fig = plt.figure(figsize=(8.27, 11.69))

                filled_rows = int(math.ceil(len(page_rois) / float(ncols))) if page_rois else 0

                for i, roi in enumerate(page_rois, start=1):
                    ax = fig.add_subplot(nrows, ncols, i)
                    row_idx = int(math.ceil(i / float(ncols)))
                    is_bottom_row = row_idx == filled_rows
                    col_idx = ((i - 1) % ncols) + 1

                    name = str(roi.get("ROI Name", f"ROI{i}"))
                    mask = self._roi_mask(roi)
                    if mask is None:
                        ax.set_title(f"{name}\n(no background)", fontsize=9, pad=10)
                        ax.axis("off")
                        continue

                    if plot_type == "XY plot":
                        show_left = col_idx == 1
                        self._apply_sci_left_ticks(ax, show_left)
                        if show_left:
                            ax.set_ylabel(y_label)

                        xv = self._flatten_roi(X, mask)
                        yv = self._flatten_roi(Y, mask)

                        if shared_xy is not None:
                            ax.set_xlim(shared_xy[0], shared_xy[1])
                            ax.set_ylim(shared_xy[2], shared_xy[3])

                        if density_mode:
                            hb = ax.hexbin(xv, yv, gridsize=30, mincnt=1, cmap="viridis")
                            cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
                            cb.ax.tick_params(labelsize=7)
                        else:
                            ax.plot(xv, yv, ".", markersize=2)

                        if chosen_corr != "None" and xv.size and yv.size:
                            if chosen_corr == "Pearson":
                                r = self._pearson_r(xv, yv)
                                ax.text(
                                    0.02,
                                    0.98,
                                    f"Pearson r = {r:.3f}",
                                    transform=ax.transAxes,
                                    va="top",
                                    ha="left",
                                    fontsize=7,
                                )
                            else:
                                r = self._spearman_r(xv, yv)
                                ax.text(
                                    0.02,
                                    0.98,
                                    f"Spearman s = {r:.3f}",
                                    transform=ax.transAxes,
                                    va="top",
                                    ha="left",
                                    fontsize=7,
                                )

                        self._apply_sci_bottom_ticks(ax, is_bottom_row)
                        if is_bottom_row:
                            ax.set_xlabel(x_label)
                        ax.set_title(name, fontsize=9, pad=10)

                    elif plot_type == "Dual axis plot":
                        show_left = col_idx == 1
                        show_right = col_idx == 3
                        left_color = "#1f77b4"
                        right_color = "#d62728"
                        self._apply_sci_left_ticks(ax, show_left)
                        if show_left:
                            ax.set_ylabel(l_label, color=left_color)
                        else:
                            ax.set_ylabel("")
                        ax.tick_params(axis="y", colors=left_color)
                        if "left" in ax.spines:
                            ax.spines["left"].set_color(left_color)
                        ax2 = ax.twinx()
                        if show_right:
                            ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
                            fmty = ScalarFormatter(useMathText=True)
                            fmty.set_powerlimits((-2, 3))
                            ax2.yaxis.set_major_formatter(fmty)
                            ax2.set_ylabel(r_label, color=right_color)
                            ax2.tick_params(axis="y", labelright=True, colors=right_color)
                        else:
                            ax2.set_ylabel("")
                            ax2.tick_params(axis="y", labelright=False, colors=right_color)
                        if "right" in ax2.spines:
                            ax2.spines["right"].set_color(right_color)

                        xv = self._flatten_roi(XB, mask)
                        lv = self._flatten_roi(XL, mask)
                        rv = self._flatten_roi(XR, mask)

                        if shared_dual is not None:
                            xmin, xmax, lmin, lmax, rmin, rmax = shared_dual
                        else:
                            xmin = np.nanmin(xv) if xv.size else 0.0
                            xmax = np.nanmax(xv) if xv.size else 1.0
                            lmin = np.nanmin(lv) if lv.size else 0.0
                            lmax = np.nanmax(lv) if lv.size else 1.0
                            rmin = np.nanmin(rv) if rv.size else 0.0
                            rmax = np.nanmax(rv) if rv.size else 1.0
                            dx = 0.02 * (xmax - xmin + 1e-12)
                            dl = 0.02 * (lmax - lmin + 1e-12)
                            dr = 0.02 * (rmax - rmin + 1e-12)
                            xmin, xmax = xmin - dx, xmax + dx
                            lmin, lmax = lmin - dl, lmax + dl
                            rmin, rmax = rmin - dr, rmax + dr

                        ax.set_xlim(xmin, xmax)
                        ax.set_ylim(lmin, lmax)
                        ax2.set_ylim(rmin, rmax)
                        ax2.set_xlim(xmin, xmax)

                        if density_mode:
                            from mpl_toolkits.axes_grid1.inset_locator import (
                                inset_axes as _inset_axes,
                            )

                            ax.patch.set_alpha(0.0)
                            ax2.patch.set_alpha(0.0)
                            hb_left = ax.hexbin(
                                xv,
                                lv,
                                gridsize=32,
                                mincnt=1,
                                cmap="viridis",
                                linewidths=0,
                                alpha=0.90,
                            )
                            x_bins = 40
                            y_bins = 40
                            cs = None
                            if xv.size and rv.size:
                                H, xedges, yedges = np.histogram2d(
                                    xv,
                                    rv,
                                    bins=[x_bins, y_bins],
                                    range=[[xmin, xmax], [rmin, rmax]],
                                )
                                xc = 0.5 * (xedges[:-1] + xedges[1:])
                                yc = 0.5 * (yedges[:-1] + yedges[1:])
                                Xc, Yc = np.meshgrid(xc, yc, indexing="xy")
                                denom = (rmax - rmin) if (rmax - rmin) != 0 else 1.0
                                Yc_left = lmin + (Yc - rmin) * (lmax - lmin) / denom
                                levels = self._robust_contour_levels(H, n=6)
                                if levels is not None:
                                    cs = ax.contour(
                                        Xc,
                                        Yc_left,
                                        H.T,
                                        levels=levels,
                                        cmap="magma",
                                        linewidths=1.0,
                                        alpha=0.95,
                                    )
                            caxL = _inset_axes(
                                ax,
                                width="3.0%",
                                height="85%",
                                loc="center left",
                                bbox_to_anchor=(1.02, 0.0, 1, 1),
                                bbox_transform=ax.transAxes,
                                borderpad=0,
                            )
                            caxR = _inset_axes(
                                ax,
                                width="3.0%",
                                height="85%",
                                loc="center left",
                                bbox_to_anchor=(1.06, 0.0, 1, 1),
                                bbox_transform=ax.transAxes,
                                borderpad=0,
                            )
                            cbL = fig.colorbar(hb_left, cax=caxL)
                            cbL.set_label(l_label, fontsize=7)
                            if cs is not None:
                                cbR = fig.colorbar(cs, cax=caxR)
                            else:
                                import matplotlib as mpl
                                from matplotlib.cm import ScalarMappable

                                dummy = ScalarMappable(
                                    norm=mpl.colors.Normalize(0, 1),
                                    cmap="magma",
                                )
                                cbR = fig.colorbar(dummy, cax=caxR)
                            cbR.set_label(r_label, fontsize=7)
                            cbL.ax.tick_params(labelsize=7)
                            cbL.set_label(l_label, fontsize=7)
                            cbR.ax.tick_params(labelsize=7)
                            cbR.set_label(r_label, fontsize=7)
                        else:
                            ax.plot(xv, lv, ".", markersize=2, color="#1f77b4")
                            ax2.plot(xv, rv, ".", markersize=2, color="#d62728")
                            if chosen_corr != "None" and xv.size:
                                if chosen_corr == "Pearson":
                                    r_pr = self._pearson_r(xv, rv) if rv.size else np.nan
                                    r_pl = self._pearson_r(xv, lv) if lv.size else np.nan
                                    txt = f"Pearson ({r_label}) r = {r_pr:.3f}\n" f"Pearson ({l_label}) r = {r_pl:.3f}"
                                else:
                                    r_sr = self._spearman_r(xv, rv) if rv.size else np.nan
                                    r_sl = self._spearman_r(xv, lv) if lv.size else np.nan
                                    txt = (
                                        f"Spearman ({r_label}) s = {r_sr:.3f}\n" f"Spearman ({l_label}) s = {r_sl:.3f}"
                                    )
                                ax.text(
                                    0.02,
                                    0.98,
                                    txt,
                                    transform=ax.transAxes,
                                    va="top",
                                    ha="left",
                                    fontsize=7,
                                )
                        self._apply_sci_bottom_ticks(ax, is_bottom_row)
                        if is_bottom_row:
                            ax.set_xlabel(b_label)
                        ax.set_title(name, fontsize=9, pad=10)

                    else:  # Triangular
                        self._draw_triangle_frame(ax, a_label, b2_label, c_label, show_grid=False)
                        Av = self._flatten_roi(A, mask)
                        Bv = self._flatten_roi(B, mask)
                        Cv = self._flatten_roi(C, mask)
                        A2, B2, C2, valid = self._normalize_composition(Av, Bv, Cv)
                        if valid.any():
                            x, y = self._barycentric_to_cartesian(A2, B2, C2)
                            keep = self._in_triangle(A2, B2, C2)
                            x = x[keep]
                            y = y[keep]
                        else:
                            x = y = np.array([])
                        if x.size:
                            if density_mode:
                                hb = ax.hexbin(x, y, gridsize=30, mincnt=1, cmap="viridis")
                                cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
                                cb.ax.tick_params(labelsize=7)
                            else:
                                ax.scatter(x, y, s=2.0, alpha=0.6)
                        ax.set_title(name, fontsize=9, pad=10)

                try:
                    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
                except Exception:
                    fig.subplots_adjust(
                        left=0.07,
                        right=0.93,
                        top=0.96,
                        bottom=0.07,
                        wspace=0.30,
                        hspace=0.40,
                    )
                pdf.savefig(fig)
                plt.close(fig)

        self.log(f"Saved report PDF: {save_path}")
        QMessageBox.information(self, "Report", f"Report saved:\n{save_path}")

    # ---------- Ticks helpers ----------
    def _apply_sci_bottom_ticks(self, ax, is_bottom_row: bool):
        if not is_bottom_row:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-2, 3))
            ax.xaxis.set_major_formatter(fmt)

    def _apply_sci_left_ticks(self, ax, show_left: bool):
        if not show_left:
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
            fmty = ScalarFormatter(useMathText=True)
            fmty.set_powerlimits((-2, 3))
            ax.yaxis.set_major_formatter(fmty)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    path = None
    if len(sys.argv) > 1:
        path = sys.argv[1]
    viewer = VisualizationStep(data_path=path)
    viewer.show()
    sys.exit(app.exec_())
