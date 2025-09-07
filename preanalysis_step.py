import os
import re
import subprocess
import sys

import cv2
import h5py
import numpy as np
from PIL import Image
from PyQt5.QtCore import QEvent, QPoint, QSize, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap, QPolygon
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def lock_groupbox_size(gb: QGroupBox):
    # compute final size from contents
    gb.adjustSize()
    hint = gb.sizeHint()
    gb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    gb.setFixedSize(hint)  # lock both width and height


# ---------------------------- GroupBox & Banner helpers -----------------
def apply_groupbox_style(gb: QGroupBox):
    gb.setStyleSheet(
        """
        QGroupBox {
            font: 600 13px "Arial";
            border: 1px solid #c8c8c8;
            border-radius: 8px;
            /* Leave space for the title; use em so it scales with font/DPI */
            margin-top: 1.3em;
            padding: 12px 10px 10px 10px;
            background: #ffffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;  /* anchor title properly */
            left: 8px;
            /* No negative top offset; prevents clipping */
            padding: 0 6px;
            background: #fafafa;
        }
    """
    )


# ---------------------------- Arrow widget ----------------------------


class ArrowLabel(QLabel):
    def __init__(self, direction="down", w=200, h=50, parent=None):
        super().__init__(parent)
        self._direction = str(direction).lower()
        self._w = int(w)
        self._h = int(h)
        self.setFixedSize(self._w, self._h)

    def setDirection(self, direction: str):
        self._direction = str(direction).lower()
        self.update()

    def sizeHint(self):
        return QSize(self._w, self._h)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.black)

        w = self.width()
        h = self.height()
        cx = w * 0.5
        cy = h * 0.5

        # Triangle size relative to the shorter side
        s = min(w, h) * 0.70  # overall triangle "span"
        half_base = max(2, int(s * 0.40))
        half_height = max(3, int(s * 0.50))

        d = self._direction
        if d == "right":
            p1 = QPoint(int(cx + half_height), int(cy))  # apex
            p2 = QPoint(int(cx - half_height), int(cy - half_base))  # base top
            p3 = QPoint(int(cx - half_height), int(cy + half_base))  # base bottom
        elif d == "left":
            p1 = QPoint(int(cx - half_height), int(cy))
            p2 = QPoint(int(cx + half_height), int(cy - half_base))
            p3 = QPoint(int(cx + half_height), int(cy + half_base))
        elif d == "up":
            p1 = QPoint(int(cx), int(cy - half_height))
            p2 = QPoint(int(cx - half_base), int(cy + half_height))
            p3 = QPoint(int(cx + half_base), int(cy + half_height))
        else:  # "down"
            p1 = QPoint(int(cx), int(cy + half_height))
            p2 = QPoint(int(cx - half_base), int(cy - half_height))
            p3 = QPoint(int(cx + half_base), int(cy - half_height))

        painter.drawPolygon(QPolygon([p1, p2, p3]))


# ---------------------------- Main Widget ----------------------------


class PreanalysisStep(QWidget):
    def __init__(self):
        super().__init__()
        self.working_dir = os.environ.get("MAPEX_WORKDIR", "")

        # Image interaction state
        self.selected_color_rgb = None
        self.locked_point_img = None
        self.current_image_path = None
        self.original_image = None  # BGR numpy array
        self.scale = 1.0
        self.padding_x = 0.0
        self.padding_y = 0.0
        self._events_installed = False

        # Data for HDF5
        self.text_files = []
        self.image_files = []
        # (retained, but not used for HDF5 creation)
        self.cleaned_text_data = {}

        # XY → ASCII
        self.xy_files = []

        self.data_h5_path = None

        self.init_ui()
        self.setMinimumSize(960, 600)
        # Data Correction
        self.corr_files = []
        self.corr_data = {}  # path -> np.ndarray

    # ---------------------------- UI ---------------------------------

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Top-level horizontal splitter: [Log] | [Right Pane] ---
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(top_splitter, 1)

        # -------- Left: Log panel --------
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)

        self.text_area = QTextEdit(readOnly=True)
        self.text_area.setPlaceholderText("Log messages will appear here.")
        self.text_area.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)
        self.text_area.setMinimumWidth(220)
        log_layout.addWidget(self.text_area)
        top_splitter.addWidget(log_container)

        # -------- Right: Everything else (image tools + image + HDF5 + XY) ---
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        top_splitter.addWidget(right_container)

        top_splitter.setStretchFactor(0, 1)  # log
        top_splitter.setStretchFactor(1, 3)  # right
        top_splitter.setSizes([320, 880])

        # --- Image tools + display splitter: [Tools] | [Image] ---
        img_splitter = QSplitter(Qt.Horizontal)
        img_splitter.setChildrenCollapsible(False)

        img_splitter.setStretchFactor(0, 1)
        img_splitter.setStretchFactor(1, 4)
        img_splitter.setSizes([300, 900])
        right_layout.addWidget(img_splitter, 3)

        # Left: Tools column
        tools_container = QWidget()
        tools_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        tools_container.setMinimumWidth(240)
        tools_layout = QVBoxLayout(tools_container)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(10)

        def mk_btn(text, tip=None):
            BTN_WIDTH = 200
            b = QPushButton(text)
            b.setFont(QFont("Arial", 10))
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            b.setFixedWidth(BTN_WIDTH)
            b.setMinimumHeight(30)
            if tip:
                b.setToolTip(tip)
            return b

        # --- Group 1: Image & ROI ---
        grp_img = QGroupBox("Image ROI")
        apply_groupbox_style(grp_img)
        g1 = QVBoxLayout(grp_img)
        g1.setSpacing(8)

        btn_upload_img = mk_btn("Upload Image")
        btn_upload_img.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_upload_img.clicked.connect(self.upload_image)
        g1.addWidget(btn_upload_img, 0, Qt.AlignHCenter)

        arrow1 = ArrowLabel("down", 24, 24)
        arrow1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        g1.addWidget(arrow1, 0, Qt.AlignHCenter)

        btn_choose_color = mk_btn(
            "Choose Colour of ROI",
            "Hover + click on the image to lock a color for ROI extraction.",
        )
        btn_choose_color.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_choose_color.clicked.connect(self.enable_color_selection)
        g1.addWidget(btn_choose_color, 0, Qt.AlignHCenter)

        arrow2 = ArrowLabel("down", 24, 24)
        arrow2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        g1.addWidget(arrow2, 0, Qt.AlignHCenter)

        btn_extract = mk_btn("Extract")
        btn_extract.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_extract.clicked.connect(self.extract_roi)
        g1.addWidget(btn_extract, 0, Qt.AlignHCenter)

        arrow3 = ArrowLabel("down", 24, 24)
        arrow3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        g1.addWidget(arrow3, 0, Qt.AlignHCenter)

        btn_resize = mk_btn(
            "Reshaped Images \nby X-ray Maps",
            "Resize the previously clipped image to match an X-ray map size.",
        )
        btn_resize.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_resize.clicked.connect(self.resize_images)
        g1.addWidget(btn_resize, 0, Qt.AlignHCenter)

        # Hover + selected color row
        color_row = QHBoxLayout()
        color_row.setSpacing(8)
        self.color_info_label = QLabel("Hover color")
        self.color_info_label.setAlignment(Qt.AlignCenter)
        self.color_info_label.setStyleSheet("border:1px solid #444; font-size:13px; padding:4px; background:#fafafa;")
        self.color_info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.selected_color_swatch = QLabel()
        self.selected_color_swatch.setFixedSize(32, 32)
        self.selected_color_swatch.setStyleSheet("border:1px solid #444; background:#ffffff;")
        self.selected_color_label = QLabel("Selected")
        self.selected_color_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.selected_color_label.setStyleSheet("font-size:13px; padding-left:4px;")
        self.selected_color_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        color_row.addWidget(self.color_info_label)
        color_row.addWidget(self.selected_color_swatch)
        color_row.addWidget(self.selected_color_label)
        g1.addLayout(color_row)

        tools_layout.addWidget(grp_img)

        # --- Group 2: Data Correction ---
        grp_corr = QGroupBox("Data Correction")
        apply_groupbox_style(grp_corr)
        g2 = QVBoxLayout(grp_corr)
        g2.setSpacing(8)

        self.btn_corr_upload = mk_btn("Upload Data")
        self.btn_corr_upload.clicked.connect(self.upload_corr_data)
        g2.addWidget(self.btn_corr_upload, 0, Qt.AlignHCenter)

        g2.addWidget(ArrowLabel("down", 24, 24), 0, Qt.AlignHCenter)

        self.btn_corr_run = mk_btn("Correct Data")
        self.btn_corr_run.setToolTip("Open correction dialog (EPMA/WDS, EDS, micro-XRF-EDS, TEM-EDS, Synchrotron).")
        self.btn_corr_run.clicked.connect(self.open_correction_dialog)
        g2.addWidget(self.btn_corr_run, 0, Qt.AlignHCenter)

        tools_layout.addWidget(grp_corr)

        # --- Group 3: Convert XY → 2D grid ---
        grp_xy = QGroupBox("XY to 2D Format Converter")
        apply_groupbox_style(grp_xy)
        g3 = QVBoxLayout(grp_xy)
        g3.setSpacing(8)

        btn_xy_upload = mk_btn("Upload XY file(s)")
        btn_xy_upload.clicked.connect(self.upload_xy_files)
        g3.addWidget(btn_xy_upload, 0, Qt.AlignHCenter)

        g3.addWidget(ArrowLabel("down", 24, 24), 0, Qt.AlignHCenter)

        btn_xy_save = mk_btn(
            "Save as 2D grid format",
            "If multiple files are uploaded, choose an output folder; each will be saved as *_grid.txt",
        )
        btn_xy_save.clicked.connect(self.save_xy_as_ascii)
        g3.addWidget(btn_xy_save, 0, Qt.AlignHCenter)

        tools_layout.addWidget(grp_xy, 0)

        # Keep row tidy
        tools_layout.addStretch(1)

        # -----------------------------------------------

        tools_scroll = QScrollArea()
        tools_scroll.setWidgetResizable(True)
        tools_scroll.setFrameShape(QScrollArea.NoFrame)
        tools_scroll.setWidget(tools_container)
        tools_scroll.setMinimumWidth(260)  # includes scroll bars, etc.
        img_splitter.addWidget(tools_scroll)

        # Right: Image display
        self.image_display_label = QLabel("No Image Uploaded")
        self.image_display_label.setAlignment(Qt.AlignCenter)
        self.image_display_label.setStyleSheet("border:1px solid #444; background:#fafafa;")
        self.image_display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_display_label.setMouseTracking(True)
        self.image_display_label.setMinimumSize(320, 240)
        self.image_display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_splitter.addWidget(self.image_display_label)

        img_splitter.setStretchFactor(0, 1)  # tools
        img_splitter.setStretchFactor(1, 4)  # image display

        # --- HDF5 creation & XY helpers (grouped) ---
        # Create HDF5 group
        grp_hdf = QGroupBox("Create HDF5 File")
        apply_groupbox_style(grp_hdf)
        hdf_box = QVBoxLayout(grp_hdf)
        hdf_box.setSpacing(8)
        hdf_box.setAlignment(Qt.AlignLeft)

        hdf_row = QHBoxLayout()
        hdf_row.setSpacing(8)
        btn_up_txt = QPushButton("Upload Text File(s)")
        btn_up_txt.setFont(QFont("Arial", 10))
        btn_up_txt.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_up_txt.setFixedWidth(200)
        btn_up_txt.setMinimumHeight(30)
        btn_up_txt.clicked.connect(self.upload_text_file)
        hdf_row.addWidget(btn_up_txt)

        hdf_row.addWidget(ArrowLabel("right", 24, 24))

        btn_up_imgs = QPushButton("Upload Reshaped Image(s)")
        btn_up_imgs.setFont(QFont("Arial", 10))
        btn_up_imgs.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_up_imgs.setFixedWidth(300)
        btn_up_imgs.setMinimumHeight(30)
        btn_up_imgs.clicked.connect(self.upload_reshaped_images)
        hdf_row.addWidget(btn_up_imgs)

        hdf_row.addWidget(ArrowLabel("right", 24, 24))

        btn_create = QPushButton("Create HDF5 File")
        btn_create.setFont(QFont("Arial", 10))
        btn_create.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_create.setFixedWidth(200)
        btn_create.setMinimumHeight(30)
        btn_create.clicked.connect(self.create_combine_file)
        hdf_row.addWidget(btn_create)
        hdf_box.addLayout(hdf_row)
        right_layout.addWidget(grp_hdf, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Preanalysis Step")

        QTimer.singleShot(
            0,
            lambda: (
                lock_groupbox_size(grp_img),
                lock_groupbox_size(grp_corr),
                lock_groupbox_size(grp_xy),
            ),
        )

    def set_working_dir(self, folder: str):
        """Called by main.py to enforce a single working directory for
        all outputs."""
        self.working_dir = os.path.abspath(folder) if folder else ""

    def _wd(self, *parts: str) -> str:
        """Join path(s) relative to the working dir (or CWD if not
        set)."""
        base = self.working_dir or os.getcwd()
        return os.path.join(base, *parts)

    # ---------------------------- Image utils -------------------------
    def _update_scaled_params(self):
        if self.original_image is None:
            return
        label_width = self.image_display_label.width()
        label_height = self.image_display_label.height()
        h, w = self.original_image.shape[:2]

        if label_width / max(label_height, 1) > w / max(h, 1):
            self.scale = label_height / h
            self.padding_x = (label_width - w * self.scale) / 2.0
            self.padding_y = 0.0
        else:
            self.scale = label_width / w
            self.padding_x = 0.0
            self.padding_y = (label_height - h * self.scale) / 2.0

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_params()
        # re-scale currently shown image to fit label on resize
        if self.current_image_path and os.path.isfile(self.current_image_path):
            pixmap = QPixmap(self.current_image_path)
            self.image_display_label.setPixmap(
                pixmap.scaled(
                    self.image_display_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    def upload_corr_data(self):
        try:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Map(s) for Correction",
                "",
                "Text Files (*.txt *.csv *.dat *.asc *.tsv *.xy);;All Files (*)",
            )
            if not files:
                QMessageBox.warning(
                    self,
                    "No Files Selected",
                    "Please select at least one numeric 2D file.",
                )
                return
            self.corr_files = files
            self.corr_data.clear()
            self.text_area.append("Loaded for correction:")
            for f in files:
                arr, removed = self._load_2d_text_clean(f)
                if arr is None:
                    self.text_area.append(f"  ⚠️ Skipped (not a uniform 2D numeric matrix): {os.path.basename(f)}")
                    continue
                self.corr_data[f] = arr
                self.text_area.append(f"  ✓ {os.path.basename(f)}  shape={arr.shape}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while selecting data:\n{e}")

    def open_correction_dialog(self):
        if not self.corr_data:
            QMessageBox.warning(self, "No Data", "Upload data for correction first.")
            return
        dlg = CorrectionDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            p = dlg.params
            # Hard warning about cps double correction
            if p["already_cps"]:
                QMessageBox.information(
                    self,
                    "Pass-through",
                    "Input marked as cps. No correction will be applied; data will be saved as-is.",
                )
            try:
                self._apply_data_correction_and_save(p)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Correction failed:\n{e}")

    def _apply_data_correction_and_save(self, p):
        """Apply instrument-appropriate correction and save outputs as
        *_corrected.txt."""

        # Helper conversions
        def _tau_seconds(val, unit):
            return (val * 1e-6) if unit == "µs" else (val * 1e-9)

        # Choose save mode (one file -> ask filename; many -> ask folder)
        if len(self.corr_data) == 1:
            in_path = next(iter(self.corr_data.keys()))
            suggested = self._wd(os.path.splitext(os.path.basename(in_path))[0] + "_corrected.txt")
            out_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Corrected Map As",
                suggested,
                "Text Files (*.txt *.asc);;All Files (*)",
            )
            if not out_path:
                return
            outputs = {in_path: out_path}
        else:
            out_dir = QFileDialog.getExistingDirectory(
                self,
                "Choose Output Folder for Corrected Maps",
                self.working_dir or "",
            )
            if not out_dir:
                return
            outputs = {
                fp: os.path.join(
                    out_dir,
                    os.path.splitext(os.path.basename(fp))[0] + "_corrected.txt",
                )
                for fp in self.corr_data.keys()
            }

        # Parameters
        instr = p["instrument"]
        in_units = p["in_units"]  # 'Counts...' or 'Counts per second (cps)'
        method = p["method"]
        real_t = float(p["real_time"])
        live_t = float(p["live_time"])
        icr = float(p["icr"])
        ocr = float(p["ocr"])
        tau = _tau_seconds(p["tau"], p["tau_unit"])
        out_units = p["out_units"]  # 'cps' or 'Counts (use Real time)'
        already_cps = bool(p["already_cps"])

        # Sanity
        if method == "Livetime ratio (counts × real/live)":
            if live_t <= 0 or real_t <= 0:
                raise ValueError("Real time and Live time must be > 0 for livetime ratio.")
        if method == "ICR/OCR scaling":
            if icr <= 0 or ocr <= 0:
                raise ValueError("ICR and OCR must be > 0 for ICR/OCR scaling.")
        if method == "WDS non-paralyzable (τ)":
            if tau <= 0:
                raise ValueError("τ must be > 0 for WDS correction.")

        # Process each file
        count_done = 0
        for fp, arr in self.corr_data.items():
            arr = np.asarray(arr, dtype=float)
            out = None
            notes = []

            if already_cps and in_units.startswith("Counts per second"):
                # pass-through
                out = arr.copy()
                notes.append("pass-through (already cps)")
            else:
                # Branch by method
                if method == "Livetime ratio (counts × real/live)":
                    if in_units.startswith("Counts per second"):
                        # Most cps are already normalized by live_time; warn &
                        # pass-through
                        notes.append("input cps + livetime ratio selected -> pass-through to avoid double correction")
                        out = arr.copy()
                    else:
                        # counts -> corrected counts
                        out_counts = arr * (real_t / live_t)
                        if out_units.startswith("Counts per second"):
                            out = out_counts / max(real_t, 1e-12)
                            notes.append("counts × real/live -> cps")
                        else:
                            out = out_counts
                            notes.append("counts × real/live -> counts")
                elif method == "ICR/OCR scaling":
                    scale = icr / ocr
                    if in_units.startswith("Counts per second"):
                        out_cps = arr * scale
                        if out_units.startswith("Counts per second"):
                            out = out_cps
                            notes.append("cps × (ICR/OCR) -> cps")
                        else:
                            out = out_cps * max(real_t, 1e-12)
                            notes.append("cps × (ICR/OCR) -> counts via real time")
                    else:
                        out_counts = arr * scale
                        if out_units.startswith("Counts per second"):
                            out = out_counts / max(real_t, 1e-12)
                            notes.append("counts × (ICR/OCR) -> cps via real time")
                        else:
                            out = out_counts
                            notes.append("counts × (ICR/OCR) -> counts")
                elif method == "WDS non-paralyzable (τ)":
                    # Need cps for the model
                    if in_units.startswith("Counts per second"):
                        icps = arr
                    else:
                        # counts -> cps
                        base_time = live_t if live_t > 0 else real_t
                        if base_time <= 0:
                            raise ValueError("Need live/real time to convert counts to cps for WDS model.")
                        icps = arr / base_time
                    # Apply I = i / (1 - i*tau)
                    denom = 1.0 - icps * tau
                    bad = denom <= 0
                    if np.any(bad):
                        # Avoid invalid; clip at tiny positive denom
                        denom = np.where(bad, 1e-12, denom)
                        notes.append(f"WDS: {np.count_nonzero(bad)} pixels exceeded safe rate (i*τ≥1); clipped.")
                    Icps = icps / denom
                    if out_units.startswith("Counts per second"):
                        out = Icps
                        notes.append("WDS τ model -> cps")
                    else:
                        out = Icps * max(real_t, 1e-12)
                        notes.append("WDS τ model -> counts via real time")
                else:
                    raise ValueError(f"Unknown method: {method}")

            # Save
            np.savetxt(outputs[fp], out, fmt="%.10e", delimiter=" ")
            count_done += 1
            self.text_area.append(
                f"Corrected: {os.path.basename(fp)}  ->  {outputs[fp]}\n"
                f"Method={method}, Instrument={instr}, In={in_units}, Out={out_units}; {'; '.join(notes)}"
            )

        QMessageBox.information(self, "Done", f"Corrected {count_done} map(s).")

    def upload_image(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select an Image",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff);;All Files (*.*)",
            )
            if not file_path:
                QMessageBox.warning(self, "No File Selected", "Please select an image file.")
                return
            self.current_image_path = file_path
            self.original_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if self.original_image is None:
                QMessageBox.critical(self, "Error", "Failed to load the image.")
                return

            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                self.image_display_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.image_display_label.setPixmap(scaled_pixmap)
            self.image_display_label.setAlignment(Qt.AlignCenter)

            self._update_scaled_params()
            self.text_area.append(f"Image loaded: {file_path}")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while uploading the image:\n{e}",
            )

    def enable_color_selection(self):
        if self.original_image is None:
            QMessageBox.warning(self, "No Image", "Please upload an image first.")
            return

        if not self._events_installed:
            self.image_display_label.installEventFilter(self)
            self._events_installed = True
        self.image_display_label.setMouseTracking(True)
        self.image_display_label.setCursor(Qt.CrossCursor)
        self.text_area.append("Hover over the image to preview colors. Click to lock the color for ROI extraction.")

    def eventFilter(self, source, event):
        if source is self.image_display_label and self.original_image is not None:
            if event.type() == QEvent.MouseMove:
                self._update_color_info(event.pos(), lock=False)
                return True
            elif event.type() == QEvent.MouseButtonPress:
                self._update_color_info(event.pos(), lock=True)
                return True
        return super().eventFilter(source, event)

    def _label_to_image_xy(self, pos):
        x_label = pos.x()
        y_label = pos.y()
        x_img = int((x_label - self.padding_x) / max(self.scale, 1e-6))
        y_img = int((y_label - self.padding_y) / max(self.scale, 1e-6))
        return x_img, y_img

    def _update_color_info(self, pos, lock=False):
        x_img, y_img = self._label_to_image_xy(pos)
        h, w = self.original_image.shape[:2]
        if 0 <= x_img < w and 0 <= y_img < h:
            b, g, r = self.original_image[y_img, x_img].tolist()
            hex_color = QColor(r, g, b).name()

            self.color_info_label.setStyleSheet(
                f"border:1px solid #444; font-size:13px; padding:4px; background:{hex_color};"
            )
            self.color_info_label.setText(f"Hover: {r},{g},{b}")

            if lock:
                self.selected_color_rgb = (r, g, b)
                self.locked_point_img = (x_img, y_img)
                self.selected_color_swatch.setStyleSheet(f"border:1px solid #444; background:{hex_color};")
                self.selected_color_label.setText(f"Selected Colour: {r},{g},{b}")
                self.text_area.append(
                    f"Locked color: RGB={self.selected_color_rgb}, HEX={hex_color} at (x={x_img}, y={y_img})"
                )
        else:
            self.color_info_label.setStyleSheet("border:1px solid #444; font-size:13px; padding:4px;")
            self.color_info_label.setText("Hover color")

    # ---------------------------- ROI extraction (Lab flood-fill) -----------

    def _find_nearby_seed_in_mask(self, mask, seed_xy, search_radius=8):
        x0, y0 = seed_xy
        h, w = mask.shape
        if 0 <= x0 < w and 0 <= y0 < h and mask[y0, x0] != 0:
            return x0, y0
        for r in range(1, search_radius + 1):
            y_min = max(0, y0 - r)
            y_max = min(h, y0 + r + 1)
            x_min = max(0, x0 - r)
            x_max = min(w, x0 + r + 1)
            roi = mask[y_min:y_max, x_min:x_max]
            ys, xs = np.where(roi != 0)
            if ys.size:
                dy = ys - (y0 - y_min)
                dx = xs - (x0 - x_min)
                idx = np.argmin(dx * dx + dy * dy)
                return int(x_min + xs[idx]), int(y_min + ys[idx])
        return None

    def extract_roi(self):
        if self.original_image is None:
            QMessageBox.warning(self, "No Image", "Please upload an image first.")
            return
        if self.locked_point_img is None:
            QMessageBox.warning(
                self,
                "No Color Selected",
                "Hover and click on the image to lock a color first.",
            )
            return

        lab = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2Lab)

        sx, sy = self.locked_point_img
        h, w = lab.shape[:2]
        if not (0 <= sx < w and 0 <= sy < h):
            QMessageBox.warning(
                self,
                "Click out of bounds",
                "Please click inside the image area.",
            )
            return

        tolerances = [(6, 6, 6), (10, 12, 12), (14, 16, 16)]
        component_mask = None
        bbox = None

        for loL, loA, loB in tolerances:
            mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            retval, lab_filled, mask_out, rect = cv2.floodFill(
                lab.copy(),
                mask,
                (sx, sy),
                (0, 0, 0),
                loDiff=(loL, loA, loB),
                upDiff=(loL, loA, loB),
                flags=cv2.FLOODFILL_FIXED_RANGE,
            )
            filled = mask[1:-1, 1:-1]
            if retval > 0 and np.any(filled):
                component_mask = filled
                bbox = rect
                break

        if component_mask is None:
            hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
            b, g, r = self.original_image[sy, sx].tolist()
            selected_rgb_np = np.array([[(r, g, b)]], dtype=np.uint8)
            selected_hsv = cv2.cvtColor(selected_rgb_np, cv2.COLOR_RGB2HSV)[0, 0]
            sel = selected_hsv.astype(int)
            lower = np.array(
                [
                    max(sel[0] - 18, 0),
                    max(sel[1] - 50, 0),
                    max(sel[2] - 50, 0),
                ],
                dtype=np.uint8,
            )
            upper = np.array(
                [
                    min(sel[0] + 18, 179),
                    min(sel[1] + 50, 255),
                    min(sel[2] + 50, 255),
                ],
                dtype=np.uint8,
            )
            pre_mask = cv2.inRange(hsv, lower, upper)
            near = self._find_nearby_seed_in_mask(pre_mask, (sx, sy), search_radius=10)
            if near is None:
                QMessageBox.warning(
                    self,
                    "No matching region",
                    "No region near the clicked color could be isolated. Try clicking on a solid area.",
                )
                return
            sx, sy = near
            mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            retval, lab_filled, mask_out, rect = cv2.floodFill(
                lab.copy(),
                mask,
                (sx, sy),
                (0, 0, 0),
                loDiff=(10, 12, 12),
                upDiff=(10, 12, 12),
                flags=cv2.FLOODFILL_FIXED_RANGE,
            )
            filled = mask[1:-1, 1:-1]
            if retval == 0 or not np.any(filled):
                QMessageBox.warning(
                    self,
                    "No matching region",
                    "Could not isolate a connected region at/near the clicked location.",
                )
                return
            component_mask = filled
            bbox = rect

        x, y, bw, bh = bbox
        if bw < 2 or bh < 2:
            QMessageBox.warning(
                self,
                "Small ROI",
                "The detected region is too small to extract.",
            )
            return

        margin = 1
        x = max(x + margin, 0)
        y = max(y + margin, 0)
        bw = max(bw - 2 * margin, 1)
        bh = max(bh - 2 * margin, 1)

        roi = self.original_image[y : y + bh, x : x + bw]
        output_path = self._wd("clipped_image.tiff")
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        try:
            Image.fromarray(roi_rgb).save(output_path, format="TIFF")
        except Exception:
            cv2.imwrite(output_path, cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))

        self.text_area.append(
            f"Lab flood-fill from seed=({sx},{sy}). Saved ROI to {output_path}. "
            f"Bounds (x,y,w,h)=({x},{y},{bw},{bh})"
        )

    # ---------------------------- Resize by map ----------------------------

    def resize_images(self):
        try:
            map_file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Map File",
                "",
                "Text Files (*.txt *.csv *.dat *.asc *.tsv *.xy);;All Files (*)",
            )
            if not map_file_path:
                QMessageBox.warning(self, "No File Selected", "Please select a map file.")
                return

            map_data, _removed = self._load_2d_text_clean(map_file_path)
            if map_data is None:
                QMessageBox.warning(
                    self,
                    "Invalid File",
                    "The selected file does not contain a valid 2D numeric matrix.",
                )
                return

            new_h, new_w = map_data.shape

            clipped_image_path = self._wd("clipped_image.tiff")
            if not os.path.isfile(clipped_image_path):
                QMessageBox.warning(
                    self,
                    "File Not Found",
                    f"{clipped_image_path} does not exist.",
                )
                return

            img = Image.open(clipped_image_path).convert("RGB")
            img_np = np.array(img)
            reshaped = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            output_path = self._wd("reshaped_clipped_image.tiff")
            Image.fromarray(reshaped).save(output_path, format="TIFF")
            self.text_area.append(f"Reshaped image saved as {output_path} (to {new_w}x{new_h})")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while resizing the image:\n{e}",
            )

    # ---------------------------- HDF5 I/O ----------------------------

    def upload_text_file(self):
        try:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Text Files for X-ray Maps",
                "",
                "Text Files (*.txt *.csv *.dat *.asc *.tsv *.xy);;All Files (*)",
            )
            if not files:
                QMessageBox.warning(
                    self,
                    "No Files Selected",
                    "Please select at least one text file.",
                )
                return

            self.text_files = files
            self.text_area.append("Loaded text files:")
            for f in files:
                self.text_area.append("  - " + f)
                arr, removed = self._load_2d_text_clean(f)
                if removed:
                    for line in removed[:5]:  # preview a few removed headers
                        self.text_area.append(f"Removed header \n {line}")
                if arr is None:
                    self.text_area.append(f"⚠️ Skipped (not valid 2D numeric): {os.path.basename(f)}")
                else:
                    self.cleaned_text_data[f] = arr
                    self.text_area.append(f"✓ Parsed 2D data: shape={arr.shape} from {os.path.basename(f)}")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while loading text files:\n{e}",
            )

    def upload_reshaped_images(self):
        """Optional: load any number of images (or none)."""
        try:
            image_files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Image Files",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)",
            )
            if not image_files:
                QMessageBox.information(
                    self,
                    "No Images Selected",
                    "No images selected. That's OK—I'll still create data.h5 with only X-ray maps.",
                )
                return
            self.image_files = image_files
            self.text_area.append("Loaded image files:")
            for f in image_files:
                self.text_area.append(f"  - {f}")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while loading image files:\n{e}",
            )

    def create_combine_file(self):
        """
        EXACT match to the old HDF5 layout:
        - 'X-ray Maps' group
            - subgroup per text file (basename)
                - dataset 'data' with numeric matrix
        - 'Images' group
            - dataset per image (dataset name = image basename)
        Images are OPTIONAL. If none are loaded, an empty 'Images' group is still created.
        """
        try:
            # Require only text files
            if not self.text_files:
                QMessageBox.warning(self, "Files Not Loaded", "Please load text files first.")
                return

            # output_dir = os.path.dirname(self.text_files[0])
            # if not output_dir:
            #     output_dir = os.getcwd()
            # hdf5_file = os.path.join(output_dir, "data.h5")
            hdf5_file = self._wd("data.h5")

            with h5py.File(hdf5_file, "w") as hdf:
                xray_maps_group = hdf.create_group("X-ray Maps")
                images_group = hdf.create_group("Images")  # create even if empty

                # Text files → subgroups with 'data' dataset (uses OLD
                # load_data)
                n_txt_written = 0
                for text_file in self.text_files:
                    data, data_type = self.load_data(text_file)  # OLD parser
                    if data is not None and data_type == "text":
                        file_name = os.path.basename(text_file)
                        # Ensure unique subgroup name if duplicates
                        grp_name = file_name
                        i = 1
                        while grp_name in xray_maps_group:
                            root, ext = os.path.splitext(file_name)
                            grp_name = f"{root} ({i}){ext}"
                            i += 1
                        xray_subgroup = xray_maps_group.create_group(grp_name)
                        xray_subgroup.create_dataset("data", data=data)
                        n_txt_written += 1

                if n_txt_written == 0:
                    QMessageBox.warning(
                        self,
                        "No Valid Text Data",
                        "No valid X-ray map matrices were parsed from the loaded text files.",
                    )
                    # Still write the file with empty groups for compatibility

                # Images → dataset named as image basename (OPTIONAL)
                if getattr(self, "image_files", None):
                    for image_file in self.image_files:
                        data, data_type = self.load_data(image_file)  # OLD parser
                        if data is not None and data_type == "image":
                            image_name = os.path.basename(image_file)
                            ds_name = image_name
                            j = 1
                            while ds_name in images_group:
                                root, ext = os.path.splitext(image_name)
                                ds_name = f"{root} ({j}){ext}"
                                j += 1
                            images_group.create_dataset(ds_name, data=data)

            self.text_area.append(f"\nHDF5 file created successfully: {hdf5_file}")
            self.text_area.append("\nMove to Phase Classification step......")
            self.data_h5_path = hdf5_file

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while creating the HDF5 file:\n{e}",
            )

    # ---------- OLD load_data: copied to guarantee identical HDF5 content ---

    def load_data(self, file_path):
        """
        OLD behavior:
        - For *.txt/*.csv: skip headers until the FIRST line that has 5–6+ numeric
          fields in a row, then read subsequent numeric lines.
        - For images: read with PIL and return np.array(image).
        Returns (data, 'text'/'image') or (None, None).
        """
        try:
            ext = file_path.split(".")[-1].lower()

            if ext in ["txt", "csv"]:
                start_reading = False
                data = []

                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if not start_reading:
                            if re.search(
                                r"(\d+(\.\d+)?[\s,;,\t]+){5,6}\d+(\.\d+)?",
                                line,
                            ):
                                start_reading = True
                                row = list(
                                    map(
                                        float,
                                        re.split(r"[;\s,]+", line.strip()),
                                    )
                                )
                                data.append(row)
                            continue
                        try:
                            row = list(map(float, re.split(r"[;\s,]+", line.strip())))
                            data.append(row)
                        except ValueError:
                            continue

                if data:
                    return np.array(data), "text"
                else:
                    self.text_area.append(f"No valid data found in {os.path.basename(file_path)}")
                    return None, "text"

            elif ext in ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]:
                image = Image.open(file_path)
                return np.array(image), "image"

            else:
                return None, None

        except FileNotFoundError:
            self.text_area.append(f"File not found: {file_path}")
            return None, None
        except Exception as e:
            self.text_area.append(f"Read error for {os.path.basename(file_path)}: {e}")
            return None, None

    # ------------------ Helpers for preview & XY→ASCII (unchanged) ----------

    def _safe_read_image(self, file_path):
        try:
            im = Image.open(file_path)
            return np.array(im)
        except Exception:
            return None

    def _load_2d_text_clean(self, file_path):
        removed = []
        rows = []
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    line = raw.rstrip("\n")
                    s = line.strip()
                    if not s:
                        removed.append(line)
                        continue
                    if s.startswith(("#", "//", "!", "%", ";")):
                        removed.append(line)
                        continue
                    for ch in [",", ";", "\t"]:
                        s = s.replace(ch, " ")
                    parts = s.split()
                    nums = []
                    all_numeric = True
                    for p in parts:
                        try:
                            nums.append(float(p))
                        except ValueError:
                            all_numeric = False
                            break
                    if not all_numeric or not nums:
                        removed.append(line)
                        continue
                    rows.append(nums)
            if not rows:
                return None, removed
            lens = {len(r) for r in rows}
            if len(lens) != 1:
                self.text_area.append(
                    f"Ragged row lengths in {os.path.basename(file_path)}: {sorted(lens)}; treating as invalid 2D."
                )
                return None, removed
            arr = np.array(rows, dtype=float)
            return arr, removed
        except Exception as e:
            self.text_area.append(f"Read error {os.path.basename(file_path)}: {e}")
            return None, removed

    def upload_xy_files(self):
        try:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select XY Data File(s)",
                "",
                "Text Files (*.txt *.csv *.dat *.asc *.tsv *.xy);;All Files (*)",
            )
            if not files:
                QMessageBox.warning(
                    self,
                    "No Files Selected",
                    "Please select at least one XY file.",
                )
                return
            self.xy_files = files
            self.text_area.append("Loaded XY files:")
            for f in files:
                self.text_area.append(f"  - {f}")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while selecting XY files:\n{e}",
            )

    def save_xy_as_ascii(self):
        if not self.xy_files:
            QMessageBox.warning(self, "No XY Files", "Upload XY file(s) first.")
            return

        try:
            if len(self.xy_files) == 1:
                base = os.path.splitext(os.path.basename(self.xy_files[0]))[0]
                out_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save ASCII Grid As",
                    os.path.splitext(self.xy_files[0])[0] + "_grid.txt",
                    "Text Files (*.txt *.asc);;All Files (*)",
                )
                if not out_path:
                    return
                grid, xs, ys = self._read_xy_to_grid(self.xy_files[0])
                if grid is None:
                    QMessageBox.warning(
                        self,
                        "Read Error",
                        f"Could not parse: {self.xy_files[0]}",
                    )
                    return
                np.savetxt(out_path, grid, fmt="%.10e", delimiter=" ")
                self.text_area.append(f"Saved 2D grid ASCII: {out_path}  (shape={grid.shape[0]}x{grid.shape[1]})")
            else:
                out_dir = QFileDialog.getExistingDirectory(self, "Choose Output Folder", self.working_dir or "")
                if not out_dir:
                    return
                count = 0
                for fpath in self.xy_files:
                    grid, xs, ys = self._read_xy_to_grid(fpath)
                    if grid is None:
                        self.text_area.append(f"Skipped (unreadable): {fpath}")
                        continue
                    base = os.path.splitext(os.path.basename(fpath))[0]
                    out_path = os.path.join(out_dir, f"{base}_grid.txt")
                    np.savetxt(out_path, grid, fmt="%.10e", delimiter=" ")
                    count += 1
                    self.text_area.append(f"Saved: {out_path}  (shape={grid.shape[0]}x{grid.shape[1]})")
                QMessageBox.information(
                    self,
                    "Done",
                    f"Converted {count} / {len(self.xy_files)} files to 2D grids.",
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while saving ASCII:\n{e}")

    def _read_xy_to_grid(self, file_path):
        """Parse XYV scattered data into a 2D grid (rows = unique Y,
        cols = unique X).

        Improvements:
        - Does NOT assume column order.
        - Looks at the nearest header line(s) ABOVE the first numeric row,
        including COMMENTED headers (e.g. '# X  Y  Intensity').
        - Robust tokenization (commas/semicolons/tabs/units).
        - Fallback to 0,1,2 if no header hints are found.
        """
        import re

        def _split_fields(s: str):
            # normalize delimiters: comma/semicolon/tab -> space
            for ch in [",", ";", "\t"]:
                s = s.replace(ch, " ")
            return [p for p in s.strip().split() if p]

        def _is_comment_or_blank(s: str) -> bool:
            s = s.strip()
            return (not s) or s.startswith(("#", "//", "!", "%", ";"))

        def _is_numeric_row(parts):
            # treat as numeric row if >=3 tokens all parse as float
            cnt = 0
            for p in parts:
                try:
                    float(p)
                    cnt += 1
                except ValueError:
                    return False
            return cnt >= 3

        def _tokenize_header_line(line: str):
            # strip leading comment markers
            s = line.strip()
            s = re.sub(r"^\s*(#|//|;|!|%)\s*", "", s)
            # drop obvious prefixes like "columns:" etc.
            s = re.sub(r"^\s*(columns?|cols?)\s*:\s*", "", s, flags=re.I)
            parts = _split_fields(s)
            toks = []
            for t in parts:
                t = t.strip().lower()
                # remove units and punctuation in tokens like 'x(um)' or
                # 'y[px]'
                t = re.sub(r"\(.*?\)|\[.*?\]", "", t)
                t = re.sub(r"[^a-z0-9]+", "", t)  # keep alnum
                toks.append(t)
            return toks

        def _detect_cols_from_header_tokens(header_tokens, n_cols_hint=3):
            """Return (x_col, y_col, v_col) or None if not enough
            hints."""
            if not header_tokens:
                return None

            cand_x = cand_y = cand_v = None
            for k, tok in enumerate(header_tokens):
                # common X labels
                if tok.startswith("x") or tok in {"lon", "longitude"}:
                    cand_x = k if cand_x is None else cand_x
                    continue
                # common Y labels
                if tok.startswith("y") or tok in {"lat", "latitude"}:
                    cand_y = k if cand_y is None else cand_y
                    continue
                # value labels (intensity/signal/etc.)
                if tok in {
                    "z",
                    "v",
                    "val",
                    "value",
                    "i",
                    "int",
                    "intensity",
                    "signal",
                    "counts",
                    "cps",
                    "cpss",
                } or tok.startswith("inten"):
                    cand_v = k if cand_v is None else cand_v
                    continue

            # Compose result if we have at least X and Y
            if cand_x is not None and cand_y is not None:
                x_col, y_col = cand_x, cand_y
                if cand_v is not None and cand_v not in (x_col, y_col):
                    v_col = cand_v
                else:
                    # pick first non X/Y as value
                    n = max(len(header_tokens), n_cols_hint, 3)
                    for c in range(n):
                        if c not in (x_col, y_col):
                            v_col = c
                            break
                return (x_col, y_col, v_col)

            # If only one axis is labeled, infer the other as "first other col"
            if cand_x is not None or cand_y is not None:
                n = max(len(header_tokens), n_cols_hint, 3)
                if cand_x is not None:
                    x_col = cand_x
                    # Y is first column that isn't X and looks like not value
                    y_col = next(
                        (c for c in range(n) if c != x_col and c != cand_v),
                        1 if x_col != 1 else 0,
                    )
                else:
                    y_col = cand_y
                    x_col = next(
                        (c for c in range(n) if c != y_col and c != cand_v),
                        0 if y_col != 0 else 1,
                    )
                if cand_v is not None and cand_v not in (x_col, y_col):
                    v_col = cand_v
                else:
                    v_col = next((c for c in range(n) if c not in (x_col, y_col)), 2)
                return (x_col, y_col, v_col)

            # Not enough info
            return None

        # ---------- Read file ----------
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            self.text_area.append(f"Read error for {file_path}: {e}")
            return None, None, None

        # ---------- Locate first data row ----------
        first_data_idx = None
        first_data_parts = None
        for i, raw in enumerate(lines):
            s = raw.strip()
            if not s:
                continue
            parts = _split_fields(s)
            if _is_numeric_row(parts):
                first_data_idx = i
                first_data_parts = parts
                break

        if first_data_idx is None:
            self.text_area.append(f"No numeric data found in {os.path.basename(file_path)}")
            return None, None, None

        # ---------- Scan upwards (including COMMENT lines) for a header ------
        # Look back up to ~10 lines to find a header-ish line
        x_col = y_col = v_col = None
        header_found = False
        for j in range(first_data_idx - 1, max(-1, first_data_idx - 11), -1):
            line = lines[j].strip()
            if not line:
                continue
            # Try to tokenize even if it's a comment line
            toks = _tokenize_header_line(line)
            # Must not look like a numeric line
            if toks and not all(t.replace(".", "", 1).isdigit() for t in toks):
                detected = _detect_cols_from_header_tokens(toks, n_cols_hint=len(first_data_parts))
                if detected:
                    x_col, y_col, v_col = detected
                    header_found = True
                    self.text_area.append(
                        f"Header detected in {'comment' if _is_comment_or_blank(line) else 'header'} line "
                        f"(near {os.path.basename(file_path)}): X={x_col}, Y={y_col}, V={v_col}"
                    )
                    break

        # ---------- Fallback if no header hint ----------
        if not header_found:
            # default to first three columns (old behavior)
            x_col, y_col, v_col = 0, 1, 2
            self.text_area.append(
                f"No header labels found near data in {os.path.basename(file_path)}; "
                f"using default columns X=0, Y=1, V=2"
            )

        # ---------- Parse data ----------
        xs_list, ys_list, vs_list = [], [], []
        for raw in lines[first_data_idx:]:
            s = raw.strip()
            if not s:
                continue
            parts = _split_fields(s)
            # try parse floats
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    nums = []
                    break
            if not nums:
                continue
            need = max(x_col, y_col, v_col) + 1
            if len(nums) < need:
                continue
            try:
                x = nums[x_col]
                y = nums[y_col]
                v = nums[v_col]
            except Exception:
                continue
            xs_list.append(x)
            ys_list.append(y)
            vs_list.append(v)

        if not xs_list:
            return None, None, None

        # ---------- Build grid ----------
        xs = sorted(set(xs_list))
        ys = sorted(set(ys_list))
        x_index = {x: i for i, x in enumerate(xs)}
        y_index = {y: i for i, y in enumerate(ys)}
        grid = np.full((len(ys), len(xs)), np.nan, dtype=float)

        for x, y, v in zip(xs_list, ys_list, vs_list):
            ix = x_index.get(x)
            iy = y_index.get(y)
            if ix is None or iy is None:
                continue
            grid[iy, ix] = v

        return grid, np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    # ---------------------------- External ---------------------------

    def open_calibration(self):
        try:
            subprocess.run([sys.executable, "Calibration.py"], check=True)
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while running Calibration.py:\n{e}",
            )


class CorrectionDialog(QDialog):
    """Instrument- and method-aware correction dialog.

    - EPMA (WDS): only "WDS non-paralyzable (τ)" is available.
      Needs τ; Real time is needed when input is in counts (for counts→cps conversion).
      Live time is NOT needed for WDS.
    - SEM-EDS / TEM-EDS / micro-XRF-EDS / Synchrotron:
      Methods: "Livetime ratio" (needs Real & Live) or "ICR/OCR" (needs ICR & OCR).

    'Already cps' greys out method-specific inputs (pass-through),
    except Real time stays enabled if the output is 'Counts' (to allow cps→counts).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Correction")
        self.params = {}

        self.form = QFormLayout(self)

        # --- Instrument
        self.cmb_instr = QComboBox()
        self.cmb_instr.addItems(
            [
                "EPMA (WDS)",
                "SEM-EDS",
                "TEM-EDS",
                "micro-XRF-EDS",
                "Synchrotron (SDD array)",
            ]
        )
        self.form.addRow("Instrument", self.cmb_instr)

        # --- Input units
        self.cmb_in_units = QComboBox()
        self.cmb_in_units.addItems(["Counts (per pixel)", "Counts per second (cps)"])
        self.form.addRow("Input units", self.cmb_in_units)

        # --- Method
        self.cmb_method = QComboBox()
        # Items are repopulated per instrument; keep the three canonical labels
        self._METHOD_LIVE = "Livetime ratio (counts × real/live)"
        self._METHOD_ICR = "ICR/OCR scaling"
        self._METHOD_WDS = "WDS non-paralyzable (τ)"
        self.cmb_method.addItems([self._METHOD_LIVE, self._METHOD_ICR, self._METHOD_WDS])
        self.form.addRow("Correction method", self.cmb_method)

        # --- Times
        self.spn_real = QDoubleSpinBox()
        self.spn_real.setDecimals(6)
        self.spn_real.setRange(0.0, 1e9)
        self.spn_real.setValue(1.0)
        self.spn_live = QDoubleSpinBox()
        self.spn_live.setDecimals(6)
        self.spn_live.setRange(0.0, 1e9)
        self.spn_live.setValue(0.8)
        self.form.addRow("Real time (s)", self.spn_real)
        self.form.addRow("Live time (s)", self.spn_live)

        # --- ICR/OCR
        self.spn_icr = QDoubleSpinBox()
        self.spn_icr.setDecimals(3)
        self.spn_icr.setRange(0.0, 1e12)
        self.spn_icr.setValue(100000.0)
        self.spn_ocr = QDoubleSpinBox()
        self.spn_ocr.setDecimals(3)
        self.spn_ocr.setRange(0.0, 1e12)
        self.spn_ocr.setValue(60000.0)
        self.form.addRow("ICR (cps)", self.spn_icr)
        self.form.addRow("OCR (cps)", self.spn_ocr)

        # --- WDS tau
        tau_row = QHBoxLayout()
        self.spn_tau = QDoubleSpinBox()
        self.spn_tau.setDecimals(6)
        self.spn_tau.setRange(0.0, 1e3)
        self.spn_tau.setValue(2.0)
        self.cmb_tau_unit = QComboBox()
        self.cmb_tau_unit.addItems(["µs", "ns"])
        tau_row.addWidget(self.spn_tau)
        tau_row.addWidget(self.cmb_tau_unit)
        self.tau_box = QWidget()
        self.tau_box.setLayout(tau_row)
        self.form.addRow("WDS τ", self.tau_box)

        # --- Output units
        self.cmb_out_units = QComboBox()
        self.cmb_out_units.addItems(["Counts per second (cps)", "Counts (use Real time)"])
        self.form.addRow("Output as", self.cmb_out_units)

        # --- Already cps
        self.chk_already_cps = QCheckBox("My input is already cps (do NOT re-correct; just pass-through).")
        self.form.addRow(self.chk_already_cps)

        warn = QLabel(
            "⚠️ If your data are already cps, avoid 'Livetime ratio' or 'ICR/OCR'.\n"
            "For WDS, ensure τ is calibrated and observed rates are within a safe range."
        )
        warn.setWordWrap(True)
        warn.setStyleSheet("color:#b00;")
        self.form.addRow(warn)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.form.addRow(buttons)

        # Wire dynamic behaviour
        self.cmb_instr.currentTextChanged.connect(self._on_instrument_changed)
        self.cmb_method.currentTextChanged.connect(self._refresh_field_enabling)
        self.cmb_in_units.currentTextChanged.connect(self._refresh_field_enabling)
        self.cmb_out_units.currentTextChanged.connect(self._refresh_field_enabling)
        self.chk_already_cps.toggled.connect(self._refresh_field_enabling)

        # Initial state
        self._on_instrument_changed(self.cmb_instr.currentText())

    # ---------- dynamic methods ----------
    def _set_combo_items(self, combo: QComboBox, items: list, keep_selection_if_possible=True):
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        if keep_selection_if_possible and current in items:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def _on_instrument_changed(self, instr_text: str):
        """Limit methods by instrument and refresh enable/disable
        state."""
        if instr_text == "EPMA (WDS)":
            self._set_combo_items(
                self.cmb_method,
                [self._METHOD_WDS],
                keep_selection_if_possible=False,
            )
        else:
            self._set_combo_items(
                self.cmb_method,
                [self._METHOD_LIVE, self._METHOD_ICR],
                keep_selection_if_possible=True,
            )
            # If selection was WDS, fall back to Livetime
            if self.cmb_method.currentText() not in (
                self._METHOD_LIVE,
                self._METHOD_ICR,
            ):
                self.cmb_method.setCurrentText(self._METHOD_LIVE)
        self._refresh_field_enabling()

    def _refresh_field_enabling(self):
        """Grey out everything not required for the current
        selections."""
        method = self.cmb_method.currentText()
        in_is_counts = self.cmb_in_units.currentText().startswith("Counts")
        out_is_counts = self.cmb_out_units.currentText().startswith("Counts")
        already = self.chk_already_cps.isChecked()

        is_wds = method == self._METHOD_WDS
        is_live = method == self._METHOD_LIVE
        is_icr = method == self._METHOD_ICR

        # --- Base: enable all, then selectively disable
        for w in (
            self.spn_real,
            self.spn_live,
            self.spn_icr,
            self.spn_ocr,
            self.spn_tau,
            self.cmb_tau_unit,
        ):
            w.setEnabled(True)
        self.tau_box.setEnabled(True)

        # If user says "already cps", disable method-specific inputs
        # (pass-through)
        if already:
            # Keep Real time enabled ONLY if output is 'Counts' (for cps→counts
            # conversion)
            self.spn_live.setEnabled(False)
            self.spn_icr.setEnabled(False)
            self.spn_ocr.setEnabled(False)
            self.spn_tau.setEnabled(False)
            self.cmb_tau_unit.setEnabled(False)
            self.tau_box.setEnabled(False)
            self.spn_real.setEnabled(out_is_counts)
            return

        # --- WDS: needs τ; Real time needed only if input is in counts (to get cps)
        if is_wds:
            self.spn_tau.setEnabled(True)
            self.cmb_tau_unit.setEnabled(True)
            self.tau_box.setEnabled(True)
            self.spn_live.setEnabled(False)  # live time not used by WDS model
            self.spn_icr.setEnabled(False)
            self.spn_ocr.setEnabled(False)
            self.spn_real.setEnabled(in_is_counts or out_is_counts)
            return

        # --- Livetime ratio: needs Real & Live; no ICR/OCR; no τ
        if is_live:
            self.spn_real.setEnabled(True)
            self.spn_live.setEnabled(True)
            self.spn_icr.setEnabled(False)
            self.spn_ocr.setEnabled(False)
            self.spn_tau.setEnabled(False)
            self.cmb_tau_unit.setEnabled(False)
            self.tau_box.setEnabled(False)
            return

        # --- ICR/OCR: needs ICR & OCR; times generally not required
        if is_icr:
            self.spn_icr.setEnabled(True)
            self.spn_ocr.setEnabled(True)
            self.spn_tau.setEnabled(False)
            self.cmb_tau_unit.setEnabled(False)
            self.tau_box.setEnabled(False)
            # Real time only needed if you want output in 'Counts'
            self.spn_real.setEnabled(out_is_counts)
            self.spn_live.setEnabled(False)
            return

    # ---------- accept ----------
    def accept(self):
        self.params = {
            "instrument": self.cmb_instr.currentText(),
            "in_units": self.cmb_in_units.currentText(),
            "method": self.cmb_method.currentText(),
            "real_time": float(self.spn_real.value()),
            "live_time": float(self.spn_live.value()),
            "icr": float(self.spn_icr.value()),
            "ocr": float(self.spn_ocr.value()),
            "tau": float(self.spn_tau.value()),
            "tau_unit": self.cmb_tau_unit.currentText(),
            "out_units": self.cmb_out_units.currentText(),
            "already_cps": bool(self.chk_already_cps.isChecked()),
        }
        super().accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = PreanalysisStep()
    viewer.show()
    sys.exit(app.exec_())
