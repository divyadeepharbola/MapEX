# main.py — MapEX launcher with working-directory routing and Refresh
import os
import sys

from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from analysis_step import AnalysisStep
from phase_mapping_step import PhaseMappingStep
from preanalysis_step import PreanalysisStep
from visualization_step import VisualizationStep

APP_ORG = "IIT Bombay"
APP_NAME = "MapEX"
version = "v1.0.4"


class MapEXLogo(QFrame):
    def __init__(
        self,
        version="v1.0.0",
        bg_color="#FFFFFF",
        color_map="#FFA500",
        color_ex="#2E7D32",
        color_version="#000000",
        radius=10,
        pad=2,
        min_px=6,
        max_px=800,
        ex_letter_px=1.0,
        version_scale=0.55,
        sub_offset_ratio=0.22,
        font_family="Segoe UI",
    ):
        super().__init__()
        self.version = version
        self.bg_color = QColor(bg_color)
        self.color_map = QColor(color_map)
        self.color_ex = QColor(color_ex)
        self.color_version = QColor(color_version)
        self.radius = radius
        self.pad = pad
        self.min_px = min_px
        self.max_px = max_px
        self.ex_letter_px = ex_letter_px
        self.version_scale = version_scale
        self.sub_offset_ratio = sub_offset_ratio
        self.font_family = font_family
        self._last_px = None
        self.setAttribute(Qt.WA_StyledBackground, False)

    def _fonts(self, px_main):
        f_map = QFont(self.font_family)
        f_map.setPixelSize(px_main)
        f_map.setWeight(QFont.Black)
        f_ex = QFont(self.font_family)
        f_ex.setPixelSize(px_main)
        f_ex.setWeight(QFont.Black)
        f_ex.setLetterSpacing(QFont.AbsoluteSpacing, self.ex_letter_px)
        px_ver = max(1, int(round(px_main * self.version_scale)))
        f_ver = QFont(self.font_family)
        f_ver.setPixelSize(px_ver)
        f_ver.setWeight(QFont.DemiBold)
        return f_map, f_ex, f_ver, px_ver

    def _bounds(self, px_main, inner_w, inner_h):
        f_map, f_ex, f_ver, _ = self._fonts(px_main)
        fm_map = QFontMetrics(f_map)
        fm_ex = QFontMetrics(f_ex)
        fm_ver = QFontMetrics(f_ver)
        w_map = fm_map.horizontalAdvance("Map")
        w_ex = fm_ex.horizontalAdvance("EX")
        gap = max(0, int(px_main * 0.04))
        sub_offset = int(round(px_main * self.sub_offset_ratio))
        top_main = max(fm_map.ascent(), fm_ex.ascent())
        bot_main = max(fm_map.descent(), fm_ex.descent())
        top_ver = max(0, fm_ver.ascent() - sub_offset)
        bot_ver = fm_ver.descent() + sub_offset
        height_total = max(top_main, top_ver) + max(bot_main, bot_ver)
        width_total = w_map + w_ex + gap + fm_ver.horizontalAdvance(self.version)
        fits = (width_total <= inner_w - 1) and (height_total <= inner_h - 1)
        baseline_top = max(top_main, top_ver)
        return fits, width_total, height_total, baseline_top, sub_offset

    def _choose_px(self, inner_w, inner_h):
        lo, hi, best = self.min_px, self.max_px, self.min_px
        for _ in range(20):
            mid = (lo + hi) // 2
            fits, *_ = self._bounds(mid, inner_w, inner_h)
            if fits:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        for bump in (1, 1):
            fits, *_ = self._bounds(best + bump, inner_w, inner_h)
            if fits:
                best += bump
        return best

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._last_px = None

    def paintEvent(self, e):
        w = self.width()
        h = self.height()
        inner_w = max(8, w - 2 * self.pad)
        inner_h = max(8, h - 2 * self.pad)
        px = self._last_px or self._choose_px(inner_w, inner_h)
        self._last_px = px
        f_map, f_ex, f_ver, _ = self._fonts(px)
        fits, width_total, _, baseline_top, sub_offset = self._bounds(px, inner_w, inner_h)
        del fits
        x0 = self.pad + (inner_w - width_total) // 2
        y0 = self.pad + int(baseline_top)

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(self.bg_color)
        r = self.rect()
        r.adjust(0, 0, -1, -1)
        p.drawRoundedRect(r, self.radius, self.radius)

        p.setFont(f_map)
        p.setPen(self.color_map)
        p.drawText(x0, y0, "Map")
        fm_map = QFontMetrics(f_map)
        x = x0 + fm_map.horizontalAdvance("Map")
        p.setFont(f_ex)
        p.setPen(self.color_ex)
        p.drawText(x, y0, "EX")
        fm_ex = QFontMetrics(f_ex)
        x += fm_ex.horizontalAdvance("EX")
        x += max(0, int(px * 0.04))
        p.setFont(f_ver)
        p.setPen(self.color_version)
        p.drawText(x, y0 + sub_offset, self.version)
        p.end()


class FlowChartGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.current_step = 0
        self.setWindowTitle(APP_NAME)
        self.resize(1100, 700)

        # Working directory
        self.settings = QSettings(APP_ORG, APP_NAME)
        self.working_dir = ""

        # Step widgets
        self.preanalysis_widget = None
        self.phase_mapping_widget = None
        self.analysis_widget = None
        self.visualization_widget = None

        self._build_ui()
        self._choose_working_dir_at_startup()
        self._wire_signals()
        self.update_step()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Top row: Logo + step buttons + spacer + utilities
        steps = QHBoxLayout()
        steps.setSpacing(6)

        # Step buttons
        self.btn_pre = QPushButton("Preanalysis")
        self.btn_phase = QPushButton("Phase Classification")
        self.btn_analysis = QPushButton("Analysis")
        self.btn_viz = QPushButton("Visualization")
        for b in (
            self.btn_pre,
            self.btn_phase,
            self.btn_analysis,
            self.btn_viz,
        ):
            b.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Logo (left)
        base_h = self.btn_pre.sizeHint().height()
        base_w = self.btn_pre.sizeHint().width()
        logo_h = int(round(base_h * 2.2))
        logo_w = int(round(base_w * 2.2))
        self.logo_widget = MapEXLogo(version=version)
        self.logo_widget.setFixedSize(logo_w, logo_h)

        steps.addWidget(self.logo_widget, 0, Qt.AlignVCenter)
        steps.addSpacing(8)
        for b in (
            self.btn_pre,
            self.btn_phase,
            self.btn_analysis,
            self.btn_viz,
        ):
            steps.addWidget(b)

        # Stretch pushes utilities right
        steps.addItem(QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Utility buttons (right)
        self.btn_change_folder = QPushButton("Change Folder…")
        self.btn_refresh = QPushButton("Refresh")
        self.btn_calibration = QPushButton("Calibration")
        for b in (
            self.btn_change_folder,
            self.btn_refresh,
            self.btn_calibration,
        ):
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        steps.addWidget(self.btn_change_folder)
        steps.addWidget(self.btn_refresh)
        steps.addWidget(self.btn_calibration)

        root.addLayout(steps)

        # Working dir display
        info_row = QHBoxLayout()
        info_row.setSpacing(6)
        self.lbl_workdir = QLabel("Working folder: —")
        self.lbl_workdir.setStyleSheet("QLabel { color: #444; }")
        self.lbl_workdir.setTextInteractionFlags(Qt.TextSelectableByMouse)
        info_row.addWidget(self.lbl_workdir)
        root.addLayout(info_row)

        # Stacked content
        self.stack_widget = QStackedWidget()

        # Pre-load Preanalysis (first page)
        self.preanalysis_widget = PreanalysisStep()
        self._apply_paths(self.preanalysis_widget)
        self.stack_widget.addWidget(self.preanalysis_widget)  # index 0

        # Placeholders for other pages
        self.stack_widget.addWidget(QWidget())  # index 1 (phase)
        self.stack_widget.addWidget(QWidget())  # index 2 (analysis)
        self.stack_widget.addWidget(QWidget())  # index 3 (viz)

        root.addWidget(self.stack_widget)

        self.setStyleSheet(
            "QWidget { font-family: Segoe UI, Arial; font-size: 10pt; }" "QPushButton { padding: 6px 10px; }"
        )

    def _wire_signals(self):
        self.btn_pre.clicked.connect(lambda: self.switch_step(0))
        self.btn_phase.clicked.connect(lambda: self.switch_step(1))
        self.btn_analysis.clicked.connect(lambda: self.switch_step(2))
        self.btn_viz.clicked.connect(lambda: self.switch_step(3))
        self.btn_calibration.clicked.connect(self._run_calibration)
        self.btn_change_folder.clicked.connect(self._choose_working_dir_interactive)
        self.btn_refresh.clicked.connect(self._refresh_current)

    # ---------- Working directory plumbing ----------
    def _data_path(self) -> str:
        return os.path.join(self.working_dir, "data.h5") if self.working_dir else ""

    def _roi_path(self) -> str:
        return os.path.join(self.working_dir, "roi_coordinates.xlsx") if self.working_dir else ""

    def _apply_paths(self, widget: QWidget):
        """Force widgets to save/load only inside working_dir."""
        folder = self.working_dir or ""
        dp = self._data_path() if folder else ""

        # Set common attributes if present
        for attr, val in [
            ("working_dir", folder),
            ("output_dir", folder),
            ("save_dir", folder),
            ("base_dir", folder),
            ("data_h5_path", dp),
            ("data_path", dp),
            ("h5_path", dp),
        ]:
            if hasattr(widget, attr):
                try:
                    setattr(widget, attr, val)
                except Exception:
                    pass

        # Call common setters if available
        for meth, val in [
            ("set_working_dir", folder),
            ("set_output_dir", folder),
            ("set_data_path", dp),
        ]:
            if hasattr(widget, meth):
                try:
                    getattr(widget, meth)(val)
                except Exception:
                    pass

    def _set_working_dir(self, folder: str):
        folder = os.path.abspath(folder)
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Folder Error", f"Cannot use folder:\n{folder}\n\n{e}")
            return

        self.working_dir = folder
        self.settings.setValue("working_dir", folder)
        os.environ["MAPEX_WORKDIR"] = folder  # helpful for subprocesses
        self.lbl_workdir.setText(f"Working folder: {folder}")

        # Update preanalysis paths
        if self.preanalysis_widget is not None:
            self._apply_paths(self.preanalysis_widget)

        # Reset heavy pages to avoid stale paths
        self._dispose_heavy_pages()
        self.switch_step(0)

    def _dispose_heavy_pages(self):
        for idx, attr in [
            (1, "phase_mapping_widget"),
            (2, "analysis_widget"),
            (3, "visualization_widget"),
        ]:
            w = getattr(self, attr)
            if w is not None:
                try:
                    self.stack_widget.removeWidget(w)
                    w.deleteLater()
                except Exception:
                    pass
                setattr(self, attr, None)
                self.stack_widget.insertWidget(idx, QWidget())

    def _choose_working_dir_at_startup(self):
        last = self.settings.value("working_dir", "", type=str) or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Choose Working Folder (data.h5 will be saved here)", last)
        if not folder:
            folder = last if os.path.isdir(last) else os.path.expanduser("~")
        self._set_working_dir(folder)

    def _choose_working_dir_interactive(self):
        start = self.working_dir or self.settings.value("working_dir", "", type=str) or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Choose Working Folder", start)
        if folder:
            self._set_working_dir(folder)

    # ------------- Navigation / gating -------------
    def _require_data_exists(self) -> bool:
        dp = self._data_path()
        if not dp or not os.path.exists(dp):
            self._warn("Please create or load 'data.h5' in Preanalysis first.\n\nExpected at:\n" + dp)
            return False
        return True

    def switch_step(self, index: int):
        if not isinstance(index, int):
            try:
                index = int(index)
            except Exception:
                return
        if index < 0 or index > 3:
            return

        if index in (1, 2, 3):
            if not self._require_data_exists():
                return

        if index == 3:
            roi_path = self._roi_path()
            if not (roi_path and os.path.exists(roi_path)):
                self._warn(
                    "Visualization requires 'roi_coordinates.xlsx' in the same folder as 'data.h5'.\n\n"
                    f"Expected at:\n{roi_path}"
                )
                return

        # Lazy create per index
        if index == 1 and self.phase_mapping_widget is None:
            self.phase_mapping_widget = PhaseMappingStep(data_path=self._data_path())
            self._apply_paths(self.phase_mapping_widget)
            self._replace_page(1, self.phase_mapping_widget)

        if index == 2 and self.analysis_widget is None:
            self.analysis_widget = AnalysisStep(data_path=self._data_path())
            self._apply_paths(self.analysis_widget)
            self._replace_page(2, self.analysis_widget)

        if index == 3 and self.visualization_widget is None:
            self.visualization_widget = VisualizationStep(data_path=self._data_path())
            self._apply_paths(self.visualization_widget)
            self._replace_page(3, self.visualization_widget)

        self.current_step = index
        self.update_step()

    # ------------- Refresh logic -------------
    def _refresh_current(self):
        """Re-create the current page and rebind paths."""
        idx = self.current_step

        # Gate checks
        if idx in (1, 2, 3) and not self._require_data_exists():
            return
        if idx == 3:
            if not (self._roi_path() and os.path.exists(self._roi_path())):
                self._warn(
                    "Visualization requires 'roi_coordinates.xlsx' in the working folder.\n\n"
                    f"Expected at:\n{self._roi_path()}"
                )
                return

        # Recreate per page
        if idx == 0:
            neww = PreanalysisStep()
            self._apply_paths(neww)
            self.preanalysis_widget = neww
            self._replace_page(0, neww)

        elif idx == 1:
            neww = PhaseMappingStep(data_path=self._data_path())
            self._apply_paths(neww)
            self.phase_mapping_widget = neww
            self._replace_page(1, neww)

        elif idx == 2:
            neww = AnalysisStep(data_path=self._data_path())
            self._apply_paths(neww)
            self.analysis_widget = neww
            self._replace_page(2, neww)

        elif idx == 3:
            neww = VisualizationStep(data_path=self._data_path())
            self._apply_paths(neww)
            self.visualization_widget = neww
            self._replace_page(3, neww)

        # Keep visible page the same
        self.stack_widget.setCurrentIndex(idx)

    # ------------- Calibration -------------
    def _run_calibration(self):
        try:
            if self.preanalysis_widget is not None and hasattr(self.preanalysis_widget, "open_calibration"):
                self.preanalysis_widget.open_calibration()
            else:
                import subprocess
                import sys as _sys

                subprocess.run(
                    [_sys.executable, "Calibration.py"],
                    check=True,
                    cwd=self.working_dir or os.getcwd(),
                )
        except Exception as e:
            QMessageBox.critical(self, "Calibration", f"Could not launch Calibration.py:\n{e}")

    # ------------- Helpers -------------
    def _replace_page(self, idx: int, widget: QWidget):
        old = self.stack_widget.widget(idx)
        self.stack_widget.removeWidget(old)
        old.deleteLater()
        self.stack_widget.insertWidget(idx, widget)

    def update_step(self):
        self.stack_widget.setCurrentIndex(self.current_step)
        buttons = [
            self.btn_pre,
            self.btn_phase,
            self.btn_analysis,
            self.btn_viz,
        ]
        for i, btn in enumerate(buttons):
            if i == self.current_step:
                btn.setStyleSheet("QPushButton { background-color: lightblue; font-weight: bold; }")
            else:
                btn.setStyleSheet("QPushButton { background-color: none; font-weight: normal; }")

    def _warn(self, text: str):
        QMessageBox.warning(self, "Missing requirement", text, QMessageBox.Ok)

    # Keyboard: Ctrl+Left/Right (or Ctrl+A/D) + F5 refresh
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F5:
            self._refresh_current()
            return
        if event.modifiers() & Qt.ControlModifier:
            if event.key() in (Qt.Key_Right, Qt.Key_D):
                self.switch_step(min(3, self.current_step + 1))
                return
            if event.key() in (Qt.Key_Left, Qt.Key_A):
                self.switch_step(max(0, self.current_step - 1))
                return
        super().keyPressEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setOrganizationName(APP_ORG)
    app.setApplicationName(APP_NAME)
    app.setFont(QFont("Segoe UI", 10))
    win = FlowChartGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
