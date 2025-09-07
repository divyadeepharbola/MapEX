import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    RANSACRegressor,
)


# ---------------------- LOG PANEL ----------------------
class LogPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("Log messages…")
        self.setMaximumWidth(420)
        layout.addWidget(self.log_text)
        self.setLayout(layout)

    def log(self, message):
        self.log_text.append(str(message))


# ---------------------- PERIODIC TABLE ----------------------
class PeriodicTablePanel(QWidget):
    """Clean, readable periodic table with simple group-based coloring.

    Clicking an element selects X = element, Y = "element (intensity)".
    """

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setMinimumHeight(230)

        self.grid = QGridLayout(self)
        self.grid.setSpacing(4)
        self.grid.setContentsMargins(10, 8, 10, 8)

        # Layout matrix (strings or "" for blanks)
        self.matrix = [
            [
                "H",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "He",
            ],
            [
                "Li",
                "Be",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "B",
                "C",
                "N",
                "O",
                "F",
                "Ne",
            ],
            [
                "Na",
                "Mg",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "Al",
                "Si",
                "P",
                "S",
                "Cl",
                "Ar",
            ],
            [
                "K",
                "Ca",
                "Sc",
                "Ti",
                "V",
                "Cr",
                "Mn",
                "Fe",
                "Co",
                "Ni",
                "Cu",
                "Zn",
                "Ga",
                "Ge",
                "As",
                "Se",
                "Br",
                "Kr",
            ],
            [
                "Rb",
                "Sr",
                "Y",
                "Zr",
                "Nb",
                "Mo",
                "Tc",
                "Ru",
                "Rh",
                "Pd",
                "Ag",
                "Cd",
                "In",
                "Sn",
                "Sb",
                "Te",
                "I",
                "Xe",
            ],
            [
                "Cs",
                "Ba",
                "La",
                "Hf",
                "Ta",
                "W",
                "Re",
                "Os",
                "Ir",
                "Pt",
                "Au",
                "Hg",
                "Tl",
                "Pb",
                "Bi",
                "Po",
                "At",
                "Rn",
            ],
            [
                "Fr",
                "Ra",
                "Ac",
                "Rf",
                "Db",
                "Sg",
                "Bh",
                "Hs",
                "Mt",
                "Ds",
                "Rg",
                "Cn",
                "Nh",
                "Fl",
                "Mc",
                "Lv",
                "Ts",
                "Og",
            ],
            [
                "",
                "",
                "",
                "Ce",
                "Pr",
                "Nd",
                "Pm",
                "Sm",
                "Eu",
                "Gd",
                "Tb",
                "Dy",
                "Ho",
                "Er",
                "Tm",
                "Yb",
                "Lu",
            ],
            [
                "",
                "",
                "",
                "Th",
                "Pa",
                "U",
                "Np",
                "Pu",
                "Am",
                "Cm",
                "Bk",
                "Cf",
                "Es",
                "Fm",
                "Md",
                "No",
                "Lr",
            ],
        ]

        # Simple group palette
        self.palette = {
            "alkali": "#ffd6a5",
            "alkaline": "#fdffb6",
            "transition": "#caffbf",
            "post": "#9bf6ff",
            "metalloid": "#a0c4ff",
            "nonmetal": "#bdb2ff",
            "noble": "#ffc6ff",
            "lan_act": "#eeeeee",
        }

        # quick membership by symbol
        self.group = {
            # rough mapping
            **{s: "alkali" for s in ["Li", "Na", "K", "Rb", "Cs", "Fr"]},
            **{s: "alkaline" for s in ["Be", "Mg", "Ca", "Sr", "Ba", "Ra"]},
            **{s: "noble" for s in ["He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"]},
            **{
                s: "nonmetal"
                for s in [
                    "H",
                    "C",
                    "N",
                    "O",
                    "F",
                    "P",
                    "S",
                    "Cl",
                    "Se",
                    "Br",
                    "I",
                ]
            },
            **{s: "metalloid" for s in ["B", "Si", "Ge", "As", "Sb", "Te"]},
        }
        # fallback: transition/post/lan_act based on row/col
        self.buttons = {}
        self._build()

    def _btn_style(self, key):
        bg = self.palette.get(key, "#eaeaea")
        return f"""
        QPushButton {{
            background-color: {bg};
            border: 1px solid #9a9a9a;
            border-radius: 6px;
            padding: 4px 0;
            font-size: 12px;
            font-weight: 600;
            min-width: 34px; min-height: 28px;
        }}
        QPushButton:hover {{ filter: brightness(0.95); }}
        QPushButton:checked {{
            background-color: #228be6; color: white;
            border: 1px solid #1c6dc1;
        }}
        """

    def _class_for_cell(self, r, c, sym):
        if sym in self.group:
            return self.group[sym]
        # lanth/act rows
        if r >= 7:
            return "lan_act"
        # crude default
        return "transition"

    def _click(self, sym):
        # exclusive selection
        for s, btn in self.buttons.items():
            btn.setChecked(s == sym)
        self.parent.selected_x = sym
        self.parent.selected_y = f"{sym} (intensity)"
        self.parent.log(f"X-Axis selected: {self.parent.selected_x}, Y-Axis selected: {self.parent.selected_y}")
        self.parent.plot_scatter()

    def _build(self):
        for r, row in enumerate(self.matrix):
            for c, sym in enumerate(row):
                if not sym:
                    continue
                cls = self._class_for_cell(r, c, sym)
                btn = QPushButton(sym)
                btn.setCheckable(True)
                btn.setStyleSheet(self._btn_style(cls))
                btn.setToolTip(sym)
                btn.clicked.connect(lambda _, s=sym: self._click(s))
                self.grid.addWidget(btn, r if r < 7 else r + 1, c)  # keep gap
                self.buttons[sym] = btn


# ---------------------- TABLE PANEL ----------------------
class DataTablePanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.table_widget = QTableWidget()
        layout.addWidget(self.table_widget)

    def update_table(self, data: pd.DataFrame):
        self.table_widget.clear()
        self.table_widget.setColumnCount(len(data.columns))
        self.table_widget.setRowCount(len(data))
        self.table_widget.setHorizontalHeaderLabels(data.columns.astype(str).tolist())
        for row_idx, row_data in data.iterrows():
            for col_idx, value in enumerate(row_data):
                self.table_widget.setItem(
                    row_idx,
                    col_idx,
                    QTableWidgetItem("" if pd.isna(value) else str(value)),
                )


# ---------------------- CHECKBOX/REGRESSION PANEL ----------------------
class CheckboxPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.checkboxes = {}

        left = QVBoxLayout()
        self.checkbox_label = QLabel("Select Standards")
        self.checkbox_label.setStyleSheet("font-weight: 700;")
        left.addWidget(self.checkbox_label)

        # scroll
        self.checkbox_scroll = QScrollArea()
        self.checkbox_scroll.setWidgetResizable(True)
        self.checkbox_scroll_content = QWidget()
        self.checkbox_scroll_layout = QVBoxLayout(self.checkbox_scroll_content)
        self.checkbox_scroll.setWidget(self.checkbox_scroll_content)
        left.addWidget(self.checkbox_scroll, 1)

        # right (regression)
        right = QVBoxLayout()
        self.hover_label = QLabel("Hovered: —")
        self.hover_label.setStyleSheet("font-weight: 700; color: #1f4f99;")
        right.addWidget(self.hover_label)

        self.regression_label = QLabel("Regression")
        self.regression_label.setStyleSheet("font-weight: 700;")
        right.addWidget(self.regression_label)

        self.regression_dropdown = QComboBox()
        self.regression_dropdown.addItems(
            [
                "Linear (OLS)",
                "Linear (Huber)",
                "Linear (RANSAC)",
                "Quadratic (OLS)",
            ]
        )
        self.regression_dropdown.currentIndexChanged.connect(self._update_regression)
        right.addWidget(self.regression_dropdown)

        self.force_zero_checkbox = QCheckBox("Force line through (0, 0)")
        self.force_zero_checkbox.stateChanged.connect(self._toggle_force_zero)
        right.addWidget(self.force_zero_checkbox)

        self.equation_label = QLabel("Equation: —")
        self.equation_label.setStyleSheet("font-style: italic; font-weight: 700;")
        right.addWidget(self.equation_label)

        self.stats_label = QLabel("Stats: —")
        right.addWidget(self.stats_label)
        right.addStretch(1)

        # layout
        self.main = QHBoxLayout(self)
        self.main.addLayout(left, 1)
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        self.main.addWidget(sep)
        self.main.addLayout(right, 1)

    def populate_checkboxes(self, rows):
        for row in rows:
            cb = QCheckBox(str(row))
            cb.setChecked(True)
            cb.toggled.connect(self._toggled)
            self.checkboxes[row] = cb
            self.checkbox_scroll_layout.addWidget(cb)

    def _toggle_force_zero(self, *_):
        self.parent.force_zero = self.force_zero_checkbox.isChecked()
        self.parent.plot_scatter()

    def _toggled(self, *_):
        self.parent.update_checked_rows()

    def _update_regression(self, *_):
        self.parent.update_regression_type(self.regression_dropdown.currentText())

    def update_equation(self, eq, stats):
        self.equation_label.setText(f"Equation: {eq}")
        self.stats_label.setText(stats)


# ---------------------- SCATTER PLOT PANEL ----------------------
class ScatterPlotPanel(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.data = None

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        # hover annotation
        self.annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", ec="#333"),
            arrowprops=dict(arrowstyle="->", color="#333"),
        )
        self.annot.set_visible(False)

        # index within self.data (masked/included order)
        self._last_hover_idx = None
        self._hover_marker = None

        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_press_event", self.on_button)

    def _data_to_pixel_distance(self, x, y, mx, my):
        """Return pixel distance between (x,y) and mouse (mx,my) using
        transform."""
        xy_disp = self.ax.transData.transform(np.column_stack([x, y]))
        m_disp = self.ax.transData.transform(np.array([[mx, my]]))[0]
        d = np.sqrt(((xy_disp - m_disp) ** 2).sum(axis=1))
        return d

    def on_motion(self, event):
        if event.inaxes != self.ax or self.data is None:
            return
        mx, my = event.xdata, event.ydata
        if mx is None or my is None:
            return

        x = self.data[self.parent.selected_x].values
        y = self.data[self.parent.selected_y].values
        std = self.data["Standard Number"].astype(str).values

        d = self._data_to_pixel_distance(x, y, mx, my)
        idx = int(np.argmin(d))
        # threshold ~8 px
        if d[idx] <= 8:
            self._last_hover_idx = idx
            label = "Excluded" if std[idx] in self.parent.excluded_standards else "Included"
            txt = f"Std: {std[idx]} ({label})\\nX={x[idx]:.4g}, Y={y[idx]:.4g}"
            self.annot.set_visible(True)
            self.annot.xy = (x[idx], y[idx])
            self.annot.set_text(txt)
            # update panel label
            self.parent.checkbox_panel.hover_label.setText(f"Hovered: Standard {std[idx]} ({label})")

            # highlight marker
            if self._hover_marker:
                self._hover_marker.remove()
                self._hover_marker = None
            self._hover_marker = self.ax.plot(
                [x[idx]],
                [y[idx]],
                marker="o",
                markersize=9,
                mfc="none",
                mec="#222",
            )[0]
            self.canvas.draw_idle()
        else:
            self._last_hover_idx = None
            self.annot.set_visible(False)
            if self._hover_marker:
                self._hover_marker.remove()
                self._hover_marker = None
            self.parent.checkbox_panel.hover_label.setText("Hovered: —")
            self.canvas.draw_idle()

    def on_button(self, event):
        if event.button == 3 and event.inaxes == self.ax:
            # context menu for include/exclude
            menu = QMenu(self)
            if self._last_hover_idx is not None:
                std_value = str(self.data["Standard Number"].iloc[self._last_hover_idx])
                is_excluded = std_value in self.parent.excluded_standards
                act_toggle = menu.addAction(("Include" if is_excluded else "Exclude") + f" Standard {std_value}")
                act_toggle.triggered.connect(lambda: self.parent.toggle_point_exclusion(std_value))

                menu.addSeparator()
            act_inc_all = menu.addAction("Include all")
            act_inc_all.triggered.connect(self.parent.include_all_points)
            act_exc_all = menu.addAction("Exclude all")
            act_exc_all.triggered.connect(self.parent.exclude_all_points)
            act_clear_exc = menu.addAction("Clear all exclusions")
            act_clear_exc.triggered.connect(self.parent.clear_exclusions)

            menu.exec_(self.mapToGlobal(QPoint(int(event.x), int(event.y))))

    def plot(self, included_df, excluded_df, x_fit, y_fit, eq_text, stats_text):
        self.ax.clear()

        if len(included_df):
            self.ax.scatter(
                included_df[self.parent.selected_x],
                included_df[self.parent.selected_y],
                label="Included",
                alpha=0.9,
            )
        if len(excluded_df):
            self.ax.scatter(
                excluded_df[self.parent.selected_x],
                excluded_df[self.parent.selected_y],
                label="Excluded",
                marker="x",
                alpha=0.8,
            )

        if x_fit is not None and y_fit is not None:
            self.ax.plot(x_fit, y_fit, linestyle="--", label="Fit")

        self.ax.set_xlabel(self.parent.selected_x)
        self.ax.set_ylabel(self.parent.selected_y)
        self.ax.set_title(f"{self.parent.selected_x} vs {self.parent.selected_y}")
        self.ax.legend(loc="best")

        self.parent.checkbox_panel.update_equation(eq_text, stats_text)

        self.canvas.draw_idle()


# ---------------------- MAIN COMPUTE/SCATTER ----------------------
class ComputeScatter(QWidget):
    def __init__(self, table_widget=None, parent_gui=None):
        super().__init__()
        self.setWindowTitle("Scatter Plot & Calibration")
        self.setGeometry(100, 100, 1300, 820)

        self.table_widget = table_widget
        self.parent_gui = parent_gui

        self.data = None
        self.selected_x = None
        self.selected_y = None
        self.checked_rows = []
        self.regression_type = "Linear (OLS)"
        self.force_zero = False
        self.excluded_standards = set()

        # --- Layout ---
        main = QHBoxLayout(self)

        # left: log
        self.log_panel = LogPanel()
        main.addWidget(self.log_panel, 1)

        # right grid: table, periodic, scatter, controls
        grid = QGridLayout()
        self.data_table = DataTablePanel()
        self.scatter_panel = ScatterPlotPanel(self)
        self.periodic_panel = PeriodicTablePanel(self)
        self.checkbox_panel = CheckboxPanel(self)

        grid.addWidget(self.data_table, 0, 0)
        grid.addWidget(self.periodic_panel, 1, 0)
        grid.addWidget(self.scatter_panel, 0, 1)
        grid.addWidget(self.checkbox_panel, 1, 1)

        right_container = QWidget()
        right_container.setLayout(grid)
        main.addWidget(right_container, 3)

        # auto-load
        self.load_data("Calibration.xlsx")

    # ---------------- helpers ----------------
    def log(self, message):
        self.log_panel.log(message)

    def include_all_points(self):
        self.excluded_standards.clear()
        self.log("Included all points.")
        self.plot_scatter()

    def exclude_all_points(self):
        if self.data is None:
            return
        self.excluded_standards = set(self.data["Standard Number"].astype(str).tolist())
        self.log("Excluded all points.")
        self.plot_scatter()

    def clear_exclusions(self):
        self.excluded_standards.clear()
        self.log("Cleared all exclusions.")
        self.plot_scatter()

    def toggle_point_exclusion(self, standard_value):
        s = str(standard_value)
        if s in self.excluded_standards:
            self.excluded_standards.remove(s)
            self.log(f"Included point: Standard {s}")
        else:
            self.excluded_standards.add(s)
            self.log(f"Excluded point: Standard {s}")
        self.plot_scatter()

    # ---------------- data ----------------
    def load_data(self, file_name):
        try:
            self.data = pd.read_excel(file_name)
            self.data_table.update_table(self.data)

            if "Standard Number" in self.data.columns:
                self.checkbox_panel.populate_checkboxes(self.data["Standard Number"].unique())

            # default checked-rows = all
            self.checked_rows = [str(v) for v in self.data["Standard Number"].unique()]

            self.log(f"Loaded file: {file_name}")
        except Exception as e:
            self.log(f"Error loading file: {e}")

    def update_checked_rows(self):
        self.checked_rows = [name for name, cb in self.checkbox_panel.checkboxes.items() if cb.isChecked()]
        self.plot_scatter()

    def update_regression_type(self, t):
        self.regression_type = t
        self.plot_scatter()

    # ---------------- plotting & regression ----------------
    def _prepare_xy(self):
        """Return filtered dataframe with finite x,y and within checked
        rows."""
        if self.data is None or not self.selected_x or not self.selected_y:
            return None

        df = self.data.copy()
        df["Standard Number"] = df["Standard Number"].astype(str)
        df = df[df["Standard Number"].isin([str(v) for v in self.checked_rows])]

        if df.empty:
            return None

        x = pd.to_numeric(df[self.selected_x], errors="coerce")
        y = pd.to_numeric(df[self.selected_y], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        df = df.loc[mask].copy()

        return df

    def _fit_regression(self, df_included):
        """Fit based on self.regression_type; return x_fit, y_fit,
        eq_text, stats_text."""
        x = df_included[self.selected_x].values
        y = df_included[self.selected_y].values

        if len(x) < 2:
            return None, None, "—", "Not enough points"

        x_min, x_max = float(np.min(x)), float(np.max(x))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or not (x_max > x_min):
            return None, None, "—", "Invalid range"

        x_fit = np.linspace(x_min, x_max, 400)

        eq_text = "—"
        stats_text = "—"

        # prepare features
        if "Quadratic" in self.regression_type:
            X = np.vstack([x, x**2]).T  # columns: x, x^2
            X_fit = np.vstack([x_fit, x_fit**2]).T
        else:
            X = x.reshape(-1, 1)
            X_fit = x_fit.reshape(-1, 1)

        # force through zero for linear
        if self.force_zero and "Quadratic" not in self.regression_type:
            # closed-form slope
            slope = float(np.sum(x * y) / np.sum(x * x))
            y_fit = slope * x_fit
            # R2 on included points
            y_pred = slope * x
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
            eq_text = f"y = {slope:.6g} x  (forced 0)"
            stats_text = f"R²={r2:.4f}, RMSE={rmse:.4g}, n={len(x)}"
            return x_fit, y_fit, eq_text, stats_text

        # choose model
        model = None
        if self.regression_type == "Linear (OLS)":
            model = LinearRegression(fit_intercept=True)
        elif self.regression_type == "Linear (Huber)":
            model = HuberRegressor(fit_intercept=True, epsilon=1.35)
        elif self.regression_type == "Linear (RANSAC)":
            base = LinearRegression()
            model = RANSACRegressor(base_estimator=base, min_samples=2, residual_threshold=None)
        elif self.regression_type == "Quadratic (OLS)":
            model = LinearRegression(fit_intercept=True)
        else:
            model = LinearRegression(fit_intercept=True)

        model.fit(X, y)
        y_fit = model.predict(X_fit)
        y_pred = model.predict(X)

        # stats
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

        # equation text
        if "Quadratic" in self.regression_type:
            # solve coefficients for y = a*x + b*x^2 + c (order matches X)
            # derive via linear regression coefs:
            a = float(model.coef_[0])
            b = float(model.coef_[1])
            c = float(model.intercept_)
            eq_text = f"y = {b:.6g} x² + {a:.6g} x + {c:.6g}"
        else:
            a = float(model.coef_[0])
            c = float(model.intercept_)
            eq_text = f"y = {a:.6g} x + {c:.6g}"

        stats_text = f"R²={r2:.4f}, RMSE={rmse:.4g}, n={len(x)}"
        return x_fit, y_fit, eq_text, stats_text

    def plot_scatter(self):
        df = self._prepare_xy()
        if df is None:
            self.log("Warning: No matching or valid rows in dataset.")
            return

        # split into included/excluded by "Standard Number"
        std = df["Standard Number"].astype(str)
        inc_mask = ~std.isin(self.excluded_standards)
        df_included = df.loc[inc_mask].copy()
        df_excluded = df.loc[~inc_mask].copy()

        # fit on included only
        if len(df_included) >= 2:
            x_fit, y_fit, eq_text, stats_text = self._fit_regression(df_included)
        else:
            x_fit, y_fit, eq_text, stats_text = (
                None,
                None,
                "—",
                "Not enough included points",
            )

        # store for hover (use entire df to allow hover over excluded too)
        self.scatter_panel.data = df.copy()

        # draw
        self.scatter_panel.plot(df_included, df_excluded, x_fit, y_fit, eq_text, stats_text)


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ComputeScatter()
    w.show()
    sys.exit(app.exec_())
