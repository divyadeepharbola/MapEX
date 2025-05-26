import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QGridLayout, QCheckBox, QFrame, QScrollArea, QTableWidget, QTableWidgetItem
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QPixmap, QPalette, QColor
from PyQt5.QtWidgets import QWidget, QPushButton, QGridLayout
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox, QLabel
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy.polynomial.polynomial import Polynomial
from PyQt5.QtWidgets import QWidget
from scipy.optimize import curve_fit
from scipy.stats import linregress
import mplcursors

class LogPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("Log messages...")
        # self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)
        self.setMaximumWidth(400)  # Set a max width
        layout.addWidget(self.log_text)
        self.setLayout(layout)
    
    def log(self, message):
        self.log_text.append(message)

class PeriodicTablePannel(QWidget):
    element_selected = pyqtSignal(str)  # Signal to emit selected element

    def __init__(self, parent, scatter_plot):
        super().__init__()
        self.setFixedSize(600, 300)
        self.grid_layout = self.create_periodic_table()
        self.setLayout(self.grid_layout)
        self.parent = parent
        # self.table_widget = table_widget  # Reference to the table widget
        self.scatter_plot = scatter_plot
        self.selected_elements = set()  # Track selected elements
        self.last_selected_button = None

    def element_clicked(self, symbol, button):
        """Handles element selection and updates the scatter plot."""
        if self.last_selected_button:
            self.last_selected_button.setStyleSheet("")  # Reset previous button color

        if symbol in self.selected_elements:
            self.selected_elements.clear()
            self.last_selected_button = None  # No selection now
            self.parent.selected_x = None  # Reset X-axis selection
        else:
            self.selected_elements.clear()
            self.selected_elements.add(symbol)
            button.setStyleSheet("background-color: green; color: white;")
            self.last_selected_button = button  # Store new selected button
            self.parent.selected_x = symbol  # Update selected X-axis element
            self.parent.selected_y = f"{symbol} (intensity)"  # Set corresponding Y-axis

        # print(f"Element selected: {symbol}")  # Debugging
        self.parent.log(f"X-Axis selected: {self.parent.selected_x}, Y-Axis selected: {self.parent.selected_y}")
        self.parent.plot_scatter()  # Update scatter plot

    def create_periodic_table(self):
        elements = [
            ["H", "", "", "", "", "", "", "", "", "", "", "","", "", "", "", "", "He"],
            ["Li", "Be", "", "", "", "", "", "", "", "", "","", "B", "C", "N", "O", "F", "Ne"],
            ["Na", "Mg", "", "", "", "", "", "", "", "","", "", "Al", "Si", "P", "S", "Cl", "Ar"],
            ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
            ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"],
            ["Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"],
            ["Fr", "Ra", "Ac", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"],
            ["", "", "", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"],
            ["", "", "", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]
        ]
        
        grid_layout = QGridLayout()
        grid_layout.setSpacing(0)
        grid_layout.setContentsMargins(10, 10, 10, 10)

        self.default_style = """
            QPushButton {
                background-color: #f0f0f0;
                border: 2px solid #b0b0b0;
                border-radius: 6px;
                padding: 5px;
                font-size: 12px;
                font-weight: bold;
                color: black;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
                border-style: inset;
            }
        """

        for r, row in enumerate(elements):
            for c, element in enumerate(row):
                if element:
                    btn = QPushButton(element)
                    btn.setFixedSize(30, 30)
                    btn.setStyleSheet(self.default_style)
                    btn.clicked.connect(lambda _, e=element, b=btn: self.element_clicked(e, b))
                    grid_layout.addWidget(btn, r if r < 7 else r + 1, c, 1, 1)

        return grid_layout

    
class ScatterPlotPanel(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.data = None  # DataFrame to store plotted data

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Annotation for hover effect
        self.annot = self.ax.annotate("", xy=(0,0), xytext=(10,10),
                                      textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="yellow"),
                                      arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

        # Connect hover event
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

    def plot(self, data, x_col, y_col, x_fit, y_fit):
        """ Plots scatter data and regression line """
        self.ax.clear()
        self.data = data  # Store for reference
        
        # Scatter plot
        self.scatter = self.ax.scatter(data[x_col], data[y_col], color='blue', picker=True)

        # Plot regression line
        self.ax.plot(x_fit, y_fit, color='red', linestyle='--')

        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col)
        self.ax.set_title(f"{x_col} vs {y_col}")

        self.canvas.draw()

    def on_hover(self, event):
        """ Update hover label in CheckboxPanel when hovering over a point """
        if event.inaxes != self.ax:
            return

        mouse_x, mouse_y = event.xdata, event.ydata
        if mouse_x is None or mouse_y is None:
            return

        if self.data is not None:
            x_vals = self.data[self.parent.selected_x].values
            y_vals = self.data[self.parent.selected_y].values
            standard_numbers = self.data["Standard Number"].values  

            # 🔹 Remove NaN and Inf values
            mask = ~np.isnan(x_vals) & ~np.isnan(y_vals) & ~np.isinf(x_vals) & ~np.isinf(y_vals)
            x_vals, y_vals, standard_numbers = x_vals[mask], y_vals[mask], standard_numbers[mask]

            # 🔹 Compute distance
            distances = np.sqrt((x_vals - mouse_x) ** 2 + (y_vals - mouse_y) ** 2)
            closest_idx = np.argmin(distances)

            # 🔹 Increase sensitivity to improve detection
            if distances[closest_idx] < 10:  
                point_name = f"Standard Number: {standard_numbers[closest_idx]}"
                self.annot.xy = (x_vals[closest_idx], y_vals[closest_idx])
                self.annot.set_text(point_name)
                self.annot.set_visible(True)

                # 🔹 Update the hover label in CheckboxPanel
                self.parent.checkbox_panel.hover_label.setText(point_name)
            else:
                self.annot.set_visible(False)
                self.parent.checkbox_panel.hover_label.setText("Hovered Point: None")  

            self.canvas.draw_idle()

class DataTablePanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.table_widget = QTableWidget()
        layout.addWidget(self.table_widget)
        self.setLayout(layout)
    
    def update_table(self, data):
        self.table_widget.setColumnCount(len(data.columns))
        self.table_widget.setRowCount(len(data))
        self.table_widget.setHorizontalHeaderLabels(data.columns.astype(str).tolist())
        for row_idx, row_data in data.iterrows():
            for col_idx, value in enumerate(row_data):
                self.table_widget.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

class CheckboxPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.checkboxes = {}

        # 🔹 Left Layout (Checkboxes)
        self.checkbox_layout = QVBoxLayout()
        self.checkbox_label = QLabel("Select Rows:")
        self.checkbox_label.setStyleSheet("font-weight: bold;")  
        self.checkbox_layout.insertWidget(0, self.checkbox_label)  # Ensures it stays at the top

        # 🔹 Scroll Area for Checkboxes (Prevents Overflow)
        self.checkbox_scroll = QScrollArea()
        self.checkbox_scroll.setWidgetResizable(True)
        self.checkbox_scroll_content = QWidget()
        self.checkbox_scroll_layout = QVBoxLayout(self.checkbox_scroll_content)
        self.checkbox_scroll.setWidget(self.checkbox_scroll_content)
        
        self.checkbox_layout.addWidget(self.checkbox_scroll)  # Add scroll area to the layout

        # 🔹 Vertical Separator Line
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.VLine)
        self.separator.setFrameShadow(QFrame.Sunken)

        # 🔹 Right Layout (Regression Selection)
        self.regression_layout = QVBoxLayout()
                                            
        self.regression_label = QLabel("Select Regression Type:")
        self.regression_label.setStyleSheet("font-weight: bold;")  
        self.regression_layout.insertWidget(0, self.regression_label)  # Ensures it stays at the top

                # Add force zero checkbox
        self.force_zero_checkbox = QCheckBox("Force line through (0,0)")
        self.force_zero_checkbox.stateChanged.connect(self.toggle_force_zero)
        self.regression_layout.addWidget(self.force_zero_checkbox)

        self.regression_dropdown = QComboBox()
        self.regression_dropdown.addItems(["Linear", "First-Order", "Higher-Order"])
        self.regression_dropdown.currentIndexChanged.connect(self.update_regression)
        
        # 🔹 Equation Display Label
        self.equation_label = QLabel("Equation: ")
        self.equation_label.setStyleSheet("font-style: italic; font-weight: bold;")
        
        self.regression_layout.addWidget(self.regression_dropdown)
        self.regression_layout.addWidget(self.equation_label)  # Display equation below dropdown
        self.regression_layout.addStretch()  # Pushes everything to the top

                # 🔹 NEW: Hover Label (Above Regression Type)
        self.hover_label = QLabel("Hovered Point: None")
        self.hover_label.setStyleSheet("font-weight: bold; color: blue;")
        self.regression_layout.insertWidget(0, self.hover_label)

        # 🔹 Combine into Horizontal Layout
        self.main_layout = QHBoxLayout()
        self.main_layout.addLayout(self.checkbox_layout, 1)  # Left: Checkboxes
        self.main_layout.addWidget(self.separator)           # Middle: Vertical Line
        self.main_layout.addLayout(self.regression_layout, 1) # Right: Regression Type

        self.setLayout(self.main_layout)

    def populate_checkboxes(self, rows):
        """ Populate checkboxes based on the available rows """
        for row in rows:
            checkbox = QCheckBox(str(row))
            checkbox.setChecked(True)
            checkbox.toggled.connect(self.checkbox_toggled)
            self.checkboxes[row] = checkbox
            self.checkbox_scroll_layout.addWidget(checkbox)  # Adds checkboxes inside scroll area

    def toggle_force_zero(self):
        """ Update force zero setting and replot """
        self.parent.force_zero = self.force_zero_checkbox.isChecked()
        self.parent.plot_scatter()

    def checkbox_toggled(self):
        """ Update the checked rows whenever a checkbox is toggled """
        self.parent.update_checked_rows()

    def update_regression(self):
        """ Update the selected regression type and refresh plot """
        self.parent.update_regression_type(self.regression_dropdown.currentText())

    def update_equation(self, equation):
        """ Display the equation below the dropdown """
        self.equation_label.setText(f"Equation: {equation}")

class ComputeScatter(QWidget):
    def __init__(self, table_widget=None, parent_gui=None):
        super().__init__()
        self.setWindowTitle("Scatter Plot GUI")
        self.setGeometry(100, 100, 1200, 800)

        # Store references
        self.table_widget = table_widget
        self.parent_gui = parent_gui  # Needed for "Back" button functionality
        self.data = None
        self.selected_element = None
        self.selected_x = None
        self.selected_y = None
        self.checked_rows = []
        self.regression_type = "Linear"
        self.force_zero = False  # Option for forced zero condition

        # 🏗️ Main Layout (Horizontal)
        main_layout = QHBoxLayout()

        # 📜 Log Panel (Left Side)
        log_layout = QVBoxLayout()
        self.log_panel = LogPanel()
        log_layout.addWidget(self.log_panel)

        # 🔙 Back Button (Below Log Panel)
        self.back_button = QPushButton("Back")
        self.back_button.setFixedWidth(400) 
        self.back_button.clicked.connect(self.go_back)  # Ensure function is defined
        log_layout.addWidget(self.back_button, alignment=Qt.AlignBottom)

        # 🔹 Add log_layout to main layout
        log_container = QWidget()
        log_container.setLayout(log_layout)
        main_layout.addWidget(log_container, 1)

        # 📊 Right Layout (Scatter Plot & Table)
        right_layout = QGridLayout()
        self.data_table = DataTablePanel()
        self.scatter_panel = ScatterPlotPanel(self)
        self.periodic_panel = PeriodicTablePannel(self, self.scatter_panel)
        self.checkbox_panel = CheckboxPanel(self)

        # 🔹 Arrange widgets in a grid
        right_layout.addWidget(self.data_table, 0, 0)
        right_layout.addWidget(self.periodic_panel, 1, 0)
        right_layout.addWidget(self.scatter_panel, 0, 1)
        right_layout.addWidget(self.checkbox_panel, 1, 1)

        # 🔹 Add right panel to main layout
        right_container = QWidget()
        right_container.setLayout(right_layout)
        main_layout.addWidget(right_container, 3)

        # 🔹 Set final layout
        self.setLayout(main_layout)

        # 🚀 Auto-load Calibration.xlsx
        self.load_data("Calibration.xlsx")

    def go_back(self):
        """Switch back to the main panel"""
        if self.parent_gui:  # Ensure parent reference exists
            print("⬅️ Returning to the main panel...")
            self.parent_gui.stack.setCurrentWidget(self.parent_gui.main_panel)  # Switch view

    def log(self, message):
        self.log_panel.log(message)

    
    def load_data(self, file_name):
        try:
            self.data = pd.read_excel(file_name)
            self.data_table.update_table(self.data)

            # Ensure 'Standard Number' column exists and populate checkboxes
            if "Standard Number" in self.data.columns:
                self.checkbox_panel.populate_checkboxes(self.data["Standard Number"].unique())
            
            self.log(f"Loaded file: {file_name}")
        except Exception as e:
            self.log(f"Error loading file: {e}")

    
    def update_checked_rows(self):
        self.checked_rows = [name for name, checkbox in self.checkbox_panel.checkboxes.items() if checkbox.isChecked()]
        # self.log(f"Selected Rows: {self.checked_rows}")
        self.plot_scatter()
    
    def compute(self):
        intensity_columns = [col for col in self.data.columns if "intensity" in col.lower()]
        if intensity_columns:
            self.selected_y = intensity_columns[0]
            self.log(f"Y-Axis selected: {self.selected_y}")
            self.plot_scatter()

    def update_regression_type(self, regression_type):
        """ Update the selected regression type """
        self.regression_type = regression_type
        self.plot_scatter()
    
    def plot_scatter(self):
        if self.data is not None and self.selected_x and self.selected_y:
            filtered_data = self.data[self.data["Standard Number"].isin(self.checked_rows)]
            
            if filtered_data.empty:
                self.log("Warning: No matching rows found in the selected dataset.")
                return

            x = filtered_data[self.selected_x].values
            y = filtered_data[self.selected_y].values

            # Remove NaN and Inf values
            mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
            x, y = x[mask], y[mask]

            if len(x) < 2:
                self.log("Warning: Not enough valid data points for regression.")
                return

            equation = ""
            if self.regression_type == "Linear":
                if self.force_zero:
                    # Fit line forcing y = mx (no intercept)
                    slope = np.sum(x * y) / np.sum(x ** 2)
                    equation = f"y = {slope:.4f}x (Forced Zero)"
                    y_fit = slope * x
                else:
                    # Regular linear regression
                    slope, intercept, r_value, _, _ = linregress(x, y)
                    equation = f"y = {slope:.4f}x + {intercept:.4f} (R²={r_value**2:.4f})"
                    y_fit = slope * x + intercept

            elif self.regression_type == "First-Order":
                coeffs = np.polyfit(x, y, 1)
                if self.force_zero:
                    coeffs[1] = 0  # Set intercept to zero
                equation = f"y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}"
                y_fit = np.polyval(coeffs, x)

            elif self.regression_type == "Higher-Order":
                coeffs = np.polyfit(x, y, 2)
                if self.force_zero:
                    coeffs[2] = 0  # Set constant term to zero
                equation = f"y = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.4f}"
                y_fit = np.polyval(coeffs, x)

            # Update equation label
            self.checkbox_panel.update_equation(equation)

            # Plot
            self.scatter_panel.plot(filtered_data, self.selected_x, self.selected_y, x, y_fit)

    def plot_linear_regression(self, x, y):
        """ Perform linear regression and plot """
        model = LinearRegression()
        x_reshaped = x.reshape(-1, 1)  # Reshape for sklearn
        model.fit(x_reshaped, y)
        
        # Predict and plot
        y_pred = model.predict(x_reshaped)
        self.scatter_panel.plot(x, y, x_label=self.selected_x, y_label=self.selected_y, regression_line=y_pred)
        
        # Display equation
        equation = f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}"
        self.log(f"Linear Regression Equation: {equation}")
    
    def plot_polynomial_regression(self, x, y, degree=1):
        """ Perform polynomial regression (first-order or higher-order) """
        model = Polynomial.fit(x, y, degree)
        
        # Predict and plot
        y_pred = model(x)
        self.scatter_panel.plot(x, y, x_label=self.selected_x, y_label=self.selected_y, regression_line=y_pred)
        
        # Display equation
        coefficients = model.convert().coef
        equation = f"y = {' + '.join([f'{coef:.2f}x^{i}' for i, coef in enumerate(coefficients)])}"
        self.log(f"Polynomial Regression Equation: {equation}")

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = ComputeScatter()
#     window.show()
#     sys.exit(app.exec_())
