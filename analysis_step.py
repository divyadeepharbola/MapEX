import os
import h5py
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QTextEdit, QRadioButton, QButtonGroup, QLabel, QSlider,QTableWidget, QTableWidgetItem,
                             QPushButton, QMessageBox, QDialog, QLineEdit, QDialogButtonBox, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import RectangleSelector, LassoSelector
import csv
import openpyxl
import os

# Set the number of CPU cores to use
# os.environ["LOKY_MAX_CPU_COUNT"] = "12"

plt.rcParams['font.family'] = 'Times New Roman'

class DataLoader(QThread):
    data_loaded = pyqtSignal(list, list)
    
    def __init__(self, data_path=None):
        super().__init__()
        self.data_path = data_path or "data.h5"

    def run(self):
        try:
            # self.data_path = data_path or "data.h5"
            file_path = self.data_path or "data.h5"
            data_list, image_list = [], []

            with h5py.File(file_path, 'r') as f:
                if "X-ray Maps" in f:
                    for group_name, group in f["X-ray Maps"].items():
                        if 'data' in group:
                            data_list.append((group_name, group['data'][()]))

                if "Images" in f:
                    for image_name, dataset in f["Images"].items():
                        image_list.append((image_name, dataset[()]))

            self.data_loaded.emit(data_list, image_list)
        except Exception as e:
            self.data_loaded.emit([], [])


class AnalysisStep(QWidget):
    def __init__(self, data_path=None):
        super().__init__()
        self.data_list = []  # Initialize data list
        self.image_list = []
        self.latest_roi_coords = [None, None, None, None]
        self.selected_data = None
        self.selected_names = None
        self.rectangle_selectors = []  # To store RectangleSelector instances
        self.lines = []  # List to store lines for all subplots
        self.line=None
        self.line_start = None  # Initialize line_start attribute
        self.line_selectors = []  # To store LineSelector instances
        self.scatter_plot_limits = None  # To store initial scatter plot limits
        self.data_path = data_path or "data.h5"
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Left panel with radio buttons and log
        left_layout = QVBoxLayout()

        self.log_screen = QTextEdit()
        self.log_screen.setReadOnly(True)
        self.log_screen.setFixedHeight(400)  # Adjust the height as needed
        self.log_screen.setFixedWidth(200)   # Adjust the width as needed
        self.log_screen.setPlaceholderText("Log messages will appear here.")
        left_layout.addWidget(self.log_screen)

        self.radio_buttons = QButtonGroup(self)
        self.radio_correlation = QRadioButton("Correlation plot")
        self.radio_line_scan = QRadioButton("Line scan")
        self.radio_pointdata = QRadioButton("Point Data")
        left_layout.addWidget(self.radio_correlation)
        left_layout.addWidget(self.radio_line_scan)
        left_layout.addWidget(self.radio_pointdata)
        self.radio_buttons.addButton(self.radio_correlation)
        self.radio_buttons.addButton(self.radio_line_scan)
        self.radio_buttons.addButton(self.radio_pointdata)

        # Dropdowns for data selection
        self.dropdown_menus = []
        self.colormap_dropdowns = []
        for i in range(3):
            label = QLabel(f"Elements {i + 1}")
            left_layout.addWidget(label)

            dropdown = QComboBox()
            self.dropdown_menus.append(dropdown)
            left_layout.addWidget(dropdown)

        for dropdown in self.dropdown_menus:
            dropdown.currentIndexChanged.connect(self.dropdown_changed)

        # Right panel with plots
        right_layout = QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        # Add brightness sliders
        self.e1_brightness_slider = QSlider(Qt.Horizontal)
        self.e1_brightness_slider.setRange(0, 200)
        self.e1_brightness_slider.setValue(100)
        self.e1_brightness_slider.setFixedWidth(200)  # Set fixed width to make the slider smaller
        # self.e1_brightness_slider.valueChanged.connect(self.adjust_brightness)
        
        self.e2_brightness_slider = QSlider(Qt.Horizontal)
        self.e2_brightness_slider.setRange(0, 200)
        self.e2_brightness_slider.setValue(100)
        self.e2_brightness_slider.setFixedWidth(200)  # Set fixed width to make the slider smaller
        # self.e2_brightness_slider.valueChanged.connect(self.adjust_brightness)
        
        self.e3_brightness_slider = QSlider(Qt.Horizontal)
        self.e3_brightness_slider.setRange(0, 200)
        self.e3_brightness_slider.setValue(100)
        self.e3_brightness_slider.setFixedWidth(200)  # Set fixed width to make the slider smaller
        # self.e3_brightness_slider.valueChanged.connect(self.adjust_brightness)

        left_layout.addWidget(QLabel('Element 1 Brightness'))
        left_layout.addWidget(self.e1_brightness_slider)
        left_layout.addWidget(QLabel('Element 2 Brightness'))
        left_layout.addWidget(self.e2_brightness_slider)
        left_layout.addWidget(QLabel('Element 3 Brightness'))
        left_layout.addWidget(self.e3_brightness_slider)

        # Connect sliders to the update function
        self.e1_brightness_slider.valueChanged.connect(self.update_combined_image)
        self.e2_brightness_slider.valueChanged.connect(self.update_combined_image)
        self.e3_brightness_slider.valueChanged.connect(self.update_combined_image)


        # Add interactive buttons
        button_layout = QHBoxLayout()
        self.save_roi_button = QPushButton("Save ROI")
        self.save_correlation_button = QPushButton("Save Correlation")
        self.manual_input_button = QPushButton("Manual Input")
        self.quick_reset_button = QPushButton("Quick Reset")
        self.full_reset_button = QPushButton("Refresh")
        self.calibration_button = QPushButton("Load Calibration Data")

        button_layout.addWidget(self.save_roi_button)
        button_layout.addWidget(self.save_correlation_button)
        button_layout.addWidget(self.manual_input_button)
        button_layout.addWidget(self.quick_reset_button)
        button_layout.addWidget(self.full_reset_button)
        # button_layout.addWidget(self.calibration_button)
        # button_layout.addWidget(self.add_calibration_button())
        right_layout.addLayout(button_layout)

        # Combine layouts
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

        # Connect signals
        self.radio_correlation.toggled.connect(self.show_correlation_plot)
        self.radio_line_scan.toggled.connect(self.show_line_scan)
        self.radio_pointdata.toggled.connect(self.show_point_data)
        self.save_roi_button.clicked.connect(self.save_roi)
        self.save_correlation_button.clicked.connect(self.save_correlation)
        self.manual_input_button.clicked.connect(self.manual_input)
        self.quick_reset_button.clicked.connect(self.quick_reset)
        self.full_reset_button.clicked.connect(self.full_reset)
        # self.calibration_button.clicked.connect(self.load_calibration_data)

        # self.add_calibration_button()

        self.loader = DataLoader(data_path=self.data_path)
        self.loader.data_loaded.connect(self.data_loaded)
        self.loader.start()
    def dropdown_changed(self):
        if self.radio_correlation.isChecked():
            self.show_correlation_plot(True)
        elif self.radio_line_scan.isChecked():
            self.show_line_scan(True)

    def data_loaded(self, data_list, image_list):
        self.data_list = data_list
        self.image_list = image_list
        if data_list and image_list is not []:
            self.log_message("Data and images loaded successfully.")
            for dropdown in self.dropdown_menus:
                dropdown.clear()
                dropdown.addItems([name for name, _ in data_list])
        else:
            self.log_message("Failed to load data and images.")


    def log_message(self, message):
        self.log_screen.append(message)

    def show_point_data(self, checked):
        if checked:
            self.remove_rectangle_selector()
            self.remove_line_selector()
            self.figure.clear()
            self.log_message("Switching to Point Data mode...")
            self.load_calibration_data()  # Automatically load calibration data
            self.axs = self.figure.subplots(2, 3)
            self.plot_point_data()
            self.add_pixel_selector()
            self.canvas.draw()

    def plot_point_data(self):
        if len(self.data_list) < 3 or len(self.image_list) < 2:
            QMessageBox.warning(self, "Error", "Not enough data or images loaded for plotting.")
            return

        self.selected_data = [data[1] for data in self.data_list]  # Use all available maps
        self.selected_names = [os.path.splitext(data[0])[0] for data in self.data_list]

        self.axs[0, 0].imshow(self.selected_data[0], cmap='Reds')
        self.axs[0, 0].set_title(f"{self.selected_names[0]} Map")

        self.axs[0, 1].imshow(self.selected_data[1], cmap='Greens')
        self.axs[0, 1].set_title(f"{self.selected_names[1]} Map")

        color_image = self.combine_data(self.selected_data[:3])
        self.axs[0, 2].imshow(color_image)
        self.axs[0, 2].set_title(f"{self.selected_names[0]}-{self.selected_names[1]}-{self.selected_names[2]} (RGB) Map")

        self.axs[1, 0].imshow(self.image_list[0][1], cmap='gray')
        self.axs[1, 0].set_title(f"{os.path.splitext(self.image_list[0][0])[0]}")

        self.axs[1, 1].imshow(self.image_list[1][1], cmap='gray')
        self.axs[1, 1].set_title(f"{os.path.splitext(self.image_list[1][0])[0]}")

        self.create_table()
        self.canvas.draw()

    def create_table(self):
        if hasattr(self, 'table_widget'):
            self.table_widget.deleteLater()

        self.table_widget = QTableWidget()
        self.table_widget.setRowCount(len(self.selected_names))
        self.table_widget.setColumnCount(4)  # 4 columns now
        self.table_widget.setHorizontalHeaderLabels(["Element", "Intensity", "Quantification", "Quantification (Normalized)"])
        self.table_widget.setFixedWidth(400)  # Adjust table width

        for i, name in enumerate(self.selected_names):
            self.table_widget.setItem(i, 0, QTableWidgetItem(name))
            self.table_widget.setItem(i, 1, QTableWidgetItem("--"))
            self.table_widget.setItem(i, 2, QTableWidgetItem("--"))
            self.table_widget.setItem(i, 3, QTableWidgetItem("--"))  # New column

        self.layout().addWidget(self.table_widget)


    def add_pixel_selector(self):
        def on_click(event):
            if event.inaxes:
                x, y = int(event.xdata), int(event.ydata)
                self.update_table(x, y)
                self.log_message(f"Selected pixel: ({x}, {y})")

        self.cid_click = self.figure.canvas.mpl_connect('button_press_event', on_click)

    def update_table(self, x, y):
        calibration_data = self.load_calibration_data()
        quant_values = []

        for i, dataset in enumerate(self.selected_data):
            intensity_value = dataset[y, x]  # Extract intensity
            self.table_widget.setItem(i, 1, QTableWidgetItem(f"{intensity_value:.2f}"))

            # Perform quantification
            if self.selected_names[i] in calibration_data:
                m, c = calibration_data[self.selected_names[i]]
                if m != 0:
                    quantification_value = (intensity_value - c) / m
                    if quantification_value < 0:
                        quantification_value = 0  # Treat negative values as zero
                        self.table_widget.setItem(i, 2, QTableWidgetItem(""))  # Do not display negative values
                    else:
                        self.table_widget.setItem(i, 2, QTableWidgetItem(f"{quantification_value:.2f}"))
                    quant_values.append(quantification_value)
                else:
                    self.table_widget.setItem(i, 2, QTableWidgetItem("Undefined"))
                    quant_values.append(0)  # Treat undefined as zero
            else:
                quant_values.append(0)  # Treat missing calibration data as zero

        # Normalize quantification values so their sum is 100%
        total_q = sum(quant_values) if sum(quant_values) != 0 else 1  # Avoid division by zero

        for i, quantification_value in enumerate(quant_values):
            if quantification_value > 0:
                normalized_value = (quantification_value / total_q) * 100  # Convert to percentage
                self.table_widget.setItem(i, 3, QTableWidgetItem(f"{normalized_value:.2f}%"))
            else:
                self.table_widget.setItem(i, 3, QTableWidgetItem(""))  # Do not display zero values


    def load_calibration_data(self):
        calibration_data = {}
        try:
            workbook = openpyxl.load_workbook("Calibration_data.xlsx")
            sheet = workbook.active
            for row in sheet.iter_rows(min_row=2, values_only=True):
                if row and len(row) >= 3:  # Ensure row has at least 3 values
                    element, m, c = row[:3]  # Extract only first three columns
                    calibration_data[element] = (m, c)
            self.log_message("Calibration data loaded successfully.")
        except Exception as e:
            self.log_message(f"Failed to load calibration data: {e}")
        return calibration_data

    def add_calibration_button(self, button_layout):
        self.calibration_button = QPushButton("Load Calibration Data")
        self.calibration_button.clicked.connect(self.load_calibration_data)
        self.layout().addWidget(self.calibration_button)

    def show_correlation_plot(self, checked):
        if checked:
            self.remove_line_selector()
            self.figure.clear()
            self.log_message("Switching to Correlation plot mode...")
            self.axs = self.figure.subplots(2, 3)
            self.plot_correlation()
            self.add_rectangle_selector()
            self.canvas.draw()

    def show_line_scan(self, checked):
        if checked:
            self.remove_rectangle_selector()
            self.figure.clear()
            self.log_message("Switching to Line Scan mode...")
            self.axs = self.figure.subplots(2, 3)
            self.plot_linescan()
            self.add_line_selector()
            self.canvas.draw()   

    def remove_rectangle_selector(self):
        # Disable and remove all rectangle selectors
        for selector in self.rectangle_selectors:
            selector.set_active(False)
        
        # Clear the list of rectangle selectors
        self.rectangle_selectors = []

        # Log the removal
        self.log_message("Rectangle selectors have been removed.")

    def remove_line_selector(self):
        # Disconnect event handlers to disable the line selector
        if hasattr(self, 'cid_press') and self.cid_press:
            self.figure.canvas.mpl_disconnect(self.cid_press)
            self.cid_press = None

        if hasattr(self, 'cid_motion') and self.cid_motion:
            self.figure.canvas.mpl_disconnect(self.cid_motion)
            self.cid_motion = None

        if hasattr(self, 'cid_release') and self.cid_release:
            self.figure.canvas.mpl_disconnect(self.cid_release)
            self.cid_release = None

        # Remove the line if it exists
        if hasattr(self, 'line') and self.line:
            self.line.remove()
            self.line = None
            self.figure.canvas.draw()

        # Log the removal
        self.log_message("Line selector has been removed.")   

    def plot_linescan(self):
        # self.dropdown_names=[]
        if len(self.data_list) < 3 or len(self.image_list) < 2:
            QMessageBox.warning(self, "Error", "Not enough data or images loaded for plotting.")
            return

        # Get selected datasets
        selected_data = [self.data_list[dropdown.currentIndex()][1] for dropdown in self.dropdown_menus]
        selected_names = [os.path.splitext(self.data_list[dropdown.currentIndex()][0])[0] for dropdown in self.dropdown_menus]
        self.selected_data = selected_data
        self.selected_names = selected_names


        # Plot individual datasets
        self.axs[0, 0].imshow(selected_data[0], cmap='Reds')
        self.axs[0, 0].set_title(f"{selected_names[0]} Map")

        self.axs[0, 1].imshow(selected_data[1], cmap='Greens')
        self.axs[0, 1].set_title(f"{selected_names[1]} Map") 

        # Combine datasets into an RGB image
        color_image = self.combine_data(selected_data)
        self.axs[0, 2].imshow(color_image)
        titile=f"{selected_names[0]}-{selected_names[1]}-{selected_names[2]} (RGB) Map"
        self.axs[0, 2].set_title(titile)
        # self.axs[0, 2].set_title("Combined RGB")

        # Plot loaded images
        self.axs[1, 0].imshow(self.image_list[0][1], cmap='gray')
        self.axs[1, 0].set_title(f"{os.path.splitext(self.image_list[0][0])[0]}")

        self.axs[1, 1].imshow(self.image_list[1][1], cmap='gray')
        self.axs[1, 1].set_title(f"{os.path.splitext(self.image_list[1][0])[0]}")

                # Plot line scan data
        rows, cols = selected_data[0].shape

        # Extract the centerline (middle row) from each dataset
        center_line_0 = selected_data[0][rows // 2, :]
        center_line_1 = selected_data[1][rows // 2, :]
        center_line_2 = selected_data[2][rows // 2, :]

        # Normalize the centerlines by the maximum intensity in each dataset
        max_intensity_0 = np.max(selected_data[0])  # Maximum intensity in dataset 0
        max_intensity_1 = np.max(selected_data[1])  # Maximum intensity in dataset 1
        max_intensity_2 = np.max(selected_data[2])  # Maximum intensity in dataset 2

        # Normalize the centerline values
        center_line_0_normalized = center_line_0 / max_intensity_0 if max_intensity_0 != 0 else center_line_0
        center_line_1_normalized = center_line_1 / max_intensity_1 if max_intensity_1 != 0 else center_line_1
        center_line_2_normalized = center_line_2 / max_intensity_2 if max_intensity_2 != 0 else center_line_2

        # Create the pixel indices for the x-axis
        pixel_indices = np.arange(cols)

        # Plot the line scan on the specified subplot
        self.axs[1, 2].plot(pixel_indices, center_line_0_normalized, label=selected_names[0],color="Red", alpha=0.7)
        self.axs[1, 2].plot(pixel_indices, center_line_1_normalized, label=selected_names[1], color="Green",alpha=0.7)
        self.axs[1, 2].plot(pixel_indices, center_line_2_normalized, label=selected_names[2], color="Blue", alpha=0.7)

        # Update plot appearance
        self.axs[1, 2].set_title("Line Scan")
        self.axs[1, 2].set_xlabel("Pixels")
        self.axs[1, 2].set_ylabel("Normalized Intensity")
        self.axs[1, 2].legend()

        self.canvas.draw()

    def normalize_data(self, data):
        min_data = np.min(data)
        max_data = np.max(data)
        
        if max_data == min_data:
            # If max and min are the same, return original data (or some other handling)
            return data
        
        return (data - min_data) / (max_data - min_data)  
    
    def combine_data(self, selected_data, r_brightness=1.0, g_brightness=1.0, b_brightness=1.0):
        # Normalize the selected datasets
        red = self.normalize_data(selected_data[0]) * r_brightness
        green = self.normalize_data(selected_data[1]) * g_brightness
        blue = self.normalize_data(selected_data[2]) * b_brightness

        # Clip the values to the range [0, 1]
        adjusted_red = np.clip(red, 0, 1)
        adjusted_green = np.clip(green, 0, 1)
        adjusted_blue = np.clip(blue, 0, 1)

        # Combine into an RGB image
        color_image = np.stack((adjusted_red, adjusted_green, adjusted_blue), axis=-1)
        return color_image

    def update_combined_image(self):
        # Get brightness values from sliders (assuming range is 0-100, normalize to 0-1)
        r_brightness = self.e1_brightness_slider.value() / 100
        g_brightness = self.e2_brightness_slider.value() / 100
        b_brightness = self.e3_brightness_slider.value() / 100

        titile=f"{self.selected_names[0]}-{self.selected_names[1]}-{self.selected_names[2]} (RGB) Map"

        # Recombine data with updated brightness values
        if len(self.selected_data) >= 3:  # Ensure there are at least 3 datasets
            color_image = self.combine_data(self.selected_data, r_brightness, g_brightness, b_brightness)

            # Update the RGB image plot
            self.axs[0, 2].imshow(color_image)
            self.axs[0, 2].set_title(titile)

            # Redraw the canvas
            self.canvas.draw()
        else:
            QMessageBox.warning(self, "Error", "Not enough datasets selected for RGB combination.")

    def plot_correlation(self):
        if len(self.data_list) < 3 or len(self.image_list) < 2:
            QMessageBox.warning(self, "Error", "Not enough data or images loaded for plotting.")
            return

        # Get selected datasets
        selected_data = [self.data_list[dropdown.currentIndex()][1] for dropdown in self.dropdown_menus]
        selected_names = [os.path.splitext(self.data_list[dropdown.currentIndex()][0])[0] for dropdown in self.dropdown_menus]
        self.selected_data = selected_data
        self.selected_names = selected_names

        # Plot individual datasets
        self.axs[0, 0].imshow(selected_data[0], cmap='Reds')
        self.axs[0, 0].set_title(f"{selected_names[0]} Map")

        self.axs[0, 1].imshow(selected_data[1], cmap='Greens')
        self.axs[0, 1].set_title(f"{selected_names[1]} Map")

        # Plot correlation
        scatter = self.axs[1, 2].scatter(selected_data[0].flatten(), selected_data[1].flatten(), alpha=0.5)
        self.scatter_plot_limits = self.axs[1, 2].axis()  # Store initial scatter plot limits
        self.axs[1, 2].set_title("Correlation Plot")
        self.axs[1, 2].set_xlabel(f"{selected_names[0]} Intensity")
        self.axs[1, 2].set_ylabel(f"{selected_names[1]} Intensity")

        # Combine datasets into an RGB image
        color_image = self.combine_data(selected_data)
        self.axs[0, 2].imshow(color_image)
        titile=f"{selected_names[0]}-{selected_names[1]}-{selected_names[2]} (RGB) Map"
        self.axs[0, 2].set_title(titile)

        # Plot loaded images
        self.axs[1, 0].imshow(self.image_list[0][1], cmap='gray')
        self.axs[1, 0].set_title(f"{os.path.splitext(self.image_list[0][0])[0]}")

        self.axs[1, 1].imshow(self.image_list[1][1], cmap='gray')
        self.axs[1, 1].set_title(f"{os.path.splitext(self.image_list[1][0])[0]}")

        # Redraw the canvas
        self.canvas.draw()


    def add_rectangle_selector(self):
        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)

            # Update ROI coordinates
            self.latest_roi_coords = [x1, y1, x2, y2]

            # Log the selection
            self.log_message(f"Selected region: ({x1}, {y1}) to ({x2}, {y2})")

            # Update plots
            self.update_zoom(x1, y1, x2, y2)

        # Clear previous selectors to avoid duplicates
        for selector in self.rectangle_selectors:
            selector.set_active(False)

        self.rectangle_selectors = []

        for ax in self.axs.flat:
            selector = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels')
            selector.set_active(True)
            self.rectangle_selectors.append(selector)

    def add_line_selector(self):
        """Enable line drawing in subplots and update line scan."""
        def on_press(event):
            if event.inaxes:
                self.line_start = (event.xdata, event.ydata)

                # Remove any previous lines in all subplots
                for line in self.lines:
                    line.remove()
                self.lines.clear()

                # Define colors for each subplot
                colors = ['blue', 'red', 'yellow', 'white', 'blue']

                # Start a new line in the active subplot
                ax_index = list(self.axs.flat).index(event.inaxes)
                line_color = colors[ax_index % len(colors)]
                line = event.inaxes.plot([event.xdata], [event.ydata], color=line_color, linestyle='--')[0]
                self.lines.append(line)
                self.figure.canvas.draw()

        def on_motion(event):
            if event.inaxes and self.line_start:
                x0, y0 = self.line_start

                # Update the line dynamically in the active subplot
                self.lines[-1].set_data([x0, event.xdata], [y0, event.ydata])
                self.figure.canvas.draw()

        def on_release(event):
            if event.inaxes and self.line_start:
                x0, y0 = self.line_start
                x1, y1 = event.xdata, event.ydata
                self.line_start = None
                self.latest_line_coords = [int(x0), int(y0), int(x1), int(y1)]

                # Finalize the colored lines in all subplots
                colors = ['blue', 'red', 'yellow', 'white', 'blue']
                for i, ax in enumerate(self.axs.flat):
                    line_color = colors[i % len(colors)]
                    line = ax.plot([x0, x1], [y0, y1], color=line_color, linestyle='--')[0]
                    self.lines.append(line)

                self.figure.canvas.draw()

                # Update the line scan plot
                self.plot_line_scan(int(x0), int(y0), int(x1), int(y1))

        # Connect mouse events
        self.cid_press = self.figure.canvas.mpl_connect('button_press_event', on_press)
        self.cid_motion = self.figure.canvas.mpl_connect('motion_notify_event', on_motion)
        self.cid_release = self.figure.canvas.mpl_connect('button_release_event', on_release)

    def plot_line_scan(self, x1, y1, x2, y2):
        """Plot the line scan on subplot (1, 2)."""
        if not self.selected_data:
            QMessageBox.warning(self, "Error", "No datasets selected for line scan.")
            return

        line_values = []
        for dataset in self.selected_data:
            values = self.extract_line_values(dataset, x1, y1, x2, y2)
            # Normalize values by the maximum intensity of the dataset
            max_intensity = np.max(dataset)  # Maximum intensity in the dataset
            normalized_values = values / max_intensity if max_intensity != 0 else values  # Avoid division by zero
            line_values.append(normalized_values)

        # Clear the line scan subplot
        ax = self.axs[1, 2]
        ax.clear()

        # Plot the line scan for each dataset
        ax.plot(range(len(values)), line_values[0], label=self.selected_names[0],color="Red", alpha=0.7)
        ax.plot(range(len(values)), line_values[1], label=self.selected_names[1], color="Green", alpha=0.7)
        ax.plot(range(len(values)), line_values[2], label=self.selected_names[2], color="Blue", alpha=0.7)
        # for i, values in enumerate(line_values):
        #     ax.plot(range(len(values)), values, label=f"Dataset {i + 1}", alpha=0.5)

        # Update plot appearance
        ax.set_title("Line Scan")
        ax.set_xlabel("Pixels")
        ax.set_ylabel("Normalized Intensity")
        ax.legend()
        self.canvas.draw()

    def extract_line_values(self, dataset, x1, y1, x2, y2):
        """Extract pixel values along a line from (x1, y1) to (x2, y2)."""
        num_points = max(abs(x2 - x1), abs(y2 - y1))
        x_coords = np.linspace(x1, x2, num_points).astype(int)
        y_coords = np.linspace(y1, y2, num_points).astype(int)

        x_coords = np.clip(x_coords, 0, dataset.shape[1] - 1)
        y_coords = np.clip(y_coords, 0, dataset.shape[0] - 1)

        return dataset[y_coords, x_coords]

    def update_zoom(self, x1, y1, x2, y2):
        # Update all axes with the selected region
        selected_data = self.selected_data
        if selected_data is None:
            return

        zoomed_data_0 = selected_data[0][y1:y2, x1:x2]
        zoomed_data_1 = selected_data[1][y1:y2, x1:x2]
        color_image = self.combine_data(selected_data)
        zoomed_data_2 = color_image[y1:y2, x1:x2]

        self.axs[0, 0].imshow(zoomed_data_0, cmap='Reds')
        self.axs[0, 1].imshow(zoomed_data_1, cmap='Greens')
        self.axs[0, 2].imshow(zoomed_data_2)
        for ax in self.axs.flat:
            ax.set_xlim(x1, x2)
            ax.set_ylim(y2, y1)

        # Update correlation but retain original axis limits
        self.axs[1, 2].clear()
        self.axs[1, 2].scatter(zoomed_data_0.flatten(), zoomed_data_1.flatten(), alpha=0.5)
        self.axs[1, 2].set_title("Correlation Plot")
        self.axs[1, 2].set_xlabel(f"{self.selected_names[0]} Intensity")
        self.axs[1, 2].set_ylabel(f"{self.selected_names[1]} Intensity")
        if self.scatter_plot_limits:
            self.axs[1, 2].axis(self.scatter_plot_limits)

        # Update images
        self.axs[1, 0].imshow(self.image_list[0][1][y1:y2, x1:x2], cmap='gray')
        self.axs[1, 1].imshow(self.image_list[1][1][y1:y2, x1:x2], cmap='gray')

        self.canvas.draw()

    def save_roi(self):
        if self.latest_roi_coords == [None, None, None, None]:
            QMessageBox.warning(self, "Warning", "No ROI selected.")
            return

        export_dir = os.path.dirname(self.data_path)
        roi_filename = os.path.join(export_dir, 'roi_coordinates.xlsx')
        try:
            if os.path.exists(roi_filename):
                workbook = openpyxl.load_workbook(roi_filename)
                sheet = workbook.active
            else:
                workbook = openpyxl.Workbook()
                sheet = workbook.active
                sheet.append(["X1", "Y1", "X2", "Y2"])

            sheet.append(self.latest_roi_coords)
            workbook.save(roi_filename)
            self.log_message(f"ROI saved to {roi_filename}.")   
            # QMessageBox.information(self, "Success", f"ROI saved to {roi_filename}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save ROI: {e}")

    def save_correlation(self):
        if self.latest_roi_coords == [None, None, None, None]:
            QMessageBox.warning(self, "Warning", "No ROI selected for correlation.")
            return

        export_dir = os.path.dirname(self.data_path)
        correlation_filename = os.path.join(export_dir, 'correlation_data.csv')
        try:
            zoomed_data_0 = self.selected_data[0][self.latest_roi_coords[1]:self.latest_roi_coords[3], self.latest_roi_coords[0]:self.latest_roi_coords[2]].flatten()
            zoomed_data_1 = self.selected_data[1][self.latest_roi_coords[1]:self.latest_roi_coords[3], self.latest_roi_coords[0]:self.latest_roi_coords[2]].flatten()

            with open(correlation_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Dataset 1", "Dataset 2"])
                writer.writerows(zip(zoomed_data_0, zoomed_data_1))

            
            self.log_message(f"Correlation data saved to {correlation_filename}.")   
            # QMessageBox.information(self, "Success", f"Correlation data saved to {correlation_filename}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save correlation data: {e}")

    def manual_input(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Manual Input")
        layout = QVBoxLayout(dialog)

        label = QLabel("Enter coordinates (x1, y1, x2, y2):")
        layout.addWidget(label)

        input_field = QLineEdit()
        layout.addWidget(input_field)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        buttons.accepted.connect(lambda: self.process_manual_input(input_field.text(), dialog))
        buttons.rejected.connect(dialog.reject)

        dialog.exec_()

    def process_manual_input(self, text, dialog):
        try:
            coords = list(map(int, text.split(',')))
            if len(coords) != 4:
                raise ValueError("Invalid number of coordinates.")

            self.latest_roi_coords = coords
            self.log_message(f"Manual input received: {coords}")
            self.update_zoom(*coords)
            dialog.accept()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter four integer values separated by commas.")

    def quick_reset(self):
        # Proceed with resetting the axes and other operations
        for i, ax in enumerate(self.axs.flat):
            ax.autoscale()
            # Check if the plot is in the second row (index 1-2) and third column
            # if (i // len(self.axs[0])) != 1 or (i % len(self.axs[0])) != 2:
            #     ax.set_ylim(ax.get_ylim()[::-1])  # Reverses y-axis to normal orientation
            
        self.canvas.draw_idle()

    def full_reset(self):
        # Check if the 'show_correlation_plot' radio button is toggled
        if self.radio_correlation.isChecked():
            self.show_correlation_plot(True)
        
        # Check if the 'show_line_scan' radio button is toggled
        elif self.radio_line_scan.isChecked():
            self.show_line_scan(True)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = AnalysisStep()
    window.show()
    sys.exit(app.exec_())