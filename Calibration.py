import sys
import os
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFileDialog, QTextEdit, QMessageBox, QTableWidget, QTableWidgetItem, QStackedWidget,QGridLayout, QCheckBox
from PyQt5.QtGui import QPixmap, QPalette, QColor
from PyQt5.QtCore import Qt, QProcess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import re
import cv2
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PIL import Image
from functools import partial
import matplotlib.patches as patches
import openpyxl
import matplotlib
matplotlib.use('Qt5Agg')
from compute_scatter import ComputeScatter
import pandas as pd

class CustomTableWidget(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def keyPressEvent(self, event):
        """ Delete selected row(s) when 'Delete' key is pressed """
        if event.key() == Qt.Key_Delete:
            self.delete_selected_rows()
        else:
            super().keyPressEvent(event)  # Handle other key events normally

    def delete_selected_rows(self):
        """ Deletes the selected row(s) from the table """
        selected_rows = set(item.row() for item in self.selectedItems())  # Get selected row indices
        for row in sorted(selected_rows, reverse=True):  # Delete rows in reverse order
            self.removeRow(row)

class ImageTextViewer(QWidget):
    def __init__(self, table_widget):  # Pass the table widget
        super().__init__()
        self.table_widget = table_widget  # Store the reference
        self.text_files = []
        # self.histogram = Histogram  # Store reference
        
        self.canvas = FigureCanvas(plt.figure(figsize=(5, 5)))
        # self.histogram = Histogram()  # Create histogram instance
        self.load_image_button = QPushButton("Load Image")
        self.load_text_button = QPushButton("Load 2D Text Files")
        self.left_button = QPushButton("\u2190")
        self.right_button = QPushButton("\u2192")
        self.add_standard_button = QPushButton("Add as Standard")
        self.reset_zoom_button = QPushButton("Reset Zoom")
        # self.status_label = QLabel("Select an ROI to zoom in.")  # Added status label
        self.left_button.setEnabled(False)
        self.right_button.setEnabled(False)

        self.load_image_button.clicked.connect(self.load_image)
        self.load_text_button.clicked.connect(self.load_text_files)
        self.left_button.clicked.connect(self.show_previous_text)
        self.right_button.clicked.connect(self.show_next_text)
        self.add_standard_button.clicked.connect(self.add_standard)
        self.reset_zoom_button.clicked.connect(self.reset_zoom)

        self.text_files = []
        self.current_index = -1
        self.image = None
        self.roi = None  # Store ROI coordinates
        self.start_point = None  # Mouse press start point
        self.end_point = None  # Mouse release end point
        self.rect = None  # Rectangle patch for visualization
        self.standard_count = 0  # Counter for standard numbers
        
        layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_image_button)
        button_layout.addWidget(self.load_text_button)
        layout.addLayout(button_layout)

        layout.addWidget(self.canvas)
        
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.left_button)
        nav_layout.addWidget(self.right_button)
        nav_layout.addWidget(self.add_standard_button)
        nav_layout.addWidget(self.reset_zoom_button)
        layout.addLayout(nav_layout)
        # layout.addWidget(self.status_label)  # Add status label to layout

        self.setLayout(layout)
        self.setWindowTitle("Image and 2D Text Viewer")

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
    
    def clear_roi(self):
        if self.image is None:
            return

        self.image = self.original_image.copy()  # Restore original image
        self.roi = None  # Clear ROI
        self.plot_image_and_text()  # Refresh the displayed image

    def on_press(self, event):
        if event.inaxes:
            self.start_point = (int(event.xdata), int(event.ydata))
            if self.rect:
                self.rect.remove()
            self.rect = None
            self.canvas.draw()

    def on_motion(self, event):
        if event.inaxes and self.start_point is not None and event.button == 1:  # Only draw if mouse is pressed
            x1, y1 = self.start_point
            x2, y2 = int(event.xdata), int(event.ydata)
            width, height = abs(x2 - x1), abs(y2 - y1)
            x_min, y_min = min(x1, x2), min(y1, y2)

            self.canvas.figure.clear()
            ax = self.canvas.figure.add_subplot(111)
            self.plot_image_and_text(ax)

            self.rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(self.rect)
            self.canvas.draw()

    def on_release(self, event):
        if event.inaxes and self.start_point:
            self.end_point = (int(event.xdata), int(event.ydata))
            x1, y1 = self.start_point
            x2, y2 = self.end_point

            # Convert ROI relative to the full original image/text data
            if hasattr(self, "roi") and self.roi:
                x1 += self.roi[0]
                y1 += self.roi[1]
                x2 += self.roi[0]
                y2 += self.roi[1]

            self.roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            self.reset_zoom_button.setEnabled(True)

            self.canvas.figure.clear()
            ax = self.canvas.figure.add_subplot(111)
            self.plot_image_and_text(ax)
            
            self.rect = patches.Rectangle((self.roi[0], self.roi[1]), self.roi[2], self.roi[3], 
                                        linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(self.rect)
            self.canvas.draw()


    def reset_zoom(self):
        self.roi = None
        self.reset_zoom_button.setEnabled(False)
        # self.status_label.setText("Zoom reset. Select a new ROI.")  # Update status label
        self.plot_image_and_text()

    def add_standard(self):
        self.standard_count += 1
        standard_number = f"CS{self.standard_count}"  # Ensure it's a number

        row_position = self.table_widget.rowCount()
        self.table_widget.insertRow(row_position)

        standard_col, coordinate_col = None, None
        for col in range(self.table_widget.columnCount()):
            header_item = self.table_widget.horizontalHeaderItem(col)
            if header_item:
                header_text = header_item.text()
                if header_text == "Standard Number":
                    standard_col = col
                elif header_text == "Coordinates":
                    coordinate_col = col

        if standard_col is not None:
            self.table_widget.setItem(row_position, standard_col, QTableWidgetItem(standard_number))  # Fix

        if coordinate_col is not None and self.roi:
            self.table_widget.setItem(row_position, coordinate_col, QTableWidgetItem(str(self.roi)))

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp *.jpeg *.gif *.tiff *.tif)")
        if file_name:
            self.image, _ = self.load_data(file_name)  # Load color image
            self.original_image = self.image.copy()  # Store original full image
            self.image_path = file_name
            self.current_index = -1  # Reset to show image first
            self.plot_image_and_text()

    def load_text_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Text Files", "", "Text Files (*.txt *.csv)")
        if files:
            self.text_files = files
            self.text_files = files
            self.current_index = 0
            self.left_button.setEnabled(len(files) > 1)
            self.right_button.setEnabled(len(files) > 1)
            self.plot_image_and_text()
        return self.text_files

    def plot_image_and_text(self, ax=None):
        if self.image is None and not self.text_files:
            return  # Nothing to plot

        if ax is None:
            self.canvas.figure.clear()
            ax = self.canvas.figure.add_subplot(111)

        title = ""

        # Load the text data dimensions to reshape the image
        target_shape = None
        text_data = None
        if self.text_files and self.current_index >= 0:
            text_data, _ = self.load_data(self.text_files[self.current_index])
            if text_data is not None:
                target_shape = text_data.shape  # Store the shape of the text data

        if self.current_index == -1 and self.image is not None:
            image_to_plot = self.image
            if target_shape and self.image.shape[:2] != target_shape:
                image_to_plot = cv2.resize(self.image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            
            if self.roi:
                x, y, w, h = self.roi
                # Ensure cropping is within bounds
                if x < 0 or y < 0 or x + w > image_to_plot.shape[1] or y + h > image_to_plot.shape[0]:
                    self.status_label.setText("ROI out of bounds. Resetting zoom.")
                    self.roi = None
                else:
                    image_to_plot = image_to_plot[y:y+h, x:x+w]  # Crop ROI
                
            ax.imshow(image_to_plot)
            title = "Image: " + self.image_path.split("/")[-1]

        elif self.text_files and self.current_index >= 0:
            if text_data is not None:
                # Step 1: Store vmin and vmax only once (before zooming)
                if not hasattr(self, 'fixed_vmin'):
                    self.fixed_vmin = text_data.min()
                    self.fixed_vmax = text_data.max()

                if self.roi:
                    x, y, w, h = self.roi
                    if x < 0 or y < 0 or x + w > text_data.shape[1] or y + h > text_data.shape[0]:
                        self.status_label.setText("ROI out of bounds. Resetting zoom.")
                        self.roi = None
                    else:
                        # Step 2: Crop the ROI but keep the original vmin and vmax
                        text_data = text_data[y:y+h, x:x+w]

                # 🔹 Step 3: Apply fixed vmin and vmax to ensure consistent colors
                ax.imshow(text_data, cmap="viridis", vmin=self.fixed_vmin, vmax=self.fixed_vmax, alpha=1)
                title = "Text Data: " + self.text_files[self.current_index].split("/")[-1]
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        self.canvas.draw()

    def draw_rois(self, roi_list):
        if self.image is None:
            return

        image_with_rois = self.image.copy()
        
        for x, y, w, h, label in roi_list:
            cv2.rectangle(image_with_rois, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_with_rois, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.image = image_with_rois  # Update displayed image
        self.plot_image_and_text()  # Refresh plot

    def show_previous_text(self):
        """Show the previous text file."""
        if self.current_index > -1:
            self.current_index -= 1
            self.update_status_label()
            self.plot_image_and_text()

    def show_next_text(self):
        """Show the next text file or the image."""
        if self.current_index < len(self.text_files) - 1:
            self.current_index += 1
        elif self.image is not None and self.current_index == len(self.text_files) - 1:
            self.current_index = -1  # Go back to the image
        self.update_status_label()
        self.plot_image_and_text()

    def update_status_label(self):
        """Update the status label with the current file index."""
        if self.current_index == -1:
            # self.status_label.setText("Viewing Image")
            pass
        else:
            pass
            # self.status_label.setText(f"File {self.current_index + 1}/{len(self.text_files)}")

    def load_data(self, file_path):
        try:
            file_extension = file_path.split('.')[-1].lower()
            if file_extension in ['txt', 'csv']:
                start_reading = False
                data = []
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        if not start_reading:
                            # Check for 5+ continuous numbers separated by `;`, `,`, ` `, or `\t`
                            if re.search(r'(\d+(\.\d+)?[\s,;,\t]+){5,6}\d+(\.\d+)?', line):
                                start_reading = True
                                row = list(map(float, re.split(r'[;\s,]+', line.strip())))
                                data.append(row)
                            continue
                        try:
                            row = list(map(float, re.split(r'[;\s,]+', line.strip())))
                            data.append(row)
                        except ValueError:
                            continue  # Skip non-numeric lines
                if data:
                    return np.array(data), 'text'
                else:
                    self.log_message(f"No valid data found in {file_path}")
                    return None, 'text'
            elif file_extension in ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff']:
                image = cv2.imread(file_path)  # Load image (BGR format)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                return image, 'image'
            else:
                return None, None
        except FileNotFoundError:
            self.log_message(f"File not found: {file_path}")
            return None, None

class PeriodicTable(QWidget):
    def __init__(self, table_widget, scatter_plot, log_function):
        super().__init__()
        self.log_message = log_function
        self.setWindowTitle("Periodic Table GUI")
        self.setFixedSize(600, 300)
        self.set_palette()
        self.grid_layout = self.create_periodic_table()
        self.setLayout(self.grid_layout)
        self.table_widget = table_widget  # Reference to the table widget
        self.scatter_plot = scatter_plot
        self.selected_elements = set()  # Track selected elements

    def set_palette(self):
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("white"))
        self.setPalette(palette)

    def element_clicked(self, symbol, button):
        if symbol in self.selected_elements:
            self.selected_elements.remove(symbol)
            button.setStyleSheet(self.default_style)
            self.remove_element_column(symbol)
            self.log_message(f"{symbol} unselected")  # Log unselection
        else:
            self.selected_elements.add(symbol)
            button.setStyleSheet("background-color: green; color: white;")
            self.add_element_column(symbol)
            self.log_message(f"{symbol} selected")  # Log selection

    
    def update_scatter_plot(self, element):
        """Update scatter plot with the selected element's data."""
        if self.scatter_plot:
            self.log_message(f"Updating scatter plot for: {element}")  # Debugging
            self.scatter_plot.set_selected_element(element)


    def add_element_column(self, symbol):
        column_count = self.table_widget.columnCount()
        self.table_widget.insertColumn(column_count)
        self.table_widget.setHorizontalHeaderItem(column_count, QTableWidgetItem(symbol))

    def remove_element_column(self, symbol):
        for col in range(self.table_widget.columnCount()):
            if self.table_widget.horizontalHeaderItem(col).text() == symbol:
                self.table_widget.removeColumn(col)
                break

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


class scatterplot(FigureCanvas):
    def __init__(self, table_widget, standard_table, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)
        self.table_widget = table_widget
        self.standard_table = standard_table
        self.selected_elements = set()
        self.plot_empty()

    def plot_empty(self):
        self.ax.clear()
        self.ax.set_title("Select an Element")
        self.ax.set_xlabel("Standard Number")
        self.ax.set_ylabel("Element Intensity")
        self.draw()

    def update_plot(self, selected_elements):
        self.selected_elements = selected_elements
        self.ax.clear()
        
        checked_standards = []
        for row in range(self.standard_table.rowCount()):
            checkbox = self.standard_table.cellWidget(row, 1)
            if checkbox and checkbox.isChecked():
                checked_standards.append(self.standard_table.item(row, 0).text())
        
        for element in self.selected_elements:
            x_data, y_data = [], []
            col_index = self.get_element_column_index(element)
            for row in range(self.table_widget.rowCount()):
                std_item = self.table_widget.item(row, 0)
                y_item = self.table_widget.item(row, col_index)
                if std_item and y_item and std_item.text() in checked_standards:
                    try:
                        x_data.append(float(std_item.text()))
                        y_data.append(float(y_item.text()))
                    except ValueError:
                        continue
            if x_data and y_data:
                self.ax.scatter(x_data, y_data, label=element)
        
        self.ax.legend()
        self.draw()

    def get_element_column_index(self, element):
        self.log_message(f"🔍 Searching for column: {element}")
        for col in range(self.table_widget.columnCount()):
            header_item = self.table_widget.horizontalHeaderItem(col)
            if header_item:
                self.log_message(f"Checking column {col}: {header_item.text()}")  # Debugging output
            if header_item and header_item.text() == element:
                self.log_message(f"✅ Found column {col} for {element}")
                return col
        self.log_message(f"❌ Column for {element} not found.")
        return None
    
class CalibrationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.roi = None  # Initialize ROI as None
        # self.compute_panel = None  # Initialize ComputePanel as None
        self.scatter_plot = None  
        self.init_ui()

    def init_ui(self):
        self.stack = QStackedWidget()
        self.main_panel = QWidget()
        main_layout = QVBoxLayout(self.main_panel)
        top_layout = QHBoxLayout()

        # Log Panel on the left
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setPlaceholderText("Log messages will appear here.")
        self.text_area.setFixedWidth(250)
        top_layout.addWidget(self.text_area)

        right_layout = QVBoxLayout()

        left_right_layout = QVBoxLayout()
        self.table_widget = CustomTableWidget()
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["Standard Number", "Coordinates", "Name"])
        # self.table_widget.setSortingEnabled(True)
        header = self.table_widget.horizontalHeader()
        header.setSectionsClickable(True)  # Allow clicking on headers to trigger sorting
        header.setSortIndicatorShown(True)  # Show the sorting arrow

        # Connect header click event to sorting function
        # header.sectionClicked.connect(self.table_widget.sortItems)

        # Track sorting states
        self.sort_states = {}  # Dictionary to track sorting state per column

        # Connect the header click event to custom sorting function
        header.sectionClicked.connect(self.custom_sort)

        self.image_viewer = ImageTextViewer(self.table_widget)
        self.text_files = []  # Store text files

        self.standard_table = QTableWidget(5, 2)  # Create standard table
        self.standard_table.setHorizontalHeaderLabels(["Name", "Select"])
        for row in range(5):
            self.standard_table.setItem(row, 0, QTableWidgetItem(f"Standard {row+1}"))
            self.standard_table.setCellWidget(row, 1, QCheckBox())
        self.scatter_plot = scatterplot(self.table_widget,self.standard_table)
        self.periodic_table = PeriodicTable(self.table_widget, None, getattr(self, "log_message", lambda msg: None))
        left_right_layout.addWidget(self.image_viewer)
        left_right_layout.addWidget(self.periodic_table)

        right_layout.addWidget(self.table_widget)

        horizontal_layout = QHBoxLayout()
        horizontal_layout.addLayout(left_right_layout)
        horizontal_layout.addLayout(right_layout)

        top_layout.addLayout(horizontal_layout)
        main_layout.addLayout(top_layout)

        # Create a horizontal layout for buttons
        button_layout = QHBoxLayout()

        # Add buttons to the horizontal layout
        self.load_roi_button = QPushButton("Load ROI")
        self.load_roi_button.setFixedSize(100, 30)
        self.load_roi_button.clicked.connect(self.load_roi_data)
        button_layout.addWidget(self.load_roi_button)

        self.load_calibration_button = QPushButton("Load Calibration Data")
        self.load_calibration_button.setFixedSize(170, 30)
        self.load_calibration_button.clicked.connect(self.load_calibration_data)
        button_layout.addWidget(self.load_calibration_button)

        self.draw_standard_button = QPushButton("Draw Standard")
        self.draw_standard_button.setFixedSize(140, 30)
        self.draw_standard_button.clicked.connect(self.draw_standard_roi)
        button_layout.addWidget(self.draw_standard_button)

        self.remove_roi_button = QPushButton("Remove ROI")
        self.remove_roi_button.setFixedSize(130, 30)
        self.remove_roi_button.clicked.connect(self.remove_standard_roi)
        button_layout.addWidget(self.remove_roi_button)

        self.compute_button = QPushButton("Compute")
        self.compute_button.setFixedSize(100, 30)
        self.compute_button.clicked.connect(self.show_compute_panel)
        button_layout.addWidget(self.compute_button)

        # Add the horizontal button layout to the main layout
        main_layout.addLayout(button_layout)
        main_layout.setAlignment(button_layout, Qt.AlignRight)

        self.stack.addWidget(self.main_panel)  # Add main_panel to the stack

        main_stack_layout = QVBoxLayout(self)
        main_stack_layout.addWidget(self.stack)
        self.setLayout(main_stack_layout)

    def custom_sort(self, column):
        """ Implements three-state sorting for the selected column. """
        if column not in self.sort_states:
            self.sort_states[column] = 0  # Initialize sorting state

        # Cycle through: Ascending (1) → Descending (2) → No Sorting (0)
        self.sort_states[column] = (self.sort_states[column] + 1) % 3

        if self.sort_states[column] == 1:
            self.table_widget.sortItems(column, Qt.AscendingOrder)
            self.table_widget.horizontalHeader().setSortIndicator(column, Qt.AscendingOrder)
        elif self.sort_states[column] == 2:
            self.table_widget.sortItems(column, Qt.DescendingOrder)
            self.table_widget.horizontalHeader().setSortIndicator(column, Qt.DescendingOrder)
        else:
            self.table_widget.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)  # Hide sorting indicator
            self.restore_original_order()  # Restore original row order

    def restore_original_order(self):
        """ Restores the original row order in the table. """
        for i in range(self.table_widget.rowCount()):
            self.table_widget.setRowHidden(i, False)  # Ensure all rows are visible

    def remove_standard_roi(self):
        if not hasattr(self.image_viewer, "image") or self.image_viewer.image is None:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return

        self.image_viewer.clear_roi()
        QMessageBox.information(self, "Success", "ROIs removed successfully.")

    def draw_standard_roi(self):
        if not hasattr(self.image_viewer, "image") or self.image_viewer.image is None:
            QMessageBox.critical(self, "Error", "No image loaded. Please load an image first.")
            return

        roi_list = []  # Store all ROIs to draw
        
        for row in range(self.table_widget.rowCount()):
            standard_number_item = self.table_widget.item(row, 0)
            coordinate_item = self.table_widget.item(row, 1)

            if standard_number_item and coordinate_item:
                standard_number = standard_number_item.text()
                coordinates_text = coordinate_item.text().strip("()")  # Remove brackets
                
                try:
                    x, y, w, h = map(int, coordinates_text.split(","))
                    roi_list.append((x, y, w, h, standard_number))  # Store ROI + label
                except ValueError:
                    QMessageBox.warning(self, "Warning", f"Invalid coordinates format in row {row+1}: {coordinates_text}")

        if roi_list:
            self.image_viewer.draw_rois(roi_list)  # Draw ROI(s) on image
            QMessageBox.information(self, "Success", "ROIs drawn successfully.")
        else:
            QMessageBox.warning(self, "Warning", "No valid ROIs found in table.")


    def load_calibration_data(self):
        file_path = "Calibration.xlsx"  # Set fixed filename

        if not os.path.exists(file_path):
            QMessageBox.critical(self, "Error", "Calibration.xlsx file not found in the current directory.")
            return

        try:
            df = pd.read_excel(file_path)  # Load Excel file

            # Exclude columns containing '(intensity)'
            filtered_columns = [col for col in df.columns if "(intensity)" not in col]

            if not filtered_columns:
                QMessageBox.critical(self, "Error", "No valid columns found in Calibration.xlsx.")
                return

            # Populate self.table_widget with filtered columns
            self.table_widget.setRowCount(len(df))  
            self.table_widget.setColumnCount(len(filtered_columns))  
            self.table_widget.setHorizontalHeaderLabels(filtered_columns)  

            for row, (_, row_data) in enumerate(df.iterrows()):
                for col, column_name in enumerate(filtered_columns):
                    self.table_widget.setItem(row, col, QTableWidgetItem(str(row_data[column_name])))

            QMessageBox.information(self, "Success", "Calibration data loaded successfully.")

            # Automatically switch to ComputeScatter panel
            self.show_compute_panel()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load Calibration data: {str(e)}")
    

    def load_roi_data(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load ROI Data", "", "Excel Files (*.xlsx *.xls);;All Files (*)", options=options)
        
        if not file_path:
            return  # User canceled the dialog

        try:
            df = pd.read_excel(file_path)  # Read Excel file

            # Ensure required columns exist
            required_columns = ["Standard Number", "Coordinates", "Name"]
            if not all(col in df.columns for col in required_columns):
                QMessageBox.critical(self, "Error", "Excel file must contain 'Standard Number', 'Coordinates', and 'Name' columns.")
                return

            # Populate self.table_widget
            self.table_widget.setRowCount(len(df))  # Set row count
            self.table_widget.setColumnCount(3)  # Ensure correct column count
            self.table_widget.setHorizontalHeaderLabels(required_columns)  # Set headers

            for row, (_, row_data) in enumerate(df.iterrows()):
                for col, column_name in enumerate(required_columns):
                    self.table_widget.setItem(row, col, QTableWidgetItem(str(row_data[column_name])))

            QMessageBox.information(self, "Success", "ROI data loaded successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load ROI data: {str(e)}")

    def get_all_roi(self):
        """Extract all ROIs from the table."""
        rows = self.table_widget.rowCount()
        if rows == 0:
            return []

        # Find the "Coordinate" column (it can be "Coordinates" or similar)
        coordinate_col = None
        for col in range(self.table_widget.columnCount()):
            header_item = self.table_widget.horizontalHeaderItem(col)
            if header_item and "Coordinate" in header_item.text():
                coordinate_col = col
                break

        if coordinate_col is None:
            return []  # No coordinate column found

        rois = []
        for row in range(rows):
            item = self.table_widget.item(row, coordinate_col)
            if item:
                try:
                    # Convert "(x, y, w, h)" -> (x, y, w, h)
                    roi_values = tuple(map(int, item.text().strip("()").split(",")))
                    if len(roi_values) == 4:
                        rois.append(roi_values)
                except ValueError:
                    continue  # Skip rows with incorrect format

        return rois  # Return list of all ROIs

    def show_compute_panel(self):
        self.roi = self.get_all_roi()
        if not self.roi:
            QMessageBox.warning(self, "Warning", "Please select an ROI before computing.")
            return

        text_files = self.image_viewer.text_files
        if not text_files:
            QMessageBox.warning(self, "Warning", "No text files loaded.")
            return
        self.log_message(f"✅ Processing text file: ")
        for file in text_files:
            self.log_message(f"{file} ✅")


        mean_intensities = {}

        for file_path in text_files:
            file_name = os.path.basename(file_path).split(".")[0]
            column_name = f"{file_name} (intensity)"

            text_data, _ = self.image_viewer.load_data(file_path)
            if text_data is None:
                self.log_message(f"❌ No valid data for {file_name}")
                continue

            mean_values = []
            for x, y, w, h in self.roi:
                if x < 0 or y < 0 or x + w > text_data.shape[1] or y + h > text_data.shape[0]:
                    self.log_message(f"⚠️ ROI out of bounds for {file_name}. Skipping.")
                    mean_values.append(None)
                else:
                    roi_data = text_data[y:y+h, x:x+w]
                    mean_values.append(np.mean(roi_data))

            mean_intensities[column_name] = mean_values

        # Ensure all required columns exist in the table
        existing_headers = [self.table_widget.horizontalHeaderItem(i).text() 
                            for i in range(self.table_widget.columnCount())]
        for column_name in mean_intensities.keys():
            if column_name not in existing_headers:
                col_index = self.table_widget.columnCount()
                self.table_widget.insertColumn(col_index)
                self.table_widget.setHorizontalHeaderItem(col_index, QTableWidgetItem(column_name))
                existing_headers.append(column_name)

        # Write intensities into the table
        for row_index in range(len(self.roi)):
            for col in range(self.table_widget.columnCount()):
                header_item = self.table_widget.horizontalHeaderItem(col)
                if header_item and header_item.text() in mean_intensities:
                    value = mean_intensities[header_item.text()][row_index]
                    if value is not None:
                        self.table_widget.setItem(
                            row_index, col, QTableWidgetItem(f"{value:.3f}")
                        )
                        self.log_message(f"✅ Writing {value:.3f} to row {row_index}, column {col} ({header_item.text()})")

        # Save data to Excel
        self.save_to_excel()

        if not hasattr(self, "compute_scatter_panel"):
            self.log_message("📊 Creating ComputeScatter panel...")
            self.compute_scatter_panel = ComputeScatter(self.table_widget, parent_gui=self)  # ✅ Pass self as parent
            self.stack.addWidget(self.compute_scatter_panel)

        self.log_message("🔄 Switching to ComputeScatter panel...")
        self.stack.setCurrentWidget(self.compute_scatter_panel)  # ✅ Switch to ComputeScatter view

    def save_to_excel(self):
        file_name = "Calibration.xlsx"

        # Create a new workbook (Overwrites the existing file)
        workbook = openpyxl.Workbook()
        sheet = workbook.active

        # Write headers
        table_headers = [self.table_widget.horizontalHeaderItem(i).text() 
                        for i in range(self.table_widget.columnCount())]
        sheet.append(table_headers)

        # Write table data
        for row in range(self.table_widget.rowCount()):
            row_data = []
            for col in range(self.table_widget.columnCount()):
                item = self.table_widget.item(row, col)
                row_data.append(item.text() if item else "")
            sheet.append(row_data)

        # Save the workbook (Overwrites the file completely)
        workbook.save(file_name)
        QMessageBox.information(self, "Success", f"Data overwritten in {file_name}")

    def show_main_panel(self):
        self.stack.setCurrentWidget(self.main_panel)

    def log_message(self, message):
        """Append log messages to the log panel."""
        self.text_area.append(message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CalibrationGUI()
    gui.show()
    sys.exit(app.exec_())
