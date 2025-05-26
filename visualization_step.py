import sys
import os
import h5py
import pandas as pd
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFileDialog, QTextEdit, QColorDialog, QComboBox

# os.environ["LOKY_MAX_CPU_COUNT"] = "12"

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


class VisualizationStep(QWidget):
    def __init__(self, data_path=None):
        super().__init__()
        self.data_path = data_path
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Image Viewer with ROI")
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        self.main_layout = QHBoxLayout()

        # Log section
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setFixedWidth(400)
        self.main_layout.addWidget(self.log_text)

        # Right-side layout
        self.right_layout = QVBoxLayout()

        # Image display area
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(self.image_label)

        # Buttons and Color Picker at the bottom
        self.bottom_layout = QVBoxLayout()  # New layout for color dropdown and buttons

        # Color dropdown
        self.color_dropdown = QComboBox(self)
        self.color_dropdown.addItems(["Red", "Green", "Blue", "Black", "Custom"])
        self.color_dropdown.currentIndexChanged.connect(self.color_changed)
        self.bottom_layout.addWidget(self.color_dropdown)

        self.roi_color = Qt.red  # Default color for ROI

        # Button layout
        self.button_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous", self)
        self.prev_button.clicked.connect(self.show_previous_image)
        self.button_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.show_next_image)
        self.button_layout.addWidget(self.next_button)

        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.upload_image)
        self.button_layout.addWidget(self.upload_button)

        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_current_image_with_roi)
        self.button_layout.addWidget(self.save_button)

        # Add button layout to bottom layout
        self.bottom_layout.addLayout(self.button_layout)

        # Add a spacer to push buttons and dropdown to the bottom
        self.right_layout.addStretch()
        self.right_layout.addLayout(self.bottom_layout)

        # Add the right layout to the main layout
        self.main_layout.addLayout(self.right_layout)
        self.setLayout(self.main_layout)

        # Initialize data variables
        self.image_list = []
        self.roi_coordinates = None
        self.current_image_index = -1
        self.target_width = 512  # Default fallback dimensions
        self.target_height = 512

        # Start loading data
        self.loader = DataLoader(data_path=self.data_path)
        self.loader.data_loaded.connect(self.data_loaded)
        self.loader.start()

        self.load_data()

    def color_changed(self):
        color_name = self.color_dropdown.currentText()
        if color_name == "Red":
            self.roi_color = Qt.red
        elif color_name == "Green":
            self.roi_color = Qt.green
        elif color_name == "Blue":
            self.roi_color = Qt.blue
        elif color_name == "Black":
            self.roi_color = Qt.black
        elif color_name == "Custom":
            color = QColorDialog.getColor()
            if color.isValid():
                self.roi_color = color

    def data_loaded(self, data_list, image_list):
        self.data_list = data_list
        self.image_list = image_list
        if data_list and image_list:
            self.log_message("Data and images loaded successfully.")
            if image_list:
                _, first_image = image_list[0]
                self.target_height, self.target_width, _ = first_image.shape  # Get dimensions from the first image
        else:
            self.log_message("Failed to load data and images.")

    def color_changed(self):
        color_name = self.color_dropdown.currentText()
        if color_name == "Red":
            self.roi_color = Qt.red
        elif color_name == "Green":
            self.roi_color = Qt.green
        elif color_name == "Blue":
            self.roi_color = Qt.blue
        elif color_name == "Black":
            self.roi_color = Qt.black
        elif color_name == "Custom":
            color = QColorDialog.getColor()
            if color.isValid():
                self.roi_color = color

                # Immediately refresh the current image with the new color
        if 0 <= self.current_image_index < len(self.image_list):
            self.display_image(self.current_image_index)

    def log_message(self, message):
        self.log_text.append(message)

    def load_data(self, data_path=None):
        self.log_message("Starting data loading...")

        # Use passed-in data_path or fallback to self.data_path
        data_path = data_path or self.data_path

        # Validate the path to data.h5
        if not data_path or not os.path.exists(data_path):
            self.log_message("Invalid or missing data.h5 path.")
            return

        data_dir = os.path.dirname(data_path)
        roi_file = os.path.join(data_dir, "roi_coordinates.xlsx")

        if os.path.exists(roi_file):
            try:
                self.roi_coordinates = pd.read_excel(roi_file)
                self.log_message(f"Loaded ROI coordinates from {roi_file}")
                self.update_roi_coordinates()
            except Exception as e:
                self.log_message(f"Failed to load ROI file: {e}")
        else:
            self.log_message(f"ROI file not found: {roi_file}")

    def display_image(self, index):
        if 0 <= index < len(self.image_list):
            image_name, image_data = self.image_list[index]

            if isinstance(image_data, np.ndarray):
                height, width, channels = image_data.shape
                bytes_per_line = channels * width
                format_type = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888
                q_image = QImage(bytes(image_data.data), width, height, bytes_per_line, format_type)
                pixmap = QPixmap.fromImage(q_image)
            else:
                pixmap = QPixmap()

            if pixmap.isNull():
                self.log_message(f"Failed to load image: {image_name}")
            else:
                # Draw rectangles on the image
                if self.roi_coordinates is not None:
                    pixmap_with_rois = self.draw_rectangles(pixmap)
                    self.image_label.setPixmap(pixmap_with_rois)
                    self.image_label.setFixedSize(pixmap_with_rois.size())  # Match QLabel to image size
                else:
                    self.image_label.setPixmap(pixmap)
                    self.image_label.setFixedSize(pixmap.size())
                self.log_message(f"Displaying image: {image_name}")

    def draw_rectangles(self, pixmap):
        pixmap_copy = pixmap.copy()
        painter = QPainter(pixmap_copy)
        pen = QPen(self.roi_color, 2, Qt.SolidLine)
        painter.setPen(pen)

        # Font for ROI labels
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(self.roi_color)  # Use the same color for text

        for idx, row in self.roi_coordinates.iterrows():
            x1, y1, x2, y2 = row['X1'], row['Y1'], row['X2'], row['Y2']
            roi_name = row.get('ROI Name', f"ROI {idx + 1}")  # Fallback if 'ROI Name' is missing
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawText(x1, y1 - 5, roi_name)  # Add label above the rectangle

        painter.end()
        return pixmap_copy

    def update_roi_coordinates(self):
        if self.roi_coordinates is not None and len(self.image_list) > 0:
            _, first_image_data = self.image_list[0]
            original_height, original_width, _ = first_image_data.shape

            updated_coordinates = []

            for idx, row in self.roi_coordinates.iterrows():
                # Ensure coordinates are within the original image dimensions
                x1 = max(0, min(original_width, int(row['X1'])))
                y1 = max(0, min(original_height, int(row['Y1'])))
                x2 = max(0, min(original_width, int(row['X2'])))
                y2 = max(0, min(original_height, int(row['Y2'])))
                updated_coordinates.append({'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2})

            # Create a DataFrame with updated coordinates
            updated_roi_df = pd.DataFrame(updated_coordinates)

            # Add ROI Name column
            updated_roi_df['ROI Name'] = [f"ROI {i + 1}" for i in range(len(updated_coordinates))]

            # Save updated DataFrame to a new Excel file
            updated_file = os.path.join(os.getcwd(), "roi_coordinates_updated.xlsx")
            updated_roi_df.to_excel(updated_file, index=False)
            self.log_message(f"Updated ROI coordinates saved to {updated_file}")

            # Update self.roi_coordinates
            self.roi_coordinates = updated_roi_df

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp);;All Files (*)")
        if file_path:
            image_data = QImage(file_path)

            if image_data.isNull():
                self.log_message("Failed to load the selected image.")
                return

            # Preserve aspect ratio and resize with padding to match target dimensions
            scaled_image = image_data.scaled(self.target_width, self.target_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Create a blank QImage with target dimensions and draw the scaled image onto it
            final_image = QImage(self.target_width, self.target_height, QImage.Format_ARGB32)
            final_image.fill(Qt.black)  # Fill with black background

            painter = QPainter(final_image)
            x_offset = (self.target_width - scaled_image.width()) // 2
            y_offset = (self.target_height - scaled_image.height()) // 2
            painter.drawImage(x_offset, y_offset, scaled_image)
            painter.end()

            # Convert final QImage to numpy array
            width = final_image.width()
            height = final_image.height()
            channels = 4 if final_image.hasAlphaChannel() else 3

            try:
                buffer = final_image.bits().asstring(width * height * channels)
                np_image = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, channels))
            except Exception as e:
                self.log_message(f"Error converting image to numpy array: {e}")
                return

            # Swap red and blue channels
            if channels == 3:
                np_image = np_image[..., [2, 1, 0]]
            elif channels == 4:
                np_image = np_image[..., [2, 1, 0, 3]]

            self.image_list.append((os.path.basename(file_path), np_image))
            self.current_image_index = len(self.image_list) - 1

            self.display_image(self.current_image_index)
            self.log_message(f"Uploaded and reshaped image: {file_path}")

    def save_current_image_with_roi(self):
        if 0 <= self.current_image_index < len(self.image_list):
            image_name, image_data = self.image_list[self.current_image_index]

            if isinstance(image_data, np.ndarray):
                height, width, channels = image_data.shape
                bytes_per_line = channels * width
                format_type = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888
                q_image = QImage(image_data.data, width, height, bytes_per_line, format_type)
                pixmap = QPixmap.fromImage(q_image)

                pixmap_with_rois = self.draw_rectangles(pixmap)  # Draw ROIs

                save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", f"{image_name}_with_roi.png", "PNG Files (*.png);;All Files (*)")
                if save_path:
                    pixmap_with_rois.save(save_path)
                    self.log_message(f"Image saved with ROI at: {save_path}")

    def show_next_image(self):
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.display_image(self.current_image_index)

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image(self.current_image_index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = VisualizationStep()
    viewer.show()
    sys.exit(app.exec_())
