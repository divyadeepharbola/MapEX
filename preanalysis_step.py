import sys
import os
import h5py
import pandas as pd
import numpy as np
import re
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal, QEvent
from PyQt5.QtGui import QPainter, QPolygon
from PyQt5.QtGui import QFont, QPixmap, QColor, QPainter, QImage
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFileDialog, QTextEdit, QMessageBox, QWidget,  QPushButton
import textwrap
import subprocess
os.environ["LOKY_MAX_CPU_COUNT"] = "12"

def print_banner(program_name, version, acknowledgment, banner_width=80):
    # Format the version string
    banner_version = f"{program_name} version {version}"

    # Calculate padding for the version string
    padding_width = (banner_width - len(banner_version) - 2) // 2  # Subtract 2 for borders
    left_padding = padding_width
    right_padding = banner_width - len(banner_version) - left_padding - 2  # Subtract 2 for borders

    # Start the HTML output with proper string concatenation
    html_banner = "<pre style='line-height: 1.0; margin: 0; font-family: Courier New, monospace;'>" + "╔" + "═" * (banner_width - 2) + "╗" + "</pre>"
    html_banner += "<pre style='line-height: 1.0; margin: 0; font-family: Courier New, monospace;'>" + " " + ' ' * left_padding + banner_version + ' ' * right_padding + " " + "</pre>"
    html_banner += "<pre style='line-height: 1.0; margin: 0; font-family: Courier New, monospace;'>" + "╠" + "═" * (banner_width - 2) + "╣" + "</pre>"

    # Wrap the acknowledgment text to fit within the banner width
    wrapped_acknowledgment = textwrap.fill(acknowledgment, width=banner_width - 4)  # Subtract 4 for borders and padding

    # Add each line of acknowledgment text with consistent padding
    for line in wrapped_acknowledgment.split('\n'):
        # Calculate padding for each line
        line_length = len(line)
        total_padding = banner_width - line_length - 2  # Subtract 2 for borders
        left_padding_line = total_padding // 2
        right_padding_line = total_padding - left_padding_line

        # Ensure the total length of the line matches the banner width
        html_banner += "<pre style='line-height: 1.0; margin: 0; font-family: Courier New,monospace;'>" + "" + '  ' * left_padding_line + line + '  ' * right_padding_line + "" + "</pre>"

    # Bottom border
    html_banner += "<pre style='line-height: 1.0; margin: 0; font-family: Courier New,monospace;'>" + "╚" + "═" * (banner_width - 2) + "╝" + "</pre>"

    return html_banner



class ArrowLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(50, 50)  # Adjust size as needed

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.black)

        # Define points for a left-pointing arrow
        points = QPolygon([
            QPoint(40, 25),  # Right middle
            QPoint(10, 10),  # Left top
            QPoint(10, 40)   # Left bottom
        ])

        painter.drawPolygon(points)

class PreanalysisStep(QWidget):

    def paint_arrow(self, event, label, direction):
        painter = QPainter(label)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.black)

        w = label.width()
        h = label.height()
        cx = w // 2

        if direction == "down":
            points = QPolygon([
                QPoint(cx, h - 10),        # Bottom center
                QPoint(cx - 15, 10),       # Top left
                QPoint(cx + 15, 10)        # Top right
            ])
        elif direction == "right":
            cy = h // 2
            points = QPolygon([
                QPoint(10, cy),            # Left middle
                QPoint(w - 10, cy - 15),   # Right top
                QPoint(w - 10, cy + 15)    # Right bottom
            ])

        painter.drawPolygon(points)
    def __init__(self):
        super().__init__()
        self.selected_color = None
        self.current_image_path = None
        self.original_image = None
        self.scale = None
        self.padding_x = 0
        self.padding_y = 0
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Banner at the top
        program_name = "MapEX"
        version = "1.0.0"
        acknowledgment = (
            "This program is funded by Prime Minister's Research Fellowship (PMRF) from Ministry of Education, Government of India to Divyadeep Harbola. DH thanks to the Department of Earth Scinces, Indian Institute of Technology Bombay for providing necessary support as a PhD student."
        )

        banner_html = print_banner(program_name, version, acknowledgment)
        banner = QLabel()
        banner.setText(banner_html)
        banner.setAlignment(Qt.AlignCenter)
            # Apply styles: larger font size, white text, blue background
        banner.setStyleSheet("""
            font-size: 20px;
            color: white;
            background-color: #4682B4;
            padding: 10px;
            border-radius: 5px;
            line-height: 1.5; 
        """)
        banner.setFixedHeight(200)
        main_layout.addWidget(banner)

        # Middle section layout
        middle_layout = QHBoxLayout()

        # Log Box on the left
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setPlaceholderText("Log messages will appear here.")
        self.text_area.setFixedWidth(250)
        middle_layout.addWidget(self.text_area)

        # Right side with two vertical sections
        right_layout = QVBoxLayout()

        # Image Modification Section
        image_modification_layout = QHBoxLayout()

        # Left side for buttons
   
        button_layout = QVBoxLayout()
        button_label = QLabel("Image Modification and ROI Extraction")
        button_label.setFont(QFont("Arial", 14, QFont.Bold))
        button_layout.addWidget(button_label)

        upload_image_button = QPushButton("Upload Image")
        upload_image_button.setFont(QFont("Arial", 10))
        upload_image_button.setFixedWidth(200)
        upload_image_button.clicked.connect(self.upload_image)
        button_layout.addWidget(upload_image_button)

        arrow_down1 = ArrowLabel()
        arrow_down1.setFixedSize(200, 50)
        arrow_down1.paintEvent = lambda event: self.paint_arrow(event, arrow_down1, "down")
        button_layout.addWidget(arrow_down1)

        choose_color_button = QPushButton("Choose Colour of ROI")
        choose_color_button.setFont(QFont("Arial", 10))
        choose_color_button.setFixedWidth(200)
        choose_color_button.clicked.connect(self.enable_color_selection)
        button_layout.addWidget(choose_color_button)

        arrow_down2 = ArrowLabel()
        arrow_down2.setFixedSize(200, 50)
        arrow_down2.paintEvent = lambda event: self.paint_arrow(event, arrow_down2, "down")
        button_layout.addWidget(arrow_down2)

        extract_button = QPushButton("Extract")
        extract_button.setFont(QFont("Arial", 10))
        extract_button.setFixedWidth(200)
        extract_button.clicked.connect(self.extract_roi)
        button_layout.addWidget(extract_button)

        arrow_down3 = ArrowLabel()
        arrow_down3.setFixedSize(200, 50)
        arrow_down3.paintEvent = lambda event: self.paint_arrow(event, arrow_down3, "down")
        button_layout.addWidget(arrow_down3)

        upload_text_button = QPushButton("Reshaped Images \nby X-ray Maps")
        upload_text_button.setFont(QFont("Arial", 10))
        upload_text_button.setFixedWidth(200)
        upload_text_button.setToolTip("Resizes the image using X-ray map dimensions. \nUpload a text file (X-ray map)\nImage will be resized to match the dimensions of the map.")
        upload_text_button.clicked.connect(self.resize_images)
        button_layout.addWidget(upload_text_button)

        button_layout.addStretch()
        image_modification_layout.addLayout(button_layout)

        # Color Information Section
        self.color_info_label = QLabel("Hover over the image to see color details.")
        self.color_info_label.setStyleSheet("border: 1px solid black; font-size: 14px; padding: 5px;")
        self.color_info_label.setAlignment(Qt.AlignCenter)
        self.color_info_label.setFixedHeight(50)
        button_layout.addWidget(self.color_info_label)

        # Right side for image display
        self.image_display_label = QLabel("No Image Uploaded")
        self.image_display_label.setAlignment(Qt.AlignCenter)
        self.image_display_label.setStyleSheet("border: 1px solid black;")
        self.image_display_label.setMinimumSize(800, 600)
        image_modification_layout.addWidget(self.image_display_label)

        right_layout.addLayout(image_modification_layout)

        # Create HDF5 File Section
        hdf5_creation_layout = QVBoxLayout()
        hdf5_label = QLabel("Create HDF5 File")
        hdf5_label.setFont(QFont("Arial", 14, QFont.Bold))
        hdf5_creation_layout.addWidget(hdf5_label)

        # Horizontal layout for buttons with arrows
        hdf5_buttons_layout = QHBoxLayout()

        upload_text_button = QPushButton("Upload Text File")
        upload_text_button.setFont(QFont("Arial", 10))
        upload_text_button.clicked.connect(self.upload_text_file)
        hdf5_buttons_layout.addWidget(upload_text_button)

        arrow1 = ArrowLabel()
        hdf5_buttons_layout.addWidget(arrow1)

        upload_reshaped_images_button = QPushButton("Upload Reshaped Images")
        upload_reshaped_images_button.setFont(QFont("Arial", 10))
        upload_reshaped_images_button.clicked.connect(self.upload_reshaped_images)
        hdf5_buttons_layout.addWidget(upload_reshaped_images_button)

        arrow2 = ArrowLabel()
        hdf5_buttons_layout.addWidget(arrow2)

        create_combine_file_button = QPushButton("Create Combine File")
        create_combine_file_button.setFont(QFont("Arial", 10))
        create_combine_file_button.clicked.connect(self.create_combine_file)
        hdf5_buttons_layout.addWidget(create_combine_file_button)

        hdf5_creation_layout.addLayout(hdf5_buttons_layout)
        hdf5_creation_layout.addStretch()
        right_layout.addLayout(hdf5_creation_layout)

        # Add right layout to the middle layout
        middle_layout.addLayout(right_layout)

        # Add middle layout to the main layout
        main_layout.addLayout(middle_layout)

        # Set the layout for the widget
        self.setLayout(main_layout)

        # Add the calibration button to the main layout
        self.calibration_button = QPushButton("Calibration")
        self.calibration_button.clicked.connect(self.open_calibration)
        main_layout.addWidget(self.calibration_button)

    def upload_image(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
            self, "Select an Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)"
            )
            if file_path:
                self.current_image_path = file_path
                self.original_image = cv2.imread(file_path)
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
                self.text_area.append(f"Image loaded: {file_path}")
            else:
                    QMessageBox.warning(self, "No File Selected", "Please select an image file.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while uploading the image:\n{e}")

    def resize_images(self):
        try:
            # Prompt user to select the map file (text file)
            map_file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Map File", "", "Text Files (*.txt *.csv)"
            )
            if not map_file_path:
                QMessageBox.warning(self, "No File Selected", "Please select a map file.")
                return

            # Load the map file data
            map_data, data_type = self.load_data(map_file_path)
            if map_data is None or data_type != 'text':
                QMessageBox.warning(self, "Invalid File", "The selected file is not a valid map file.")
                return

            new_height, new_width = map_data.shape

            # Check if "Clipped_image.tiff" exists
            clipped_image_path = "clipped_image.tiff"
            if not os.path.isfile(clipped_image_path):
                QMessageBox.warning(self, "File Not Found", f"{clipped_image_path} does not exist.")
                return

            # Load the clipped image
            img = cv2.imread(clipped_image_path, cv2.IMREAD_COLOR)
            if img is None:
                QMessageBox.warning(self, "Error", f"Could not load image at {clipped_image_path}")
                return

            # Resize the image to match the map file dimensions
            reshaped_img = cv2.resize(img, (new_width, new_height))

            # Save the resized image in TIFF format
            output_path = "reshaped_clipped_image.tiff"
            cv2.imwrite(output_path, reshaped_img)
            self.text_area.append(f"\n\nReshaped image saved as {output_path}")

            # Display the original and reshaped images
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Original Clipped Image')
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(reshaped_img, cv2.COLOR_BGR2RGB))
            plt.title('Reshaped Clipped Image')
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while resizing the image:\n{e}")

    def enable_color_selection(self):
        if self.original_image is None:
            QMessageBox.warning(self, "No Image", "Please upload an image first.")
            return

        # Calculate scaling factors and padding
        self.calculate_scaling_factors()

        # Enable mouse events
        self.image_display_label.installEventFilter(self)
        self.image_display_label.setCursor(Qt.CrossCursor)

    def calculate_scaling_factors(self):
        label_width = self.image_display_label.width()
        label_height = self.image_display_label.height()
        image_height, image_width = self.original_image.shape[:2]

        if label_width / label_height > image_width / image_height:
            self.scale = label_height / image_height
            self.padding_x = (label_width - image_width * self.scale) / 2
            self.padding_y = 0
        else:
            self.scale = label_width / image_width
            self.padding_x = 0
            self.padding_y = (label_height - image_height * self.scale) / 2

    def eventFilter(self, source, event):
        if source == self.image_display_label and event.type() == QEvent.MouseMove:
            self.update_color_info(event)
            return True
        return super().eventFilter(source, event)

    def update_color_info(self, event):
        if self.original_image is None:
            return

        label_x = event.pos().x()
        label_y = event.pos().y()

        x = int((label_x - self.padding_x) / self.scale)
        y = int((label_y - self.padding_y) / self.scale)

        if 0 <= x < self.original_image.shape[1] and 0 <= y < self.original_image.shape[0]:
            color = self.original_image[y, x]
            color_hex = QColor(color[2], color[1], color[0]).name()
            self.color_info_label.setStyleSheet(
                f"border: 1px solid black; background-color: {color_hex}; font-size: 14px; padding: 5px;"
            )
            self.color_info_label.setText(f"RGB: {color[2]}, {color[1]}, {color[0]}")
        else:
            self.color_info_label.setStyleSheet(
                "border: 1px solid black; font-size: 14px; padding: 5px;"
            )
            self.color_info_label.setText("Hover over the image to see color details.")

    def extract_roi(self):
        if self.original_image is None:
            QMessageBox.warning(self, "No Image", "Please upload an image first.")
            return

        hover_color = self.color_info_label.styleSheet()
        match = re.search(r'background-color: (#\w+);', hover_color)
        if not match:
            QMessageBox.warning(self, "No Color Selected", "Please hover over the image to select a color.")
            return

        selected_color_hex = match.group(1)
        selected_color_rgb = QColor(selected_color_hex).getRgb()[:3]


        # Convert the selected RGB color to HSV
        selected_color_rgb = np.array([[selected_color_rgb]], dtype=np.uint8)  # Reshape for cv2.cvtColor
        selected_color_hsv = cv2.cvtColor(selected_color_rgb, cv2.COLOR_RGB2HSV)[0][0]
        self.text_area.append(f"\n\nSelected Color (RGB): {selected_color_rgb}")

        # Apply Gaussian blur to reduce noise
        hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        # blurred_image = cv2.GaussianBlur(self.original_image, (5, 5), 0)

        # # Convert the blurred image to HSV
        # hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

        # Define a tolerance for the selected color in HSV space
        hue_tolerance = 30
        saturation_tolerance = 50
        value_tolerance = 50

        # Convert selected_color_hsv to int to avoid overflow during arithmetic operations
        selected_color_hsv = selected_color_hsv.astype(int)

    # Calculate lower and upper bounds
        lower_bound = np.array([
            max(selected_color_hsv[0] - hue_tolerance, 0),
            max(selected_color_hsv[1] - saturation_tolerance, 0),
            max(selected_color_hsv[2] - value_tolerance, 0)
        ], dtype=np.uint8)

        upper_bound = np.array([
            min(selected_color_hsv[0] + hue_tolerance, 179),  # Hue range is 0-179 in OpenCV
            min(selected_color_hsv[1] + saturation_tolerance, 255),
            min(selected_color_hsv[2] + value_tolerance, 255)
        ], dtype=np.uint8)

        # Create a mask using the tolerance range in HSV space
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Display the mask for debugging
        cv2.imshow("Debug Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Find contours in the mask
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.imshow("HSV Mask", mask)
        # cv2.imshow("Detected Contours", cv2.drawContours(hsv_image.copy(), contours, -1, (0, 255, 0), 2))
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) < 100:  # Ignore very small areas
                QMessageBox.warning(self, "Small ROI", "The detected region is too small to extract.")
                return

            x, y, w, h = cv2.boundingRect(largest_contour)
            # Define a margin to exclude the green border
            margin = 5  # Adjust this value as needed
            x += margin
            y += margin
            w -= 2 * margin
            h -= 2 * margin

            roi = self.original_image[y:y + h, x:x + w]
            output_path = "clipped_image.tiff"
            cv2.imwrite(output_path, roi)

            cv2.imshow("Extracted ROI", roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            self.text_area.append(f"\n\nRegion extracted and saved to {output_path}.")

        else:
            QMessageBox.warning(self, "No ROI Found", "No region matching the selected color was found.")

    def upload_text_file(self):
        try:
            text_files, _ = QFileDialog.getOpenFileNames(
                self, "Select Text Files for X-ray Maps", "", "Text Files (*.txt *.csv)"
            )
            if not text_files:
                QMessageBox.warning(self, "No Files Selected", "Please select at least one text file.")
                return
            self.text_files = text_files
            self.text_area.append("Loaded text files:")
            for file in text_files:
                self.text_area.append(f"\n{file}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while loading text files:\n{e}")

    def upload_reshaped_images(self):
        try:
            image_files, _ = QFileDialog.getOpenFileNames(
                self, "Select Image Files", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)"
            )
            if not image_files:
                QMessageBox.warning(self, "No Files Selected", "Please select at least one image file.")
                return
            self.image_files = image_files
            self.text_area.append("\n\nLoaded image files:")
            for file in image_files:
                self.text_area.append(f"\n{file}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while loading image files:\n{e}")

    def create_combine_file(self):
        try:
            if not hasattr(self, 'text_files') or not hasattr(self, 'image_files'):
                QMessageBox.warning(self, "Files Not Loaded", "Please load text and image files first.")
                return
    
            # Define the file path for saving as 'data.h5'
            output_dir = os.path.dirname(self.text_files[0])
            hdf5_file = os.path.join(output_dir, "data.h5")
    
            # Save the HDF5 file directly
            with h5py.File(hdf5_file, 'w') as hdf:
                xray_maps_group = hdf.create_group('X-ray Maps')
                images_group = hdf.create_group('Images')
    
                # Add text file data to X-ray Maps group
                for text_file in self.text_files:
                    data, data_type = self.load_data(text_file)
                    if data is not None and data_type == 'text':
                        # Extract the file name from the full path
                        file_name = os.path.basename(text_file)
                        
                        # Create a subgroup for the file name under X-ray Maps
                        xray_subgroup = xray_maps_group.create_group(file_name)
                        
                        # Store the data as a dataset in the subgroup (using the default dtype)
                        xray_subgroup.create_dataset("data", data=data)
    
                # Add image data to Images group
                for image_file in self.image_files:
                    data, data_type = self.load_data(image_file)
                    if data is not None and data_type == 'image':
                        # Extract the file name from the full image path (to avoid directories in the name)
                        image_name = os.path.basename(image_file)
                        
                        # Create the dataset with just the image file name, not the full path
                        images_group.create_dataset(image_name, data=data)
    
            self.text_area.append(f"\n\n\n HDF5 file created successfully: {hdf5_file}")
            self.data_h5_path = hdf5_file
    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while creating the HDF5 file:\n{e}")

    def load_data(self, file_path):
        try:
            # Check the file extension to determine if it's a text file or an image file
            file_extension = file_path.split('.')[-1].lower()
    
            if file_extension in ['txt', 'csv']:
                # Read text data
                start_reading = False
                data = []
    
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        # Skip the header lines until we find a line with at least 5-6 continuous numbers
                        if not start_reading:
                            # Use regex to check for 5-6 continuous numbers separated by ; , space, or tab
                            if re.search(r'(\d+(\.\d+)?[\s,;,\t]+){5,6}\d+(\.\d+)?', line):
                                start_reading = True  # Found the first set of continuous numbers
                                # Convert the first line into a list of numbers and append to data
                                row = list(map(float, re.split(r'[;\s,]+', line.strip())))
                                data.append(row)
                                # print(f"Reading data: {row}")  # Debugging print to verify the data
                            continue  # Skip this line if not starting to read
    
                        # Once we start reading, process the line as numeric data
                        try:
                            # Convert line into a list of numbers
                            row = list(map(float, re.split(r'[;\s,]+', line.strip())))
                            data.append(row)
                            # print(f"Reading data: {row}")  # Debugging print to verify the data
                        except ValueError:
                            # In case of lines that cannot be converted to numbers, continue reading
                            continue
    
                # Convert the data list to a numpy array
                if data:
                    data = np.array(data)
                    return data, 'text'
                else:
                    print(f"No valid data found in {file_path}")
                    return None, 'text'
    
            elif file_extension in ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff']:
                # Read image data
                image = Image.open(file_path)
                data = np.array(image)
                return data, 'image'
    
            else:
                return None, None
    
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None, None

    def open_calibration(self):
        # Run Calibration.py as a separate process
        try:
            subprocess.run(['python', 'Calibration.py'], check=True)
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"An error occurred while running Calibration.py:\n{e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = PreanalysisStep()
    viewer.show()
    sys.exit(app.exec_())