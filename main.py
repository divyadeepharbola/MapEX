import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QMessageBox
from PyQt5.QtGui import QFont
from preanalysis_step import PreanalysisStep
from phase_mapping_step import PhaseMappingStep
from analysis_step import AnalysisStep
from visualization_step import VisualizationStep  # Import the new visualization step
import os

# Set the number of CPU cores to use
# os.environ["LOKY_MAX_CPU_COUNT"] = "12"

class FlowChartGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.current_step = 0  # Track the current step
        self.steps = ["Preanalysis", "Phase Mapping", "Analysis", "Visualization"]  # Add the new step
        self.required_files = {
            "Phase Mapping": "data.h5",
            "Analysis": "data.h5",
            "Visualization": "roi_coordinates.xlsx"  
        }
        self.init_ui()

    def init_ui(self):
        # Set up the main window
        self.setWindowTitle("MapEX")
        self.setGeometry(100, 100, 600, 400)

        # Main vertical layout
        self.main_layout = QVBoxLayout()

        # Step buttons layout
        self.steps_layout = QHBoxLayout()
        self.step_buttons = []

        for i, step in enumerate(self.steps):
            button = QPushButton(step)
            button.setFont(QFont("Arial", 10, QFont.Bold if i == self.current_step else QFont.Normal))
            button.clicked.connect(lambda checked, index=i: self.switch_step(index))  # Connect to step switching
            self.step_buttons.append(button)
            self.steps_layout.addWidget(button)

        # Content area for each step
        self.stack_widget = QStackedWidget()

        # Add the imported step widgets
        self.preanalysis_widget = PreanalysisStep()
        self.phase_mapping_widget = None
        self.analysis_widget = None
        self.visualization_widget = None

        self.stack_widget.addWidget(self.preanalysis_widget)
        self.stack_widget.addWidget(QWidget())  # Placeholder for Phase Mapping
        self.stack_widget.addWidget(QWidget())  # Placeholder for Analysis
        self.stack_widget.addWidget(QWidget())  # Placeholder for Visualization

        # Add layouts and widgets to the main layout
        self.main_layout.addLayout(self.steps_layout)
        self.main_layout.addWidget(self.stack_widget)

        # Set layout to the main window
        self.setLayout(self.main_layout)

        # Initialize with the first step
        self.update_step()

    def update_step(self):
        # Update the stack widget and highlight the current step button
        self.stack_widget.setCurrentIndex(self.current_step)
        for i, button in enumerate(self.step_buttons):
            button.setFont(QFont("Arial", 10, QFont.Bold if i == self.current_step else QFont.Normal))

    def switch_step(self, index):
        step_name = self.steps[index]
        file_path = None

        # Try to get the data.h5 path from Preanalysis step
        data_path = getattr(self.preanalysis_widget, 'data_h5_path', None)

        # Common check for data.h5 requirement
        if step_name in ["Phase Mapping", "Analysis", "Visualization"]:
            if not data_path or not os.path.exists(data_path):
                QMessageBox.warning(self, "File Missing", f"The required file 'data.h5' for step '{step_name}' is missing.")
                return
            file_path = data_path  # Use as input to other widgets

        # Additional check for roi_coordinates.xlsx in Visualization step
        if step_name == "Visualization":
            folder = os.path.dirname(data_path)
            roi_path = os.path.join(folder, "roi_coordinates.xlsx")
            if not os.path.exists(roi_path):
                QMessageBox.warning(self, "File Missing", "The required file 'roi_coordinates.xlsx' for Visualization is missing in the same folder as data.h5.")
                return

        # Instantiate and inject the correct step widget
        if step_name == "Phase Mapping":
            self.phase_mapping_widget = PhaseMappingStep(data_path=file_path)
            self.stack_widget.removeWidget(self.stack_widget.widget(1))
            self.stack_widget.insertWidget(1, self.phase_mapping_widget)

        elif step_name == "Analysis":
            self.analysis_widget = AnalysisStep(data_path=file_path)
            self.stack_widget.removeWidget(self.stack_widget.widget(2))
            self.stack_widget.insertWidget(2, self.analysis_widget)

        elif step_name == "Visualization":
            self.visualization_widget = VisualizationStep(data_path=file_path)
            self.stack_widget.removeWidget(self.stack_widget.widget(3))
            self.stack_widget.insertWidget(3, self.visualization_widget)

        self.current_step = index
        self.update_step()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FlowChartGUI()
    window.show()
    sys.exit(app.exec_())
