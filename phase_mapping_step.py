import sys
import h5py
import pandas as pd
import os
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox,
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QMessageBox, QProgressBar, QSplitter, QComboBox, QLabel, QInputDialog,QLineEdit
)
from PyQt5.QtGui import QIntValidator
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, silhouette_score, davies_bouldin_score, pairwise_distances_argmin_min
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage
from sklearn.cluster import MeanShift, DBSCAN, AffinityPropagation
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.ndimage import gaussian_filter
from PIL import Image
from skfuzzy import cmeans
import math
import os
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from joblib import Parallel, delayed

# Set the number of CPU cores to use
# os.environ["LOKY_MAX_CPU_COUNT"] = "12" 


class PhaseMappingStep(QWidget):
    def __init__(self, data_path=None):
        super().__init__()
        self.init_ui()
        self.data_list = []  # Initialize data_list here
        self.data_path = data_path or "data.h5"
        self.results = None

        # Automatically load data.h5
        self.load_data_automatically()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Create splitter for left and right sections
        main_splitter = QSplitter()

        # Left side layout
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        # Create splitter for log and buttons on the left side
        left_splitter = QSplitter()
        left_splitter.setOrientation(Qt.Vertical)

        # Instructions / Logs Area
        self.text_area = QTextEdit()
        self.text_area.setPlaceholderText("Logs and instructions will appear here.")
        self.text_area.setReadOnly(True)
        left_splitter.addWidget(self.text_area)
        left_splitter.setStretchFactor(0, 2)  # Set stretch factor for the log area

        # Buttons for user actions
        button_widget = QWidget()
        button_layout = QVBoxLayout()
        self.phase_quant_checkbox = QCheckBox("Phase mapping by quantification")
        self.phase_quant_checkbox.stateChanged.connect(self.apply_quantification_if_checked) # Connect checkbox to function
        button_layout.addWidget(self.phase_quant_checkbox)
        # Phase method dropdown
        self.phase_method_label = QLabel("Select Phase Method:")
        self.phase_method_combo = QComboBox()  
        self.phase_method_combo.addItems(["kmeans", "gmm", "fcm", "hierarchical", "pca", "mean_shift", "dbscan", "affinity_propagation"])
        button_layout.addWidget(self.phase_method_label)
        button_layout.addWidget(self.phase_method_combo)

        self.method_combo_label = QLabel("Select Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["elbow", "gap", "davies_bouldin"])
        button_layout.addWidget(self.method_combo_label)
        button_layout.addWidget(self.method_combo)

        self.method_combo2_label = QLabel("Number of Components:")
        self.method_combo2_input = QLineEdit()
        self.method_combo2_input.setPlaceholderText("Enter number of components")
        # Set the QIntValidator to only accept integers
        int_validator = QIntValidator(1, 100)  # Set the range of acceptable numbers
        self.method_combo2_input.setValidator(int_validator)
        button_layout.addWidget(self.method_combo2_label)
        button_layout.addWidget(self.method_combo2_input)

        # Run and save buttons
        self.run_phase_mapping_button = QPushButton("Run Phase Mapping")
        self.run_phase_mapping_button.clicked.connect(self.run_phase_mapping)
        button_layout.addWidget(self.run_phase_mapping_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Progress: %p%")
        self.progress_bar.setValue(0)
        self.progress_bar.hide()  # initially hidden
        button_layout.addWidget(self.progress_bar)


        self.save_results_button = QPushButton("Save Results")
        self.save_results_button.clicked.connect(self.save_results)
        button_layout.addWidget(self.save_results_button)

        button_widget.setLayout(button_layout)
        left_splitter.addWidget(button_widget)

        left_layout.addWidget(left_splitter)
        left_widget.setLayout(left_layout)
        main_splitter.addWidget(left_widget)
        main_splitter.setStretchFactor(0, 1)  # Set stretch factor for the left side

        # Right side layout for the plot
        right_widget = QWidget()
        right_layout = QVBoxLayout()

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        right_widget.setLayout(right_layout)
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(1, 2)  # Set stretch factor for the right side

        main_layout.addWidget(main_splitter)
        self.setLayout(main_layout)

        # Connect the phase method combo box to update the method combo box state
        self.phase_method_combo.currentIndexChanged.connect(self.update_method_combo_state)

    def apply_quantification_if_checked(self):
        """Apply quantification only if checkbox is checked."""
        if not self.phase_quant_checkbox.isChecked():
            self.log_message("🔵 Using raw intensity values for phase mapping.")
            return

        # Load calibration data if not already loaded
        if not hasattr(self, 'calibration_dict'):
            self.load_calibration_data()

        if not hasattr(self, 'calibration_dict'):
            self.log_message(" No calibration data available. Using raw intensity values.")
            return

        # Apply quantification formula to each element map
        new_data_list = []
        for i, element_name in enumerate(self.element_names):
            clean_name = element_name.replace(".txt", "")  # Remove ".txt" from the name
            if clean_name in self.calibration_dict:
                m, c = self.calibration_dict[clean_name]
                self.log_message(f"🔄 Applying quantification for {clean_name}: (Intensity - {c}) / {m}")
                new_data_list.append((self.data_list[i] - c) / m)
            else:
                self.log_message(f"⚠️ No calibration data for {element_name}. Using raw intensity.")
                new_data_list.append(self.data_list[i])

        self.data_list = new_data_list  # Replace raw data with quantified data
        self.log_message(" Quantification applied successfully.")

    def load_calibration_data(self):
        """Load calibration data from Calibration_data.xlsx if available."""
        calibration_file = "Calibration_data.xlsx"

        if not os.path.exists(calibration_file):
            QMessageBox.critical(self, "Error", f"Calibration file {calibration_file} not found.")
            self.log_message(f" Calibration file {calibration_file} not found.")
            return

        try:
            df = pd.read_excel(calibration_file)

            # Ensure required columns exist
            required_columns = {'Elements', 'm', 'c'}
            if not required_columns.issubset(df.columns):
                QMessageBox.critical(self, "Error", "Calibration file must contain 'Elements', 'm', and 'c' columns.")
                self.log_message(" Calibration file must contain 'Elements', 'm', and 'c' columns.")
                return

            # Store calibration values in a dictionary (Element → (m, c))
            self.calibration_dict = {row['Elements']: (row['m'], row['c']) for _, row in df.iterrows()}
            self.log_message(f"✅ Calibration data loaded: {len(self.calibration_dict)} elements found.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load calibration data: {e}")
            self.log_message(f" Error loading calibration data: {e}")

    def update_method_combo_state(self):
        """Enable/Disable the method combo box based on the selected phase method."""
        self.method_combo.clear()  # Clear existing items before adding new ones
        if self.phase_method_combo.currentText() == "kmeans":
            self.method_combo.setEnabled(True)
            self.method_combo2_input.setEnabled(False)
            self.method_combo.addItems(["elbow", "gap", "davies_bouldin"])
        elif self.phase_method_combo.currentText() == "gmm":
            self.method_combo.setEnabled(True)
            self.method_combo2_input.setEnabled(False)
            self.method_combo.addItems(["elbow", "gap", "davies_bouldin", "bic_aic"])
        elif self.phase_method_combo.currentText() == "fcm":
            self.method_combo.setEnabled(True)
            self.method_combo2_input.setEnabled(False)
            self.method_combo.addItems(["fcm_objective", "fcm_silhouette", "fuzzy_partition_coefficient"])
        elif self.phase_method_combo.currentText() in ["hierarchical", "mean_shift", "dbscan", "affinity_propagation"]:
            self.method_combo.setEnabled(False)
            self.method_combo2_input.setEnabled(False)
        elif self.phase_method_combo.currentText() == "pca":
            self.method_combo.setEnabled(False)  # Disable method combo if PCA is selected
            self.method_combo2_input.setEnabled(True)

    def log_message(self, message):
        """Append messages to the text area for user feedback."""
        self.text_area.append(message)

    def load_data_automatically(self):
        """Automatically load the data.h5 file and extract the X-ray Maps group."""
        # file_path = "data.h5"  # Path to the data file
        file_path = self.data_path or "data.h5" # Instead of hardcoded "data.h5"
        
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "Error", f"Data file {file_path} does not exist.")
            self.log_message(f"Data file {file_path} does not exist.")
            return

        try:
            with h5py.File(file_path, 'r') as f:
                if "X-ray Maps" not in f:
                    raise KeyError("Group 'X-ray Maps' not found in the file.")

                xray_maps_group = f["X-ray Maps"]
                self.data_list = []
                self.element_names = []  # Store element names

                for group_name in xray_maps_group.keys():
                    group = xray_maps_group[group_name]
                    if isinstance(group, h5py.Group) and 'data' in group:
                        dataset = group['data']
                        self.data_list.append(dataset[()])
                        self.element_names.append(group_name)
                        self.log_message(f"Loaded dataset '{group_name}'.")

            self.log_message(f"Loaded {len(self.data_list)} intensity maps.")

            # Apply quantification if checkbox is checked
            self.apply_quantification_if_checked()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {e}")
            self.log_message(f" Error loading data: {e}")
    
    def run_phase_mapping(self):
        """Run phase mapping analysis on the loaded data."""
        if not self.data_list:
            QMessageBox.warning(self, "Warning", "No data loaded. Please check the file.")
            self.log_message("No data loaded. Please check the file.")
            return

        # Setup progress bar
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Progress: %p%")
        self.progress_bar.show()

        try:
            combined_data = np.stack(self.data_list, axis=-1)
            self.log_message(" Running phase mapping on quantified data" if self.phase_quant_checkbox.isChecked() else "Running phase mapping on raw intensity data.")

            # Get selected method settings
            phase_method = self.phase_method_combo.currentText()
            method = self.method_combo.currentText()
            n_component = self.method_combo2_input.text()

            if n_component.strip():
                n_component = int(n_component)
                self.log_message(f"Number of components: {n_component}")
            else:
                self.log_message("Number of components is not required")

            # Call mapping logic (will update progress internally)
            self.results, sorted_indices = self.execute_phase_mapping(
                combined_data,
                phase_method=phase_method,
                method=method,
                n_pca_components=n_component,
                max_clusters=10
            )

            self.progress_bar.hide()  # Hide when done
            self.display_results(self.results, sorted_indices)
            self.log_message(" Phase mapping completed successfully.")
            QMessageBox.information(self, "Success", "Phase mapping completed successfully.")

        except Exception as e:
            self.progress_bar.hide()
            QMessageBox.critical(self, "Error", f"Phase mapping failed: {e}")
            self.log_message(f"Error running phase mapping: {e}")

    
    def execute_phase_mapping(
        self,
        element_files,
        phase_method="kmeans",
        method='elbow',
        max_clusters=10,
        preprocess_sigma=1,
        normalize=True,
        n_pca_components=3,
        cmap="jet",
        phase_method_params=None
    ):
        """Execute phase mapping using different clustering or dimensionality reduction methods."""

        if not isinstance(element_files, np.ndarray):
            raise TypeError("Data should be a NumPy array.")

        phase_method_params = phase_method_params or {}

        # ----------------------------
        # STEP 1: Parallel Preprocessing
        # ----------------------------
        self.log_message("🔄 Preprocessing element maps...")

        def preprocess_single(element_map):
            smoothed = gaussian_filter(element_map, sigma=preprocess_sigma)
            if normalize:
                return (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))
            return smoothed

        # Parallel loop (now using only numpy arrays, not class methods)
        maps = []
        # Set progress bar range
        n_maps = element_files.shape[-1]
        self.progress_bar.setRange(0, n_maps)
        self.progress_bar.setValue(0)
        results = Parallel(n_jobs=-1)(
            delayed(PhaseMappingStep.preprocess_map)(element_files[:, :, i], sigma=preprocess_sigma, normalize=normalize)
            for i in range(n_maps)
            )

        for i, result in enumerate(results):
            maps.append(result)
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()

        # ----------------------------
        # STEP 2: Stack and Flatten
        # ----------------------------
        data = np.stack(maps, axis=-1)
        self.log_message(f" Preprocessed data shape: {data.shape}")

        data_flat = data.reshape(-1, data.shape[-1])
        self.log_message(f"Flattened data shape: {data_flat.shape}")

        # ----------------------------
        # STEP 3: Phase Mapping Logic
        # ----------------------------
        if phase_method in ["kmeans", "gmm", "fcm", "hierarchical"]:
            self.log_message(f" Determining optimal clusters for '{phase_method}' using '{method}'...")
            n_clusters = self.optimal_clusters(data_flat, phase_method=phase_method, method=method, max_clusters=max_clusters)
            self.log_message(f"Optimal number of clusters: {n_clusters}")

            self.log_message(f"Running {phase_method} clustering...")
            phase_map, model, sorted_indices = self.create_phase_map(data, phase_method, n_clusters=n_clusters, method=method)
            phase_map_reshaped = phase_map.reshape(data.shape[:-1])
            self.log_message(f"Phase map shape: {phase_map_reshaped.shape}")

        elif phase_method == "pca":
            self.log_message(f"Running PCA with {n_pca_components} components...")
            pca = PCA(n_components=n_pca_components)
            pca_data = pca.fit_transform(data_flat)
            pca_phase_map = np.argmax(pca_data, axis=1)
            phase_map_reshaped = pca_phase_map.reshape(data.shape[:-1])

            cluster_means = [pca_data[pca_phase_map == i].mean(axis=0) for i in range(n_pca_components)]
            sorted_indices = np.argsort(cluster_means)

            explained_variance = np.sum(pca.explained_variance_ratio_)
            self.log_message(f"Explained Variance Ratio: {explained_variance:.2f}")

        elif phase_method == "mean_shift":
            self.log_message("Running Mean Shift clustering...")
            bandwidth = phase_method_params.get("bandwidth", None)
            model = MeanShift(bandwidth=bandwidth)
            model_labels = model.fit_predict(data_flat)
            phase_map_reshaped = model_labels.reshape(data.shape[:-1])
            sorted_indices = np.unique(model_labels)

        elif phase_method == "dbscan":
            self.log_message("Running DBSCAN clustering...")
            eps = phase_method_params.get("eps", 0.5)
            min_samples = phase_method_params.get("min_samples", 5)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan_labels = dbscan.fit_predict(data_flat)
            phase_map_reshaped = dbscan_labels.reshape(data.shape[:-1])
            sorted_indices = np.unique(dbscan_labels)

        elif phase_method == "affinity_propagation":
            self.log_message("Running Affinity Propagation clustering...")
            damping = phase_method_params.get("damping", 0.5)
            preference = phase_method_params.get("preference", None)
            affinity = AffinityPropagation(damping=damping, preference=preference)
            affinity_labels = affinity.fit_predict(data_flat)
            phase_map_reshaped = affinity_labels.reshape(data.shape[:-1])
            sorted_indices = np.unique(affinity_labels)

        else:
            raise ValueError(f"Invalid phase_method '{phase_method}'.")

        # ----------------------------
        # STEP 4: Display Results
        # ----------------------------
        self.log_message("Plotting phase map...")
        self.display_results(phase_map_reshaped, sorted_indices, cmap=cmap)

        return phase_map_reshaped, sorted_indices

    
    def optimal_clusters(self, data, phase_method="kmeans", method='elbow', max_clusters=10):
        """Determine the optimal number of clusters using the specified method."""
        if phase_method in ["kmeans", "gmm"]:
            if method == 'elbow':
                return self.elbow_method(data, phase_method, max_clusters)
            elif method == 'gap':
                return self.gap_statistic(data,phase_method, max_clusters)
            elif method == 'davies_bouldin':
                return self.davies_bouldin_index(data, phase_method,max_clusters)
            elif method == 'bic_aic':
                return self.bic_aic_selection(data, max_clusters)
            else:
                raise ValueError("Invalid method for 'kmeans' or 'gmm'. Choose from 'elbow', 'gap', 'davies_bouldin', or 'bic_aic'.")
        
        elif phase_method == "fcm":
            if method == 'fcm_objective':
                return self.fcm_objective(data, max_clusters)
            elif method == 'fcm_silhouette':
                return self.fcm_silhouette(data, max_clusters)
            elif method == 'fuzzy_partition_coefficient':
                return self.fuzzy_partition_coefficient(data, max_clusters)
            else:
                raise ValueError("Invalid method for 'fcm'. Choose from 'fcm_objective', 'fcm_silhouette', or 'fuzzy_partition_coefficient'.")
        
        elif phase_method == "hierarchical":
            if method == 'hierarchical_clustering':
                return self.hierarchical_clustering(data)
            else:
                raise ValueError("Invalid method for 'hierarchical'. Choose from 'hierarchical_clustering'.")
        
        else:
            raise ValueError("Invalid phase_method. Choose from 'kmeans', 'gmm', 'fcm', or 'hierarchical'.")
    
    def create_phase_map(self, data, phase_method="kmeans", method="elbow", n_clusters=None, **phase_method_params):
        """Create a phase map using various clustering methods."""
        # Flatten data for clustering
        data_flat = data.reshape(-1, data.shape[-1])
        
        # Determine the number of clusters if not provided
        if n_clusters is None:
            n_clusters = self.optimal_clusters(data, phase_method)
    
        model = None
        sorted_indices = None
        model_labels = None  # Ensure model_labels is defined
    
        # K-Means Clustering
        if phase_method == "kmeans":
            self.log_message("Performing K-Means clustering...")
            model = KMeans(n_clusters=n_clusters, random_state=42)
            model_labels = model.fit_predict(data_flat)
            cluster_means = [data_flat[model_labels == i].mean(axis=0) for i in range(n_clusters)]
            sorted_indices = np.argsort(cluster_means)
    
        # Gaussian Mixture Model (GMM)
        elif phase_method == "gmm":
            self.log_message("Performing Gaussian Mixture Model (GMM) clustering...")
            covariance_type = phase_method_params.get("covariance_type", "full")
            model = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, random_state=42)
            model_labels = model.fit_predict(data_flat)
            sorted_indices = np.argsort(model.means_.sum(axis=1))

            # Fuzzy C-Means (FCM)
        elif phase_method == "fcm":
            self.log_message("Performing Fuzzy C-Means clustering...")
            m = phase_method_params.get("fuzziness", 2)
            error = phase_method_params.get("error", 1e-4)
            max_iter = phase_method_params.get("max_iter", 100)
            cntr, u, _, _, _, _, _ = cmeans(data_flat.T, n_clusters, m, error, max_iter)
            model_labels = np.argmax(u, axis=0)
            sorted_indices = np.argsort(cntr.sum(axis=1))
        
        # Hierarchical Clustering
        elif phase_method == "hierarchical":
            self.log_message("Performing Hierarchical clustering...")
            linkage_method = phase_method_params.get("linkage_method", "ward")
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            model_labels = model.fit_predict(data_flat)
            sorted_indices = np.unique(model_labels)
    
        else:
            raise ValueError(f"Invalid phase_method '{phase_method}'. Supported methods: 'kmeans', 'gmm', 'fcm', 'hierarchical'.")
    
        # Reshape the labels to the original map shape
        phase_map = model_labels.reshape(data.shape[:-1])
    
        return phase_map, model, sorted_indices
    
    @staticmethod
    def preprocess_map(map_data, sigma=1, normalize=True):
        smoothed = gaussian_filter(map_data, sigma=sigma)
        if normalize:
            return (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))
        return smoothed

    def elbow_method(self, data, phase_method="kmeans", max_clusters=10):
        """Determine the optimal number of clusters using the Elbow Method."""
        inertia = []
        self.progress_bar.setRange(0, max_clusters)
        self.progress_bar.setValue(0)

        for n in range(1, max_clusters + 1):
            if phase_method == "kmeans":
                model = KMeans(n_clusters=n, n_init=10)
            elif phase_method == "gmm":
                model = GaussianMixture(n_components=n, n_init=10)
            else:
                raise ValueError("Unsupported phase_method. Use 'kmeans' or 'gmm'.")

            model.fit(data)
            inertia.append(model.inertia_ if hasattr(model, 'inertia_') else -model.score(data))

            self.progress_bar.setValue(n)
            QApplication.processEvents()

        # Plot the Elbow curve
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_clusters + 1), inertia, marker='o')
        plt.title(f'Elbow Method for {phase_method.upper()}')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia' if phase_method == "kmeans" else 'Negative Log-Likelihood')
        plt.grid(True)
        plt.show()

        optimal_n, ok = QInputDialog.getInt(self, 'Input', 'Enter the number of clusters from the Elbow plot:', min=1, max=max_clusters, step=1)

        if ok:
            return optimal_n
        else:
            self.log_message("User canceled the input.")
            return None

            
    def gap_statistic(self, data, phase_method="kmeans", max_clusters=10, n_ref=10):
        """Determine the optimal number of clusters using the Gap Statistic."""
        data_flat = data.reshape(-1, data.shape[-1])

        scaler = StandardScaler()
        data_flat = scaler.fit_transform(data_flat)

        self.progress_bar.setRange(0, max_clusters + n_ref)
        self.progress_bar.setValue(0)

        Wk = []
        for k in range(1, max_clusters + 1):
            if phase_method == "kmeans":
                model = KMeans(n_clusters=k, n_init=10)
            elif phase_method == "gmm":
                model = GaussianMixture(n_components=k, n_init=10)
            else:
                raise ValueError("Unsupported phase_method. Use 'kmeans' or 'gmm'.")
            model.fit(data_flat)
            if phase_method == "kmeans":
                Wk.append(np.sum(np.min(pairwise_distances_argmin_min(data_flat, model.cluster_centers_)[1])))
            elif phase_method == "gmm":
                log_likelihood = model.score_samples(data_flat).sum()
                Wk.append(-log_likelihood)

            self.progress_bar.setValue(k)
            QApplication.processEvents()

        ref_disps = []
        for i in range(n_ref):
            ref_data = np.random.random_sample(size=data_flat.shape)
            ref_disp = []
            for k in range(1, max_clusters + 1):
                if phase_method == "kmeans":
                    model = KMeans(n_clusters=k, n_init=10)
                    model.fit(ref_data)
                    ref_disp.append(np.sum(np.min(pairwise_distances_argmin_min(ref_data, model.cluster_centers_)[1])))
                elif phase_method == "gmm":
                    model = GaussianMixture(n_components=k, n_init=10)
                    model.fit(ref_data)
                    log_likelihood = model.score_samples(ref_data).sum()
                    ref_disp.append(-log_likelihood)
            ref_disps.append(ref_disp)

            self.progress_bar.setValue(max_clusters + i + 1)
            QApplication.processEvents()

        ref_disps = np.mean(ref_disps, axis=0)
        gaps = np.log(ref_disps) - np.log(Wk)
        optimal_n = np.argmax(gaps) + 1
        self.log_message(f"Optimum number of clusters in Gap statistics is {optimal_n}")
        return optimal_n

    

    def davies_bouldin_index(self, data, phase_method="kmeans", max_clusters=10):
        """Determine the optimal number of clusters using the Davies-Bouldin Index."""
        data_flat = data.reshape(-1, data.shape[-1])
        scores = []

        self.progress_bar.setRange(0, max_clusters - 1)
        self.progress_bar.setValue(0)

        for i, n in enumerate(range(2, max_clusters + 1), start=1):
            if phase_method == "kmeans":
                model = KMeans(n_clusters=n, n_init=10)
            elif phase_method == "gmm":
                model = GaussianMixture(n_components=n, n_init=10)
            else:
                raise ValueError("Unsupported phase_method. Use 'kmeans' or 'gmm'.")
            
            labels = model.fit_predict(data_flat)
            score = davies_bouldin_score(data_flat, labels)
            scores.append((n, score))

            self.progress_bar.setValue(i)
            QApplication.processEvents()

        optimal_n = min(scores, key=lambda x: x[1])[0]
        self.log_message(f"Optimum number of cluster in Davies-Bouldin Index is {optimal_n}")
        return optimal_n

    

    def bic_aic_selection(self, data, max_clusters=10):
        """Determine the optimal number of clusters using BIC/AIC."""
        bics = []
        aics = []
        self.progress_bar.setRange(0, max_clusters)
        self.progress_bar.setValue(0)

        for n in range(1, max_clusters + 1):
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(data)
            bics.append(gmm.bic(data))
            aics.append(gmm.aic(data))

            self.progress_bar.setValue(n)
            QApplication.processEvents()

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_clusters + 1), bics, marker='o', label='BIC')
        plt.plot(range(1, max_clusters + 1), aics, marker='o', label='AIC')
        plt.title('BIC and AIC for Optimal Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.show()

        optimal_n, ok = QInputDialog.getInt(self, 'Input', 'Enter the number of clusters from the BIC/AIC plot:', min=1, max=max_clusters, step=1)
        return optimal_n if ok else None

    
    def hierarchical_clustering(self, data):
        """Perform hierarchical clustering and plot the dendrogram."""
        Z = scipy_linkage(data, method='ward')
        plt.figure(figsize=(10, 7))
        dendrogram(Z)
        plt.title('Dendrogram')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')
        plt.show()

    def fcm_objective(self, data, phase_method="fcm", max_clusters=10):
        """Calculate and plot the Fuzzy C-Means (FCM) objective function for different numbers of clusters."""
        objective_function = []
        self.progress_bar.setRange(0, max_clusters)
        self.progress_bar.setValue(0)

        for n_clusters in range(1, max_clusters + 1):
            cntr, u, _, _, _, _, _ = cmeans(data.T, n_clusters, 2, 1e-4, 100)
            distances = np.linalg.norm(data[:, None, :] - cntr[None, :, :], axis=2)
            obj_func = np.sum((u.T ** 2) * (distances ** 2))
            objective_function.append(obj_func)

            self.progress_bar.setValue(n_clusters)
            QApplication.processEvents()

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_clusters + 1), objective_function, marker='o')
        plt.title('FCM Objective Function vs Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Objective Function Value')
        plt.grid(True)
        plt.show()

        optimal_n, ok = QInputDialog.getInt(self, 'Input', 'Enter the number of clusters from the Elbow plot:', min=1, max=max_clusters, step=1)
        return optimal_n if ok else None

        
    def fcm_silhouette(self, data, phase_method="fcm", max_clusters=10):
        """Calculate and plot the silhouette scores for FCM."""
        from skfuzzy import cmeans as fuzz_cmeans
        silhouette_scores = []
        self.progress_bar.setRange(0, max_clusters - 1)
        self.progress_bar.setValue(0)

        for i, n_clusters in enumerate(range(2, max_clusters + 1), start=1):
            cntr, u, _, _, _, _, _ = fuzz_cmeans(data.T, n_clusters, 2, 1e-4, 100)
            labels = np.argmax(u, axis=0)
            score = silhouette_score(data, labels)
            silhouette_scores.append(score)

            self.progress_bar.setValue(i)
            QApplication.processEvents()

        plt.figure(figsize=(8, 6))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.title('Silhouette Score vs Number of Clusters for FCM')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.show()

        optimal_n, ok = QInputDialog.getInt(self, 'Input', 'Enter the number of clusters from the silhouette plot:', min=2, max=max_clusters, step=1)
        return optimal_n if ok else None



    def fuzzy_partition_coefficient(self, data, phase_method="fcm", max_clusters=10):
        """Calculate and plot the Fuzzy Partition Coefficient (FPC)."""
        fpc_scores = []
        self.progress_bar.setRange(0, max_clusters)
        self.progress_bar.setValue(0)

        for n_clusters in range(1, max_clusters + 1):
            cntr, u, _, _, _, _, _ = cmeans(data.T, n_clusters, 2, 1e-4, 100)
            fpc = np.sum(u ** 2) / len(data)
            fpc_scores.append(fpc)

            self.progress_bar.setValue(n_clusters)
            QApplication.processEvents()

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_clusters + 1), fpc_scores, marker='o')
        plt.title('Fuzzy Partition Coefficient vs Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('FPC Value')
        plt.grid(True)
        plt.show()

        optimal_n, ok = QInputDialog.getInt(self, 'Input', 'Enter the number of clusters from the Elbow plot:', min=1, max=max_clusters, step=1)
        return optimal_n if ok else None



    def display_results(self, phase_map, sorted_indices, phase_filename="Phase.tiff", pie_filename="PieChart.tiff", cmap='jet', results=None):
        """Display the phase mapping results in the UI and save both the phase map (without title and colorbar) and the pie chart separately."""
        try:
            jet_cmap = plt.get_cmap(cmap)
            fixed_colors = jet_cmap(np.linspace(0, 1, len(sorted_indices)))
            fixed_cmap = ListedColormap(fixed_colors)

            # Count pixels for each phase
            unique, counts = np.unique(phase_map, return_counts=True)
            total_pixels = phase_map.size
            phase_percentages = (counts / total_pixels) * 100

            # Clear figure before plotting
            self.figure.clear()
            self.figure.set_size_inches(12, 6)

            # Define layout for phase map and pie chart
            gs = self.figure.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.4)

            ax1 = self.figure.add_subplot(gs[0])  # Phase Map
            ax2 = self.figure.add_subplot(gs[1])  # Pie Chart

            # Ensure colorbar has the same height as the phase map
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            # Plot phase map in UI
            img = ax1.imshow(phase_map, cmap=fixed_cmap, vmin=0, vmax=len(sorted_indices)-1)
            ax1.set_title("Phase Map", fontsize=14, fontweight='bold')
            ax1.axis('off')
            self.figure.colorbar(img, cax=cax)

            # Plot pie chart in UI
            labels = [f"Phase {i+1}" for i in unique]
            wedges, texts, autotexts = ax2.pie(
                phase_percentages, labels=labels, autopct='%1.1f%%',
                colors=fixed_colors, startangle=90, textprops={'fontsize': 10, 'weight': 'bold'}
            )

            # Function to determine contrasting text color
            def get_contrasting_color(hex_color):
                """Returns white or black depending on background color brightness."""
                rgb = mcolors.hex2color(hex_color)
                brightness = (rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114)
                return "black" if brightness > 0.5 else "white"

            # Set percentage text color dynamically for better contrast
            for wedge, autotext in zip(wedges, autotexts):
                face_color = wedge.get_facecolor()
                hex_color = mcolors.rgb2hex(face_color[:3])
                autotext.set_color(get_contrasting_color(hex_color))

            for text in texts:
                text.set_color("black")

            ax2.set_title("Phase Distribution", fontsize=12, fontweight='bold')

            # Ensure the GUI updates
            self.canvas.draw()

            # --- SAVE SEPARATE FIGURES ---

            # Save Phase Map separately (WITHOUT title & colorbar)
            fig1, ax1_fig = plt.subplots(figsize=(8, 6))
            ax1_fig.imshow(phase_map, cmap=fixed_cmap, vmin=0, vmax=len(sorted_indices)-1)
            ax1_fig.axis('off')  # Remove axis labels, title, and colorbar
            fig1.savefig(phase_filename, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig1)

            # Save Pie Chart separately
            fig2, ax2_fig = plt.subplots(figsize=(6, 6))
            ax2_fig.pie(
                phase_percentages, labels=labels, autopct='%1.1f%%',
                colors=fixed_colors, startangle=90, textprops={'fontsize': 10, 'weight': 'bold'}
            )
            ax2_fig.set_title("Phase Distribution", fontsize=12, fontweight='bold')
            fig2.savefig(pie_filename, dpi=300, bbox_inches='tight')
            plt.close(fig2)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display results: {e}")
            self.log_message(f"Error displaying results: {e}")

        
    def show_popup(self, title, message):
        """Show a non-blocking popup that allows multiple popups to be selected."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)  # Allow selection of multiple popups
        msg_box.show()  # Non-blocking
        
    def save_results(self):
        """Save phase mapping results in TIFF format and HDF5 file."""
        tiff_filename = "Phase.tiff"
        hdf5_filename = self.data_path  # Use the path passed to PhaseMappingStep
        
        try:
            # Check if the TIFF file exists
            if not os.path.isfile(tiff_filename):
                raise FileNotFoundError(f"The file {tiff_filename} does not exist.")
            
            # Open the HDF5 file to retrieve the expected shape
            with h5py.File(hdf5_filename, 'r') as f:
                xray_maps_group = f['X-ray Maps']
                first_dataset_name = list(xray_maps_group.keys())[0]
                expected_shape = xray_maps_group[first_dataset_name+"/data"].shape
                
            # Open the TIFF file and read it as an RGB image
            image = Image.open(tiff_filename).convert('RGB')
            tiff_data = np.array(image)
            
            # Ensure the shape matches the expected shape
            if tiff_data.shape[:2] != expected_shape:
                tiff_data = np.array(Image.fromarray(tiff_data).resize((expected_shape[1], expected_shape[0])))
            
            # Ensure the reshaped data has 3 channels
            if tiff_data.shape[2] != 3:
                raise ValueError(f"Reshaped TIFF data does not have 3 channels: {tiff_data.shape}")
            
            # Save the reshaped data to the HDF5 file
            with h5py.File(hdf5_filename, 'a') as f:
                images_group = f.require_group('Images')
                if 'Phase_Map.tiff' in images_group:
                    del images_group['Phase_Map.tiff']
                images_group.create_dataset('Phase_Map.tiff', data=tiff_data)
            
            self.log_message(f"Results saved successfully to {hdf5_filename} in the Images group.")
        except FileNotFoundError as fnf_error:
            QMessageBox.critical(self, "Error", str(fnf_error))
            self.log_message(f"Error: {fnf_error}")
        except ValueError as ve:
            QMessageBox.critical(self, "Error", str(ve))
            self.log_message(f"Error: {ve}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results: {e}")
            self.log_message(f"Error saving results: {e}")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    phase_mapping_step = PhaseMappingStep()
    phase_mapping_step.show()
    sys.exit(app.exec_())  # ← This is correct