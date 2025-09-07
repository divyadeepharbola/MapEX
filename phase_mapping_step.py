# -------------------------------------------
# Phase Mapping (stable, threaded, low-memory) + Manual Editor
# -------------------------------------------
# IMPORTANT: cap threads BEFORE any NumPy/BLAS imports
import os
import sys

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import EllipseSelector, LassoSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.cluster import MiniBatchKMeans as _MBK_for_iso
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.utils import check_random_state

os.environ.setdefault("OMP_NUM_THREADS", "4")

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")


# Use Qt backend for interactive plots inside the app

try:
    matplotlib.use("Qt5Agg")
except Exception:
    pass


# ML / Stats

# Optional fuzzy c-means
try:
    from skfuzzy import cmeans

    _HAS_SKFUZZY = True
except Exception:
    _HAS_SKFUZZY = False

# Optional HDBSCAN
try:
    import hdbscan

    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False

# Optional UMAP
try:
    import umap

    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

# Optional t-SNE
try:
    from sklearn.manifold import TSNE

    _HAS_TSNE = True
except Exception:
    _HAS_TSNE = False

# Optional interactive hover for pie chart
try:
    import mplcursors

    _HAS_MPLCURSORS = True
except Exception:
    _HAS_MPLCURSORS = False

# Optional SOM (MiniSom)
try:
    from minisom import MiniSom

    _HAS_MINISOM = True
except Exception:
    _HAS_MINISOM = False

# -------------------
# Utility helpers
# -------------------
MAX_PREVIEW_SIDE = 2000  # cap longer side for on-screen preview to avoid huge Qt canvas
# keep disabled (no preview clustering UI in this build)
ENABLE_PREVIEW_PCA = False


def _preview_img(img2d):
    H, W = img2d.shape[:2]
    s = max(1, int(np.ceil(max(H, W) / MAX_PREVIEW_SIDE)))
    return img2d[::s, ::s] if s > 1 else img2d, s


def _assign_by_centers(feat, centers, batch=250_000):
    """Assign each point in feat to its nearest center (squared
    Euclidean)."""
    n = feat.shape[0]
    out = np.empty(n, dtype=np.int32)
    s = 0
    c_norm = (centers**2).sum(axis=1, keepdims=True).T  # (k,)
    while s < n:
        e = min(s + batch, n)
        Xb = feat[s:e]
        x_norm = (Xb**2).sum(axis=1, keepdims=True)  # (b,1)
        d = x_norm + c_norm - 2.0 * (Xb @ centers.T)  # (b,k)
        out[s:e] = np.argmin(d, axis=1).astype(np.int32)
        s = e
    return out


def _safe_nan_to_num(x):
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if np.any(finite):
        maxf = np.nanmax(x[finite])
    else:
        maxf = 0.0
    x = np.nan_to_num(x, nan=0.0, posinf=maxf, neginf=0.0)
    return x


def _clr_transform(arr, eps=None):
    """Centered log-ratio transform channel-wise (compositional):

    arr[..., C].
    """
    x = np.asarray(arr, dtype=np.float64)
    x[x < 0] = 0.0
    if eps is None:
        med = np.median(x[x > 0]) if np.any(x > 0) else 1.0
        eps = max(1e-12, 1e-6 * med)
    x = x + eps
    gm = np.exp(np.mean(np.log(x), axis=-1, keepdims=True))
    clr = np.log(x / gm)
    return clr


def _majority_filter(labels, size=3):
    """Simple 2D majority filter over integer labels."""
    labels = labels.astype(np.int64, copy=False)
    footprint = np.ones((size, size), dtype=bool)

    def mode_func(values):
        vals, counts = np.unique(values, return_counts=True)
        return vals[np.argmax(counts)]

    return ndi.generic_filter(labels, mode_func, footprint=footprint, mode="nearest")


def _auto_components_for_pca(n_features, n_pixels):
    cap = min(25, n_features)
    base = int(min(cap, max(2, round(min(n_features, 10 if n_pixels > 5e6 else 15)))))
    return base


def _nice_error(msg, parent=None):
    if parent is not None:
        QMessageBox.critical(parent, "Error", msg)
    else:
        print("ERROR:", msg)


def _make_consistent_cmap(n, base_name="tab20"):
    base = plt.get_cmap(base_name)
    xs = np.linspace(0, 1, max(n, base.N), endpoint=True)[:n]
    colors = base(xs)
    return ListedColormap(colors)


# -------------------
# Simple ISODATA (split-merge) on a subsample then nearest-centroid assign
# -------------------


def _isodata_centroids(X, k0=6, max_iter=10, split_std_thresh=0.8, merge_dist_thresh=1.0, rng=None):
    rng = check_random_state(42 if rng is None else rng)
    k0 = max(2, int(k0))
    mbk = _MBK_for_iso(n_clusters=k0, random_state=42)
    labels = mbk.fit_predict(X)
    cents = mbk.cluster_centers_

    for _ in range(max_iter):
        new_cents, to_split = [], []
        for j in range(len(cents)):
            pts = X[labels == j]
            if len(pts) < 2:
                new_cents.append(cents[j])
                continue
            std = pts.std(axis=0)
            if np.mean(std) > split_std_thresh:
                to_split.append((j, std))
            else:
                new_cents.append(cents[j])

        for j, std in to_split:
            c = cents[j]
            delta = 0.5 * std / (np.linalg.norm(std) + 1e-9)
            new_cents.append(c + delta)
            new_cents.append(c - delta)
        cents = np.vstack(new_cents)

        keep = np.ones(len(cents), dtype=bool)
        for a in range(len(cents)):
            if not keep[a]:
                continue
            for b in range(a + 1, len(cents)):
                if not keep[b]:
                    continue
                d = np.linalg.norm(cents[a] - cents[b])
                if d < merge_dist_thresh:
                    cents[a] = 0.5 * (cents[a] + cents[b])
                    keep[b] = False
        cents = cents[keep]

        d2 = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)

        if len(cents) > 32:
            mbk = _MBK_for_iso(n_clusters=32, random_state=42).fit(cents)
            cents = mbk.cluster_centers_

    return cents


# -------------------
# Workers (no GUI calls from worker thread)
# -------------------
class _PhaseWorker(QObject):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    # (phase_map HxW, labels N, extras dict)
    done = pyqtSignal(object, object, object)
    error = pyqtSignal(str)

    def __init__(self, owner):
        super().__init__()
        self.p = owner

    def _emit_progress(self, frac):
        try:
            val = int(round(100 * float(np.clip(frac, 0.0, 1.0))))
        except Exception:
            val = 0
        self.progress.emit(val)

    def run(self):
        p = self.p
        orig_log = getattr(p, "_log", None)
        orig_prog = getattr(p, "_update_progress_fraction", None)
        p._log = lambda s: self.log.emit(str(s))
        p._update_progress_fraction = self._emit_progress

        try:
            self.log.emit("Collecting input maps…")
            data_list = [_safe_nan_to_num(m) for m in p.data_list]
            if not data_list:
                raise RuntimeError("No data loaded.")
            stack = np.stack(data_list, axis=-1).astype(np.float32)  # (H,W,C)
            H, W, C = stack.shape
            N = H * W
            self.progress.emit(2)
            self.log.emit(f"Stack shape = {H}×{W}×{C} (N={N:,})")

            stack = p._apply_quantification_if_requested(stack, p.element_names)
            stack = np.asarray(stack, dtype=np.float32)
            self.progress.emit(5)

            try:
                sigma = float(p.sigma_edit.text())
            except Exception:
                sigma = 1.0
            if sigma > 0:
                self.log.emit(f"Gaussian smoothing σ={sigma}…")
                stack = gaussian_filter(stack, sigma=(sigma, sigma, 0.0)).astype(np.float32)
            self.progress.emit(10)

            if p.chk_clr.isChecked():
                self.log.emit("Applying CLR transform…")
                stack = _clr_transform(stack).astype(np.float32)

            flat = stack.reshape(-1, C)
            self.progress.emit(15)

            self.log.emit(f"Scaling: {p.scaling_choice.currentText()} …")
            flat = p._apply_scaling(flat, p.scaling_choice.currentText())
            flat = np.asarray(flat, dtype=np.float32, order="C")
            self.progress.emit(20)

            use_pca = p.chk_pca.isChecked()
            if use_pca:
                n_comp_txt = (p.pca_n_edit.text() or "").strip()
                if n_comp_txt.isdigit():
                    n_comp = max(2, min(int(n_comp_txt), C))
                else:
                    try:
                        n_comp = min(C, _auto_components_for_pca(C, N))
                    except Exception:
                        n_comp = min(C, 10)
                self.log.emit(f"PCA → {n_comp} components…")
                pca = PCA(
                    n_components=n_comp,
                    svd_solver="randomized",
                    random_state=42,
                )
                feat = pca.fit_transform(flat)
            else:
                feat = flat
            feat = np.asarray(feat, dtype=np.float32, order="C")
            p._feat_for_cluster = feat
            if hasattr(p, "_kcurve_cache"):
                p._kcurve_cache.clear()
            self.progress.emit(35)

            method = p.phase_method.currentText().lower()
            need_k = method in (
                "kmeans",
                "gmm",
                "gmm_diag",
                "fcm",
                "nmf_kmeans",
                "isodata",
                "spectral",
                "hierarchical",
                "som",
            )
            k = None
            if need_k:
                self.log.emit(f"Estimating k for method={method}…")
                Xk = feat
                if Xk.shape[0] > p.kcurve_max_samples:
                    rng = check_random_state(42)
                    idx = rng.choice(Xk.shape[0], size=p.kcurve_max_samples, replace=False)
                    Xk = Xk[idx]
                try:
                    p._feat_for_cluster = Xk
                    _, _, k_est = p._compute_k_curve(Xk, p.k_criterion.currentText(), method)
                finally:
                    p._feat_for_cluster = feat
                k = int(k_est) if (k_est is not None and int(k_est) >= 2) else max(2, min(8, feat.shape[1]))
                self.log.emit(f"Using k={k}")
            self.progress.emit(45)

            self.log.emit(f"Clustering with {method} …")
            labels = p._cluster(feat, method, k=k, H=H, W=W, Ceff=feat.shape[1])
            labels = np.asarray(labels, dtype=np.int32, order="C")
            if labels.ndim != 1 or labels.size != N:
                raise RuntimeError(f"Clusterer returned labels of wrong shape: {labels.shape} (expected {N},)")
            self.progress.emit(80)

            if p.chk_spatial.isChecked():
                self.log.emit("Applying 3×3 majority spatial post-filter…")
                labels = _majority_filter(labels.reshape(H, W), size=3).reshape(-1).astype(np.int32)
            self.progress.emit(88)

            phase_map = labels.reshape(H, W)
            self.progress.emit(92)

            p2 = None
            self.progress.emit(97)

            extras = {"k": k, "proj2d": p2}
            self.progress.emit(100)
            self.done.emit(phase_map, labels, extras)

        except Exception as e:
            self.error.emit(str(e))

        finally:
            try:
                if orig_log is not None:
                    p._log = orig_log
                if orig_prog is not None:
                    p._update_progress_fraction = orig_prog
            except Exception:
                pass


class _ReclusterWorker(QObject):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    done = pyqtSignal(object)  # labels (N,)
    error = pyqtSignal(str)

    def __init__(self, owner, method, k, H, W):
        super().__init__()
        self.p = owner
        self.method = method
        self.k = int(k)
        self.H = H
        self.W = W

    def _emit_progress(self, frac):
        try:
            val = int(round(100 * float(np.clip(frac, 0.0, 1.0))))
        except Exception:
            val = 0
        self.progress.emit(val)

    def run(self):
        p = self.p
        orig_prog = getattr(p, "_update_progress_fraction", None)
        p._update_progress_fraction = self._emit_progress
        try:
            self.log.emit(f"Re-clustering with k={self.k} ({self.method}) …")
            self.progress.emit(5)
            labels = p._cluster(p._feat_for_cluster, self.method, k=self.k, H=self.H, W=self.W)
            if p.chk_spatial.isChecked() and self.H is not None and self.W is not None:
                labels = _majority_filter(labels.reshape(self.H, self.W), size=3).reshape(-1)
            self.progress.emit(98)
            self.done.emit(labels)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            try:
                if orig_prog is not None:
                    p._update_progress_fraction = orig_prog
            except Exception:
                pass


# -------------------
# Main Widget
# -------------------
class PhaseMappingStep(QWidget):
    def __init__(self, data_path=None):
        super().__init__()
        self.working_dir = os.environ.get("MAPEX_WORKDIR", "")
        self.data_path = data_path or "data.h5"
        if not os.path.isabs(self.data_path):
            self.data_path = self._wd(os.path.basename(self.data_path))
        self.data_list = []
        self.element_names = []
        self.results = None
        self._feat_for_cluster = None
        self._labels = None
        self._cmap = None
        self._phase_names = None
        self._kcurve_cache = {}
        self.kcurve_max_samples = 200_000

        self._build_ui()
        self._load_data_h5()
        self._load_calibration_maybe()
        self._log("Ready. Select options and run.")

    def set_working_dir(self, folder: str):
        """Called by main.py to enforce one working directory for all
        inputs/outputs."""
        self.working_dir = os.path.abspath(folder) if folder else ""
        # If data_path is relative (or default), pin it into the working dir.
        if self.data_path and not os.path.isabs(self.data_path):
            base = os.path.basename(self.data_path) or "data.h5"
            self.data_path = os.path.join(self.working_dir or os.getcwd(), base)

    def _wd(self, *parts: str) -> str:
        """Join path parts under working_dir (fallback: CWD)."""
        base = self.working_dir or os.getcwd()
        return os.path.join(base, *parts)

    # ---------- UI ----------
    def _build_ui(self):
        self.setWindowTitle("Phase Mapping")

        root = QHBoxLayout(self)
        self.splitter_main = QSplitter(self)
        root.addWidget(self.splitter_main)

        # Left splitter: Logs / Controls
        self.splitter_left = QSplitter(self)
        self.splitter_left.setOrientation(Qt.Vertical)
        self.splitter_main.addWidget(self.splitter_left)

        # Logs
        self.text_area = QTextEdit(self)
        self.text_area.setReadOnly(True)
        self.text_area.setPlaceholderText("Logs will appear here...")
        self.splitter_left.addWidget(self.text_area)

        # Controls panel
        controls = QWidget(self)
        controls_layout = QVBoxLayout(controls)

        # Method row
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Phase method:", self))
        self.phase_method = QComboBox(self)
        self.phase_method.addItems(
            [
                "kmeans",
                "gmm",
                "gmm_diag",
                "fcm",
                "hierarchical",
                "spectral",
                "birch",
                "optics",
                "affinity_propagation",
                "meanshift",
                "dbscan",
                "hdbscan",
                "isodata",
                "som",
                "nmf_kmeans",
            ]
        )
        method_row.addWidget(self.phase_method)
        self.auto_cluster = QCheckBox("Auto-Clustering", self)
        self.auto_cluster.setChecked(False)
        method_row.addWidget(self.auto_cluster)
        controls_layout.addLayout(method_row)

        # Scaling + smoothing
        crit_row = QHBoxLayout()
        crit_row.addWidget(QLabel("Scaling:", self))
        self.scaling_choice = QComboBox(self)
        self.scaling_choice.addItems(["None", "Standard (count/mean)", "MinMax [0,1]", "Robust (5-95%)"])
        crit_row.addWidget(self.scaling_choice)
        crit_row.addWidget(QLabel("Pre-smooth σ:", self))
        self.sigma_edit = QLineEdit(self)
        self.sigma_edit.setText("1.0")
        crit_row.addWidget(self.sigma_edit)
        controls_layout.addLayout(crit_row)

        # Advanced
        self.adv = QGroupBox("Advanced settings", self)
        self.adv.setCheckable(True)
        self.adv.setChecked(False)
        adv_form = QFormLayout(self.adv)
        self.chk_quant = QCheckBox("Quantify using Calibration_data.xlsx", self)
        adv_form.addRow(self.chk_quant)
        self.chk_clr = QCheckBox("CLR (compositional) transform", self)
        adv_form.addRow(self.chk_clr)
        self.chk_pca = QCheckBox("PCA before clustering", self)
        self.chk_pca.setChecked(True)
        adv_form.addRow(self.chk_pca)
        self.pca_n_edit = QLineEdit(self)
        self.pca_n_edit.setPlaceholderText("auto")
        self.pca_n_edit.setValidator(QIntValidator(2, 128, self))
        adv_form.addRow("PCA components:", self.pca_n_edit)
        self.chk_spatial = QCheckBox("Spatial post-smoothing (3×3 majority)", self)
        self.chk_spatial.setChecked(True)
        adv_form.addRow(self.chk_spatial)
        controls_layout.addWidget(self.adv)

        # Run & progress
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Phase Mapping", self)
        self.btn_run.clicked.connect(self._on_run)
        run_row.addWidget(self.btn_run)
        self.btn_save = QPushButton("Save Results", self)
        self.btn_save.clicked.connect(self._save_results)
        run_row.addWidget(self.btn_save)
        controls_layout.addLayout(run_row)
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.hide()
        controls_layout.addWidget(self.progress)

        # K panel
        self.kpanel = QGroupBox("K selection", self)
        kpanel_layout = QVBoxLayout(self.kpanel)
        self.kplot_fig = Figure(figsize=(4, 2))
        self.kplot_canvas = FigureCanvas(self.kplot_fig)
        self.kplot_ax = self.kplot_fig.add_subplot(111)
        kpanel_layout.addWidget(self.kplot_canvas)
        krow = QHBoxLayout()
        krow.addWidget(QLabel("K-criterion:", self))
        self.k_criterion = QComboBox(self)
        self.k_criterion.addItems(
            [
                "elbow",
                "silhouette",
                "davies_bouldin",
                "gap",
                "bic_aic",
                "consensus",
            ]
        )
        self.k_criterion.currentIndexChanged.connect(self._refresh_kpanel_curve)
        krow.addWidget(self.k_criterion)
        krow.addWidget(QLabel("k:", self))
        self.k_spin = QSpinBox(self)
        self.k_spin.setRange(2, 128)
        self.k_spin.setValue(3)
        self.k_spin.valueChanged.connect(self._on_k_spin_changed)
        krow.addWidget(self.k_spin)
        self.btn_recluster = QPushButton("Recluster", self)
        self.btn_recluster.setToolTip("Run clustering with current k and method")
        self.btn_recluster.clicked.connect(self._on_recluster_clicked)
        krow.addWidget(self.btn_recluster)
        krow.addStretch(1)
        kpanel_layout.addLayout(krow)
        self.kpanel.hide()
        controls_layout.addWidget(self.kpanel)

        # Manual editor launcher (new window)
        self.manual_grp = QGroupBox("Manual clustering (new window)", self)
        mg_v = QVBoxLayout(self.manual_grp)
        self.btn_manual = QPushButton("Open Manual Editor", self)
        self.btn_manual.setToolTip("Draw lasso / ellipse / rectangle / circle to reassign phases")
        self.btn_manual.clicked.connect(self._open_manual_editor)
        self.btn_manual.setEnabled(False)
        mg_v.addWidget(self.btn_manual)
        controls_layout.addWidget(self.manual_grp)

        controls_layout.addStretch(1)
        self.splitter_left.addWidget(controls)

        # Right plot
        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        self.figure = plt.figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        self.splitter_main.addWidget(right)

        # Stretch behavior
        self.splitter_main.setStretchFactor(0, 1)
        self.splitter_main.setStretchFactor(1, 3)
        self.splitter_left.setStretchFactor(0, 3)
        self.splitter_left.setStretchFactor(1, 2)

        # React to method changes
        self.phase_method.currentIndexChanged.connect(self._on_method_changed)
        self._on_method_changed()

    def _on_method_changed(self):
        m = self.phase_method.currentText()
        need_k = m in (
            "kmeans",
            "gmm",
            "fcm",
            "nmf_kmeans",
            "isodata",
            "spectral",
            "hierarchical",
            "som",
        )
        self.kpanel.setEnabled(need_k)
        if not need_k:
            self.kpanel.hide()
        elif self.results is not None:
            self.kpanel.show()

    def _log(self, msg):
        self.text_area.append(msg)

    # ---------- Data loading ----------
    def _load_data_h5(self):
        path = self.data_path
        if not os.path.isfile(path):
            _nice_error(f"Data file not found: {path}", self)
            return
        try:
            with h5py.File(path, "r") as f:
                if "X-ray Maps" not in f:
                    raise KeyError("Group 'X-ray Maps' not found")
                grp = f["X-ray Maps"]
                self.data_list = []
                self.element_names = []
                for name in grp.keys():
                    ds = grp[name].get("data", None)
                    if ds is None:
                        continue
                    arr = np.array(ds)
                    self.data_list.append(arr)
                    self.element_names.append(str(name))
                    self._log(f"Loaded map: {name}  shape={arr.shape}")
            if not self.data_list:
                _nice_error("No maps found under 'X-ray Maps/*/data'.", self)
                return
            self._log(f"Total maps: {len(self.data_list)}")
        except Exception as e:
            _nice_error(f"Failed to load HDF5: {e}", self)

    def _load_calibration_maybe(self):
        # Prefer the working dir; fall back to the folder of data_path.
        base_dir = self.working_dir or os.path.dirname(os.path.abspath(self.data_path))
        cfile = os.path.join(base_dir, "Calibration_data.xlsx")
        if not os.path.isfile(cfile):
            self.calibration = None
            self._log("Calibration_data.xlsx not found in working directory (optional).")
            return
        try:
            df = pd.read_excel(cfile)
            need = {"Elements", "m", "c"}
            if not need.issubset(df.columns):
                self._log("Calibration file missing columns Elements/m/c; ignored.")
                self.calibration = None
                return
            self.calibration = {str(r["Elements"]).strip(): (float(r["m"]), float(r["c"])) for _, r in df.iterrows()}
            self._log(f"Loaded calibration for {len(self.calibration)} elements.")
        except Exception as e:
            self.calibration = None
            self._log(f"Failed to load calibration: {e}")

    def _apply_quantification_if_requested(self, stack, names):
        if not self.chk_quant.isChecked():
            self._log("Using raw intensities (no quantification).")
            return stack
        if not getattr(self, "calibration", None):
            self._log("Quantification requested but calibration not loaded; using raw.")
            return stack
        out = np.empty_like(stack, dtype=np.float64)
        for i, nm in enumerate(names):
            key = nm.strip()
            if key in self.calibration:
                m, c = self.calibration[key]
                m = m if m != 0 else 1.0
                out[..., i] = (stack[..., i] - c) / m
                self._log(f"Quantified {key}: (I - {c}) / {m}")
            else:
                out[..., i] = stack[..., i]
                self._log(f"No calibration for {key}; raw used.")
        return out

    def _apply_scaling(self, flat, choice):
        eps = 1e-12
        choice = (choice or "None").lower()
        if choice.startswith("none"):
            self._log("Scaling: None")
            return flat
        if choice.startswith("standard"):
            mu = np.mean(flat, axis=0)
            mu[mu == 0] = eps
            self._log("Scaling: Standard (count/mean)")
            return flat / mu
        if choice.startswith("minmax"):
            mn = np.min(flat, axis=0)
            mx = np.max(flat, axis=0)
            rng = np.maximum(mx - mn, eps)
            self._log("Scaling: MinMax [0,1]")
            return (flat - mn) / rng
        if choice.startswith("robust"):
            self._log("Scaling: Robust (5-95%)")
            rs = RobustScaler(
                with_centering=True,
                with_scaling=True,
                quantile_range=(5.0, 95.0),
            )
            return rs.fit_transform(flat)
        return flat

    # ---------- K metrics (embedded) ----------
    def _compute_k_curve(self, feat, criterion, method):
        c = criterion.lower()
        key = (c,)
        if key in self._kcurve_cache:
            return self._kcurve_cache[key]

        N = feat.shape[0]
        rng = check_random_state(42)
        maxn = getattr(self, "kcurve_max_samples", 200_000)
        X = feat[rng.choice(N, size=maxn, replace=False)] if N > maxn else feat

        if c in ("bic_aic", "bic", "aic"):
            ks = list(range(2, 11))
            vals = []
            for i, k in enumerate(ks):
                self._update_progress_fraction(0.20 + 0.50 * i / max(1, len(ks) - 1))
                g = GaussianMixture(n_components=k, random_state=42)
                g.fit(X)
                vals.append(g.bic(X))
            suggested = ks[int(np.argmin(vals))]
            self._kcurve_cache[key] = (ks, vals, suggested)
            return ks, vals, suggested

        if c == "elbow":
            ks = list(range(2, 11))
            inertia = []
            for i, k in enumerate(ks):
                self._update_progress_fraction(0.20 + 0.50 * i / max(1, len(ks) - 1))
                km = MiniBatchKMeans(n_clusters=k, random_state=42).fit(X)
                inertia.append(km.inertia_)
            diffs = np.diff(inertia)
            suggested = ks[np.argmin(np.gradient(diffs))] if len(diffs) >= 2 else 3
            self._kcurve_cache[key] = (ks, inertia, suggested)
            return ks, inertia, suggested

        if c == "silhouette":
            ks = list(range(2, 11))
            scores = []
            best_k, best_s = 2, -1.0
            for i, k in enumerate(ks):
                self._update_progress_fraction(0.20 + 0.50 * i / max(1, len(ks) - 1))
                km = MiniBatchKMeans(n_clusters=k, random_state=42).fit(X)
                lbl = km.labels_
                try:
                    s = silhouette_score(X, lbl, sample_size=min(5000, X.shape[0]))
                except Exception:
                    s = -1.0
                scores.append(s)
                if s > best_s:
                    best_s, best_k = s, k
            self._kcurve_cache[key] = (ks, scores, best_k)
            return ks, scores, best_k

        if c == "davies_bouldin":
            ks = list(range(2, 11))
            dbs = []
            best_k, best_db = 2, np.inf
            for i, k in enumerate(ks):
                self._update_progress_fraction(0.20 + 0.50 * i / max(1, len(ks) - 1))
                km = MiniBatchKMeans(n_clusters=k, random_state=42).fit(X)
                lbl = km.labels_
                try:
                    db = davies_bouldin_score(X, lbl)
                except Exception:
                    db = np.inf
                dbs.append(db)
                if db < best_db:
                    best_db, best_k = db, k
            self._kcurve_cache[key] = (ks, dbs, best_k)
            return ks, dbs, best_k

        if c == "gap":
            ks = list(range(2, 11))
            disp = []
            for i, k in enumerate(ks):
                self._update_progress_fraction(0.20 + 0.50 * i / max(1, len(ks) - 1))
                km = MiniBatchKMeans(n_clusters=k, random_state=42).fit(X)
                disp.append(km.inertia_)
            gaps = -np.gradient(np.log(disp))
            suggested = ks[int(np.argmax(gaps))]
            self._kcurve_cache[key] = (ks, gaps.tolist(), suggested)
            return ks, gaps.tolist(), suggested

        if c == "consensus":
            base = ("elbow", "silhouette", "davies_bouldin", "gap", "bic_aic")
            suggs = []
            for m in base:
                _, _, s = self._compute_k_curve(feat, m, method)
                if s is not None:
                    suggs.append(int(s))
            ks = list(range(2, 11))
            if not suggs:
                self._kcurve_cache[key] = (ks, [0] * len(ks), 5)
                return ks, [0] * len(ks), 5
            votes = [suggs.count(k) for k in ks]
            suggested = ks[int(np.argmax(votes))] if max(votes) > 1 else int(np.median(suggs))
            self._kcurve_cache[key] = (ks, votes, suggested)
            return ks, votes, suggested

        return [2, 3, 4, 5, 6, 7, 8], [np.nan] * 7, 5

    def _refresh_kpanel_curve(self):
        if self._feat_for_cluster is None or not self.kpanel.isVisible():
            return
        method = self.phase_method.currentText().lower()
        if method not in (
            "kmeans",
            "gmm",
            "fcm",
            "nmf_kmeans",
            "isodata",
            "spectral",
            "hierarchical",
            "som",
        ):
            return
        crit = self.k_criterion.currentText()
        ks, vals, suggested = self._compute_k_curve(self._feat_for_cluster, crit, method)
        self.kplot_ax.clear()
        self.kplot_ax.plot(ks, vals, marker="o")
        self.kplot_ax.set_xlabel("k")
        ylabel = {
            "elbow": "Inertia (lower better)",
            "silhouette": "Silhouette (higher better)",
            "davies_bouldin": "Davies-Bouldin (lower better)",
            "gap": "Gap (higher better)",
            "bic_aic": "BIC (lower better)",
            "consensus": "Votes (higher better)",
        }.get(crit.lower(), crit)
        self.kplot_ax.set_ylabel(ylabel)
        self.kplot_ax.grid(True, alpha=0.3)
        if suggested is not None:
            self.kplot_ax.axvline(suggested, linestyle="--", alpha=0.6)
        self.kplot_canvas.draw_idle()
        if self.auto_cluster.isChecked() and suggested is not None:
            self.k_spin.blockSignals(True)
            self.k_spin.setValue(int(suggested))
            self.k_spin.blockSignals(False)

    # ---------- Processing pipeline ----------
    def _on_run(self):
        if not self.data_list:
            _nice_error("No data loaded.", self)
            return
        self.progress.show()
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_manual.setEnabled(False)

        self._thr = QThread(self)
        self._worker = _PhaseWorker(self)
        self._worker.moveToThread(self._thr)

        self._thr.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.log.connect(self._append_log, type=Qt.QueuedConnection)

        def _on_done(phase_map, labels, extras):
            try:
                self.results = phase_map
                self._labels = labels
                n_phases = int(np.max(phase_map)) + 1
                self._cmap = _make_consistent_cmap(n_phases, base_name="tab20")
                self._display_results(phase_map)
                method = self.phase_method.currentText().lower()
                if method in (
                    "kmeans",
                    "gmm",
                    "fcm",
                    "nmf_kmeans",
                    "isodata",
                    "spectral",
                    "hierarchical",
                    "som",
                ):
                    self.kpanel.show()
                    if extras.get("k", None) is not None:
                        self.k_spin.blockSignals(True)
                        self.k_spin.setValue(int(extras["k"]))
                        self.k_spin.blockSignals(False)
                    self._refresh_kpanel_curve()
                self.progress.setValue(100)
                QMessageBox.information(self, "Done", "Phase mapping completed.")
                self.btn_manual.setEnabled(True)
            finally:
                self.progress.hide()
                self.btn_run.setEnabled(True)
                self._thr.quit()
                self._thr.wait()
                self._worker.deleteLater()
                self._thr.deleteLater()

        def _on_error(msg):
            try:
                _nice_error(f"Phase mapping failed: {msg}", self)
            finally:
                self.progress.hide()
                self.btn_run.setEnabled(True)
                self._thr.quit()
                self._thr.wait()
                self._worker.deleteLater()
                self._thr.deleteLater()

        self._worker.done.connect(_on_done)
        self._worker.error.connect(_on_error)
        self._thr.start()

    def _cluster(self, feat, method, k=None, H=None, W=None, Ceff=None):
        self._update_progress_fraction(0.75)
        method = method.lower()
        N = feat.shape[0]
        rng = check_random_state(42)

        if method == "kmeans":
            algo = MiniBatchKMeans(n_clusters=int(k), random_state=42)
            labels = algo.fit_predict(feat)

        elif method == "gmm":
            n_fit = min(N, 300_000)
            idx = rng.choice(N, size=n_fit, replace=False) if N > n_fit else np.arange(N)
            g = GaussianMixture(n_components=int(k), random_state=42)
            g.fit(feat[idx])
            labels = g.predict(feat)

        elif method == "gmm_diag":
            n_fit = min(N, 200_000)
            idx = rng.choice(N, size=n_fit, replace=False) if N > n_fit else np.arange(N)
            g = GaussianMixture(
                n_components=int(k),
                covariance_type="diag",
                reg_covar=1e-6,
                max_iter=200,
                random_state=42,
            )
            g.fit(feat[idx])
            labels = np.empty(N, dtype=np.int32)
            bs = 100_000
            s = 0
            while s < N:
                e = min(s + bs, N)
                labels[s:e] = g.predict(feat[s:e])
                s = e

        elif method == "fcm":
            if not _HAS_SKFUZZY:
                raise RuntimeError("scikit-fuzzy not installed; cannot run FCM.")
            n_fit = min(N, 250_000)
            idx = rng.choice(N, size=n_fit, replace=False) if N > n_fit else np.arange(N)
            cntr, u, *_ = cmeans(feat[idx].T, int(k), 2, 1e-4, 150)
            d2 = ((feat[:, None, :] - cntr[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(d2, axis=1)

        elif method == "hierarchical":
            n_samp = min(20_000, N)
            idx = rng.choice(N, size=n_samp, replace=False)
            sub = feat[idx]
            k_use = int(k) if k else 6
            agg = AgglomerativeClustering(n_clusters=k_use, linkage="ward")
            sub_lbl = agg.fit_predict(sub)
            cents = np.vstack([sub[sub_lbl == j].mean(axis=0) for j in range(k_use)])
            labels = _assign_by_centers(feat, cents)

        elif method == "spectral":
            n_samp = min(30_000, N)
            idx = rng.choice(N, size=n_samp, replace=False)
            sub = feat[idx]
            k_use = int(k) if k else 6
            sc = SpectralClustering(
                n_clusters=k_use,
                random_state=42,
                n_init=10,
                affinity="nearest_neighbors",
            )
            sub_lbl = sc.fit_predict(sub)
            cents = np.vstack([sub[sub_lbl == j].mean(axis=0) for j in range(k_use)])
            labels = _assign_by_centers(feat, cents)

        elif method == "birch":
            X = feat.astype(np.float32, copy=False)
            n_fit = min(100_000, N)
            idx_fit = rng.choice(N, size=n_fit, replace=False) if N > n_fit else np.arange(N)
            X_fit = X[idx_fit]
            br = Birch(
                n_clusters=None,
                threshold=1.0,
                branching_factor=200,
                compute_labels=False,
            )
            br.fit(X_fit)
            subc = br.subcluster_centers_.astype(np.float32, copy=False)
            if subc.shape[0] == 0:
                cents = np.mean(X_fit, axis=0, keepdims=True)
            else:
                k_final = int(k) if (k is not None) else max(2, min(8, subc.shape[0]))
                try:
                    mbk = MiniBatchKMeans(
                        n_clusters=k_final,
                        batch_size=4096,
                        random_state=42,
                        n_init="auto",
                    )
                except TypeError:
                    mbk = MiniBatchKMeans(
                        n_clusters=k_final,
                        batch_size=4096,
                        random_state=42,
                        n_init=10,
                    )
                mbk.fit(subc)
                cents = mbk.cluster_centers_.astype(np.float32, copy=False)
            labels = _assign_by_centers(X, cents, batch=100_000)

        elif method == "optics":
            op = OPTICS(min_samples=50, xi=0.05, min_cluster_size=0.02, n_jobs=-1)
            lbl = op.fit_predict(feat)
            uniq = np.unique(lbl)
            mapping = {u: i for i, u in enumerate(uniq)}
            labels = np.vectorize(mapping.get)(lbl)

        elif method == "affinity_propagation":
            n_samp = min(30_000, N)
            idx = rng.choice(N, size=n_samp, replace=False) if N > n_samp else np.arange(N)
            sub = feat[idx]
            af = AffinityPropagation(random_state=42)
            lbl_sub = af.fit_predict(sub)
            uniq = np.unique(lbl_sub)
            cents = (
                np.mean(sub, axis=0, keepdims=True)
                if len(uniq) == 0
                else np.vstack([sub[lbl_sub == j].mean(axis=0) for j in uniq])
            )
            labels = _assign_by_centers(feat, cents)

        elif method == "meanshift":
            n_samp = min(30_000, N)
            idx = rng.choice(N, size=n_samp, replace=False)
            sub = feat[idx]
            ms = MeanShift(bin_seeding=True)
            _ = ms.fit_predict(sub)
            cents = ms.cluster_centers_
            labels = _assign_by_centers(feat, cents)

        elif method == "dbscan":
            n_samp = min(20_000, N)
            idx = rng.choice(N, size=n_samp, replace=False) if N > n_samp else np.arange(N)
            sub = feat[idx]
            nn = NearestNeighbors(n_neighbors=5).fit(sub)
            dists, _ = nn.kneighbors(sub)
            kdist = np.sort(dists[:, -1])
            eps_grid = np.percentile(kdist, [70, 75, 80, 85, 90, 95]).astype(float)
            k_target = int(k) if k else None
            best = None
            for eps in eps_grid:
                try:
                    db = DBSCAN(eps=float(eps), min_samples=5, n_jobs=-1)
                except TypeError:
                    db = DBSCAN(eps=float(eps), min_samples=5)
                lbl_sub = db.fit_predict(sub)
                ncl = int(len(np.unique(lbl_sub[lbl_sub >= 0])))
                if ncl == 0:
                    continue
                score = abs(ncl - k_target) if k_target is not None else abs(ncl - (len(eps_grid) // 2 + 1))
                if (best is None) or (score < best[0]):
                    best = (score, eps, lbl_sub, ncl)
            if best is None:
                cents = np.mean(sub, axis=0, keepdims=True)
            else:
                _, eps, lbl_sub, ncl = best
                self._log(f"DBSCAN: auto-selected eps={eps:.4f} (clusters={ncl})")
                cents = np.vstack([sub[lbl_sub == j].mean(axis=0) for j in np.unique(lbl_sub) if j >= 0])
            labels = _assign_by_centers(feat, cents)

        elif method == "hdbscan":
            if not _HAS_HDBSCAN:
                raise RuntimeError("hdbscan is not installed. Install with `pip install hdbscan`.")
            n_samp = min(50_000, N)
            idx = rng.choice(N, size=n_samp, replace=False) if N > n_samp else np.arange(N)
            sub = feat[idx]
            min_cluster_size = max(10, n_samp // 500)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=5,
                core_dist_n_jobs=-1,
            )
            lbl_sub = clusterer.fit_predict(sub)
            cents = (
                np.mean(sub, axis=0, keepdims=True)
                if not np.any(lbl_sub >= 0)
                else np.vstack([sub[lbl_sub == j].mean(axis=0) for j in np.unique(lbl_sub) if j >= 0])
            )
            labels = _assign_by_centers(feat, cents)

        elif method == "isodata":
            n_samp = min(50_000, N)
            idx = rng.choice(N, size=n_samp, replace=False)
            sub = feat[idx]
            k0 = int(k) if k else 6
            cents = _isodata_centroids(sub, k0=k0, rng=rng)
            labels = _assign_by_centers(feat, cents)

        elif method == "som":
            if not _HAS_MINISOM:
                raise RuntimeError("MiniSom not installed; install with `pip install minisom`.")
            k_use = int(k) if k else 6
            side = max(2, int(np.sqrt(k_use)))
            som = MiniSom(
                side,
                side,
                feat.shape[1],
                sigma=1.0,
                learning_rate=0.5,
                random_seed=42,
            )
            n_samp = min(50_000, N)
            idx = rng.choice(N, size=n_samp, replace=False)
            som.random_weights_init(feat[idx])
            som.train_random(feat[idx], num_iteration=500)
            bmus = [som.winner(x) for x in feat]
            labels = np.array([i * side + j for (i, j) in bmus], dtype=int)

        elif method == "nmf_kmeans":
            k_use = int(k) if k else 6
            X = feat.copy()
            minv = X.min()
            if minv < 0:
                X = X - minv + 1e-9
            nmf = NMF(
                n_components=k_use,
                init="nndsvda",
                max_iter=300,
                random_state=42,
            )
            W = nmf.fit_transform(X)
            algo = MiniBatchKMeans(n_clusters=k_use, random_state=42)
            labels = algo.fit_predict(W)

        else:
            raise ValueError(f"Unknown method: {method}")

        self._update_progress_fraction(0.90)
        return labels

    # ---------- Display & Save ----------
    def _display_results(self, phase_map):
        try:
            uniq = np.unique(phase_map)
            n = int(max(int(uniq.max()) + 1, len(uniq)))

            if self._cmap is None or self._cmap.N != n:
                self._cmap = _make_consistent_cmap(n, base_name="tab20")

            H, W = phase_map.shape
            total = H * W
            counts = np.array([(phase_map == i).sum() for i in range(n)])
            pct = 100.0 * counts / max(1, total)

            self.figure.clear()
            gs = self.figure.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.28)
            ax1 = self.figure.add_subplot(gs[0])
            ax2 = self.figure.add_subplot(gs[1])

            view, _s = _preview_img(phase_map)
            im = ax1.imshow(view, cmap=self._cmap, vmin=0, vmax=n - 1)
            ax1.set_title("Phase Map", fontsize=12, fontweight="bold")
            ax1.axis("off")
            div = make_axes_locatable(ax1)
            cax = div.append_axes("right", size="5%", pad=0.05)
            cb = self.figure.colorbar(im, cax=cax)
            cb.set_ticks(range(n))
            cb.set_ticklabels([f"P{i+1}" for i in range(n)])

            labels_txt = [f"P{i+1}" for i in range(n)]
            colors = [self._cmap(i) for i in range(n)]
            wedges, texts, autotexts = ax2.pie(
                pct,
                labels=labels_txt,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
            )
            ax2.set_title("Phase Distribution", fontsize=11, fontweight="bold")

            if _HAS_MPLCURSORS:
                cursor = mplcursors.cursor(wedges, hover=True)

                @cursor.connect("add")
                def on_add(sel):
                    i = wedges.index(sel.artist)
                    sel.annotation.set_text(f"{labels_txt[i]}: {pct[i]:.2f}% ({counts[i]})")

            else:
                for w in wedges:
                    w.set_picker(True)

                def on_pick(event):
                    w = event.artist
                    i = wedges.index(w)
                    self._log(f"Clicked {labels_txt[i]}: {pct[i]:.2f}% ({counts[i]})")
                    w.set_alpha(0.6 if w.get_alpha() in (None, 1.0) else 1.0)
                    self.canvas.draw_idle()

                self.canvas.mpl_connect("pick_event", on_pick)

            self.canvas.draw()

            # Save full-resolution images/text
            out_phase = self._wd("Phase.tiff")
            out_pie = self._wd("PieChart.tiff")
            fig1, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(phase_map, cmap=self._cmap, vmin=0, vmax=n - 1)
            ax.axis("off")
            fig1.savefig(out_phase, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(fig1)

            fig2, ax = plt.subplots(figsize=(5, 5))
            ax.pie(
                pct,
                labels=labels_txt,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
            )
            ax.set_title("Phase Distribution")
            fig2.savefig(out_pie, dpi=300, bbox_inches="tight")
            plt.close(fig2)

            self._log(f"Saved {out_phase} and {out_pie}")
            try:
                np.savetxt(
                    self._wd("PhaseMap.txt"),
                    phase_map.astype(np.int32),
                    fmt="%d",
                    delimiter="\t",
                )
                stats_df = pd.DataFrame(
                    {
                        "phase_id": np.arange(1, n + 1, dtype=int),
                        "pixels": counts.astype(int),
                        "percent": pct,
                    }
                )
                stats_df.to_csv(self._wd("PhaseStats.txt"), sep="\t", index=False)
                self._log("Saved PhaseMap.txt and PhaseStats.txt")
            except Exception as e:
                self._log(f"Warning: failed to write TXT exports: {e}")
        except Exception as e:
            _nice_error(f"Failed to display/save results: {e}", self)

    @pyqtSlot(str)
    def _append_log(self, msg: str):
        self.text_area.append(msg)

    def _save_results(self):
        try:
            tiff_filename = self._wd("Phase.tiff")
            if not os.path.isfile(tiff_filename):
                _nice_error(
                    f"Expected {tiff_filename} not found. Run phase mapping first.",
                    self,
                )
                return
            hdf5_filename = self.data_path
            with h5py.File(hdf5_filename, "r") as f:
                if "X-ray Maps" not in f:
                    raise KeyError("Group 'X-ray Maps' not found in HDF5.")
                xmaps = f["X-ray Maps"]
                first_name = next(iter(xmaps.keys()), None)
                if first_name is None or "data" not in xmaps[first_name]:
                    raise KeyError("No datasets found under 'X-ray Maps/*/data'.")
                expected_shape = xmaps[first_name]["data"].shape  # (H, W)

            from PIL import Image

            img = Image.open(tiff_filename).convert("RGB")
            arr = np.array(img)
            if arr.shape[:2] != expected_shape:
                img = Image.fromarray(arr).resize((expected_shape[1], expected_shape[0]))
                arr = np.array(img)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Phase.tiff must be RGB; got shape {arr.shape}")

            with h5py.File(hdf5_filename, "a") as f:
                images = f.require_group("Images")
                if "Phase_Map.tiff" in images:
                    del images["Phase_Map.tiff"]
                images.create_dataset("Phase_Map.tiff", data=arr)

            self._log(f"Results saved to {hdf5_filename} under /Images/Phase_Map.tiff")
            QMessageBox.information(
                self,
                "Saved",
                "Saved phase map TIFF to HDF5 (/Images/Phase_Map.tiff).",
            )
        except Exception as e:
            _nice_error(f"Failed to save results to HDF5: {e}", self)

    # ---------- Interactive K tweak ----------
    def _on_k_spin_changed(self, value):
        try:
            if self._feat_for_cluster is None:
                return
            method = self.phase_method.currentText().lower()
            if method not in (
                "kmeans",
                "gmm",
                "fcm",
                "nmf_kmeans",
                "isodata",
                "spectral",
                "hierarchical",
                "som",
            ):
                return
            if not self.auto_cluster.isChecked():
                self._log(f"Selected k={value}. Click 'Recluster' to apply.")
                try:
                    self.kplot_ax.axvline(int(value), linestyle=":", alpha=0.4)
                    self.kplot_canvas.draw_idle()
                except Exception:
                    pass
                return

            # Auto-apply path also shows progress bar
            self._log(f"Re-clustering with k={value} ({method}) …")
            H, W = self.results.shape if self.results is not None else (None, None)
            self._start_recluster_thread(method, int(value), H, W)
        except Exception as e:
            _nice_error(f"Failed to update map for k={value}: {e}", self)

    def _on_recluster_clicked(self):
        try:
            if self._feat_for_cluster is None:
                _nice_error("Run phase mapping first, then choose k.", self)
                return
            method = self.phase_method.currentText().lower()
            if method not in (
                "kmeans",
                "gmm",
                "fcm",
                "nmf_kmeans",
                "isodata",
                "spectral",
                "hierarchical",
                "som",
            ):
                _nice_error(f"'{method}' does not use k.", self)
                return
            k = int(self.k_spin.value())
            H, W = self.results.shape if self.results is not None else (None, None)
            self._start_recluster_thread(method, k, H, W)
        except Exception as e:
            _nice_error(f"Failed to recluster: {e}", self)

    def _start_recluster_thread(self, method, k, H, W):
        # Show progress bar during recluster (non-blocking)
        self.progress.show()
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_recluster.setEnabled(False)
        self.btn_manual.setEnabled(False)

        self._thr_r = QThread(self)
        self._worker_r = _ReclusterWorker(self, method, k, H, W)
        self._worker_r.moveToThread(self._thr_r)

        self._thr_r.started.connect(self._worker_r.run)
        self._worker_r.progress.connect(self.progress.setValue)
        self._worker_r.log.connect(self._append_log, type=Qt.QueuedConnection)

        def _on_done(labels):
            try:
                self._labels = labels
                self.results = labels.reshape(self.results.shape)
                self._display_results(self.results)
            finally:
                self.progress.hide()
                self.btn_run.setEnabled(True)
                self.btn_recluster.setEnabled(True)
                self.btn_manual.setEnabled(True)
                self._thr_r.quit()
                self._thr_r.wait()
                self._worker_r.deleteLater()
                self._thr_r.deleteLater()

        def _on_error(msg):
            try:
                _nice_error(f"Recluster failed: {msg}", self)
            finally:
                self.progress.hide()
                self.btn_run.setEnabled(True)
                self.btn_recluster.setEnabled(True)
                self.btn_manual.setEnabled(True)
                self._thr_r.quit()
                self._thr_r.wait()
                self._worker_r.deleteLater()
                self._thr_r.deleteLater()

        self._worker_r.done.connect(_on_done)
        self._worker_r.error.connect(_on_error)
        self._thr_r.start()

    # ---------- Manual editor (new window) ----------
    def _open_manual_editor(self):
        if self.results is None or self._feat_for_cluster is None:
            _nice_error("Run phase mapping first.", self)
            return
        dlg = ManualEditor(
            self,
            self.results.copy(),
            self._cmap,
            self._feat_for_cluster,
            self._labels,
        )
        dlg.exec_()

    # ---------- Progress helper ----------
    def _update_progress_fraction(self, frac):
        frac = float(np.clip(frac, 0.0, 1.0))
        self.progress.setValue(int(round(frac * 100)))
        QApplication.processEvents()

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, "_splitters_initialized", False):
            return
        try:
            self.splitter_main.setCollapsible(0, False)
            self.splitter_main.setCollapsible(1, False)
            self.splitter_left.setCollapsible(0, False)
            self.splitter_left.setCollapsible(1, False)
        except Exception:
            pass
        left_width = 350
        total_w = max(self.width(), 1000)
        self.splitter_main.setSizes([left_width, max(600, total_w - left_width)])
        self.splitter_main.moveSplitter(left_width, 1)
        total_h = max(self.height(), 800)
        logs_h = int(total_h * 0.60)
        self.splitter_left.setSizes([logs_h, max(240, total_h - logs_h)])
        self.splitter_main.widget(0).setMinimumWidth(300)
        self.splitter_left.widget(0).setMinimumHeight(220)
        self.splitter_left.widget(1).setMinimumHeight(180)
        self._splitters_initialized = True


# ---- Manual Editor Dialog ----
class ManualEditor(QDialog):
    """Manual clustering on a *multidimensional* view.

    Shows a 2D embedding (PCA/UMAP/t-SNE) of feature space so the user
    can draw an outer polygon (lasso/ellipse) around clusters;
    assignment updates the full map.
    """

    def __init__(
        self,
        owner: PhaseMappingStep,
        phase_map: np.ndarray,
        cmap: ListedColormap,
        feat: np.ndarray,
        labels1d: np.ndarray,
    ):
        super().__init__(owner)
        self.owner = owner
        self.map = phase_map  # live reference to full-res labels (H,W)
        self.cmap = cmap
        self.feat = feat.astype(np.float32, copy=False)  # (N,C)
        self.labels1d = labels1d.astype(np.int32, copy=False) if labels1d is not None else None
        self.N = self.feat.shape[0]
        self.H, self.W = self.map.shape

        self.setWindowTitle("Manual Clustering Editor (Feature Space)")
        self.resize(1100, 800)

        # State for embedding & selection
        self.sample_idx = None
        self.embed = None  # (Ns, 2)
        self._proj_kind = "PCA"
        self._last_path = None  # matplotlib.path.Path of last polygon selection
        self._selector = None

        # UI layout
        v = QVBoxLayout(self)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Projection:", self))
        self.proj_choice = QComboBox(self)
        proj_opts = ["PCA"]
        if _HAS_UMAP:
            proj_opts.append("UMAP")
        if _HAS_TSNE:
            proj_opts.append("t-SNE")
        self.proj_choice.addItems(proj_opts)
        ctrl.addWidget(self.proj_choice)

        ctrl.addWidget(QLabel("Sample size:", self))
        self.sample_spin = QSpinBox(self)
        self.sample_spin.setRange(1000, 500000)
        self.sample_spin.setSingleStep(5000)
        self.sample_spin.setValue(min(100000, max(5000, self.N // 20)))
        ctrl.addWidget(self.sample_spin)
        self.btn_compute = QPushButton("Compute / Refresh", self)
        self.btn_compute.clicked.connect(self._compute_embedding)
        ctrl.addWidget(self.btn_compute)

        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Shape:", self))
        self.shape_choice = QComboBox(self)
        # polygon tools on scatter
        self.shape_choice.addItems(["Lasso", "Ellipse"])
        ctrl.addWidget(self.shape_choice)

        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Assign to phase:", self))
        self.phase_spin = QSpinBox(self)
        self.phase_spin.setRange(1, 1024)
        self.phase_spin.setValue(int(np.max(self.map)) + 1)
        ctrl.addWidget(self.phase_spin)
        self.btn_new_phase = QPushButton("New Phase", self)
        self.btn_new_phase.clicked.connect(lambda: self.phase_spin.setValue(int(np.max(self.map)) + 2))
        ctrl.addWidget(self.btn_new_phase)

        self.chk_full = QCheckBox("Apply to full map (accurate)", self)
        self.chk_full.setChecked(True)
        ctrl.addWidget(self.chk_full)

        self.btn_assign = QPushButton("Assign Selection", self)
        self.btn_assign.clicked.connect(self._assign_selection)
        ctrl.addWidget(self.btn_assign)
        self.btn_clear = QPushButton("Clear Selection", self)
        self.btn_clear.clicked.connect(self._clear_selection)
        ctrl.addWidget(self.btn_clear)
        self.btn_reset = QPushButton("Reset Map", self)
        self.btn_reset.clicked.connect(self._reset_map)
        ctrl.addWidget(self.btn_reset)
        ctrl.addStretch(1)

        v.addLayout(ctrl)

        # Figures: left scatter (embedding), right live preview map
        figs = QHBoxLayout()
        self.fig_sc = Figure(figsize=(6, 5))
        self.canvas_sc = FigureCanvas(self.fig_sc)
        self.ax_sc = self.fig_sc.add_subplot(111)
        figs.addWidget(self.canvas_sc, stretch=3)

        self.fig_map = Figure(figsize=(5, 5))
        self.canvas_map = FigureCanvas(self.fig_map)
        self.ax_map = self.fig_map.add_subplot(111)
        view, _ = _preview_img(self.map)
        _ = self.ax_map.imshow(view, cmap=self.cmap, vmin=0, vmax=int(np.max(self.map)))
        self.ax_map.set_title("Live Phase Map Preview", fontsize=11, fontweight="bold")
        self.ax_map.axis("off")
        figs.addWidget(self.canvas_map, stretch=2)
        v.addLayout(figs)

        # Bottom: progress + close
        self.prog = QProgressBar(self)
        self.prog.setRange(0, 100)
        self.prog.hide()
        v.addWidget(self.prog)
        bb = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

        # Initial embedding
        self._compute_embedding()

    # ---------------- Embedding & plotting ----------------
    def _compute_embedding(self):
        # Clear any prior selector
        self._disconnect_selector()
        self._last_path = None
        self.prog.show()
        self.prog.setValue(0)
        QApplication.processEvents()

        rng = check_random_state(42)
        Ns = int(self.sample_spin.value())
        if self.N <= Ns:
            self.sample_idx = np.arange(self.N, dtype=np.int64)
        else:
            self.sample_idx = rng.choice(self.N, size=Ns, replace=False)
        Xs = self.feat[self.sample_idx]
        labs = self.map.reshape(-1)[self.sample_idx]

        kind = self.proj_choice.currentText()
        self._proj_kind = kind

        try:
            if kind == "PCA":
                self._pca2 = PCA(n_components=2, svd_solver="randomized", random_state=42).fit(Xs)
                self.embed = self._pca2.transform(Xs)
            elif kind == "UMAP" and _HAS_UMAP:
                self._umap = umap.UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=30,
                    min_dist=0.05,
                )
                self.embed = self._umap.fit_transform(Xs)
            elif kind == "t-SNE" and _HAS_TSNE:
                self.embed = TSNE(
                    n_components=2,
                    init="pca",
                    random_state=42,
                    learning_rate="auto",
                    perplexity=30,
                ).fit_transform(Xs)
            else:  # fallback
                self._pca2 = PCA(n_components=2, svd_solver="randomized", random_state=42).fit(Xs)
                self.embed = self._pca2.transform(Xs)
                self._proj_kind = "PCA"
        except Exception as e:
            _nice_error(f"Projection failed ({kind}): {e}", self)
            self.prog.hide()
            return

        self.prog.setValue(60)
        QApplication.processEvents()

        # Plot scatter colored by current phase labels
        self.ax_sc.clear()
        sc = self.ax_sc.scatter(
            self.embed[:, 0],
            self.embed[:, 1],
            c=labs,
            s=2,
            cmap=self.cmap,
            vmin=0,
            vmax=int(np.max(self.map)),
        )
        self.ax_sc.set_title(
            f"{self._proj_kind} of features (N={len(self.sample_idx):,})",
            fontsize=11,
            fontweight="bold",
        )
        self.ax_sc.set_xlabel("Dim 1")
        self.ax_sc.set_ylabel("Dim 2")
        self.fig_sc.colorbar(sc, ax=self.ax_sc, fraction=0.046, pad=0.04)
        self.canvas_sc.draw()

        # Activate default selector (lasso/ellipse) on scatter
        self._activate_selector()

        # Refresh map preview
        self._redraw_preview()
        self.prog.setValue(100)
        self.prog.hide()

    def _activate_selector(self):
        self._disconnect_selector()
        tool = self.shape_choice.currentText().lower()
        if tool == "lasso":
            self._selector = LassoSelector(self.ax_sc, onselect=self._on_lasso_scatter)
        else:
            self._selector = EllipseSelector(self.ax_sc, onselect=self._on_ellipse_scatter, interactive=True)

    def _disconnect_selector(self):
        if self._selector is not None:
            try:
                self._selector.disconnect_events()
            except Exception:
                pass
            self._selector = None

    def _on_lasso_scatter(self, verts):
        self._last_path = Path(verts)
        self._highlight_selection_on_scatter()

    def _on_ellipse_scatter(self, eclick, erelease):
        x0, y0 = float(min(eclick.xdata, erelease.xdata)), float(min(eclick.ydata, erelease.ydata))
        x1, y1 = float(max(eclick.xdata, erelease.xdata)), float(max(eclick.ydata, erelease.ydata))
        # approximate ellipse by polygon
        t = np.linspace(0, 2 * np.pi, 200)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        verts = np.c_[cx + rx * np.cos(t), cy + ry * np.sin(t)]
        self._last_path = Path(verts)
        self._highlight_selection_on_scatter()

    def _highlight_selection_on_scatter(self):
        if self._last_path is None or self.embed is None:
            return
        pts = self.embed
        inside = self._last_path.contains_points(pts)
        # overlay highlight
        self.ax_sc.collections = [c for c in self.ax_sc.collections if getattr(c, "is_base", False)]
        hl = self.ax_sc.scatter(
            pts[inside, 0],
            pts[inside, 1],
            s=8,
            facecolors="none",
            edgecolors="k",
            linewidths=0.7,
        )
        hl.is_base = False
        self.canvas_sc.draw_idle()

    # ---------------- Assignment ----------------
    def _assign_selection(self):
        if self._last_path is None or self.embed is None or self.sample_idx is None:
            _nice_error("Draw a selection first.", self)
            return
        target_phase = int(self.phase_spin.value()) - 1  # 1-based to 0-based

        # Determine selected indices in the SAMPLE set
        inside = self._last_path.contains_points(self.embed)
        if not np.any(inside):
            _nice_error("Selection contains no points.", self)
            return
        sel_sample_idx = self.sample_idx[inside]

        # Build full-resolution mask
        apply_full = self.chk_full.isChecked()
        full_mask = np.zeros(self.N, dtype=bool)

        if apply_full:
            self.prog.show()
            self.prog.setValue(0)
            QApplication.processEvents()
            if self._proj_kind == "PCA" and hasattr(self, "_pca2"):
                # Accurate: transform full dataset in batches, test polygon in
                # 2D
                bs = 400_000
                total = self.N
                done = 0
                for s in range(0, total, bs):
                    e = min(s + bs, total)
                    Xb = self.feat[s:e]
                    Yb = self._pca2.transform(Xb)
                    full_mask[s:e] = self._last_path.contains_points(Yb)
                    done = e
                    self.prog.setValue(int(100 * done / total))
                    QApplication.processEvents()
            else:
                # Fallback: nearest-neighbor propagate from selected SAMPLE points in ORIGINAL FEATURE SPACE
                # Points whose nearest sample neighbor is selected will be
                # assigned.
                self.prog.setValue(5)
                QApplication.processEvents()
                from sklearn.neighbors import NearestNeighbors

                Xs = self.feat[self.sample_idx]
                sel_flags = np.zeros(len(self.sample_idx), dtype=bool)
                # map sample_idx positions to flags
                sel_positions = np.nonzero(inside)[0]
                sel_flags[sel_positions] = True
                nn = NearestNeighbors(n_neighbors=1).fit(Xs)
                bs = 300_000
                total = self.N
                done = 0
                for s in range(0, total, bs):
                    e = min(s + bs, total)
                    Xb = self.feat[s:e]
                    _, ind = nn.kneighbors(Xb, n_neighbors=1, return_distance=True)
                    ind = ind.reshape(-1)
                    full_mask[s:e] = sel_flags[ind]
                    done = e
                    self.prog.setValue(int(100 * done / total))
                    QApplication.processEvents()
        else:
            # Fast: affect only the sampled points
            full_mask[sel_sample_idx] = True

        # Apply assignment
        n_before = int(np.max(self.map)) + 1
        self.map.reshape(-1)[full_mask] = target_phase
        # Update owner's labels too if present
        if self.labels1d is not None and self.labels1d.size == self.N:
            self.labels1d[full_mask] = target_phase
        # Ensure colormap covers new phase id
        n_after = int(np.max(self.map)) + 1
        if n_after > n_before:
            self.owner._cmap = _make_consistent_cmap(n_after, base_name="tab20")
            self.cmap = self.owner._cmap

        # Reflect in owner & preview
        self.owner.results = self.map.copy()
        self.owner._display_results(self.owner.results)
        self._redraw_preview()
        self._recolor_scatter()
        self.prog.hide()

    def _recolor_scatter(self):
        # recolor the scatter by new labels for sampled points
        if self.embed is None or self.sample_idx is None:
            return
        labs = self.map.reshape(-1)[self.sample_idx]
        self.ax_sc.collections.clear()
        sc = self.ax_sc.scatter(
            self.embed[:, 0],
            self.embed[:, 1],
            c=labs,
            s=2,
            cmap=self.cmap,
            vmin=0,
            vmax=int(np.max(self.map)),
        )
        sc.is_base = True
        self.canvas_sc.draw_idle()

    def _redraw_preview(self):
        self.ax_map.clear()
        view, _ = _preview_img(self.map)
        self.ax_map.imshow(view, cmap=self.cmap, vmin=0, vmax=int(np.max(self.map)))
        self.ax_map.set_title("Live Phase Map Preview", fontsize=11, fontweight="bold")
        self.ax_map.axis("off")
        self.canvas_map.draw_idle()

    # ---------------- Misc ----------------
    def _clear_selection(self):
        self._last_path = None
        self._recolor_scatter()

    def _reset_map(self):
        # reset to owner's current map stored before editor opened? Here we
        # revert to the map state at dialog launch
        self.map[:] = self.owner.results
        self.owner._display_results(self.owner.results)
        self._recolor_scatter()
        self._redraw_preview()


# Entrypoint
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = PhaseMappingStep()
    w.show()
    sys.exit(app.exec_())
