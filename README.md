# MapEX

**MapEX** is a graphical application for performing **phase mapping** of elemental maps using secondary X-rays using advanced clustering and dimensionality reduction techniques.

**Key features**
- Unsupervised clustering: **K-Means, GMM, FCM, Hierarchical**
- Dimensionality reduction: **PCA**
- Density methods: **Mean Shift, DBSCAN, Affinity Propagation**
- Interactive **correlation plots** and **ternary plots** (user-selected elements)
- ROI tools and **report-ready exports**
- HDF5 packaging; scalable from laptop to HPC

---

## 🔧 Requirements

To run MapEX, you need **Python 3.7+** and the following dependencies:

```bash
pip install pyqt5 numpy scikit-learn matplotlib scipy joblib
```

Install the Program by 

```bash
git clone https://github.com/divyadeepharbola/MapEX.git
cd MapEX
python main.py
```

install required dependencies by 
```
pip install -r requirements.txt
```
## 🚀 Quick Start (one command)

Works on **Windows, macOS, and Linux**. Requires Python installed.

```bash
# Clone and install
git clone https://github.com/divyadeepharbola/MapEX.git
cd MapEX
python install.py  
```

Release executable soon !

## About

**MapEX** is a Python/Qt toolkit for quantitative multi-channel X-ray map analysis (µ-XRF, EPMA-WDS, SEM/TEM-EDS, synchrotron). It packages raw exports into HDF5, provides calibration from intensity→composition, supports interactive phase classification (PCA + clustering), correlation plots, ROI tools, and publication-quality exports. The goal is a reproducible, GUI-driven workflow that scales from laptop to HPC.

## License

This project is licensed under the **MIT License** — see [LICENSE](./LICENSE) for details.
