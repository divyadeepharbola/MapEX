# MapEX

**MapEX** is a graphical application for performing **phase mapping** of elemental maps using secondary X-rays using advanced clustering and dimensionality reduction techniques.

It supports common methods like:
- KMeans, GMM, FCM, Hierarchical clustering
- PCA-based dimensionality reduction
- Density-based clustering: Mean Shift, DBSCAN, Affinity Propagation

MapEX helps researchers preprocess, analyze, and visualize complex elemental data interactively.

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


Release executable soon !

## About

**MapEX** is a Python/Qt toolkit for quantitative multi-channel X-ray map analysis (µ-XRF, EPMA-WDS, SEM/TEM-EDS, synchrotron). It packages raw exports into HDF5, provides calibration from intensity→composition, supports interactive phase classification (PCA + clustering), correlation plots, ROI tools, and publication-quality exports. The goal is a reproducible, GUI-driven workflow that scales from laptop to HPC.

## License

This project is licensed under the **MIT License** — see [LICENSE](./LICENSE) for details.
