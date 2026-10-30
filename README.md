# 🔬 Machine Learning Studio

A desktop GUI application for interactive data analysis, clustering, and classification — built with Python, Tkinter, and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

### 📁 Data Loading & Preprocessing
- Import datasets from **CSV** and **Excel** files
- Automatic **ID column removal**
- Interactive **descriptive statistics** viewer
- Configurable preprocessing pipeline:
  - Aberrant value detection (text in numeric columns → NaN)
  - **Label encoding** for categorical features
  - Missing value imputation (`mean`, `median`, `drop`)
  - Normalization (`Z-Score`, `Min-Max`)
  - **PCA** dimensionality reduction
- Live **data preview table** with scrolling
- **2D scatter plot** visualization (auto PCA projection for high-dimensional data)

### 📊 Clustering
- **5 algorithms**: KMeans, K-Medoids, AGNES, DIANA, DBSCAN
- Dynamic parameter panel (K for partitional methods, ε/MinPts for DBSCAN)
- **Auto-detect optimal K** using the geometric elbow method
- Evaluation metrics: **Silhouette Score** and **Inertia (SSE)**
- Visualization tools:
  - PCA-projected cluster scatter plot with centroids
  - **Elbow method** curve
  - **Dendrogram** (Ward linkage)
  - **Algorithm comparison** bar chart

### 🎯 Classification
- **4 algorithms**: k-NN, Naive Bayes, C4.5 (Decision Tree), SVM
- Automatic 80/20 train-test split
- Metrics dashboard: **Accuracy**, **Precision**, **Recall**, **F1-Score**
- **Confusion matrix** heatmap
- **K-Precision curve** for k-NN (K = 1–10)

---

## 📂 Project Structure

```
├── app_gui.py                  # Main application (GUI entry point)
├── src/
│   ├── data_manager.py         # Data loading, preprocessing & splitting
│   ├── clustering_engine.py    # Clustering algorithms & metrics
│   └── classification_engine.py# Classification algorithms & metrics
├── data/
│   └── horse-colic.csv         # Sample dataset
├── test_DM.py                  # CLI test script for clustering pipeline
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Machine-Learning-Studio.git
   cd Machine-Learning-Studio
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn scipy
   ```

3. **Run the application**
   ```bash
   python app_gui.py
   ```

### Quick Start

1. Click **📂 Open Dataset** and load a CSV/Excel file (try `data/horse-colic.csv`)
2. Select a **Target column** if you plan to use Classification
3. Configure preprocessing options and click **⚙️ APPLY PREPROCESSING**
4. Switch to the **CLUSTERING** or **CLASSIFICATION** tab to analyze your data

---

## 🧪 Running Tests

A command-line test script is included to validate the clustering pipeline without the GUI:

```bash
python test_DM.py
```

This runs all 5 clustering algorithms on the sample dataset and prints a comparison table of Silhouette scores and Inertia values.

---

## 📸 Screenshots

> _Add screenshots of the application here._

---

## 🛠️ Tech Stack

| Layer          | Technology                                      |
|----------------|--------------------------------------------------|
| GUI            | Tkinter + ttk                                    |
| Visualization  | Matplotlib, Seaborn                              |
| ML Algorithms  | scikit-learn, SciPy                              |
| Data Handling  | pandas, NumPy                                    |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
