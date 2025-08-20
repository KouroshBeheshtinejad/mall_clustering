# Mall Customer Segmentation

![Clustering](https://img.shields.io/badge/Status-Completed-green)

## 🔹 Overview
This project performs **customer segmentation** for a mall dataset using **unsupervised machine learning algorithms**. The goal is to group customers into meaningful clusters based on their **Age**, **Annual Income**, and **Spending Score**.

This is a complete end-to-end project including:
- Data preprocessing
- Feature scaling
- Multiple clustering algorithms: **KMeans**, **Agglomerative Hierarchical**, **DBSCAN**
- Evaluation with **Silhouette Score**
- **PCA visualization**
- Cluster profiling and plotting

---

## 🔹 Features

- **Preprocessing**
  - Dropped unnecessary columns
  - Encoded categorical features (Gender)
- **Clustering**
  - KMeans (k=2..10, final k=6)
  - Agglomerative Hierarchical
  - DBSCAN
- **Evaluation**
  - Silhouette scores comparison
  - Visual comparison with PCA 2D projection
- **Visualization**
  - Scatter plots for clusters
  - Bar charts for cluster sizes
  - Silhouette score plots
- **Cluster Profiling**
  - Mean values of features per cluster
  - Customer counts per cluster

---

## 🔹 Directory Structure
```bash
mall_clustering/
│
├─ data/ # Dataset (not included in GitHub, sample CSV can be added)
├─ reports/
│ ├─ figures/ # Generated plots (scatter, bar, PCA)
│ └─ cluster_summary.csv
├─ src/
│ └─ clustering_analysis.py
├─ .gitignore # Ignore data, figures, reports
└─ README.md
```

---

## 🔹 Getting Started

### **1. Clone the repository**
```bash
git clone https://github.com/KouroshBeheshtinejad/mall_clustering.git
cd mall_clustering/src
```

### **2. Create virtual environment & install dependencies**
```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### **3. Add your dataset**
Place your CSV file in:
```bash
data/Customer.csv
```

Format:
| CustomerID | Gender | Age | Annual Income (k\$) | Spending Score (1-100) |
| ---------- | ------ | --- | ------------------- | ---------------------- |

### **4. Run the analysis**
```bash
python clustering_analysis.py
```

- Outputs:
 - `reports/cluster_summary.csv` → Cluster statistics
 - `reports/figures/` → All generated plots


## 🔹 Results

- **Optimal K for KMeans: 6**
- **Silhouette Scores**
 - KMeans (k=6): 0.428
 - Agglomerative: 0.420
 - DBSCAN: 0.482

DBSCAN achieved the highest silhouette score but creates some noise points (`-1`).

- **Cluster Insights**
 - Cluster 0 → Older, mid-income, medium spending
 - Cluster 1 → Younger, mid-income, medium spending
 - Cluster 2 → Old, high-income, low spending
 - Cluster 3 → Young, high-income, high spending
 - Cluster 4 → Younger, low-income, high spending
 - Cluster 5 → Old, low-income, low spending

- **Visualizations**
 - PCA projection showing KMeans, Agglomerative, DBSCAN
 - Cluster scatter and bar plots


## 🔹 Algorithms Used

1. **KMeans**
 - Centroid-based clustering
 - Number of clusters optimized with silhouette scores

2. **Agglomerative Hierarchical Clustering**
 - Bottom-up clustering
 - Euclidean distance metric

3. **DBSCAN**
 - Density-based clustering
 - Detects noise/outliers
 - `eps=0.5`, `min_samples=5`

4. **PCA**
 - Reduces 3D features into 2D for visualization


## 🔹 Notes

- `.gitignore` excludes
 - Dataset files(`data/`)
 - Generated plots and reports(`reports/`)
- You can still run the project by adding your own dataset.
- The code is modular: you can easily replace algorithms or tweak parameters.


## 🔹 Requirements
```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
```


## 🔹 Author
**Kourosh Beheshtinejad**
