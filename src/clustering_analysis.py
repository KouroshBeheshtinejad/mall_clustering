import os
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ================== CONFIG ==================
SEED = 42
np.random.seed(SEED)

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
REPORTS_DIR = os.path.join(BASE_DIR, "..", "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ================== UTILS ==================
def load_and_preprocess():
    df = pd.read_csv(os.path.join(DATA_DIR, "Customer.csv"))
    df = df.drop(columns=["CustomerID"])
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    logging.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    return df

def scale_features(df, features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return pd.DataFrame(X_scaled, columns=features)

def plot_clusters(df, x_col, y_col, labels, title, filename):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=df[x_col], y=df[y_col], hue=labels, palette="tab10", s=60, alpha=0.7, edgecolor="k"
    )
    plt.title(title)
    plt.legend(title="Cluster")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def bar_plot_clusters(cluster_counts, filename):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="tab10")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")
    plt.title("Customer count per Cluster")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def plot_silhouette_scores(X_scaled_df, filename):
    scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        labels = kmeans.fit_predict(X_scaled_df)
        score = silhouette_score(X_scaled_df, labels)
        scores.append(score)
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 11), scores, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores for different k")
    plt.grid(True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    return scores

# ================== MAIN ==================
def main():
    df = load_and_preprocess()
    features = ["Annual Income (k$)", "Spending Score (1-100)", "Age"]
    X_scaled_df = scale_features(df, features)

    # Silhouette scores plot for KMeans k=2..10
    sil_scores = plot_silhouette_scores(X_scaled_df, os.path.join(FIGURES_DIR, "silhouette_scores.png"))
    logging.info(f"Silhouette scores (KMeans k=2..10): {sil_scores}")

    # Fit clustering models
    kmeans = KMeans(n_clusters=6, n_init=10, random_state=SEED)
    labels_kmeans = kmeans.fit_predict(X_scaled_df)

    agg = AgglomerativeClustering(n_clusters=6)
    labels_agg = agg.fit_predict(X_scaled_df)

    db = DBSCAN(eps=0.5, min_samples=5)
    labels_db = db.fit_predict(X_scaled_df)

    # Silhouette scores
    sil_kmeans = silhouette_score(X_scaled_df, labels_kmeans)
    sil_agg = silhouette_score(X_scaled_df, labels_agg)

    # For DBSCAN, remove noise points (-1)
    mask = labels_db != -1
    if np.sum(mask) > 1:
        sil_db = silhouette_score(X_scaled_df[mask], labels_db[mask])
    else:
        sil_db = np.nan
    silhouettes = {
        "KMeans (k=6)": sil_kmeans,
        "Agglomerative (k=6)": sil_agg,
        "DBSCAN": sil_db
    }
    logging.info(f"Silhouette comparison: {silhouettes}")

    # PCA Visualization
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled_df)
    fig_pca = os.path.join(FIGURES_DIR, "pca_comparison.png")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap="tab10", s=30)
    axs[0].set_title("KMeans (k=6)")
    axs[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_agg, cmap="tab10", s=30)
    axs[1].set_title("Agglomerative (k=6)")
    axs[2].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_db, cmap="tab10", s=30)
    axs[2].set_title("DBSCAN")
    plt.savefig(fig_pca, bbox_inches="tight")
    plt.close()

    # Cluster Profiling (KMeans)
    df["Cluster"] = labels_kmeans
    cluster_summary = df.groupby("Cluster")[features].mean()
    cluster_counts = df["Cluster"].value_counts().sort_index()
    cluster_summary["count"] = cluster_counts
    cluster_summary.to_csv(os.path.join(REPORTS_DIR, "cluster_summary.csv"))
    logging.info("\n" + str(cluster_summary))

    # ================== Plots for all clusters ==================
    # KMeans plots
    plot_clusters(df, "Annual Income (k$)", "Spending Score (1-100)", labels_kmeans,
                  "Customer Segments - KMeans (k=6)",
                  os.path.join(FIGURES_DIR, "kmeans_scatter.png"))
    bar_plot_clusters(cluster_counts, os.path.join(FIGURES_DIR, "kmeans_customers_per_cluster.png"))

    # Agglomerative plots
    df["Cluster_Agg"] = labels_agg
    agg_counts = df["Cluster_Agg"].value_counts().sort_index()
    plot_clusters(df, "Annual Income (k$)", "Spending Score (1-100)", labels_agg,
                  "Customer Segments - Agglomerative (k=6)",
                  os.path.join(FIGURES_DIR, "agg_scatter.png"))
    bar_plot_clusters(agg_counts, os.path.join(FIGURES_DIR, "agg_customers_per_cluster.png"))

    # DBSCAN plots
    df_db = df.copy()
    df_db["Cluster_DB"] = labels_db
    db_counts = df_db[df_db["Cluster_DB"] != -1]["Cluster_DB"].value_counts().sort_index()
    plot_clusters(df_db, "Annual Income (k$)", "Spending Score (1-100)", labels_db,
                  "Customer Segments - DBSCAN",
                  os.path.join(FIGURES_DIR, "dbscan_scatter.png"))
    bar_plot_clusters(db_counts, os.path.join(FIGURES_DIR, "dbscan_customers_per_cluster.png"))

# ================== RUN ==================
if __name__ == "__main__":
    main()