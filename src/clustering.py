"""
Customer clustering module for eCommerce transactions analysis.

This module implements K-Means clustering for customer segmentation
with evaluation metrics and visualization capabilities.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config
from . import data_loader
from . import preprocessing

# Configure logging
logger = logging.getLogger(__name__)


def prepare_clustering_features(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    feature_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Prepare feature matrix for clustering.
    
    Args:
        customers: Customers DataFrame
        transactions: Transactions DataFrame
        feature_columns: Optional list of feature columns to use
    
    Returns:
        Tuple of (feature DataFrame, scaled features array, feature column names)
    
    Example:
        >>> df, scaled, cols = prepare_clustering_features(customers, transactions)
        >>> print(f"Prepared {len(cols)} features for clustering")
    """
    # Prepare customer features
    features = preprocessing.prepare_customer_features(customers, transactions)
    
    # Encode regions
    features = preprocessing.encode_categorical(features, columns=["Region"])
    
    # Get feature columns
    if feature_columns is None:
        feature_columns = [
            "TotalSpend", "TransactionCount", "AvgTransactionValue",
            "DaysSinceSignup", "Frequency", "Monetary"
        ]
        # Add region columns
        region_cols = [col for col in features.columns if col.startswith("Region_")]
        feature_columns.extend(region_cols)
    
    # Filter to existing columns
    feature_columns = [col for col in feature_columns if col in features.columns]
    
    # Handle missing values
    features[feature_columns] = features[feature_columns].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features[feature_columns])
    
    logger.info(f"Prepared {len(feature_columns)} features for {len(features)} customers")
    return features, scaled_features, feature_columns


def find_optimal_clusters(
    scaled_features: np.ndarray,
    max_clusters: int = 10,
    plot: bool = True,
    save_path: Optional[Path] = None
) -> Tuple[int, plt.Figure]:
    """
    Find optimal number of clusters using the elbow method and silhouette scores.
    
    Args:
        scaled_features: Scaled feature array
        max_clusters: Maximum number of clusters to test
        plot: Whether to generate visualization
        save_path: Optional path to save the plot
    
    Returns:
        Tuple of (optimal cluster count, Figure object)
    
    Example:
        >>> optimal_k, fig = find_optimal_clusters(scaled_features, max_clusters=10)
        >>> print(f"Optimal clusters: {optimal_k}")
    """
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(
            n_clusters=k,
            random_state=config.CLUSTERING_CONFIG["random_state"],
            n_init=config.CLUSTERING_CONFIG["n_init"]
        )
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))
    
    # Find optimal k based on silhouette score
    optimal_k = K_range[np.argmax(silhouette_scores)]
    
    fig = None
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow plot
        axes[0].plot(K_range, inertias, "bo-", linewidth=2, markersize=8)
        axes[0].set_xlabel("Number of Clusters (k)", fontsize=12)
        axes[0].set_ylabel("Inertia", fontsize=12)
        axes[0].set_title("Elbow Method for Optimal k", fontsize=14, fontweight="bold")
        axes[0].axvline(x=optimal_k, color="r", linestyle="--", label=f"Optimal k={optimal_k}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette plot
        axes[1].plot(K_range, silhouette_scores, "go-", linewidth=2, markersize=8)
        axes[1].set_xlabel("Number of Clusters (k)", fontsize=12)
        axes[1].set_ylabel("Silhouette Score", fontsize=12)
        axes[1].set_title("Silhouette Analysis for Optimal k", fontsize=14, fontweight="bold")
        axes[1].axvline(x=optimal_k, color="r", linestyle="--", label=f"Optimal k={optimal_k}")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved elbow plot to {save_path}")
    
    logger.info(f"Optimal number of clusters: {optimal_k} (silhouette score: {max(silhouette_scores):.3f})")
    return optimal_k, fig


def perform_clustering(
    scaled_features: np.ndarray,
    n_clusters: int = None,
    random_state: int = None
) -> Tuple[np.ndarray, KMeans]:
    """
    Perform K-Means clustering on customer data.
    
    Args:
        scaled_features: Scaled feature array
        n_clusters: Number of clusters (uses config default if None)
        random_state: Random state for reproducibility
    
    Returns:
        Tuple of (cluster labels array, fitted KMeans model)
    
    Example:
        >>> labels, model = perform_clustering(scaled_features, n_clusters=4)
        >>> print(f"Cluster sizes: {np.bincount(labels)}")
    """
    if n_clusters is None:
        n_clusters = config.CLUSTERING_CONFIG["n_clusters"]
    if random_state is None:
        random_state = config.CLUSTERING_CONFIG["random_state"]
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=config.CLUSTERING_CONFIG["n_init"],
        max_iter=config.CLUSTERING_CONFIG["max_iter"]
    )
    
    labels = kmeans.fit_predict(scaled_features)
    
    logger.info(f"Performed clustering with {n_clusters} clusters")
    logger.info(f"Cluster sizes: {np.bincount(labels)}")
    
    return labels, kmeans


def evaluate_clustering(
    scaled_features: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate clustering quality using multiple metrics.
    
    Computes:
    - Davies-Bouldin Index (lower is better)
    - Silhouette Score (higher is better, range -1 to 1)
    
    Args:
        scaled_features: Scaled feature array
        labels: Cluster labels
    
    Returns:
        Dictionary containing evaluation metrics
    
    Example:
        >>> metrics = evaluate_clustering(scaled_features, labels)
        >>> print(f"DB Index: {metrics['davies_bouldin']:.3f}")
    """
    db_index = davies_bouldin_score(scaled_features, labels)
    silhouette = silhouette_score(scaled_features, labels)
    
    n_clusters = len(np.unique(labels))
    cluster_sizes = np.bincount(labels)
    
    metrics = {
        "davies_bouldin": db_index,
        "silhouette_score": silhouette,
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes.tolist(),
        "min_cluster_size": int(cluster_sizes.min()),
        "max_cluster_size": int(cluster_sizes.max()),
    }
    
    logger.info(f"Clustering evaluation: DB Index={db_index:.3f}, Silhouette={silhouette:.3f}")
    return metrics


def get_cluster_profiles(
    features: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Generate cluster profiles showing mean feature values for each cluster.
    
    Args:
        features: Feature DataFrame
        labels: Cluster labels
        feature_columns: List of feature columns to include
    
    Returns:
        DataFrame with cluster profiles
    
    Example:
        >>> profiles = get_cluster_profiles(features, labels, feature_cols)
        >>> print(profiles.T)  # Transpose for better readability
    """
    features_with_clusters = features.copy()
    features_with_clusters["Cluster"] = labels
    
    # Filter to numeric feature columns only
    numeric_cols = [col for col in feature_columns if features[col].dtype in ["int64", "float64"]]
    
    profiles = features_with_clusters.groupby("Cluster")[numeric_cols].mean()
    profiles["Size"] = features_with_clusters.groupby("Cluster").size()
    profiles["Percentage"] = (profiles["Size"] / len(features) * 100).round(1)
    
    return profiles


def visualize_clusters(
    scaled_features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create visualization of customer clusters using PCA projection.
    
    Args:
        scaled_features: Scaled feature array
        labels: Cluster labels
        feature_names: Names of features (for axis labels)
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib Figure object
    """
    from sklearn.decomposition import PCA
    
    # Reduce to 2D for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(scaled_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = config.COLOR_PALETTES["clusters"]
    unique_labels = np.unique(labels)
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=colors[i % len(colors)], label=f"Cluster {label}",
            alpha=0.7, s=50, edgecolors="black", linewidth=0.5
        )
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=12)
    ax.set_title("Customer Segments Visualization (PCA)", fontsize=14, fontweight="bold")
    ax.legend(title="Segment", loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved cluster visualization to {save_path}")
    
    return fig


def save_cluster_results(
    features: pd.DataFrame,
    labels: np.ndarray,
    output_path: Optional[Path] = None
) -> None:
    """
    Save clustering results to CSV file.
    
    Args:
        features: Feature DataFrame with CustomerID
        labels: Cluster labels
        output_path: Path to save results (uses config default if None)
    """
    if output_path is None:
        output_path = config.OUTPUT_FILES["cluster_assignments"]
    
    results = features[["CustomerID"]].copy()
    results["Cluster"] = labels
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results.to_csv(output_path, index=False)
    logger.info(f"Saved cluster assignments to {output_path}")


def main() -> Dict:
    """
    Run the complete customer clustering pipeline.
    
    Returns:
        Dictionary containing clustering results and metrics
    """
    logger.info("Starting customer clustering pipeline...")
    
    # Load data
    customers, products, transactions = data_loader.load_all_data()
    
    # Prepare features
    features, scaled_features, feature_cols = prepare_clustering_features(
        customers, transactions
    )
    
    # Find optimal clusters
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    optimal_k, _ = find_optimal_clusters(
        scaled_features,
        save_path=config.FIGURES_DIR / "elbow_plot.png"
    )
    
    # Perform clustering
    labels, model = perform_clustering(scaled_features, n_clusters=optimal_k)
    
    # Evaluate
    metrics = evaluate_clustering(scaled_features, labels)
    
    # Generate profiles
    profiles = get_cluster_profiles(features, labels, feature_cols)
    
    # Visualize
    visualize_clusters(
        scaled_features, labels,
        save_path=config.FIGURES_DIR / "cluster_visualization.png"
    )
    
    # Save results
    save_cluster_results(features, labels)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CUSTOMER CLUSTERING RESULTS")
    print("=" * 60)
    print(f"\nNumber of clusters: {metrics['n_clusters']}")
    print(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.3f}")
    print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
    print(f"\nCluster sizes: {metrics['cluster_sizes']}")
    print("\nCluster Profiles:")
    print(profiles.to_string())
    print("=" * 60)
    
    return {
        "metrics": metrics,
        "profiles": profiles,
        "model": model,
        "labels": labels,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
