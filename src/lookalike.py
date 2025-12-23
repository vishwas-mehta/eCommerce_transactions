"""
Lookalike customer modeling module for eCommerce transactions analysis.

This module implements a similarity-based lookalike model to identify
customers with similar behavior patterns for targeted marketing.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from . import config
from . import data_loader
from . import preprocessing

# Configure logging
logger = logging.getLogger(__name__)


def build_similarity_matrix(
    scaled_features: np.ndarray,
    metric: str = "cosine"
) -> np.ndarray:
    """
    Build a customer similarity matrix.
    
    Args:
        scaled_features: Scaled feature array (n_customers x n_features)
        metric: Similarity metric to use ('cosine' or 'euclidean')
    
    Returns:
        Similarity matrix (n_customers x n_customers)
    
    Example:
        >>> similarity = build_similarity_matrix(scaled_features)
        >>> print(f"Matrix shape: {similarity.shape}")
    """
    if metric == "cosine":
        similarity_matrix = cosine_similarity(scaled_features)
    elif metric == "euclidean":
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(scaled_features)
        # Convert distances to similarities
        similarity_matrix = 1 / (1 + distances)
    else:
        raise ValueError(f"Invalid metric: {metric}. Use 'cosine' or 'euclidean'")
    
    logger.info(f"Built {metric} similarity matrix of shape {similarity_matrix.shape}")
    return similarity_matrix


def get_top_lookalikes(
    customer_idx: int,
    similarity_matrix: np.ndarray,
    customer_ids: pd.Series,
    n: int = 3,
    min_similarity: float = 0.0
) -> List[Tuple[str, float]]:
    """
    Get top N lookalike customers for a given customer.
    
    Args:
        customer_idx: Index of the target customer in the similarity matrix
        similarity_matrix: Pre-computed similarity matrix
        customer_ids: Series mapping indices to CustomerIDs
        n: Number of lookalikes to return
        min_similarity: Minimum similarity threshold
    
    Returns:
        List of tuples (CustomerID, similarity_score)
    
    Example:
        >>> lookalikes = get_top_lookalikes(0, similarity_matrix, customer_ids, n=3)
        >>> print(lookalikes)
        [('C0045', 0.95), ('C0089', 0.92), ('C0123', 0.88)]
    """
    # Get similarity scores for this customer
    similarities = similarity_matrix[customer_idx]
    
    # Create array of (index, similarity) pairs, excluding self
    candidates = [
        (idx, sim) for idx, sim in enumerate(similarities)
        if idx != customer_idx and sim >= min_similarity
    ]
    
    # Sort by similarity (descending)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N
    top_n = candidates[:n]
    
    # Convert indices to CustomerIDs
    lookalikes = [
        (customer_ids.iloc[idx], round(sim, 4))
        for idx, sim in top_n
    ]
    
    return lookalikes


def find_all_lookalikes(
    features: pd.DataFrame,
    similarity_matrix: np.ndarray,
    n_lookalikes: int = 3,
    customer_subset: Optional[List[str]] = None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Find lookalike customers for all (or a subset of) customers.
    
    Args:
        features: Customer features DataFrame with CustomerID column
        similarity_matrix: Pre-computed similarity matrix
        n_lookalikes: Number of lookalikes per customer
        customer_subset: Optional list of CustomerIDs to process (if None, process all)
    
    Returns:
        Dictionary mapping CustomerID to list of (lookalike_id, similarity) tuples
    
    Example:
        >>> lookalikes = find_all_lookalikes(features, similarity_matrix, n_lookalikes=3)
        >>> print(lookalikes['C0001'])
    """
    customer_ids = features["CustomerID"]
    
    if customer_subset is not None:
        # Get indices for subset
        indices_to_process = [
            idx for idx, cid in enumerate(customer_ids)
            if cid in customer_subset
        ]
    else:
        indices_to_process = range(len(customer_ids))
    
    lookalikes = {}
    for idx in indices_to_process:
        customer_id = customer_ids.iloc[idx]
        lookalikes[customer_id] = get_top_lookalikes(
            idx, similarity_matrix, customer_ids, n=n_lookalikes
        )
    
    logger.info(f"Found lookalikes for {len(lookalikes)} customers")
    return lookalikes


def export_lookalikes(
    lookalikes: Dict[str, List[Tuple[str, float]]],
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Export lookalike results to CSV file.
    
    Creates a formatted CSV with columns:
    - CustomerID: Target customer
    - Lookalike_1, Lookalike_1_Score: First lookalike
    - Lookalike_2, Lookalike_2_Score: Second lookalike
    - Lookalike_3, Lookalike_3_Score: Third lookalike
    
    Args:
        lookalikes: Dictionary from find_all_lookalikes()
        output_path: Path to save CSV (uses config default if None)
    
    Returns:
        DataFrame with formatted lookalike results
    """
    if output_path is None:
        output_path = config.OUTPUT_FILES["lookalike"]
    
    rows = []
    for customer_id, similar_customers in lookalikes.items():
        row = {"CustomerID": customer_id}
        for i, (lookalike_id, score) in enumerate(similar_customers, 1):
            row[f"Lookalike_{i}"] = lookalike_id
            row[f"Lookalike_{i}_Score"] = score
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved lookalike results to {output_path}")
    
    return df


def get_lookalike_summary(
    lookalikes: Dict[str, List[Tuple[str, float]]]
) -> Dict[str, any]:
    """
    Generate summary statistics for lookalike results.
    
    Args:
        lookalikes: Dictionary from find_all_lookalikes()
    
    Returns:
        Dictionary containing summary statistics
    """
    all_scores = []
    for customer_id, similar_list in lookalikes.items():
        all_scores.extend([score for _, score in similar_list])
    
    summary = {
        "n_customers_processed": len(lookalikes),
        "n_lookalikes_per_customer": len(next(iter(lookalikes.values()))) if lookalikes else 0,
        "avg_similarity_score": np.mean(all_scores) if all_scores else 0,
        "min_similarity_score": np.min(all_scores) if all_scores else 0,
        "max_similarity_score": np.max(all_scores) if all_scores else 0,
        "median_similarity_score": np.median(all_scores) if all_scores else 0,
    }
    
    return summary


def visualize_similarity_distribution(
    similarity_matrix: np.ndarray,
    save_path: Optional[Path] = None
) -> "plt.Figure":
    """
    Visualize the distribution of similarity scores.
    
    Args:
        similarity_matrix: Customer similarity matrix
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get upper triangle of similarity matrix (excluding diagonal)
    upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(upper_tri, bins=50, color="#3498DB", edgecolor="black", alpha=0.7)
    axes[0].axvline(x=np.mean(upper_tri), color="red", linestyle="--", 
                    label=f"Mean: {np.mean(upper_tri):.3f}")
    axes[0].set_xlabel("Similarity Score", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Distribution of Customer Similarity Scores", fontsize=14, fontweight="bold")
    axes[0].legend()
    
    # Heatmap (sample for visualization)
    sample_size = min(50, similarity_matrix.shape[0])
    sample_matrix = similarity_matrix[:sample_size, :sample_size]
    
    sns.heatmap(sample_matrix, ax=axes[1], cmap="YlOrRd", 
                xticklabels=False, yticklabels=False)
    axes[1].set_title(f"Similarity Heatmap (First {sample_size} Customers)", 
                     fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved similarity visualization to {save_path}")
    
    return fig


def recommend_marketing_targets(
    lookalikes: Dict[str, List[Tuple[str, float]]],
    features: pd.DataFrame,
    high_value_threshold: float = None
) -> pd.DataFrame:
    """
    Recommend marketing targets based on lookalike analysis.
    
    Identifies customers who are similar to high-value customers
    but haven't achieved high-value status yet.
    
    Args:
        lookalikes: Dictionary from find_all_lookalikes()
        features: Customer features DataFrame
        high_value_threshold: Threshold for high-value customers
                              (uses 75th percentile if None)
    
    Returns:
        DataFrame with recommended targets and their similar high-value customers
    """
    if high_value_threshold is None:
        high_value_threshold = features["TotalSpend"].quantile(0.75)
    
    # Identify high-value customers
    high_value_customers = set(
        features[features["TotalSpend"] >= high_value_threshold]["CustomerID"]
    )
    
    # Find customers similar to high-value customers
    recommendations = []
    for customer_id, similar_list in lookalikes.items():
        if customer_id in high_value_customers:
            continue  # Skip already high-value customers
            
        # Check if any lookalikes are high-value
        hv_lookalikes = [
            (lid, score) for lid, score in similar_list
            if lid in high_value_customers
        ]
        
        if hv_lookalikes:
            customer_spend = features[features["CustomerID"] == customer_id]["TotalSpend"].values[0]
            recommendations.append({
                "CustomerID": customer_id,
                "CurrentSpend": customer_spend,
                "PotentialUplift": high_value_threshold - customer_spend,
                "SimilarHVCustomers": len(hv_lookalikes),
                "AvgSimilarityToHV": np.mean([s for _, s in hv_lookalikes]),
            })
    
    recommendations_df = pd.DataFrame(recommendations)
    
    if len(recommendations_df) > 0:
        recommendations_df = recommendations_df.sort_values(
            "AvgSimilarityToHV", ascending=False
        )
    
    logger.info(f"Identified {len(recommendations_df)} potential marketing targets")
    return recommendations_df


def main() -> Dict:
    """
    Run the complete lookalike model pipeline.
    
    Returns:
        Dictionary containing lookalike results and summary
    """
    logger.info("Starting lookalike model pipeline...")
    
    # Load data
    customers, products, transactions = data_loader.load_all_data()
    
    # Prepare features
    features = preprocessing.prepare_customer_features(customers, transactions)
    features = preprocessing.encode_categorical(features, columns=["Region"])
    
    # Get feature columns
    feature_cols = preprocessing.get_feature_columns(features)
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features[feature_cols].fillna(0))
    
    # Build similarity matrix
    similarity_matrix = build_similarity_matrix(scaled_features)
    
    # Find lookalikes for first 20 customers (as in original)
    customer_subset = customers["CustomerID"][:20].tolist()
    lookalikes = find_all_lookalikes(
        features, similarity_matrix,
        n_lookalikes=config.LOOKALIKE_CONFIG["n_lookalikes"],
        customer_subset=customer_subset
    )
    
    # Export results
    results_df = export_lookalikes(lookalikes)
    
    # Generate summary
    summary = get_lookalike_summary(lookalikes)
    
    # Create visualization
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    visualize_similarity_distribution(
        similarity_matrix,
        save_path=config.FIGURES_DIR / "similarity_distribution.png"
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("LOOKALIKE MODEL RESULTS")
    print("=" * 60)
    print(f"\nCustomers processed: {summary['n_customers_processed']}")
    print(f"Lookalikes per customer: {summary['n_lookalikes_per_customer']}")
    print(f"Average similarity score: {summary['avg_similarity_score']:.4f}")
    print(f"Score range: {summary['min_similarity_score']:.4f} - {summary['max_similarity_score']:.4f}")
    print(f"\nResults saved to: {config.OUTPUT_FILES['lookalike']}")
    print("=" * 60)
    
    return {
        "lookalikes": lookalikes,
        "summary": summary,
        "results_df": results_df,
        "similarity_matrix": similarity_matrix,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
