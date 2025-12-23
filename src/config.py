"""
Configuration settings for the eCommerce Transactions Analysis project.

This module contains all configuration parameters, file paths, and hyperparameters
used throughout the analysis pipeline.
"""

from pathlib import Path
from typing import Dict, List, Any

# =============================================================================
# Path Configuration
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# =============================================================================
# Data Files
# =============================================================================

DATA_FILES: Dict[str, Path] = {
    "customers": RAW_DATA_DIR / "Customers.csv",
    "products": RAW_DATA_DIR / "Products.csv",
    "transactions": RAW_DATA_DIR / "Transactions.csv",
}

OUTPUT_FILES: Dict[str, Path] = {
    "lookalike": PROCESSED_DATA_DIR / "Lookalike.csv",
    "cluster_assignments": PROCESSED_DATA_DIR / "customer_clusters.csv",
    "customer_features": PROCESSED_DATA_DIR / "customer_features.csv",
}

# =============================================================================
# Column Definitions
# =============================================================================

CUSTOMER_COLUMNS: List[str] = ["CustomerID", "CustomerName", "Region", "SignupDate"]
PRODUCT_COLUMNS: List[str] = ["ProductID", "ProductName", "Category", "Price"]
TRANSACTION_COLUMNS: List[str] = [
    "TransactionID", "CustomerID", "ProductID", 
    "TransactionDate", "Quantity", "TotalValue", "Price"
]

# =============================================================================
# Analysis Parameters
# =============================================================================

# Clustering configuration
CLUSTERING_CONFIG: Dict[str, Any] = {
    "n_clusters": 4,
    "max_clusters_to_test": 10,
    "random_state": 42,
    "n_init": 10,
    "max_iter": 300,
}

# Lookalike model configuration
LOOKALIKE_CONFIG: Dict[str, Any] = {
    "n_lookalikes": 3,
    "min_similarity_threshold": 0.5,
}

# Feature engineering configuration
FEATURE_CONFIG: Dict[str, Any] = {
    "date_features": True,
    "region_encoding": "onehot",
    "scaling_method": "standard",
}

# =============================================================================
# Visualization Settings
# =============================================================================

PLOT_STYLE: Dict[str, Any] = {
    "figure_size": (12, 8),
    "dpi": 100,
    "style": "whitegrid",
    "palette": "husl",
    "font_scale": 1.2,
}

# Color palettes for different analyses
COLOR_PALETTES: Dict[str, List[str]] = {
    "regions": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
    "categories": ["#FFEAA7", "#DFE6E9", "#74B9FF", "#A29BFE", "#FD79A8"],
    "clusters": ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"],
}

# =============================================================================
# Validation Settings
# =============================================================================

VALIDATION_CONFIG: Dict[str, Any] = {
    "required_customer_fields": ["CustomerID", "Region"],
    "required_transaction_fields": ["TransactionID", "CustomerID", "TotalValue"],
    "min_transactions_per_customer": 1,
    "date_format": "%Y-%m-%d",
}
