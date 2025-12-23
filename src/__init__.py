"""
eCommerce Transactions Analysis Package

This package provides tools for analyzing eCommerce transaction data,
including customer segmentation, lookalike modeling, and business insights.

Modules:
    - config: Configuration settings and paths
    - data_loader: Data loading and validation utilities
    - preprocessing: Data preprocessing and feature engineering
    - eda: Exploratory data analysis functions
    - clustering: Customer clustering algorithms
    - lookalike: Lookalike customer modeling
"""

__version__ = "1.0.0"
__author__ = "Vishwas Mehta"

from . import config
from . import data_loader
from . import preprocessing
from . import clustering
from . import lookalike

__all__ = [
    "config",
    "data_loader",
    "preprocessing",
    "clustering",
    "lookalike",
]
