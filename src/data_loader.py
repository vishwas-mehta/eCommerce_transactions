"""
Data loading utilities for eCommerce transactions analysis.

This module provides functions for loading, validating, and preparing
the raw data files for analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from . import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_customers(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load customer data from CSV file.
    
    Args:
        filepath: Optional path to customers CSV file. 
                  Defaults to config.DATA_FILES["customers"].
    
    Returns:
        DataFrame containing customer information with columns:
        - CustomerID: Unique customer identifier
        - CustomerName: Customer's full name
        - Region: Geographic region (Asia, Europe, North America, South America)
        - SignupDate: Date of account creation
    
    Raises:
        FileNotFoundError: If the data file doesn't exist.
        ValueError: If required columns are missing.
    
    Example:
        >>> customers = load_customers()
        >>> print(customers.shape)
        (200, 4)
    """
    if filepath is None:
        filepath = config.DATA_FILES["customers"]
    
    logger.info(f"Loading customers from {filepath}")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Customers file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Validate required columns
    required_cols = set(config.CUSTOMER_COLUMNS)
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Parse dates
    df["SignupDate"] = pd.to_datetime(df["SignupDate"])
    
    logger.info(f"Loaded {len(df)} customers")
    return df


def load_products(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load product catalog from CSV file.
    
    Args:
        filepath: Optional path to products CSV file.
                  Defaults to config.DATA_FILES["products"].
    
    Returns:
        DataFrame containing product information with columns:
        - ProductID: Unique product identifier
        - ProductName: Product name
        - Category: Product category (Books, Electronics, Clothing, Home Decor)
        - Price: Product price in USD
    
    Raises:
        FileNotFoundError: If the data file doesn't exist.
        ValueError: If required columns are missing.
    
    Example:
        >>> products = load_products()
        >>> print(products["Category"].unique())
        ['Books', 'Electronics', 'Clothing', 'Home Decor']
    """
    if filepath is None:
        filepath = config.DATA_FILES["products"]
    
    logger.info(f"Loading products from {filepath}")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Products file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Validate required columns
    required_cols = set(config.PRODUCT_COLUMNS)
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Loaded {len(df)} products across {df['Category'].nunique()} categories")
    return df


def load_transactions(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load transaction records from CSV file.
    
    Args:
        filepath: Optional path to transactions CSV file.
                  Defaults to config.DATA_FILES["transactions"].
    
    Returns:
        DataFrame containing transaction records with columns:
        - TransactionID: Unique transaction identifier
        - CustomerID: Reference to customer
        - ProductID: Reference to product
        - TransactionDate: Date and time of transaction
        - Quantity: Number of items purchased
        - TotalValue: Total transaction value in USD
        - Price: Unit price of product
    
    Raises:
        FileNotFoundError: If the data file doesn't exist.
        ValueError: If required columns are missing.
    
    Example:
        >>> transactions = load_transactions()
        >>> print(f"Total revenue: ${transactions['TotalValue'].sum():,.2f}")
    """
    if filepath is None:
        filepath = config.DATA_FILES["transactions"]
    
    logger.info(f"Loading transactions from {filepath}")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Transactions file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Validate required columns
    required_cols = set(config.TRANSACTION_COLUMNS)
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Parse dates
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    
    logger.info(f"Loaded {len(df)} transactions totaling ${df['TotalValue'].sum():,.2f}")
    return df


def load_all_data(
    data_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all datasets (customers, products, transactions).
    
    Args:
        data_dir: Optional base directory containing data files.
                  Defaults to config.RAW_DATA_DIR.
    
    Returns:
        Tuple of (customers_df, products_df, transactions_df)
    
    Example:
        >>> customers, products, transactions = load_all_data()
        >>> print(f"Customers: {len(customers)}, Products: {len(products)}, Transactions: {len(transactions)}")
    """
    if data_dir is not None:
        customers_path = data_dir / "Customers.csv"
        products_path = data_dir / "Products.csv"
        transactions_path = data_dir / "Transactions.csv"
    else:
        customers_path = None
        products_path = None
        transactions_path = None
    
    customers = load_customers(customers_path)
    products = load_products(products_path)
    transactions = load_transactions(transactions_path)
    
    return customers, products, transactions


def validate_data_integrity(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    transactions: pd.DataFrame
) -> Dict[str, bool]:
    """
    Validate data integrity across all datasets.
    
    Checks:
    - All CustomerIDs in transactions exist in customers
    - All ProductIDs in transactions exist in products
    - No null values in critical columns
    - No duplicate primary keys
    
    Args:
        customers: Customers DataFrame
        products: Products DataFrame
        transactions: Transactions DataFrame
    
    Returns:
        Dictionary with validation results for each check.
    
    Example:
        >>> validation = validate_data_integrity(customers, products, transactions)
        >>> print(validation)
        {'customer_refs_valid': True, 'product_refs_valid': True, ...}
    """
    results = {}
    
    # Check customer references
    customer_ids = set(customers["CustomerID"])
    transaction_customer_ids = set(transactions["CustomerID"])
    results["customer_refs_valid"] = transaction_customer_ids.issubset(customer_ids)
    
    # Check product references
    product_ids = set(products["ProductID"])
    transaction_product_ids = set(transactions["ProductID"])
    results["product_refs_valid"] = transaction_product_ids.issubset(product_ids)
    
    # Check for duplicates
    results["no_duplicate_customers"] = not customers["CustomerID"].duplicated().any()
    results["no_duplicate_products"] = not products["ProductID"].duplicated().any()
    results["no_duplicate_transactions"] = not transactions["TransactionID"].duplicated().any()
    
    # Check for nulls in critical columns
    results["no_null_customer_id"] = not customers["CustomerID"].isnull().any()
    results["no_null_product_id"] = not products["ProductID"].isnull().any()
    results["no_null_transaction_id"] = not transactions["TransactionID"].isnull().any()
    
    # Log validation results
    all_valid = all(results.values())
    if all_valid:
        logger.info("✓ All data integrity checks passed")
    else:
        failed_checks = [k for k, v in results.items() if not v]
        logger.warning(f"✗ Failed checks: {failed_checks}")
    
    return results


def get_data_summary(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    transactions: pd.DataFrame
) -> Dict[str, any]:
    """
    Generate a summary of the loaded datasets.
    
    Args:
        customers: Customers DataFrame
        products: Products DataFrame
        transactions: Transactions DataFrame
    
    Returns:
        Dictionary containing summary statistics.
    """
    return {
        "n_customers": len(customers),
        "n_products": len(products),
        "n_transactions": len(transactions),
        "n_regions": customers["Region"].nunique(),
        "n_categories": products["Category"].nunique(),
        "total_revenue": transactions["TotalValue"].sum(),
        "avg_transaction_value": transactions["TotalValue"].mean(),
        "date_range": (
            transactions["TransactionDate"].min(),
            transactions["TransactionDate"].max()
        ),
        "regions": customers["Region"].unique().tolist(),
        "categories": products["Category"].unique().tolist(),
    }
