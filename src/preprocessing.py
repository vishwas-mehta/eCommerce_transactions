"""
Data preprocessing utilities for eCommerce transactions analysis.

This module provides functions for cleaning, transforming, and preparing
data for machine learning models.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Configure logging
logger = logging.getLogger(__name__)


def clean_customer_data(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate customer data.
    
    Performs the following cleaning operations:
    - Remove duplicates based on CustomerID
    - Handle missing values
    - Standardize region names
    - Validate date formats
    
    Args:
        customers: Raw customers DataFrame
    
    Returns:
        Cleaned customers DataFrame
    
    Example:
        >>> cleaned = clean_customer_data(raw_customers)
        >>> print(cleaned.isnull().sum())
    """
    df = customers.copy()
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=["CustomerID"], keep="first")
    if len(df) < initial_count:
        logger.warning(f"Removed {initial_count - len(df)} duplicate customers")
    
    # Standardize region names
    df["Region"] = df["Region"].str.strip().str.title()
    
    # Ensure SignupDate is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["SignupDate"]):
        df["SignupDate"] = pd.to_datetime(df["SignupDate"])
    
    logger.info(f"Cleaned customer data: {len(df)} records")
    return df


def clean_transaction_data(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate transaction data.
    
    Performs the following cleaning operations:
    - Remove duplicates based on TransactionID
    - Handle negative values
    - Validate date formats
    - Remove outliers (optional)
    
    Args:
        transactions: Raw transactions DataFrame
    
    Returns:
        Cleaned transactions DataFrame
    """
    df = transactions.copy()
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=["TransactionID"], keep="first")
    if len(df) < initial_count:
        logger.warning(f"Removed {initial_count - len(df)} duplicate transactions")
    
    # Ensure positive values
    df = df[df["Quantity"] > 0]
    df = df[df["TotalValue"] > 0]
    
    # Ensure TransactionDate is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["TransactionDate"]):
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    
    logger.info(f"Cleaned transaction data: {len(df)} records")
    return df


def prepare_customer_features(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    include_rfm: bool = True
) -> pd.DataFrame:
    """
    Prepare customer feature matrix for clustering and similarity analysis.
    
    Creates the following features:
    - TotalSpend: Total amount spent by customer
    - TransactionCount: Number of transactions
    - AvgTransactionValue: Average transaction value
    - Region (one-hot encoded): Geographic region
    - DaysSinceSignup: Customer tenure
    - RFM metrics (optional): Recency, Frequency, Monetary
    
    Args:
        customers: Cleaned customers DataFrame
        transactions: Cleaned transactions DataFrame
        include_rfm: Whether to include RFM metrics
    
    Returns:
        DataFrame with customer features
    
    Example:
        >>> features = prepare_customer_features(customers, transactions)
        >>> print(features.columns.tolist())
    """
    # Aggregate transaction data by customer
    customer_agg = transactions.groupby("CustomerID").agg({
        "TotalValue": "sum",
        "TransactionID": "count",
        "Quantity": "sum",
        "TransactionDate": ["min", "max"]
    })
    
    # Flatten column names
    customer_agg.columns = [
        "TotalSpend", "TransactionCount", "TotalQuantity",
        "FirstPurchase", "LastPurchase"
    ]
    customer_agg = customer_agg.reset_index()
    
    # Calculate average transaction value
    customer_agg["AvgTransactionValue"] = (
        customer_agg["TotalSpend"] / customer_agg["TransactionCount"]
    )
    
    # Merge with customer data
    features = customers.merge(customer_agg, on="CustomerID", how="left")
    
    # Fill missing values for customers without transactions
    numeric_cols = ["TotalSpend", "TransactionCount", "TotalQuantity", "AvgTransactionValue"]
    features[numeric_cols] = features[numeric_cols].fillna(0)
    
    # Calculate days since signup
    reference_date = pd.Timestamp.now()
    features["DaysSinceSignup"] = (reference_date - features["SignupDate"]).dt.days
    
    # Add RFM metrics if requested
    if include_rfm:
        features = _add_rfm_features(features, reference_date)
    
    logger.info(f"Prepared {len(features)} customer feature records with {len(features.columns)} features")
    return features


def _add_rfm_features(features: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """
    Add RFM (Recency, Frequency, Monetary) features.
    
    Args:
        features: Customer features DataFrame
        reference_date: Reference date for recency calculation
    
    Returns:
        DataFrame with RFM features added
    """
    df = features.copy()
    
    # Recency: Days since last purchase
    df["Recency"] = np.where(
        df["LastPurchase"].notna(),
        (reference_date - df["LastPurchase"]).dt.days,
        df["DaysSinceSignup"]  # Use signup date if no purchases
    )
    
    # Frequency: Number of transactions (already have this)
    df["Frequency"] = df["TransactionCount"]
    
    # Monetary: Total spend (already have this)
    df["Monetary"] = df["TotalSpend"]
    
    return df


def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "onehot"
) -> pd.DataFrame:
    """
    Encode categorical variables.
    
    Args:
        df: DataFrame with categorical columns
        columns: List of column names to encode
        method: Encoding method ('onehot' or 'label')
    
    Returns:
        DataFrame with encoded categorical variables
    
    Raises:
        ValueError: If invalid encoding method specified
    """
    result = df.copy()
    
    if method == "onehot":
        result = pd.get_dummies(result, columns=columns, prefix=columns)
    elif method == "label":
        for col in columns:
            result[col] = result[col].astype("category").cat.codes
    else:
        raise ValueError(f"Invalid encoding method: {method}. Use 'onehot' or 'label'")
    
    logger.info(f"Encoded {len(columns)} categorical columns using {method} encoding")
    return result


def scale_features(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "standard",
    return_scaler: bool = False
) -> Tuple[pd.DataFrame, Optional[object]]:
    """
    Scale numerical features.
    
    Args:
        df: DataFrame with numerical columns
        columns: List of column names to scale
        method: Scaling method ('standard' or 'minmax')
        return_scaler: Whether to return the fitted scaler
    
    Returns:
        Tuple of (scaled DataFrame, scaler object if return_scaler=True)
    
    Example:
        >>> scaled_df, scaler = scale_features(df, ['TotalSpend', 'Frequency'], return_scaler=True)
    """
    result = df.copy()
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Invalid scaling method: {method}")
    
    result[columns] = scaler.fit_transform(result[columns])
    
    logger.info(f"Scaled {len(columns)} features using {method} scaling")
    
    if return_scaler:
        return result, scaler
    return result, None


def get_feature_columns(
    df: pd.DataFrame,
    exclude: Optional[List[str]] = None
) -> List[str]:
    """
    Get list of feature columns for modeling.
    
    Excludes identifier columns, date columns, and any specified exclusions.
    
    Args:
        df: DataFrame to extract feature columns from
        exclude: Additional columns to exclude
    
    Returns:
        List of feature column names
    """
    # Default exclusions
    default_exclude = [
        "CustomerID", "CustomerName", "ProductID", "ProductName",
        "TransactionID", "SignupDate", "TransactionDate",
        "FirstPurchase", "LastPurchase"
    ]
    
    if exclude:
        default_exclude.extend(exclude)
    
    feature_cols = [
        col for col in df.columns
        if col not in default_exclude
        and df[col].dtype in ["int64", "float64", "uint8"]
    ]
    
    return feature_cols


def create_modeling_dataset(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    scale: bool = True,
    encode_regions: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Create a complete dataset ready for modeling.
    
    This is a convenience function that combines all preprocessing steps.
    
    Args:
        customers: Raw customers DataFrame
        transactions: Raw transactions DataFrame
        scale: Whether to scale numerical features
        encode_regions: Whether to one-hot encode regions
    
    Returns:
        Tuple of (original features, scaled features, feature column names)
    
    Example:
        >>> original, scaled, features = create_modeling_dataset(customers, transactions)
        >>> print(f"Created {len(features)} features for {len(scaled)} customers")
    """
    # Clean data
    customers_clean = clean_customer_data(customers)
    transactions_clean = clean_transaction_data(transactions)
    
    # Prepare features
    features = prepare_customer_features(customers_clean, transactions_clean)
    
    # Encode categorical variables
    if encode_regions:
        features = encode_categorical(features, columns=["Region"])
    
    # Get feature columns
    feature_cols = get_feature_columns(features)
    
    # Scale features
    if scale:
        scaled_features, _ = scale_features(features.copy(), feature_cols)
    else:
        scaled_features = features.copy()
    
    logger.info(f"Created modeling dataset with {len(feature_cols)} features")
    return features, scaled_features, feature_cols
