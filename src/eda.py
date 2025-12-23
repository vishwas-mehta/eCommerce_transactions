"""
Exploratory Data Analysis utilities for eCommerce transactions.

This module provides functions for generating visualizations and
statistical summaries of the eCommerce data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

# Configure logging
logger = logging.getLogger(__name__)

# Set plot style
sns.set_style(config.PLOT_STYLE["style"])
plt.rcParams["figure.figsize"] = config.PLOT_STYLE["figure_size"]
plt.rcParams["figure.dpi"] = config.PLOT_STYLE["dpi"]


def plot_revenue_by_region(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a bar chart showing revenue by geographic region.
    
    Args:
        customers: Customers DataFrame
        transactions: Transactions DataFrame
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib Figure object
    
    Example:
        >>> fig = plot_revenue_by_region(customers, transactions)
        >>> plt.show()
    """
    # Merge data to get region for each transaction
    merged = transactions.merge(customers[["CustomerID", "Region"]], on="CustomerID")
    
    # Calculate revenue by region
    revenue_by_region = merged.groupby("Region")["TotalValue"].sum().sort_values(ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = config.COLOR_PALETTES["regions"][:len(revenue_by_region)]
    bars = ax.bar(revenue_by_region.index, revenue_by_region.values, color=colors, edgecolor="black")
    
    # Add value labels on bars
    for bar, value in zip(bars, revenue_by_region.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
            f"${value:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    
    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel("Total Revenue ($)", fontsize=12)
    ax.set_title("Revenue Distribution by Region", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")
    
    return fig


def plot_category_distribution(
    products: pd.DataFrame,
    transactions: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create visualizations showing product category distribution.
    
    Creates a side-by-side plot showing:
    1. Product count by category
    2. Revenue by category
    
    Args:
        products: Products DataFrame
        transactions: Transactions DataFrame
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib Figure object
    """
    # Merge data
    merged = transactions.merge(products[["ProductID", "Category"]], on="ProductID")
    
    # Calculate metrics
    product_counts = products["Category"].value_counts()
    revenue_by_category = merged.groupby("Category")["TotalValue"].sum().sort_values(ascending=False)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = config.COLOR_PALETTES["categories"]
    
    # Product count pie chart
    axes[0].pie(
        product_counts.values, labels=product_counts.index,
        autopct="%1.1f%%", colors=colors, explode=[0.02] * len(product_counts)
    )
    axes[0].set_title("Product Distribution by Category", fontsize=12, fontweight="bold")
    
    # Revenue bar chart
    bars = axes[1].barh(revenue_by_category.index, revenue_by_category.values, color=colors)
    axes[1].set_xlabel("Total Revenue ($)")
    axes[1].set_title("Revenue by Category", fontsize=12, fontweight="bold")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")
    
    return fig


def plot_transaction_trends(
    transactions: pd.DataFrame,
    freq: str = "M",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create time series plot showing transaction trends.
    
    Args:
        transactions: Transactions DataFrame
        freq: Frequency for aggregation ('D'=daily, 'W'=weekly, 'M'=monthly)
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib Figure object
    """
    # Aggregate by time period
    transactions_ts = transactions.copy()
    transactions_ts.set_index("TransactionDate", inplace=True)
    
    monthly_stats = transactions_ts.resample(freq).agg({
        "TotalValue": "sum",
        "TransactionID": "count",
        "Quantity": "sum"
    }).rename(columns={"TransactionID": "TransactionCount"})
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Revenue trend
    axes[0].plot(
        monthly_stats.index, monthly_stats["TotalValue"],
        marker="o", linewidth=2, markersize=6, color="#3498DB"
    )
    axes[0].fill_between(
        monthly_stats.index, monthly_stats["TotalValue"],
        alpha=0.3, color="#3498DB"
    )
    axes[0].set_ylabel("Revenue ($)")
    axes[0].set_title("Monthly Revenue Trend", fontsize=12, fontweight="bold")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    axes[0].grid(True, alpha=0.3)
    
    # Transaction count trend
    axes[1].plot(
        monthly_stats.index, monthly_stats["TransactionCount"],
        marker="s", linewidth=2, markersize=6, color="#E74C3C"
    )
    axes[1].fill_between(
        monthly_stats.index, monthly_stats["TransactionCount"],
        alpha=0.3, color="#E74C3C"
    )
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Number of Transactions")
    axes[1].set_title("Monthly Transaction Volume", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")
    
    return fig


def plot_customer_distribution(
    customers: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create visualizations showing customer distribution by region and signup trends.
    
    Args:
        customers: Customers DataFrame
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Region distribution
    region_counts = customers["Region"].value_counts()
    colors = config.COLOR_PALETTES["regions"]
    
    axes[0].pie(
        region_counts.values, labels=region_counts.index,
        autopct="%1.1f%%", colors=colors, startangle=90
    )
    axes[0].set_title("Customer Distribution by Region", fontsize=12, fontweight="bold")
    
    # Signup trend
    signup_monthly = customers.set_index("SignupDate").resample("M").size()
    axes[1].bar(signup_monthly.index, signup_monthly.values, color="#3498DB", width=20)
    axes[1].set_xlabel("Signup Date")
    axes[1].set_ylabel("Number of Signups")
    axes[1].set_title("Monthly Customer Signups", fontsize=12, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")
    
    return fig


def generate_summary_stats(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    transactions: pd.DataFrame
) -> Dict[str, any]:
    """
    Generate comprehensive summary statistics for all datasets.
    
    Args:
        customers: Customers DataFrame
        products: Products DataFrame
        transactions: Transactions DataFrame
    
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        "customers": {
            "total_count": len(customers),
            "regions": customers["Region"].value_counts().to_dict(),
            "signup_date_range": (
                customers["SignupDate"].min().strftime("%Y-%m-%d"),
                customers["SignupDate"].max().strftime("%Y-%m-%d")
            ),
        },
        "products": {
            "total_count": len(products),
            "categories": products["Category"].value_counts().to_dict(),
            "price_stats": {
                "min": products["Price"].min(),
                "max": products["Price"].max(),
                "mean": products["Price"].mean(),
                "median": products["Price"].median(),
            },
        },
        "transactions": {
            "total_count": len(transactions),
            "total_revenue": transactions["TotalValue"].sum(),
            "avg_transaction_value": transactions["TotalValue"].mean(),
            "total_quantity_sold": transactions["Quantity"].sum(),
            "unique_customers": transactions["CustomerID"].nunique(),
            "unique_products": transactions["ProductID"].nunique(),
            "date_range": (
                transactions["TransactionDate"].min().strftime("%Y-%m-%d %H:%M"),
                transactions["TransactionDate"].max().strftime("%Y-%m-%d %H:%M")
            ),
            "quantity_stats": {
                "min": int(transactions["Quantity"].min()),
                "max": int(transactions["Quantity"].max()),
                "mean": transactions["Quantity"].mean(),
            },
        },
    }
    
    return summary


def print_summary_report(summary: Dict) -> None:
    """
    Print a formatted summary report to console.
    
    Args:
        summary: Dictionary from generate_summary_stats()
    """
    print("=" * 60)
    print("eCommerce Transactions - Data Summary Report")
    print("=" * 60)
    
    print("\n📊 CUSTOMERS")
    print(f"   Total Customers: {summary['customers']['total_count']:,}")
    print(f"   Regions: {', '.join(summary['customers']['regions'].keys())}")
    print(f"   Signup Period: {summary['customers']['signup_date_range'][0]} to {summary['customers']['signup_date_range'][1]}")
    
    print("\n📦 PRODUCTS")
    print(f"   Total Products: {summary['products']['total_count']:,}")
    print(f"   Categories: {', '.join(summary['products']['categories'].keys())}")
    print(f"   Price Range: ${summary['products']['price_stats']['min']:.2f} - ${summary['products']['price_stats']['max']:.2f}")
    print(f"   Average Price: ${summary['products']['price_stats']['mean']:.2f}")
    
    print("\n💰 TRANSACTIONS")
    print(f"   Total Transactions: {summary['transactions']['total_count']:,}")
    print(f"   Total Revenue: ${summary['transactions']['total_revenue']:,.2f}")
    print(f"   Average Transaction: ${summary['transactions']['avg_transaction_value']:.2f}")
    print(f"   Active Customers: {summary['transactions']['unique_customers']:,}")
    print(f"   Products Sold: {summary['transactions']['unique_products']:,}")
    
    print("\n" + "=" * 60)


def run_full_eda(
    customers: pd.DataFrame = None,
    products: pd.DataFrame = None,
    transactions: pd.DataFrame = None,
    save_figures: bool = True
) -> Dict:
    """
    Run complete EDA pipeline and generate all visualizations.
    
    Args:
        customers: Optional customers DataFrame (loads from file if not provided)
        products: Optional products DataFrame
        transactions: Optional transactions DataFrame
        save_figures: Whether to save figures to disk
    
    Returns:
        Dictionary containing summary statistics
    """
    from . import data_loader
    
    # Load data if not provided
    if customers is None or products is None or transactions is None:
        customers, products, transactions = data_loader.load_all_data()
    
    # Create figures directory
    if save_figures:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate summary statistics
    summary = generate_summary_stats(customers, products, transactions)
    print_summary_report(summary)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    plot_revenue_by_region(
        customers, transactions,
        save_path=config.FIGURES_DIR / "revenue_by_region.png" if save_figures else None
    )
    
    plot_category_distribution(
        products, transactions,
        save_path=config.FIGURES_DIR / "category_distribution.png" if save_figures else None
    )
    
    plot_transaction_trends(
        transactions,
        save_path=config.FIGURES_DIR / "transaction_trends.png" if save_figures else None
    )
    
    plot_customer_distribution(
        customers,
        save_path=config.FIGURES_DIR / "customer_distribution.png" if save_figures else None
    )
    
    logger.info("EDA complete!")
    return summary
