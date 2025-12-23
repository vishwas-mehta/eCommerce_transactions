"""
eCommerce Transactions Analysis Package

A comprehensive data analysis toolkit for eCommerce transaction data,
featuring customer segmentation, lookalike modeling, and business insights.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ecommerce-transactions",
    version="1.0.0",
    author="Vishwas Mehta",
    author_email="vishwas.mehta@example.com",
    description="Customer segmentation and lookalike modeling for eCommerce",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vishwas-mehta/eCommerce_transactions",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "flake8>=6.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ecommerce-analyze=src.clustering:main",
        ],
    },
)
