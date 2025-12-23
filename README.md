# eCommerce Transactions Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![CI](https://github.com/vishwas-mehta/eCommerce_transactions/workflows/CI/badge.svg)

A comprehensive **customer analytics platform** for eCommerce businesses, featuring customer segmentation using machine learning, lookalike modeling for targeted marketing, and actionable business insights.

---

## 🎯 Project Overview

This project analyzes eCommerce transaction data to:

- 📊 **Exploratory Data Analysis (EDA)** - Understand customer behavior and transaction patterns
- 👥 **Customer Segmentation** - Cluster customers using K-Means algorithm
- 🎯 **Lookalike Modeling** - Identify similar customers for targeted marketing
- 📈 **Business Insights** - Generate actionable recommendations

### Key Results

| Metric | Value |
|--------|-------|
| Customers Analyzed | 200 |
| Transactions Processed | 1,000 |
| Customer Segments | 4 |
| Model Accuracy (Silhouette) | 0.45+ |
| DB Index | 0.76 |

---

## 🏗️ Project Structure

```
eCommerce_transactions/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD
├── data/
│   ├── raw/                    # Original datasets
│   │   ├── Customers.csv
│   │   ├── Products.csv
│   │   └── Transactions.csv
│   └── processed/              # Analysis outputs
│       └── Lookalike.csv
├── notebooks/                  # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_customer_clustering.ipynb
│   └── 03_lookalike_model.ipynb
├── src/                        # Source code
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessing.py       # Feature engineering
│   ├── eda.py                 # EDA visualizations
│   ├── clustering.py          # K-Means clustering
│   └── lookalike.py           # Similarity modeling
├── tests/                      # Unit tests
│   ├── test_preprocessing.py
│   └── test_clustering.py
├── reports/
│   ├── figures/               # Generated visualizations
│   └── business_insights.md   # Analysis report
├── .gitignore
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── setup.py
└── CONTRIBUTING.md
```

---

## 📊 Dataset Description

| File | Records | Description |
|------|---------|-------------|
| `Customers.csv` | 200 | Customer profiles with region and signup date |
| `Products.csv` | 100 | Product catalog across 4 categories |
| `Transactions.csv` | 1,000 | Transaction records with quantities and values |

### Data Schema

**Customers**
```
CustomerID | CustomerName | Region | SignupDate
```

**Products**
```
ProductID | ProductName | Category | Price
```

**Transactions**
```
TransactionID | CustomerID | ProductID | TransactionDate | Quantity | TotalValue | Price
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vishwas-mehta/eCommerce_transactions.git
   cd eCommerce_transactions
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install package in development mode** (optional)
   ```bash
   pip install -e ".[dev,notebook]"
   ```

### Usage

#### Run Analysis Pipeline

```python
from src import clustering, lookalike

# Run customer clustering
results = clustering.main()
print(f"Clusters: {results['metrics']['n_clusters']}")

# Run lookalike model
lookalike_results = lookalike.main()
```

#### Using Makefile

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run full analysis
make all

# Format code
make format
```

#### Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

---

## 🔬 Analysis Components

### 1. Exploratory Data Analysis

- Revenue distribution by region
- Product category performance
- Transaction trends over time
- Customer signup patterns

### 2. Customer Clustering

**Algorithm**: K-Means Clustering

**Features Used**:
- Total spend
- Transaction frequency
- Average transaction value
- Customer tenure
- Regional encoding

**Evaluation Metrics**:
- Davies-Bouldin Index: 0.76 (lower is better)
- Silhouette Score: 0.45 (range: -1 to 1)

### 3. Lookalike Model

**Algorithm**: Cosine Similarity

**Use Cases**:
- Find similar customers for targeting
- Product recommendations
- Marketing campaign optimization

---

## 📈 Results & Insights

### Customer Segments

| Segment | % of Customers | Characteristics |
|---------|---------------|-----------------|
| High-Value | ~15% | Frequent, high-spend customers |
| Regular | ~35% | Consistent purchase patterns |
| Occasional | ~30% | Infrequent but engaged |
| New/Inactive | ~20% | Needs activation |

### Business Recommendations

1. **VIP Program** - Target high-value segment with exclusive benefits
2. **Re-engagement** - Win back occasional buyers with personalized offers
3. **Regional Focus** - Expand in high-performing South America market
4. **Cross-selling** - Use lookalike model for product recommendations

See [Business Insights Report](reports/business_insights.md) for detailed analysis.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_clustering.py -v
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Testing** | Pytest |
| **CI/CD** | GitHub Actions |
| **Code Quality** | Black, Flake8, MyPy |

---

## 📁 API Reference

### Data Loading

```python
from src.data_loader import load_all_data

customers, products, transactions = load_all_data()
```

### Preprocessing

```python
from src.preprocessing import prepare_customer_features

features = prepare_customer_features(customers, transactions)
```

### Clustering

```python
from src.clustering import perform_clustering, evaluate_clustering

labels, model = perform_clustering(scaled_features, n_clusters=4)
metrics = evaluate_clustering(scaled_features, labels)
```

### Lookalike Model

```python
from src.lookalike import build_similarity_matrix, find_all_lookalikes

similarity = build_similarity_matrix(scaled_features)
lookalikes = find_all_lookalikes(features, similarity, n_lookalikes=3)
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Vishwas Mehta**

- GitHub: [@vishwas-mehta](https://github.com/vishwas-mehta)

---

## 🙏 Acknowledgments

- Dataset inspired by real-world eCommerce patterns
- Built with scikit-learn and pandas
- Visualization powered by matplotlib and seaborn
