# eCommerce Transactions Analysis

A comprehensive data analysis project that explores eCommerce transaction data to derive customer insights, perform customer segmentation using clustering, and build a lookalike model for customer recommendations.

## Project Overview

This project analyzes eCommerce transaction data to:
- Perform **Exploratory Data Analysis (EDA)** to understand customer behavior and transaction patterns
- Apply **Customer Clustering** to segment customers based on their purchasing behavior
- Build a **Lookalike Model** to identify similar customers for targeted marketing

## Dataset Description

| File | Description |
|------|-------------|
| `Customers.csv` | Customer information including CustomerID, CustomerName, Region, and SignupDate |
| `Products.csv` | Product catalog with product details |
| `Transactions.csv` | Transaction records linking customers and products |
| `Lookalike.csv` | Output file containing lookalike customer recommendations |

## Project Structure

```
eCommerce_transactions/
├── Customers.csv                    # Customer data
├── Products.csv                     # Product catalog
├── Transactions.csv                 # Transaction records
├── Lookalike.csv                    # Lookalike model output
├── Vishwas_Mehta_EDA.ipynb         # Exploratory Data Analysis notebook
├── Vishwas_Mehta_EDA.pdf           # EDA report (PDF)
├── Vishwas_Mehta_Clustering.ipynb  # Customer clustering analysis
├── Vishwas_Mehta_Clustering.pdf    # Clustering report (PDF)
├── Vishwas_Mehta_Lookalike.ipynb   # Lookalike model implementation
└── README.md
```

## Analysis Components

### 1. Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Statistical analysis of transactions
- Customer behavior analysis by region
- Transaction trends and patterns
- Revenue analysis and insights

### 2. Customer Clustering
- Feature engineering from transaction data
- K-Means clustering for customer segmentation
- Cluster profiling and interpretation
- Visualization of customer segments

### 3. Lookalike Model
- Customer similarity computation
- Recommendation of similar customers
- Top 3 lookalike customers for each customer with similarity scores

## Tech Stack

- **Python 3.x**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning (clustering, similarity metrics)
- **Jupyter Notebook** - Interactive analysis

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/vishwas-mehta/eCommerce_transactions.git
   cd eCommerce_transactions
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open and run the notebooks in order:
   - `Vishwas_Mehta_EDA.ipynb`
   - `Vishwas_Mehta_Clustering.ipynb`
   - `Vishwas_Mehta_Lookalike.ipynb`

## Key Insights

- Customer segmentation reveals distinct purchasing patterns across different regions
- The lookalike model identifies similar customers based on transaction behavior
- Clustering analysis helps in targeted marketing strategies

## Author

**Vishwas Mehta**

## License

This project is for educational purposes.
