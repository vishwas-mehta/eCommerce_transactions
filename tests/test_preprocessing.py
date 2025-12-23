"""
Unit tests for the preprocessing module.
"""

import numpy as np
import pandas as pd
import pytest

from src import preprocessing


class TestCleanCustomerData:
    """Tests for clean_customer_data function."""
    
    def test_removes_duplicates(self):
        """Test that duplicate CustomerIDs are removed."""
        data = pd.DataFrame({
            "CustomerID": ["C001", "C001", "C002"],
            "CustomerName": ["John", "John Duplicate", "Jane"],
            "Region": ["Asia", "Asia", "Europe"],
            "SignupDate": ["2024-01-01", "2024-01-02", "2024-01-01"],
        })
        
        result = preprocessing.clean_customer_data(data)
        
        assert len(result) == 2
        assert result["CustomerID"].nunique() == 2
    
    def test_standardizes_region_names(self):
        """Test that region names are standardized."""
        data = pd.DataFrame({
            "CustomerID": ["C001"],
            "CustomerName": ["John"],
            "Region": ["  asia  "],
            "SignupDate": ["2024-01-01"],
        })
        
        result = preprocessing.clean_customer_data(data)
        
        assert result["Region"].iloc[0] == "Asia"
    
    def test_converts_signup_date_to_datetime(self):
        """Test that SignupDate is converted to datetime."""
        data = pd.DataFrame({
            "CustomerID": ["C001"],
            "CustomerName": ["John"],
            "Region": ["Asia"],
            "SignupDate": ["2024-01-15"],
        })
        
        result = preprocessing.clean_customer_data(data)
        
        assert pd.api.types.is_datetime64_any_dtype(result["SignupDate"])


class TestCleanTransactionData:
    """Tests for clean_transaction_data function."""
    
    def test_removes_duplicate_transactions(self):
        """Test that duplicate TransactionIDs are removed."""
        data = pd.DataFrame({
            "TransactionID": ["T001", "T001", "T002"],
            "CustomerID": ["C001", "C001", "C002"],
            "ProductID": ["P001", "P001", "P002"],
            "TransactionDate": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "Quantity": [1, 2, 1],
            "TotalValue": [100.0, 200.0, 150.0],
            "Price": [100.0, 100.0, 150.0],
        })
        
        result = preprocessing.clean_transaction_data(data)
        
        assert len(result) == 2
    
    def test_removes_negative_quantities(self):
        """Test that negative quantities are removed."""
        data = pd.DataFrame({
            "TransactionID": ["T001", "T002"],
            "CustomerID": ["C001", "C002"],
            "ProductID": ["P001", "P002"],
            "TransactionDate": ["2024-01-01", "2024-01-02"],
            "Quantity": [-1, 2],
            "TotalValue": [100.0, 200.0],
            "Price": [100.0, 100.0],
        })
        
        result = preprocessing.clean_transaction_data(data)
        
        assert len(result) == 1
        assert result["Quantity"].iloc[0] == 2


class TestPrepareCustomerFeatures:
    """Tests for prepare_customer_features function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample customer and transaction data."""
        customers = pd.DataFrame({
            "CustomerID": ["C001", "C002", "C003"],
            "CustomerName": ["John", "Jane", "Bob"],
            "Region": ["Asia", "Europe", "Asia"],
            "SignupDate": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
        })
        
        transactions = pd.DataFrame({
            "TransactionID": ["T001", "T002", "T003", "T004"],
            "CustomerID": ["C001", "C001", "C002", "C001"],
            "ProductID": ["P001", "P002", "P001", "P003"],
            "TransactionDate": pd.to_datetime([
                "2024-03-01", "2024-03-15", "2024-04-01", "2024-04-15"
            ]),
            "Quantity": [1, 2, 1, 3],
            "TotalValue": [100.0, 200.0, 150.0, 300.0],
            "Price": [100.0, 100.0, 150.0, 100.0],
        })
        
        return customers, transactions
    
    def test_creates_total_spend_column(self, sample_data):
        """Test that TotalSpend column is created correctly."""
        customers, transactions = sample_data
        
        result = preprocessing.prepare_customer_features(
            customers, transactions, include_rfm=False
        )
        
        # C001 has 3 transactions: 100 + 200 + 300 = 600
        c001_spend = result[result["CustomerID"] == "C001"]["TotalSpend"].iloc[0]
        assert c001_spend == 600.0
    
    def test_creates_transaction_count_column(self, sample_data):
        """Test that TransactionCount column is created correctly."""
        customers, transactions = sample_data
        
        result = preprocessing.prepare_customer_features(
            customers, transactions, include_rfm=False
        )
        
        # C001 has 3 transactions
        c001_count = result[result["CustomerID"] == "C001"]["TransactionCount"].iloc[0]
        assert c001_count == 3
    
    def test_fills_zero_for_customers_without_transactions(self, sample_data):
        """Test that customers without transactions get zeros."""
        customers, transactions = sample_data
        
        result = preprocessing.prepare_customer_features(
            customers, transactions, include_rfm=False
        )
        
        # C003 has no transactions
        c003_spend = result[result["CustomerID"] == "C003"]["TotalSpend"].iloc[0]
        assert c003_spend == 0.0


class TestEncodeCategorical:
    """Tests for encode_categorical function."""
    
    def test_onehot_encoding(self):
        """Test one-hot encoding of categorical columns."""
        data = pd.DataFrame({
            "CustomerID": ["C001", "C002", "C003"],
            "Region": ["Asia", "Europe", "Asia"],
        })
        
        result = preprocessing.encode_categorical(data, columns=["Region"])
        
        assert "Region_Asia" in result.columns
        assert "Region_Europe" in result.columns
        assert "Region" not in result.columns
    
    def test_invalid_method_raises_error(self):
        """Test that invalid encoding method raises ValueError."""
        data = pd.DataFrame({
            "Region": ["Asia"],
        })
        
        with pytest.raises(ValueError):
            preprocessing.encode_categorical(data, columns=["Region"], method="invalid")


class TestScaleFeatures:
    """Tests for scale_features function."""
    
    def test_standard_scaling(self):
        """Test standard scaling of features."""
        data = pd.DataFrame({
            "Feature1": [10, 20, 30, 40, 50],
            "Feature2": [100, 200, 300, 400, 500],
        })
        
        result, scaler = preprocessing.scale_features(
            data, columns=["Feature1", "Feature2"], 
            method="standard", return_scaler=True
        )
        
        # Mean should be approximately 0 for scaled columns
        assert abs(result["Feature1"].mean()) < 1e-10
        assert abs(result["Feature2"].mean()) < 1e-10
        
        # Scaler should be returned
        assert scaler is not None
    
    def test_minmax_scaling(self):
        """Test min-max scaling of features."""
        data = pd.DataFrame({
            "Feature1": [10, 20, 30, 40, 50],
        })
        
        result, _ = preprocessing.scale_features(
            data, columns=["Feature1"], method="minmax"
        )
        
        # Values should be between 0 and 1
        assert result["Feature1"].min() >= 0
        assert result["Feature1"].max() <= 1


class TestGetFeatureColumns:
    """Tests for get_feature_columns function."""
    
    def test_excludes_id_columns(self):
        """Test that ID columns are excluded."""
        data = pd.DataFrame({
            "CustomerID": ["C001"],
            "TotalSpend": [100.0],
            "Frequency": [5],
        })
        
        result = preprocessing.get_feature_columns(data)
        
        assert "CustomerID" not in result
        assert "TotalSpend" in result
        assert "Frequency" in result
    
    def test_excludes_custom_columns(self):
        """Test that custom exclusions work."""
        data = pd.DataFrame({
            "TotalSpend": [100.0],
            "CustomField": [1],
        })
        
        result = preprocessing.get_feature_columns(data, exclude=["CustomField"])
        
        assert "CustomField" not in result
        assert "TotalSpend" in result
