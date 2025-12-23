"""
Unit tests for the clustering module.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from src import clustering


class TestPrepareClusteringFeatures:
    """Tests for prepare_clustering_features function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        customers = pd.DataFrame({
            "CustomerID": ["C001", "C002", "C003", "C004", "C005"],
            "CustomerName": ["John", "Jane", "Bob", "Alice", "Charlie"],
            "Region": ["Asia", "Europe", "Asia", "North America", "Europe"],
            "SignupDate": pd.to_datetime([
                "2024-01-01", "2024-01-15", "2024-02-01", 
                "2024-02-15", "2024-03-01"
            ]),
        })
        
        transactions = pd.DataFrame({
            "TransactionID": [f"T{i:03d}" for i in range(1, 11)],
            "CustomerID": ["C001", "C001", "C002", "C002", "C002",
                          "C003", "C004", "C004", "C005", "C005"],
            "ProductID": ["P001"] * 10,
            "TransactionDate": pd.to_datetime(["2024-03-01"] * 10),
            "Quantity": [1, 2, 1, 1, 3, 2, 1, 2, 1, 1],
            "TotalValue": [100, 200, 150, 100, 300, 200, 100, 200, 150, 100],
            "Price": [100, 100, 150, 100, 100, 100, 100, 100, 150, 100],
        })
        
        return customers, transactions
    
    def test_returns_three_elements(self, sample_data):
        """Test that function returns (features, scaled, columns)."""
        customers, transactions = sample_data
        
        result = clustering.prepare_clustering_features(customers, transactions)
        
        assert len(result) == 3
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], list)
    
    def test_scaled_features_have_correct_shape(self, sample_data):
        """Test that scaled features have correct dimensions."""
        customers, transactions = sample_data
        
        features, scaled, columns = clustering.prepare_clustering_features(
            customers, transactions
        )
        
        assert scaled.shape[0] == len(customers)  # n_customers rows
        assert scaled.shape[1] == len(columns)    # n_features columns


class TestPerformClustering:
    """Tests for perform_clustering function."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample scaled features for clustering."""
        np.random.seed(42)
        # Create 3 distinct clusters
        cluster1 = np.random.randn(10, 5) + np.array([0, 0, 0, 0, 0])
        cluster2 = np.random.randn(10, 5) + np.array([5, 5, 5, 5, 5])
        cluster3 = np.random.randn(10, 5) + np.array([10, 10, 10, 10, 10])
        return np.vstack([cluster1, cluster2, cluster3])
    
    def test_returns_correct_number_of_labels(self, sample_features):
        """Test that correct number of labels is returned."""
        labels, model = clustering.perform_clustering(
            sample_features, n_clusters=3, random_state=42
        )
        
        assert len(labels) == len(sample_features)
    
    def test_returns_correct_number_of_clusters(self, sample_features):
        """Test that correct number of clusters is created."""
        n_clusters = 3
        labels, model = clustering.perform_clustering(
            sample_features, n_clusters=n_clusters, random_state=42
        )
        
        assert len(np.unique(labels)) == n_clusters
    
    def test_returns_kmeans_model(self, sample_features):
        """Test that KMeans model is returned."""
        labels, model = clustering.perform_clustering(
            sample_features, n_clusters=3, random_state=42
        )
        
        assert isinstance(model, KMeans)
    
    def test_reproducible_with_random_state(self, sample_features):
        """Test that results are reproducible with same random state."""
        labels1, _ = clustering.perform_clustering(
            sample_features, n_clusters=3, random_state=42
        )
        labels2, _ = clustering.perform_clustering(
            sample_features, n_clusters=3, random_state=42
        )
        
        np.testing.assert_array_equal(labels1, labels2)


class TestEvaluateClustering:
    """Tests for evaluate_clustering function."""
    
    @pytest.fixture
    def clustering_result(self):
        """Create sample clustering result."""
        np.random.seed(42)
        # Create distinct clusters for good evaluation metrics
        cluster1 = np.random.randn(20, 5) + np.array([0, 0, 0, 0, 0])
        cluster2 = np.random.randn(20, 5) + np.array([10, 10, 10, 10, 10])
        features = np.vstack([cluster1, cluster2])
        labels = np.array([0] * 20 + [1] * 20)
        return features, labels
    
    def test_returns_expected_metrics(self, clustering_result):
        """Test that expected metrics are returned."""
        features, labels = clustering_result
        
        metrics = clustering.evaluate_clustering(features, labels)
        
        assert "davies_bouldin" in metrics
        assert "silhouette_score" in metrics
        assert "n_clusters" in metrics
        assert "cluster_sizes" in metrics
    
    def test_davies_bouldin_is_positive(self, clustering_result):
        """Test that Davies-Bouldin index is positive."""
        features, labels = clustering_result
        
        metrics = clustering.evaluate_clustering(features, labels)
        
        assert metrics["davies_bouldin"] > 0
    
    def test_silhouette_score_in_valid_range(self, clustering_result):
        """Test that silhouette score is between -1 and 1."""
        features, labels = clustering_result
        
        metrics = clustering.evaluate_clustering(features, labels)
        
        assert -1 <= metrics["silhouette_score"] <= 1
    
    def test_good_clustering_has_high_silhouette(self, clustering_result):
        """Test that well-separated clusters have high silhouette score."""
        features, labels = clustering_result
        
        metrics = clustering.evaluate_clustering(features, labels)
        
        # Well-separated clusters should have silhouette > 0.5
        assert metrics["silhouette_score"] > 0.5


class TestGetClusterProfiles:
    """Tests for get_cluster_profiles function."""
    
    def test_returns_profile_per_cluster(self):
        """Test that profile is returned for each cluster."""
        features = pd.DataFrame({
            "CustomerID": ["C001", "C002", "C003", "C004"],
            "TotalSpend": [100, 200, 150, 250],
            "Frequency": [1, 5, 2, 7],
        })
        labels = np.array([0, 1, 0, 1])
        feature_cols = ["TotalSpend", "Frequency"]
        
        profiles = clustering.get_cluster_profiles(features, labels, feature_cols)
        
        assert len(profiles) == 2  # Two clusters
        assert "Size" in profiles.columns
        assert "Percentage" in profiles.columns
    
    def test_correct_cluster_means(self):
        """Test that cluster means are calculated correctly."""
        features = pd.DataFrame({
            "CustomerID": ["C001", "C002", "C003", "C004"],
            "TotalSpend": [100, 200, 100, 200],
        })
        labels = np.array([0, 1, 0, 1])
        
        profiles = clustering.get_cluster_profiles(features, labels, ["TotalSpend"])
        
        assert profiles.loc[0, "TotalSpend"] == 100  # Mean of cluster 0
        assert profiles.loc[1, "TotalSpend"] == 200  # Mean of cluster 1


class TestFindOptimalClusters:
    """Tests for find_optimal_clusters function."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features with known optimal clusters."""
        np.random.seed(42)
        cluster1 = np.random.randn(30, 5)
        cluster2 = np.random.randn(30, 5) + 10
        cluster3 = np.random.randn(30, 5) + 20
        return np.vstack([cluster1, cluster2, cluster3])
    
    def test_returns_optimal_k_and_figure(self, sample_features):
        """Test that function returns k and figure."""
        optimal_k, fig = clustering.find_optimal_clusters(
            sample_features, max_clusters=6, plot=True
        )
        
        assert isinstance(optimal_k, int)
        assert optimal_k >= 2
        assert fig is not None
    
    def test_optimal_k_within_range(self, sample_features):
        """Test that optimal k is within tested range."""
        max_clusters = 6
        optimal_k, _ = clustering.find_optimal_clusters(
            sample_features, max_clusters=max_clusters, plot=False
        )
        
        assert 2 <= optimal_k <= max_clusters
