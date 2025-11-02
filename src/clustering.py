"""
Clustering algorithms for bioinformatics data analysis.

This module implements K-means and hierarchical clustering algorithms
specifically designed for biological data analysis.
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Tuple, Optional


class BioinformaticsClustering:
    """
    A class for performing clustering analysis on bioinformatics data.
    
    Attributes:
        n_clusters (int): Number of clusters for K-means
        linkage_method (str): Linkage method for hierarchical clustering
        random_state (int): Random state for reproducibility
    """
    
    def __init__(self, n_clusters: int = 3, linkage_method: str = 'ward', 
                 random_state: int = 42):
        """
        Initialize the clustering object.
        
        Args:
            n_clusters: Number of clusters (default: 3)
            linkage_method: Linkage method for hierarchical clustering 
                          ('ward', 'complete', 'average', 'single')
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.random_state = random_state
        self.kmeans_model = None
        self.hierarchical_model = None
        
    def kmeans_clustering(self, data: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Perform K-means clustering on the data.
        
        Args:
            data: Input data array of shape (n_samples, n_features)
            
        Returns:
            Tuple containing:
                - cluster labels
                - silhouette score
                - Davies-Bouldin index
        """
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        
        labels = self.kmeans_model.fit_predict(data)
        
        # Calculate clustering quality metrics
        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        
        return labels, silhouette, davies_bouldin
    
    def hierarchical_clustering(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform hierarchical clustering on the data.
        
        Args:
            data: Input data array of shape (n_samples, n_features)
            
        Returns:
            Tuple containing:
                - cluster labels
                - linkage matrix for dendrogram
        """
        self.hierarchical_model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage_method
        )
        
        labels = self.hierarchical_model.fit_predict(data)
        
        # Calculate linkage matrix for dendrogram
        linkage_matrix = linkage(data, method=self.linkage_method)
        
        return labels, linkage_matrix
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Get cluster centers from K-means clustering.
        
        Returns:
            Array of cluster centers or None if K-means hasn't been fitted
        """
        if self.kmeans_model is not None:
            return self.kmeans_model.cluster_centers_
        return None
    
    def evaluate_clustering(self, data: np.ndarray, labels: np.ndarray) -> dict:
        """
        Evaluate clustering quality using multiple metrics.
        
        Args:
            data: Input data array
            labels: Cluster labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        metrics = {
            'silhouette_score': silhouette_score(data, labels),
            'davies_bouldin_index': davies_bouldin_score(data, labels),
            'n_clusters': len(np.unique(labels))
        }
        
        return metrics


def find_optimal_clusters(data: np.ndarray, max_clusters: int = 10) -> Tuple[int, list]:
    """
    Find optimal number of clusters using the elbow method.
    
    Args:
        data: Input data array
        max_clusters: Maximum number of clusters to test
        
    Returns:
        Tuple containing:
            - optimal number of clusters
            - list of inertia values for each k
    """
    inertias = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    # Simple elbow detection (can be improved with more sophisticated methods)
    # Calculate the rate of decrease
    decreases = np.diff(inertias)
    second_derivative = np.diff(decreases)
    
    # Find the point where decrease slows down most
    optimal_k = np.argmax(second_derivative) + 2  # +2 because of range starting at 2
    
    return optimal_k, inertias
