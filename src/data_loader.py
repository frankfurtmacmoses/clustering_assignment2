"""
Data loading and preprocessing utilities for bioinformatics data.

This module provides functions to load and preprocess various types of
biological data for clustering analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class BioinformaticsDataLoader:
    """
    A class for loading and preprocessing bioinformatics data.
    
    Handles various data formats commonly used in bioinformatics including
    gene expression matrices, sequence similarity matrices, etc.
    """
    
    def __init__(self, normalize: bool = True, standardize: bool = True):
        """
        Initialize the data loader.
        
        Args:
            normalize: Whether to normalize data to [0, 1] range
            standardize: Whether to standardize data (mean=0, std=1)
        """
        self.normalize = normalize
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        
    def load_gene_expression(self, filepath: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load gene expression data from CSV file.
        
        Expected format: Rows are genes, columns are samples
        First column should be gene IDs/names
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Tuple containing:
                - DataFrame with gene information
                - NumPy array of expression values
        """
        df = pd.read_csv(filepath, index_col=0)
        
        # Extract numeric data
        data = df.values.astype(float)
        
        # Preprocess if requested
        if self.normalize:
            data = self._normalize_data(data)
        
        if self.standardize:
            data = self.scaler.fit_transform(data)
        
        return df, data
    
    def load_from_dataframe(self, df: pd.DataFrame, 
                           feature_cols: Optional[list] = None) -> np.ndarray:
        """
        Load data from a pandas DataFrame.
        
        Args:
            df: Input DataFrame
            feature_cols: List of column names to use as features (None = all numeric)
            
        Returns:
            NumPy array of processed data
        """
        if feature_cols is None:
            # Select all numeric columns
            data = df.select_dtypes(include=[np.number]).values
        else:
            data = df[feature_cols].values
        
        # Preprocess
        if self.normalize:
            data = self._normalize_data(data)
            
        if self.standardize:
            data = self.scaler.fit_transform(data)
        
        return data
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to [0, 1] range.
        
        Args:
            data: Input array
            
        Returns:
            Normalized array
        """
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        normalized = (data - min_vals) / range_vals
        
        return normalized
    
    def handle_missing_values(self, data: np.ndarray, 
                             strategy: str = 'mean') -> np.ndarray:
        """
        Handle missing values in the data.
        
        Args:
            data: Input array
            strategy: Strategy for handling missing values 
                     ('mean', 'median', 'zero', 'remove')
            
        Returns:
            Array with missing values handled
        """
        if strategy == 'mean':
            col_means = np.nanmean(data, axis=0)
            inds = np.where(np.isnan(data))
            data[inds] = np.take(col_means, inds[1])
        elif strategy == 'median':
            col_medians = np.nanmedian(data, axis=0)
            inds = np.where(np.isnan(data))
            data[inds] = np.take(col_medians, inds[1])
        elif strategy == 'zero':
            data = np.nan_to_num(data, nan=0.0)
        
        return data
    
    def filter_low_variance_features(self, data: pd.DataFrame, 
                                    threshold: float = 0.01) -> pd.DataFrame:
        """
        Filter out features with low variance.
        
        Useful for removing genes with minimal expression variation.
        
        Args:
            data: Input DataFrame
            threshold: Variance threshold
            
        Returns:
            Filtered DataFrame
        """
        variances = data.var()
        high_variance_features = variances[variances > threshold].index
        
        return data[high_variance_features]


def generate_sample_gene_expression_data(n_genes: int = 100, 
                                         n_samples: int = 20,
                                         n_clusters: int = 3,
                                         random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic gene expression data for testing.
    
    Creates data with distinct clusters representing different biological conditions.
    
    Args:
        n_genes: Number of genes
        n_samples: Number of samples
        n_clusters: Number of underlying clusters
        random_state: Random seed
        
    Returns:
        DataFrame with gene expression data
    """
    np.random.seed(random_state)
    
    # Create cluster centers
    samples_per_cluster = n_samples // n_clusters
    data = []
    
    for cluster_id in range(n_clusters):
        # Generate cluster center
        center = np.random.randn(n_genes) * 2 + cluster_id * 3
        
        # Generate samples around this center
        for _ in range(samples_per_cluster):
            sample = center + np.random.randn(n_genes) * 0.5
            data.append(sample)
    
    # Add remaining samples if n_samples not divisible by n_clusters
    remaining = n_samples - len(data)
    for _ in range(remaining):
        cluster_id = np.random.randint(0, n_clusters)
        center = np.random.randn(n_genes) * 2 + cluster_id * 3
        sample = center + np.random.randn(n_genes) * 0.5
        data.append(sample)
    
    # Create DataFrame
    data_array = np.array(data).T  # Transpose so genes are rows
    gene_names = [f"Gene_{i+1}" for i in range(n_genes)]
    sample_names = [f"Sample_{i+1}" for i in range(n_samples)]
    
    df = pd.DataFrame(data_array, index=gene_names, columns=sample_names)
    
    return df
