"""
Visualization utilities for clustering analysis in bioinformatics.

This module provides functions to visualize clustering results including
heatmaps, dendrograms, and scatter plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from typing import Optional, Tuple


def plot_heatmap(data: pd.DataFrame, cluster_labels: Optional[np.ndarray] = None,
                figsize: Tuple[int, int] = (12, 8), cmap: str = 'RdYlBu_r',
                title: str = 'Gene Expression Heatmap') -> plt.Figure:
    """
    Plot a heatmap of gene expression data with optional cluster annotations.
    
    Args:
        data: Gene expression DataFrame (genes x samples)
        cluster_labels: Optional array of cluster labels for samples
        figsize: Figure size
        cmap: Colormap name
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by clusters if provided
    if cluster_labels is not None:
        # Sort columns by cluster labels
        sorted_indices = np.argsort(cluster_labels)
        sorted_data = data.iloc[:, sorted_indices]
        sorted_labels = cluster_labels[sorted_indices]
        
        # Create color bar for clusters
        unique_clusters = np.unique(sorted_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        cluster_colors = [colors[label] for label in sorted_labels]
        
        # Plot heatmap with cluster colors
        sns.heatmap(sorted_data, cmap=cmap, center=0, ax=ax,
                   xticklabels=True, yticklabels=False,
                   cbar_kws={'label': 'Expression Level'})
        
        # Add cluster color bar
        for i, color in enumerate(cluster_colors):
            ax.add_patch(plt.Rectangle((i, -0.5), 1, 0.5, 
                                      facecolor=color, edgecolor='none',
                                      clip_on=False))
    else:
        sns.heatmap(data, cmap=cmap, center=0, ax=ax,
                   xticklabels=True, yticklabels=False,
                   cbar_kws={'label': 'Expression Level'})
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Samples', fontsize=12)
    ax.set_ylabel('Genes', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_dendrogram(linkage_matrix: np.ndarray, labels: Optional[list] = None,
                   figsize: Tuple[int, int] = (12, 6),
                   title: str = 'Hierarchical Clustering Dendrogram') -> plt.Figure:
    """
    Plot a dendrogram for hierarchical clustering results.
    
    Args:
        linkage_matrix: Linkage matrix from hierarchical clustering
        labels: Optional sample labels
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    dendrogram(linkage_matrix, labels=labels, ax=ax,
              leaf_font_size=10, color_threshold=0.7*max(linkage_matrix[:, 2]))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_clusters_2d(data: np.ndarray, labels: np.ndarray,
                    centers: Optional[np.ndarray] = None,
                    feature_names: Optional[list] = None,
                    figsize: Tuple[int, int] = (10, 8),
                    title: str = 'Clustering Results (PCA)') -> plt.Figure:
    """
    Plot 2D visualization of clusters using PCA.
    
    Args:
        data: Data array (n_samples, n_features)
        labels: Cluster labels
        centers: Optional cluster centers
        feature_names: Names of the first two features if not using PCA
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # If data has more than 2 features, use PCA
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)'
        ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'
        
        if centers is not None:
            centers_2d = pca.transform(centers)
        else:
            centers_2d = None
    else:
        data_2d = data
        xlabel = feature_names[0] if feature_names else 'Feature 1'
        ylabel = feature_names[1] if feature_names else 'Feature 2'
        centers_2d = centers
    
    # Plot clusters
    scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], 
                        c=labels, cmap='viridis', 
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Plot cluster centers if provided
    if centers_2d is not None:
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
                  c='red', marker='X', s=300, edgecolors='black',
                  linewidth=2, label='Cluster Centers')
        ax.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster', fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_elbow_curve(k_values: list, inertias: list,
                    optimal_k: Optional[int] = None,
                    figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot elbow curve for determining optimal number of clusters.
    
    Args:
        k_values: List of k values tested
        inertias: List of inertia values for each k
        optimal_k: Optimal k value to highlight
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    
    if optimal_k is not None:
        optimal_idx = k_values.index(optimal_k)
        ax.plot(optimal_k, inertias[optimal_idx], 'r*', 
               markersize=20, label=f'Optimal k={optimal_k}')
        ax.legend()
    
    ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_silhouette_scores(data: np.ndarray, labels: np.ndarray,
                          n_clusters: int,
                          figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot silhouette analysis for clustering results.
    
    Args:
        data: Data array
        labels: Cluster labels
        n_clusters: Number of clusters
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute silhouette scores for each sample
    silhouette_vals = silhouette_samples(data, labels)
    
    y_lower = 10
    for i in range(n_clusters):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.viridis(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, cluster_silhouette_vals,
                        facecolor=color, edgecolor=color, alpha=0.7)
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.set_title('Silhouette Plot for Clusters', fontsize=14, fontweight='bold')
    ax.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    
    # Add average silhouette score line
    avg_silhouette = silhouette_vals.mean()
    ax.axvline(x=avg_silhouette, color='red', linestyle='--',
              label=f'Average Score: {avg_silhouette:.3f}')
    ax.legend()
    
    plt.tight_layout()
    return fig
