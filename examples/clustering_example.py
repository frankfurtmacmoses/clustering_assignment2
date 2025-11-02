"""
Example script demonstrating clustering analysis on gene expression data.

This script shows how to:
1. Load gene expression data
2. Perform K-means clustering
3. Perform hierarchical clustering
4. Visualize results
5. Evaluate clustering quality
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from clustering import BioinformaticsClustering, find_optimal_clusters
from data_loader import BioinformaticsDataLoader
from visualization import (plot_heatmap, plot_dendrogram, plot_clusters_2d, 
                          plot_elbow_curve, plot_silhouette_scores)


def main():
    """Main function demonstrating clustering workflow."""
    
    print("=" * 60)
    print("Bioinformatics Clustering Analysis Example")
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Loading gene expression data...")
    data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'gene_expression.csv')
    
    loader = BioinformaticsDataLoader(normalize=False, standardize=True)
    df, data = loader.load_gene_expression(data_file)
    
    # Transpose data so samples are rows (required for clustering)
    data_samples = data.T
    
    print(f"   Loaded {df.shape[0]} genes and {df.shape[1]} samples")
    print(f"   Data shape for clustering: {data_samples.shape}")
    
    # 2. Find optimal number of clusters
    print("\n2. Finding optimal number of clusters...")
    optimal_k, inertias = find_optimal_clusters(data_samples, max_clusters=8)
    print(f"   Optimal number of clusters: {optimal_k}")
    
    # Plot elbow curve
    k_values = list(range(2, 9))
    fig_elbow = plot_elbow_curve(k_values, inertias, optimal_k=optimal_k)
    plt.savefig('elbow_curve.png', dpi=150, bbox_inches='tight')
    print("   Saved elbow curve to 'elbow_curve.png'")
    plt.close()
    
    # 3. Perform K-means clustering
    print(f"\n3. Performing K-means clustering with k={optimal_k}...")
    clustering = BioinformaticsClustering(n_clusters=optimal_k, random_state=42)
    kmeans_labels, silhouette, davies_bouldin = clustering.kmeans_clustering(data_samples)
    
    print(f"   Silhouette Score: {silhouette:.3f} (higher is better)")
    print(f"   Davies-Bouldin Index: {davies_bouldin:.3f} (lower is better)")
    
    # Count samples per cluster
    unique, counts = np.unique(kmeans_labels, return_counts=True)
    print(f"   Samples per cluster: {dict(zip(unique, counts))}")
    
    # 4. Visualize K-means results
    print("\n4. Visualizing K-means clustering results...")
    
    # Plot 2D visualization
    centers = clustering.get_cluster_centers()
    fig_2d = plot_clusters_2d(data_samples, kmeans_labels, centers=centers,
                             title=f'K-means Clustering Results (k={optimal_k})')
    plt.savefig('kmeans_clusters_2d.png', dpi=150, bbox_inches='tight')
    print("   Saved 2D cluster plot to 'kmeans_clusters_2d.png'")
    plt.close()
    
    # Plot heatmap
    fig_heatmap = plot_heatmap(df, cluster_labels=kmeans_labels,
                              title='Gene Expression Heatmap (K-means Clusters)')
    plt.savefig('kmeans_heatmap.png', dpi=150, bbox_inches='tight')
    print("   Saved heatmap to 'kmeans_heatmap.png'")
    plt.close()
    
    # Plot silhouette analysis
    fig_silhouette = plot_silhouette_scores(data_samples, kmeans_labels, optimal_k)
    plt.savefig('silhouette_analysis.png', dpi=150, bbox_inches='tight')
    print("   Saved silhouette analysis to 'silhouette_analysis.png'")
    plt.close()
    
    # 5. Perform hierarchical clustering
    print(f"\n5. Performing hierarchical clustering with k={optimal_k}...")
    hier_labels, linkage_matrix = clustering.hierarchical_clustering(data_samples)
    
    # Evaluate hierarchical clustering
    hier_metrics = clustering.evaluate_clustering(data_samples, hier_labels)
    print(f"   Silhouette Score: {hier_metrics['silhouette_score']:.3f}")
    print(f"   Davies-Bouldin Index: {hier_metrics['davies_bouldin_index']:.3f}")
    
    # Count samples per cluster
    unique, counts = np.unique(hier_labels, return_counts=True)
    print(f"   Samples per cluster: {dict(zip(unique, counts))}")
    
    # 6. Visualize hierarchical clustering results
    print("\n6. Visualizing hierarchical clustering results...")
    
    # Plot dendrogram
    sample_labels = [f"S{i+1}" for i in range(len(hier_labels))]
    fig_dendro = plot_dendrogram(linkage_matrix, labels=sample_labels,
                                 title='Hierarchical Clustering Dendrogram')
    plt.savefig('hierarchical_dendrogram.png', dpi=150, bbox_inches='tight')
    print("   Saved dendrogram to 'hierarchical_dendrogram.png'")
    plt.close()
    
    # Plot 2D visualization
    fig_2d_hier = plot_clusters_2d(data_samples, hier_labels, centers=None,
                                   title=f'Hierarchical Clustering Results (k={optimal_k})')
    plt.savefig('hierarchical_clusters_2d.png', dpi=150, bbox_inches='tight')
    print("   Saved 2D cluster plot to 'hierarchical_clusters_2d.png'")
    plt.close()
    
    # 7. Compare clustering methods
    print("\n7. Comparing clustering methods...")
    print("\n   K-means Clustering:")
    print(f"     - Silhouette Score: {silhouette:.3f}")
    print(f"     - Davies-Bouldin Index: {davies_bouldin:.3f}")
    
    print("\n   Hierarchical Clustering:")
    print(f"     - Silhouette Score: {hier_metrics['silhouette_score']:.3f}")
    print(f"     - Davies-Bouldin Index: {hier_metrics['davies_bouldin_index']:.3f}")
    
    # Agreement between methods
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(kmeans_labels, hier_labels)
    print(f"\n   Adjusted Rand Index (agreement): {ari:.3f}")
    print("   (1.0 = perfect agreement, 0.0 = random agreement)")
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check the generated PNG files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
