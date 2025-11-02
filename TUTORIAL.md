# Tutorial: Clustering Analysis in Bioinformatics

This tutorial will guide you through performing clustering analysis on biological data using the tools provided in this repository.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Understanding the Data](#understanding-the-data)
5. [Clustering Methods](#clustering-methods)
6. [Interpreting Results](#interpreting-results)
7. [Best Practices](#best-practices)

## Introduction

Clustering is an unsupervised learning technique that groups similar samples together. In bioinformatics, it's commonly used to:

- Identify subtypes in diseases (e.g., cancer subtypes)
- Group genes with similar expression patterns
- Classify proteins into families
- Find patterns in clinical data

## Installation

1. Clone this repository:
```bash
git clone https://github.com/frankfurtmacmoses/clustering_assignment2.git
cd clustering_assignment2
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the example script to see clustering in action:

```bash
cd examples
python clustering_example.py
```

This will:
- Load sample gene expression data
- Find the optimal number of clusters
- Perform K-means and hierarchical clustering
- Generate visualization plots
- Compare the two methods

## Understanding the Data

### Gene Expression Data Format

The gene expression data is stored in CSV format with:
- **Rows**: Genes (features)
- **Columns**: Samples (observations)

Example structure:
```
         Sample_1  Sample_2  Sample_3
Gene_1      2.5       3.1       2.8
Gene_2      1.2       1.5       1.3
Gene_3      4.8       5.2       4.9
```

### Data Preprocessing

Before clustering, the data is:
1. **Transposed**: Samples become rows (required for clustering)
2. **Standardized**: Each feature scaled to mean=0, std=1
3. **Cleaned**: Missing values handled appropriately

## Clustering Methods

### 1. K-Means Clustering

**How it works:**
- Randomly initializes K cluster centers
- Assigns each sample to nearest center
- Updates centers based on assigned samples
- Repeats until convergence

**When to use:**
- You have an idea of how many clusters to expect
- Clusters are roughly spherical
- Fast computation is needed

**Code example:**
```python
from clustering import BioinformaticsClustering

clustering = BioinformaticsClustering(n_clusters=3)
labels, silhouette, davies_bouldin = clustering.kmeans_clustering(data)
```

### 2. Hierarchical Clustering

**How it works:**
- Starts with each sample as its own cluster
- Iteratively merges closest clusters
- Creates a dendrogram showing relationships
- Cut at desired height for K clusters

**When to use:**
- You want to see relationships between samples
- Number of clusters is unknown
- You need a dendrogram visualization

**Code example:**
```python
from clustering import BioinformaticsClustering

clustering = BioinformaticsClustering(n_clusters=3, linkage_method='ward')
labels, linkage_matrix = clustering.hierarchical_clustering(data)
```

## Interpreting Results

### Evaluation Metrics

**1. Silhouette Score** (range: -1 to 1)
- Measures how similar samples are to their own cluster vs. other clusters
- **Higher is better**
- > 0.5: Good clustering
- 0.25-0.5: Moderate clustering
- < 0.25: Poor clustering

**2. Davies-Bouldin Index** (range: 0 to ∞)
- Measures average similarity between clusters
- **Lower is better**
- < 1.0: Good clustering
- Higher values indicate overlapping clusters

**3. Adjusted Rand Index** (range: -1 to 1)
- Measures agreement between two clustering methods
- 1.0: Perfect agreement
- 0.0: Random agreement
- < 0: Less agreement than random

### Visualization Interpretation

**Heatmap:**
- Rows: Genes
- Columns: Samples (grouped by cluster)
- Color: Expression level (red=high, blue=low)
- Look for blocks of similar colors within clusters

**Dendrogram:**
- Height: Distance between clusters
- Cutting line: Determines number of clusters
- Longer branches: More distinct clusters

**PCA Plot:**
- Shows samples in 2D space
- Points close together: Similar samples
- Cluster centers: Average position of cluster members

**Elbow Curve:**
- X-axis: Number of clusters
- Y-axis: Within-cluster variance (inertia)
- Elbow point: Optimal number of clusters

## Best Practices

### Choosing Number of Clusters

1. **Use the elbow method**: Look for the "elbow" in the inertia plot
2. **Consider biological knowledge**: Are there known subtypes?
3. **Try multiple values**: Compare clustering quality metrics
4. **Visualize results**: Do the clusters make biological sense?

### Data Quality

1. **Check for outliers**: They can create artificial clusters
2. **Handle missing values**: Don't ignore them
3. **Normalize data**: Especially if features have different scales
4. **Filter low-variance genes**: They don't contribute to clustering

### Validation

1. **Use multiple metrics**: Don't rely on just one
2. **Compare methods**: K-means vs. hierarchical
3. **Biological validation**: Do clusters correspond to known biology?
4. **Cross-validation**: Test stability across data subsets

## Example: Complete Analysis Workflow

```python
import sys
import numpy as np
sys.path.insert(0, 'src')

from clustering import BioinformaticsClustering, find_optimal_clusters
from data_loader import BioinformaticsDataLoader
from visualization import plot_heatmap, plot_clusters_2d

# 1. Load data
loader = BioinformaticsDataLoader(standardize=True)
df, data = loader.load_gene_expression('data/gene_expression.csv')
data_samples = data.T  # Transpose: samples as rows

# 2. Find optimal clusters
optimal_k, inertias = find_optimal_clusters(data_samples, max_clusters=8)
print(f"Optimal k: {optimal_k}")

# 3. Perform clustering
clustering = BioinformaticsClustering(n_clusters=optimal_k)
labels, silhouette, db_index = clustering.kmeans_clustering(data_samples)

print(f"Silhouette Score: {silhouette:.3f}")
print(f"Davies-Bouldin Index: {db_index:.3f}")

# 4. Visualize
import matplotlib.pyplot as plt

fig = plot_heatmap(df, cluster_labels=labels)
plt.savefig('my_heatmap.png')

fig = plot_clusters_2d(data_samples, labels, 
                       centers=clustering.get_cluster_centers())
plt.savefig('my_clusters.png')
```

## Common Issues and Solutions

**Issue**: Clusters don't make biological sense
- **Solution**: Try different preprocessing, adjust number of clusters, or check for batch effects

**Issue**: Different methods give very different results
- **Solution**: Data may not have clear clusters, or preprocessing may be needed

**Issue**: All samples end up in one cluster
- **Solution**: Data may need better normalization or scaling

**Issue**: Too many small clusters
- **Solution**: Reduce number of clusters or check for outliers

## Further Reading

- [sklearn clustering documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [Gene expression clustering review](https://www.nature.com/articles/nrg2825)
- [Choosing the right clustering algorithm](https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68)

## Assignment Questions

Use this code to answer the following questions:

1. What is the optimal number of clusters in the sample data?
2. Which clustering method (K-means or hierarchical) performs better on this dataset?
3. What biological interpretation could you give to the identified clusters?
4. How would you validate that these clusters are meaningful?

## License

This project is for educational purposes in bioinformatics courses.
