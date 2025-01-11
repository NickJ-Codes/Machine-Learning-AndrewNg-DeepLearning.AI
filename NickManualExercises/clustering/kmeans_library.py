"""manual implementation of k-means clustering algorithm"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_cluster_comparison(y_true, cluster_labels):
    """
    Plot comparison between true labels and cluster assignments
    Args:
        y_true: ground truth labels
        cluster_labels: predicted cluster assignments
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, cluster_labels)

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Cluster Labels vs True Labels')
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True Label')
    plt.show()

    # Print some metrics
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    ari = adjusted_rand_score(y_true, cluster_labels)
    nmi = normalized_mutual_info_score(y_true, cluster_labels)

    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Normalized Mutual Information: {nmi:.3f}")

# Usage:
# plot_cluster_comparison(y_train, cluster_labels)

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    k = 3
    kmeans = KMeans(n_clusters = k, random_state = 5, tol = 1e-10)
    cluster_labels = kmeans.fit_predict(X)

    #graph cluster_labels vs known y_train labels
    plot_cluster_comparison(y, cluster_labels)

if __name__ == '__main__':
    main()
