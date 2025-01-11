"""manual implementation of k-means clustering algorithm"""
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def initialize_centroids(X, k, seed = None):
    """
    Initialize k centroids randomly from the data points
    Args:
        X: input data (n_samples, n_features)
        k: number of clusters
    Returns:
        centroids: initial centroids (k, n_features)
    """
    #Find range of each feature
    feature_mins = X.min(axis = 0)
    feature_maxs = X.max(axis = 0)

    #Intialize centroids
    centroids = np.zeros((k,X.shape[1]))
    if seed is not None:
        np.random.seed(seed)
    for i in range(k):
        centroids[i] = [np.random.uniform(feature_mins[j], feature_maxs[j]) for j in range(X.shape[1])]
    return centroids


def assign_clusters(X, centroids):
    """
    Assign each data point to nearest centroid
    Args:
        X: input data (n_samples, n_features)
        centroids: current centroids (k, n_features)
    Returns:
        cluster_labels: cluster assignments for each data point
    """
    # calculate squared distances used vectorized operations
    # expand (a-b)^2 = a^2 + b^2 - 2ab

    #Square each element in X, sum along axis 1
    # np.newaxis is an alias of None; since it's after the comma i, it is turning a 1d vector into a 2d matrix
    squared_X = np.sum(X**2, axis = 1)[:, np.newaxis] #shape (n_samples, 1)
    square_centroids = np.sum(centroids**2, axis = 1)[np.newaxis:] # shape (1, k)
    cross_term = -2 * np.matmul(X, centroids.T) # shape (n_samples, k)

    #compute vectorized distances, no need to do square root since we're just minimizing values not finding actual numeric distance
    # Uses broadcasting since you're adding an (n,1) (1,k) and (n,k) matrices
    distances = squared_X + square_centroids + cross_term #shape (n_samples, k clusters)

    #find minimum distance cluster, and assign cluster to data point
    assignedClusters = np.argmin(distances, axis = 1)
    return assignedClusters


def update_centroids(X, cluster_labels, k):
    """
    Update centroid positions based on mean of assigned points
    Args:
        X: input data (n_samples, n_features)
        cluster_labels: current cluster assignments
        k: number of clusters
    Returns:
        new_centroids: updated centroid positions
    """
    centroids = np.zeros((k,X.shape[1]))
    # loop implementation to calculate centroid means
    # for i in range(k):
    #     X_ingroup = X[cluster_labels ==i,:]
    #     centroids[i]= np.mean(X_ingroup, axis = 0)

    #vectorized implementation of calculating centroid means
    # the np.maximum is there to prevent divide by zero errors for clusters with n data points
    counts = np.bincount(cluster_labels, minlength = k).reshape(-1,1)

    #create a sparse matrix-like representation for fast summation
    dummy = np.zeros((X.shape[0], k))
    dummy[np.arange(X.shape[0]), cluster_labels] = 1

    #he output of dummy.T @ X is a matrix where element a_ij = for cluster i, sum of feature j across all sample points that were in cluster i
    centroids = (dummy.T @ X) / np.maximum(counts, 1)
    return centroids

def kmeans(X, k, max_iters=100, tol=1e-4, verbose = False,
           break_early = True, seed = None,
           reduceClusters = True):
    """
    Main K-means clustering algorithm
    Args:
        X: input data (n_samples, n_features)
        k: number of clusters
        max_iters: maximum number of iterations
        tol: convergence tolerance
    Returns:
        cluster_labels: final cluster assignments
        centroids: final centroid positions
    """
    centroids = initialize_centroids(X, k, seed = seed)
    inertia_old = 0
    for iter in range(max_iters):
        if verbose:
            print(f'Running iteration: {iter} with k={k}, starting interia = {inertia_old}')
        cluster_labels = assign_clusters(X, centroids)

        # remove empty clusters
        if reduceClusters == True:
            counts = np.bincount(cluster_labels, minlength=k)
            active_clusters = counts > 0
            if np.sum(active_clusters) < k:
                centroids = centroids[active_clusters]
                k = centroids.shape[0]

                # assign new clusters
                cluster_labels = assign_clusters(X, centroids)

        centroids = update_centroids(X, cluster_labels, k)
        inertia_new = calculate_inertia(X, cluster_labels, centroids)
        if np.abs(inertia_new - inertia_old) < tol and break_early == True:
            if verbose:
                print(f"Breaking after iteration {iter} because reached tolerance threshold for inertia")
            break
        else:
            inertia_old = inertia_new
    return cluster_labels, centroids

def calculate_inertia(X, cluster_labels, centroids):
    """
    Calculate the sum of squared distances of samples to their closest centroid
    Args:
        X: input data
        cluster_labels: cluster assignments
        centroids: centroid positions
    Returns:
        inertia: sum of squared distances
    """
    # use advanced indexing to get the centroid for each point based its label
    assigned_centroids = centroids[cluster_labels]
    squared_distances = np.sum((X-assigned_centroids)**2, axis = 1)
    intertia = np.sum(squared_distances)
    return(intertia)

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

    k = 100000
    k_new = k
    print("Running k trimmer")
    while True:
        cluster_labels, centroids = kmeans(X, k=k_new, verbose = False,
                                           break_early = False, tol = 1e-10,
                                           seed = 5)
        k_new = centroids.shape[0]
        if k_new != k:
            print(f"K moved from {k} to {k_new}")
            k = k_new
        else:
            break

    #graph cluster_labels vs known y_train labels
    plot_cluster_comparison(y, cluster_labels)

if __name__ == '__main__':
    main()
