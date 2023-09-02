import numpy as np 
import pandas as pd
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self,n_clusters,tolerance=0.01,max_iters=1000):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit

        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.centroids = []
        self.max_iters = max_iters

        pass
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        
        mins = np.min(X,axis=0)
        maxs = np.max(X,axis=0)

        X_normalized = (X - mins)/(maxs - mins)

        for j in range(self.n_clusters):
            centroid = np.zeros(len(mins))

            for k in range(len(mins)):
                centroid[k] = np.random.uniform()

            self.centroids.append(centroid)

        self.centroids = np.array(self.centroids)

        diffs = np.full(len(self.centroids),self.tolerance+1)
        no_of_iterations = 0

        while np.max(diffs) > self.tolerance:
            no_of_iterations += 1

            # To stop the algorithm from running endlessly
            if no_of_iterations >= self.max_iters:
                print("Stopping algorithm due to too many iterations")
                break

            categorized_points = self.predict(X_normalized)

            prev_centroids = self.centroids
            self.centroids = update_centroids(self.centroids,X_normalized,categorized_points)

            for l in range(len(self.centroids)):
                diffs[l] = euclidean_distance(self.centroids[l],prev_centroids[l])

        self.centroids = self.centroids*(maxs-mins) + mins

        return self.centroids
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        
        mins = np.min(X,axis=0)
        maxs = np.max(X,axis=0)

        X_normalized = (X - mins)/(maxs - mins)

        normalized_centroids = (self.centroids - mins)/(maxs-mins)

        categorized_points = np.zeros(len(X_normalized))

        for p, point in enumerate(X_normalized):
            categorized_points[p] = find_closest_centroid(point,normalized_centroids)

        return categorized_points
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids
    
def find_closest_centroid(point,centroids):
    index_of_closest_centroid = 0
    distance_to_closest_centroid = euclidean_distance(point,centroids[0])

    for i in range(1, len(centroids)):
        if euclidean_distance(point,centroids[i]) < distance_to_closest_centroid:
            index_of_closest_centroid = i
            distance_to_closest_centroid = euclidean_distance(point,centroids[i])

    return index_of_closest_centroid
    
def update_centroids(centroids,points,points_categories):
    new_centroids = []

    for i in range(len(centroids)):
        relevant_points = points[points_categories == i]
        if not relevant_points.any():
            for j in range(len(centroids[i])):
                new_centroid = np.zeros(len(centroids[i]))
                new_centroid[j] = np.random.uniform()
        else:
            new_centroid = np.mean(relevant_points,axis=0)
        new_centroids.append(new_centroid)

    new_centroids = np.array(new_centroids)

    return new_centroids

# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum(axis=1).sum()
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z.astype(int)]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
  