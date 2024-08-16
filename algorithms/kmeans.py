import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class MyKMeans:
    def __init__(self, n_clusters=8, tol=1e-4, p=1):
        self.n_clusters = n_clusters
        self.tol = tol
        self.p = p

    def get_minkowski_distance(self, X, Y):
        return np.power(np.sum(np.abs(X - Y) ** self.p), 1 / self.p)

    def precompute_dist_matrix(self, X):
        return cdist(X, X, 'minkowski', p=self.p)

    def get_max_radius(self, X, dist_matrix):
        return np.max(dist_matrix)
    
    def can_cluster(self, X, k, radius, dist_matrix):
        indices = list(range(len(X)))
        centers = []
    
        while indices and len(centers) < k:
            i = indices.pop(np.random.randint(0, len(indices)))
            centers.append(i)
    
            to_remove = [j for j in indices if dist_matrix[i, j] <= 2 * radius]
    
            indices = [j for j in indices if j not in to_remove]
    
        return len(centers) <= k, [X[i] for i in centers]

    def approx_k_centers1(self, X, dist_matrix, interval_width=0.1):
        r_max = self.get_max_radius(X, dist_matrix)
        left, right = 0, r_max

        best_centers = []
        best_radius = r_max

        while right - left > interval_width:
            mid = (left + right) / 2
            can_cluster, centers = self.can_cluster(X, self.n_clusters, mid, dist_matrix)
            
            if len(centers) == self.n_clusters:
                return centers, mid 
            
            if can_cluster:
                best_centers = centers
                best_radius = mid
                right = mid
            else:
                left = mid

        return np.array(best_centers), best_radius

    def approx_k_centers2(self, X, dist_matrix):
        if self.n_clusters >= len(X):
            return X
        centers = [X[np.random.randint(0, len(X))]]
        center_indices = [np.random.randint(0, len(X))]

        while len(centers) < self.n_clusters:
            distances = np.min([dist_matrix[i] for i in center_indices], axis=0)
            i = np.argmax(distances)
            centers.append(X[i])
            center_indices.append(i)

        return np.array(centers), np.max(distances)