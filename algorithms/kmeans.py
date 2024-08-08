import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class MyKMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, p=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.p = p  # Parameter for Minkowski distance

    def get_minkowski_distance(self, X, Y):
        return np.power(np.sum(np.power(np.abs(X - Y), self.p)), 1/self.p)

    def get_max_radius(self, X, dist_matrix):
        return np.max(dist_matrix)

    def precompute_dist_matrix(self, X):
        n = X.shape[0]
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            dist_matrix[i, i] = 0  # Distance of a point to itself is 0
            for j in range(i + 1, n):  # Optimize: Calculate only upper triangular part
                dist_matrix[i, j] = self.get_minkowski_distance(X[i], X[j])
                dist_matrix[j, i] = dist_matrix[i, j]  # Symmetry for lower triangular part
        return dist_matrix

    def can_cluster(self, X, k, radius, dist_matrix):
        indices = list(range(len(X)))
        centers = []

        while indices:
            i = indices.pop(np.random.randint(0, len(indices)))
            centers.append(i)

            # Find all indices within 2*radius of the chosen center
            to_remove = [j for j in indices if dist_matrix[i, j] <= 2 * radius]

            # Remove these indices from the list
            indices = [j for j in indices if j not in to_remove]

        return len(centers) <= k, [X[i] for i in centers]

    def approx_k_centers1(self, X, k, dist_matrix, interval_width=0.01):
        r_max = self.get_max_radius(X, dist_matrix)
        left, right = 0, r_max

        while right - left > interval_width:
            mid = (left + right) / 2
            can_cluster, centers = self.can_cluster(X, k, mid, dist_matrix)
            if can_cluster:
                right = mid
            else:
                left = mid

        # Return the centers and the maximum radius that was used
        return centers, right

    def approx_k_centers2(self, X, k, dist_matrix):
        if k >= len(X):
            return X
        centers = [X[np.random.randint(0, len(X))]]

        while len(centers) < k:
            distances = np.array([np.min(dist_matrix[i]) for i in range(len(X))])
            i = np.argmax(distances)
            centers.append(X[i])

        # Return the centers and the maximum radius that was used
        return np.array(centers), np.max(distances)

# Read the data from the CSV file
data = pd.read_csv('combined_data.csv')
x = data.iloc[1:, 1].values.astype(float) # Skip the first row and first column, selecting the second column
y = data.iloc[1:, 2].values.astype(float) # Skip the first row and second column, selecting the third column
X = np.column_stack((x, y))  # Combine x and y into a single array

# Execute kCentrosAproximados1 algorithm
num_clusters = 5  # Number of clusters for kCentrosAproximados1
kmeans = MyKMeans(n_clusters=num_clusters)

# Compute the distance matrix
dist_matrix = kmeans.precompute_dist_matrix(X)
centers1, a = kmeans.approx_k_centers1(X, num_clusters, dist_matrix)

# Execute kCentrosAproximados2 algorithm
kmeans = MyKMeans(n_clusters=num_clusters)
dist_matrix = kmeans.precompute_dist_matrix(X)
centers2, b = kmeans.approx_k_centers2(X, num_clusters, dist_matrix)

# Print the results
print("Centers 1:", centers1)
print("Centers 2:", centers2)

# Now, compare the results with the KMeans algorithm from scikit-learn
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)
print("KMeans Centers 1:", kmeans.cluster_centers_)
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)
print("KMeans Centers 2:", kmeans.cluster_centers_)