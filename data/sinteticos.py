import time
import warnings
from itertools import cycle, islice
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

# Import the MyKMeans class
from algorithms.kmeans import MyKMeans
from sklearn.cluster import KMeans
import time
from sklearn.metrics import silhouette_score, adjusted_rand_score
import pandas as pd

# Generate synthetic datasets
n_samples = 700
seed = 42
noisy_circles_1 = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
# noisy_circles_2 = datasets.make_circles(n_samples=n_samples, factor=0.4, noise=0.07, random_state=seed+1)
noisy_moons_1 = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
noisy_moons_2 = datasets.make_moons(n_samples=n_samples, noise=0.1, random_state=seed+1)
blobs_1 = datasets.make_blobs(n_samples=n_samples, random_state=seed)
blobs_2 = datasets.make_blobs(n_samples=n_samples, centers=4, random_state=seed+1)
rng = np.random.RandomState(seed)
no_structure_1 = rng.rand(n_samples, 2), rng.randint(0, 2, n_samples)
# no_structure_2 = rng.rand(n_samples, 2), None

# Dados anisotrópicos
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso_1 = np.dot(X, transformation)
aniso_1 = (X_aniso_1, y)

X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state+1)
transformation = [[0.8, -0.2], [-0.6, 0.6]]
X_aniso_2 = np.dot(X, transformation)
aniso_2 = (X_aniso_2, y)

# Blobs com variâncias variadas
varied_1 = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
varied_2 = datasets.make_blobs(n_samples=n_samples, cluster_std=[0.5, 3.0, 1.5], random_state=random_state+1)


# List of datasets
datasets = [
    noisy_circles_1,
    noisy_moons_1,
    noisy_moons_2,
    blobs_1,
    blobs_2,
    no_structure_1,
    aniso_1, aniso_2, 
    varied_1, varied_2
]

np.random.seed(seed)
for i in range(10):
    n_clusters = np.random.randint(2, 6)  # Número de centros entre 2 e 5, pode modificar dps
    centers = np.random.uniform(-10, 10, (n_clusters, 2))  # Centros em uma faixa de -10 a 10 pode modificar dps
    covariances = []

    for _ in range(n_clusters):
        std_dev = np.random.uniform(0.1, 2.0)  # Desvio padrão variando entre 0.1 e 2.0 pode modificar dps
        cov_matrix = np.array([[std_dev, 0], [0, std_dev]])
        covariances.append(cov_matrix)

    X = []
    y = []

    for idx, center in enumerate(centers):
        points = np.random.multivariate_normal(center, covariances[idx], size=n_samples // n_clusters)
        X.append(points)
        y.extend([idx] * (n_samples // n_clusters))

    X = np.vstack(X)
    datasets.append((X, np.array(y)))

# Export datasets to CSV
for i, (X, y) in enumerate(datasets):
    df = pd.DataFrame(X, columns=["x", "y"])
    df["label"] = y
    df.to_csv(f"/home/rafaelmg/Documents/ALG2/TP2/data/synth_datasets/synthetic_dataset_{i+1}.csv", index=False)

# Now, let's add the real datasets. They are .csvs in data/real_datasets. Their first row has the name of the columns; the second row contains the number of unique labels; the third row is the first row of the dataset. The last column is the label column.
# Iterate over the real datasets and load them
real_datasets = []
nums_clusters = []
for file in os.listdir("/home/rafaelmg/Documents/ALG2/TP2/data/real_datasets"):
    if file.endswith(".csv"):
        print(f"Loading {file}...")
        df = pd.read_csv(f"/home/rafaelmg/Documents/ALG2/TP2/data/real_datasets/{file}")
        real_datasets.append((df.iloc[1:, :-1].values, df.iloc[1:, -1].values))
        nums_clusters.append(df.iloc[0, 0])
        
# Add the real datasets to the list of datasets
datasets.extend(real_datasets)

# Process each dataset
for p in [1, 2]:
    results = []
    for i, (X, y) in enumerate(datasets):
        
        # Se o dataset for um dos reais, pegar o número de clusters do arquivo
        if i >= 20:
            n = int(nums_clusters[i-20])
        else:
            n = 3
            
        print(X.shape)
        my_kmeans = MyKMeans(n_clusters=n, tol=1e-4, p=p)

        X = StandardScaler().fit_transform(X)

        dist_matrix = my_kmeans.precompute_dist_matrix(X)
        print(dist_matrix.shape)

        radii = []
        silhouettes = []
        adjusted_rands = []
        execution_times = []
        results_temp = []
        
        for run in range(30):
            result = {
                "dataset": i + 1,
                "run": run + 1,
                "centers1": None,
                "radius1": None,
                "silhouette1": None,
                "adjusted_rand1": None,
                "time1": None,
                "centers2": None,
                "radius2": None,
                "silhouette2": None,
                "adjusted_rand2": None,
                "time2": None,
            }

            start_time = time.time()
            centers1, radius1 = my_kmeans.approx_k_centers1(X, dist_matrix, interval_width=0.15)
            end_time = time.time()
            
            result["centers1"] = centers1
            result["radius1"] = radius1
            result["time1"] = end_time - start_time
            
            start_time = time.time()
            centers2, radius2 = my_kmeans.approx_k_centers2(X, dist_matrix)
            end_time = time.time()
            result["centers2"] = centers2
            result["radius2"] = radius2
            result["time2"] = end_time - start_time
            
            try:
                labels1 = np.argmin(dist_matrix[:, [np.where((X == c).all(axis=1))[0][0] for c in centers1]], axis=1)
                result["silhouette1"] = silhouette_score(X, labels1)
                result["adjusted_rand1"] = adjusted_rand_score(y, labels1)

                labels2 = np.argmin(dist_matrix[:, [np.where((X == c).all(axis=1))[0][0] for c in centers2]], axis=1)
                result["silhouette2"] = silhouette_score(X, labels2)
                result["adjusted_rand2"] = adjusted_rand_score(y, labels2)
            except ValueError as e:
                print(f"Error on dataset {i+1} run {run+1}: {e}")
                continue
            
            start_time = time.time()
            kmeans = KMeans(n_clusters=my_kmeans.n_clusters, tol=my_kmeans.tol)
            kmeans.fit(X)
            end_time = time.time()
            result["centers_sklearn"] = kmeans.cluster_centers_
            centers_sklearn = kmeans.cluster_centers_
            result["time_sklearn"] = end_time - start_time

            labels_sklearn = kmeans.labels_
            result["silhouette_sklearn"] = silhouette_score(X, labels_sklearn)
            result["adjusted_rand_sklearn"] = adjusted_rand_score(y, labels_sklearn)

            results_temp.append(result)

            print(f"Dataset {i+1} Run {run+1} completed")
            print(f"approx_k_centers1 centers: {centers1}, radius: {radius1}")
            print(f"approx_k_centers2 centers: {centers2}, radius: {radius2}")
            print(f"sklearn KMeans centers: {result['centers_sklearn']}")
            print(f"Silhouette score 1: {result['silhouette1']}")
            print(f"Silhouette score 2: {result['silhouette2']}")
            print(f"Silhouette score sklearn: {result['silhouette_sklearn']}")
            print(f"Adjusted Rand score 1: {result['adjusted_rand1']}")
            print(f"Adjusted Rand score 2: {result['adjusted_rand2']}")
            print(f"Adjusted Rand score sklearn: {result['adjusted_rand_sklearn']}")
            print(f"Execution time 1: {result['time1']} seconds")
            print(f"Execution time 2: {result['time2']} seconds")
            print(f"Execution time sklearn: {result['time_sklearn']} seconds")
            print()
        
        avg_radius1 = np.mean([result["radius1"] for result in results_temp])
        avg_radius2 = np.mean([result["radius2"] for result in results_temp])
        avg_silhouette1 = np.mean([result["silhouette1"] for result in results_temp])
        avg_silhouette2 = np.mean([result["silhouette2"] for result in results_temp])
        avg_silhouette_sklearn = np.mean([result["silhouette_sklearn"] for result in results_temp])
        avg_adjusted_rand1 = np.mean([result["adjusted_rand1"] for result in results_temp])
        avg_adjusted_rand2 = np.mean([result["adjusted_rand2"] for result in results_temp])
        avg_adjusted_rand_sklearn = np.mean([result["adjusted_rand_sklearn"] for result in results_temp])
        avg_time1 = np.mean([result["time1"] for result in results_temp])
        avg_time2 = np.mean([result["time2"] for result in results_temp])
        avg_time_sklearn = np.mean([result["time_sklearn"] for result in results_temp])
        
        std_radius1 = np.std([result["radius1"] for result in results_temp])
        std_radius2 = np.std([result["radius2"] for result in results_temp])
        std_silhouette1 = np.std([result["silhouette1"] for result in results_temp])
        std_silhouette2 = np.std([result["silhouette2"] for result in results_temp])
        std_silhouette_sklearn = np.std([result["silhouette_sklearn"] for result in results_temp])
        std_adjusted_rand1 = np.std([result["adjusted_rand1"] for result in results_temp])
        std_adjusted_rand2 = np.std([result["adjusted_rand2"] for result in results_temp])
        std_adjusted_rand_sklearn = np.std([result["adjusted_rand_sklearn"] for result in results_temp])
        std_time1 = np.std([result["time1"] for result in results_temp])
        std_time2 = np.std([result["time2"] for result in results_temp])
        std_time_sklearn = np.std([result["time_sklearn"] for result in results_temp])
        
        print(f"Average results for Dataset {i+1}:")
        print(f"approx_k_centers1 average radius: {avg_radius1} (std: {std_radius1})")
        print(f"approx_k_centers2 average radius: {avg_radius2} (std: {std_radius2})")
        print(f"approx_k_centers1 average silhouette score: {avg_silhouette1} (std: {std_silhouette1})")
        print(f"approx_k_centers2 average silhouette score: {avg_silhouette2} (std: {std_silhouette2})")
        print(f"sklearn KMeans average silhouette score: {avg_silhouette_sklearn} (std: {std_silhouette_sklearn})")
        print(f"approx_k_centers1 average adjusted Rand score: {avg_adjusted_rand1} (std: {std_adjusted_rand1})")
        print(f"approx_k_centers2 average adjusted Rand score: {avg_adjusted_rand2} (std: {std_adjusted_rand2})")
        print(f"sklearn KMeans average adjusted Rand score: {avg_adjusted_rand_sklearn} (std: {std_adjusted_rand_sklearn})")
        print(f"approx_k_centers1 average execution time: {avg_time1} seconds (std: {std_time1})")
        print(f"approx_k_centers2 average execution time: {avg_time2} seconds (std: {std_time2})")
        print(f"sklearn KMeans average execution time: {avg_time_sklearn} seconds (std: {std_time_sklearn})")
        print()
        
        results.append({
            "dataset": i + 1,
            "n_clusters": my_kmeans.n_clusters,
            "avg_radius1": avg_radius1,
            "std_radius1": std_radius1,
            "avg_radius2": avg_radius2,
            "std_radius2": std_radius2,
            "avg_silhouette1": avg_silhouette1,
            "std_silhouette1": std_silhouette1,
            "avg_silhouette2": avg_silhouette2,
            "std_silhouette2": std_silhouette2,
            "avg_silhouette_sklearn": avg_silhouette_sklearn,
            "std_silhouette_sklearn": std_silhouette_sklearn,
            "avg_adjusted_rand1": avg_adjusted_rand1,
            "std_adjusted_rand1": std_adjusted_rand1,
            "avg_adjusted_rand2": avg_adjusted_rand2,
            "std_adjusted_rand2": std_adjusted_rand2,
            "avg_adjusted_rand_sklearn": avg_adjusted_rand_sklearn,
            "std_adjusted_rand_sklearn": std_adjusted_rand_sklearn,
            "avg_time1": avg_time1,
            "std_time1": std_time1,
            "avg_time2": avg_time2,
            "std_time2": std_time2,
            "avg_time_sklearn": avg_time_sklearn,
            "std_time_sklearn": std_time_sklearn
        })
        
        if i >= 20:
            continue
        
        # Opcionalmente, plotar os resultados
        plt.figure(figsize=(8, 4))
        plt.scatter(X[:, 0], X[:, 1], s=10, label='Data points')
        plt.scatter([c[0] for c in centers1], [c[1] for c in centers1], s=100, c='red', label='Centers1')
        plt.scatter([c[0] for c in centers2], [c[1] for c in centers2], s=100, c='blue', label='Centers2')
        plt.scatter([c[0] for c in centers_sklearn], [c[1] for c in centers_sklearn], s=100, c='green', label='Sklearn Centers')
        plt.title(f"Dataset {i+1}")
        plt.legend()
        plt.show(block=False)
        
        # Salva os plots em um arquivo
        plt.savefig(f"./data/plots/dataset_{i+1}_p_{p}.png")
        
        plt.pause(0.5)
        plt.close("all")
        
    results_df = pd.DataFrame(results)

    # Exportar os resultados para um arquivo CSV
    results_df.to_csv(f"results_p{p}.csv", index=False)
    # # Group the results by dataset and algorithm
    # grouped_df = results_df.groupby(["dataset", "algorithm"])

    # # Calculate the mean and standard deviation for each metric
    # mean_df = grouped_df.mean()
    # std_df = grouped_df.std()

    # # Print the aggregated results
    # print("Aggregated Results:")
    # print(mean_df)
    # print(std_df)
