import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import pandas as pd

from data.sinteticos import datasets as synthetic_datasets
from data.reais import real_datasets, nums_clusters
from algorithms.kmeans import MyKMeans
from scipy.spatial.distance import cdist

# Combina os datasets sintéticos e reais
datasets = synthetic_datasets + real_datasets

for p in [1, 2]:
    results = []
    for i, (X, y) in enumerate(datasets):
        
        # Se o dataset for um dos reais, pegar o número de clusters do arquivo
        if i >= len(synthetic_datasets):
            n = int(nums_clusters[i - len(synthetic_datasets)])
        else:
            n = 3
            
        my_kmeans = MyKMeans(n_clusters=n, tol=1e-4, p=p)

        X = StandardScaler().fit_transform(X)

        dist_matrix = my_kmeans.precompute_dist_matrix(X)

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
            
            distances_to_centers = np.min(cdist(X, centers_sklearn), axis=1)
            r_C = np.max(distances_to_centers)
            result["radius_sklearn"] = r_C

            results_temp.append(result)

        # Média e desvio padrão dos resultados
        avg_radius1 = np.mean([result["radius1"] for result in results_temp])
        avg_radius2 = np.mean([result["radius2"] for result in results_temp])
        avg_radius_sklearn = np.mean([result["radius_sklearn"] for result in results_temp])
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
        std_radius_sklearn = np.std([result["radius_sklearn"] for result in results_temp])
        std_silhouette1 = np.std([result["silhouette1"] for result in results_temp])
        std_silhouette2 = np.std([result["silhouette2"] for result in results_temp])
        std_silhouette_sklearn = np.std([result["silhouette_sklearn"] for result in results_temp])
        std_adjusted_rand1 = np.std([result["adjusted_rand1"] for result in results_temp])
        std_adjusted_rand2 = np.std([result["adjusted_rand2"] for result in results_temp])
        std_adjusted_rand_sklearn = np.std([result["adjusted_rand_sklearn"] for result in results_temp])
        std_time1 = np.std([result["time1"] for result in results_temp])
        std_time2 = np.std([result["time2"] for result in results_temp])
        std_time_sklearn = np.std([result["time_sklearn"] for result in results_temp])
        
        # Impressão dos resultados
        print(f"Average results for dataset {i+1} with p={p}:")
        print(f" - Method 1: Avg Radius = {avg_radius1:.4f}, Avg Silhouette = {avg_silhouette1:.4f}, Avg Adjusted Rand = {avg_adjusted_rand1:.4f}, Avg Time = {avg_time1:.4f}s")
        print(f" - Method 2: Avg Radius = {avg_radius2:.4f}, Avg Silhouette = {avg_silhouette2:.4f}, Avg Adjusted Rand = {avg_adjusted_rand2:.4f}, Avg Time = {avg_time2:.4f}s")
        print(f" - Sklearn KMeans: Avg Radius = {avg_radius_sklearn:.4f}, Avg Silhouette = {avg_silhouette_sklearn:.4f}, Avg Adjusted Rand = {avg_adjusted_rand_sklearn:.4f}, Avg Time = {avg_time_sklearn:.4f}s")

        # Salvando os resultados no arquivo
        results.append({
            "dataset": i + 1,
            "n_clusters": my_kmeans.n_clusters,
            "avg_radius1": avg_radius1,
            "avg_radius2": avg_radius2,
            "avg_radius_sklearn": avg_radius_sklearn,
            "avg_radius_sklearn": avg_radius_sklearn,
            "avg_silhouette1": avg_silhouette1,
            "avg_silhouette2": avg_silhouette2,
            "avg_silhouette_sklearn": avg_silhouette_sklearn,
            "avg_adjusted_rand1": avg_adjusted_rand1,
            "avg_adjusted_rand2": avg_adjusted_rand2,
            "avg_adjusted_rand_sklearn": avg_adjusted_rand_sklearn,
            "avg_time1": avg_time1,
            "avg_time2": avg_time2,
            "avg_time_sklearn": avg_time_sklearn,
            "std_radius1": std_radius1,
            "std_radius2": std_radius2,
            "std_radius_sklearn": std_radius_sklearn,
            "std_radius_sklearn": std_radius_sklearn,
            "std_silhouette1": std_silhouette1,
            "std_silhouette2": std_silhouette2,
            "std_silhouette_sklearn": std_silhouette_sklearn,
            "std_adjusted_rand1": std_adjusted_rand1,
            "std_adjusted_rand2": std_adjusted_rand2,
            "std_adjusted_rand_sklearn": std_adjusted_rand_sklearn,
            "std_time1": std_time1,
            "std_time2": std_time2,
            "std_time_sklearn": std_time_sklearn,
        })
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results_p{p}.csv", index=False)

