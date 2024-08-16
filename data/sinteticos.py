import numpy as np
from sklearn import datasets
import pandas as pd

n_samples = 700
seed = 42

# Geração dos datasets sintéticos
noisy_circles_1 = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
noisy_moons_1 = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
noisy_moons_2 = datasets.make_moons(n_samples=n_samples, noise=0.1, random_state=seed + 1)
blobs_1 = datasets.make_blobs(n_samples=n_samples, random_state=seed)
blobs_2 = datasets.make_blobs(n_samples=n_samples, centers=4, random_state=seed + 1)
rng = np.random.RandomState(seed)
no_structure_1 = rng.rand(n_samples, 2), rng.randint(0, 2, n_samples)

# Dados anisotrópicos
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso_1 = np.dot(X, transformation)
aniso_1 = (X_aniso_1, y)

X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state + 1)
transformation = [[0.8, -0.2], [-0.6, 0.6]]
X_aniso_2 = np.dot(X, transformation)
aniso_2 = (X_aniso_2, y)

# Blobs com variâncias variadas
varied_1 = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
varied_2 = datasets.make_blobs(n_samples=n_samples, cluster_std=[0.5, 3.0, 1.5], random_state=random_state + 1)

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

# Gerar mais datasets sintéticos
np.random.seed(seed)
for i in range(10):
    n_clusters = np.random.randint(2, 6)
    centers = np.random.uniform(-10, 10, (n_clusters, 2))
    covariances = []

    for _ in range(n_clusters):
        std_dev = np.random.uniform(0.1, 2.0)
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

# Salvar os datasets sintéticos em arquivos CSV
for i, (X, y) in enumerate(datasets):
    df = pd.DataFrame(X, columns=["x", "y"])
    df["label"] = y
    df.to_csv(f"/home/rafaelmg/Documents/ALG2/TP2/data/synth_datasets/synthetic_dataset_{i+1}.csv", index=False)
