import os
import pandas as pd

real_datasets = []
nums_clusters = []

# Carregar os datasets reais
for file in os.listdir("/home/rafaelmg/Documents/ALG2/TP2/data/real_datasets"):
    if file.endswith(".csv"):
        df = pd.read_csv(f"/home/rafaelmg/Documents/ALG2/TP2/data/real_datasets/{file}")
        real_datasets.append((df.iloc[1:, :-1].values, df.iloc[1:, -1].values))
        nums_clusters.append(df.iloc[0, 0])
