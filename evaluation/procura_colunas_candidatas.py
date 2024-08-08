import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV
data = pd.read_csv('raisin_data.csv', header=0)

# Ler o número de clusters da segunda linha (o número de clusters está na primeira coluna do dataframe)
n_clusters = data.iloc[0, 0]
n_clusters = int(n_clusters)  # Converter para inteiro

# Ajustar os dados para ignorar a linha com o número de clusters
data.columns = data.iloc[0]  # Usar a primeira linha como nomes das colunas
data = data.drop(data.index[0])  # Remover a primeira linha
data = data.reset_index(drop=True)  # Resetar os índices
data = data.apply(pd.to_numeric)  # Converter todas as colunas para valores numéricos

# Examinar as primeiras linhas do DataFrame
print("Dados:\n", data.head())

# Calcular a variância de cada coluna numérica
variances = data.var()

# Calcular o intervalo de cada coluna numérica
ranges = data.max() - data.min()

# Exibir as variâncias e intervalos
print("\nVariâncias das colunas numéricas:\n", variances)
print("\nIntervalos das colunas numéricas:\n", ranges)

# Plotar a distribuição dos dados para cada coluna
data.hist(bins=20, figsize=(10, 8))
plt.suptitle("Distribuição dos Dados")
plt.show()

# Selecionar as colunas com variâncias e intervalos mais semelhantes
# Remover os thresholds e inspecionar todas as combinações de colunas
candidate_columns = []
for col1 in data.columns:
    for col2 in data.columns:
        if col1 != col2:
            variance_diff = abs(variances[col1] - variances[col2])
            range_diff = abs(ranges[col1] - ranges[col2])
            # Selecionar pares com variação semelhante e valores comparáveis
            if variance_diff < variances.mean() and range_diff < ranges.mean():
                candidate_columns.append((col1, col2))

# Exibir pares de colunas candidatas
print("\nPares de colunas candidatas para coordenadas (x, y):\n", candidate_columns)

# Plotar as primeiras duas colunas candidatas (se houver)
if candidate_columns:
    x_col, y_col = candidate_columns[0]
    x = data[x_col].values
    y = data[y_col].values

    plt.scatter(x, y)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Plot das colunas {x_col} vs {y_col}")
    plt.show()
else:
    print("Nenhuma coluna candidata encontrada.")
