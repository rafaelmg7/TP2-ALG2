# TP2-ALG2
Este repositório contém a implementação de algoritmos de clustering utilizando métodos de K-Means e K-Centers. A implementação inclui a comparação entre diferentes variações dos algoritmos, bem como a avaliação de suas performances em datasets sintéticos e reais. O projeto é organizado de forma modular, com separação clara entre processamento de dados, execução dos algoritmos e análise de resultados.

## Estrutura do Repositório

- `data/`
  - `sinteticos.py`: Script contendo a geração e manipulação de datasets sintéticos para testes dos algoritmos.
  - `reais.py`: Script contendo a manipulação de datasets reais para aplicação dos algoritmos.
  - `plots/`: Diretório que armazena os gráficos gerados durante a execução dos algoritmos.
- `main.py`: Script principal que executa os algoritmos de clustering nos datasets definidos e salva os resultados das métricas e gráficos.
- `results_p1.csv`: Arquivo CSV contendo as métricas de avaliação de todos os datasets usando `p1`.
- `results_p2.csv`: Arquivo CSV contendo as métricas de avaliação de todos os datasets usando `p2`.

## Requisitos

Este projeto requer a instalação dos seguintes pacotes Python:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Você pode instalar os pacotes utilizando o `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn