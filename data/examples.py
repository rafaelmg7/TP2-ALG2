import pandas as pd
from ucimlrepo import fetch_ucirepo

datasets_info = {
    "raisin": 850,
    "wine_quality": 186,
    "spambase": 94,
    "statlog_vehicle_silhouettes": 149,
    "banknote_authentication": 267,
    "pen_based_recognition_of_handwritten_digits ": 81,
    "optical_recognition_of_handwritten_digits ": 80,
    "magic_gamma_telescope ": 159,
    'waveform_database_generator_version_1': 107,
    "statlog_image_segmentation": 147,
    "cardiotocography ": 193,
}

datasets = {}
for name, dataset_id in datasets_info.items():
    datasets[name] = fetch_ucirepo(id=dataset_id)
    print(f"Loaded dataset: {name}")
    
print("Datasets loaded.")

for name, dataset in datasets.items():
    df = dataset.data.features
    columns = dataset.data.targets.columns
    print(f"Exporting {name} dataset...")
    print(dataset.data.targets)

    class_column = next((col for col in columns if col.lower() == 'class'), None)
    label_column = next((col for col in columns if col.lower() == 'label'), None)
    moreThanOneTarget = len(columns) > 1

    if (class_column or label_column) or not moreThanOneTarget:
        unique_count = dataset.data.targets[columns[0]].nunique()
        df['label'] = dataset.data.targets[columns[0]]

        print(df['label'])
        df['label'], _ = pd.factorize(df['label'])
        
        # Insere o número de clusters na primeira linha do DataFrame
        df.loc[-1] = [unique_count] * len(df.columns) 
        df.index = df.index + 1
        df = df.sort_index()
        
        df = df.head(2500)
            
        # Exporta o DataFrame para um arquivo CSV
        df.to_csv(f'/home/rafaelmg/Documents/ALG2/TP2/data/real_datasets/{name}_data.csv', index=False)
        print(f"Exported {name}_data.csv")
    else:
        print(f"{name} dataset invalid.")
