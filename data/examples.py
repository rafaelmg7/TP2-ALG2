import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch datasets
datasets_info = {
    "raisin": 850,
    "wine_quality": 186,
    "spambase": 94,
    "estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition": 544,
    "energy_efficiency": 242,
    "concrete_compressive_strength": 165,
    "image_segmentation": 50,
    "statlog_vehicle_silhouettes": 149,
    "banknote_authentication": 267,
    "absenteeism_at_work": 445,
    "pen_based_recognition_of_handwritten_digits ": 81,
    "optical_recognition_of_handwritten_digits ": 80,
    "magic_gamma_telescope ": 159,
    "isolet ": 54,
    "statlog_vehicle_silhouettes ": 149,
    # "statlog_shuttle ": 148,
    "cardiotocography ": 193
}

print("Fetching datasets...")
# Load datasets
# datasets = {name: fetch_ucirepo(id=dataset_id) for name, dataset_id in datasets_info.items()}
datasets = {}
for name, dataset_id in datasets_info.items():
    datasets[name] = fetch_ucirepo(id=dataset_id)
    print(f"Loaded dataset: {name}")
print("Datasets loaded.")
# Iterate over each dataset and export if 'class' or 'label' column is present
for name, dataset in datasets.items():
    df = dataset.data.features
    columns = dataset.data.targets.columns
    print(f"Exporting {name} dataset...")

    # Check if 'class' or 'label' columns exist (case insensitive)
    class_column = next((col for col in columns if col.lower() == 'class'), None)
    label_column = next((col for col in columns if col.lower() == 'label'), None)

    if class_column or label_column:
        # Determine the unique count
        if class_column:
            unique_count = dataset.data.targets[class_column].nunique()
        else:
            unique_count = dataset.data.targets[label_column].nunique()

        # Insert the unique count as the first row
        df.loc[-1] = [unique_count] * len(df.columns)  # Add a new row with unique_count
        df.index = df.index + 1  # Shift the index
        df = df.sort_index()  # Sort by index to place the new row at the top

        # Export the DataFrame to a CSV file named after the dataset
        df.to_csv(f'{name}_data.csv', index=False)
        print(f"Exported {name}_data.csv")
    else:
        print(f"No 'class' or 'label' column found in {name} dataset.")
