from datasets import load_from_disk

# Load the dataset
tokenized_datasets = load_from_disk('path_to_your_tokenized_datasets')

# Print dataset columns
print("Dataset columns:", tokenized_datasets.column_names)

# Print some rows to understand its content
print("Sample rows:", tokenized_datasets[:2])
