from datasets import load_from_disk

# Path to the directory where the tokenized dataset is saved
tokenized_dataset_path = 'path_to_your_tokenized_datasets'

# Load the tokenized dataset
tokenized_datasets = load_from_disk(tokenized_dataset_path)

# Print the dataset structure
print(tokenized_datasets)
print(tokenized_datasets.keys())
