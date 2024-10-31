from datasets import load_from_disk

# Load the tokenized dataset from the correct path
tokenized_datasets = load_from_disk('tokenized_datasets')

# Print the dataset structure
print(tokenized_datasets)
