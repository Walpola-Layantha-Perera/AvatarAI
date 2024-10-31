from datasets import load_dataset

# Load the dataset
dataset_path = 'C:/Users/laya4/Documents/LLM - Acer Laptop/Hugging-Face-Code/erfurt-data.json'
dataset = load_dataset('json', data_files={'train': dataset_path})

# Split the dataset
split = dataset['train'].train_test_split(test_size=0.1)  # Adjust test_size as needed
train_dataset = split['train']
eval_dataset = split['test']

# Save the split dataset if needed
train_dataset.save_to_disk('train_dataset')
eval_dataset.save_to_disk('eval_dataset')

# If you need to load them later
# train_dataset = load_from_disk('train_dataset')
# eval_dataset = load_from_disk('eval_dataset')
