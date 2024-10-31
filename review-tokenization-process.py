from transformers import AutoTokenizer
from datasets import Dataset

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Add a pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})

# Set the pad token (must be a string)
tokenizer.pad_token = '<PAD>'

# Define the tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# Example data
data = {
    'text': ["Example sentence 1", "Example sentence 2"]
}

# Create dataset
dataset = Dataset.from_dict(data)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Save the tokenized dataset
tokenized_datasets.save_to_disk('path_to_your_tokenized_datasets')

# Save the updated tokenizer
tokenizer.save_pretrained('path_to_your_updated_tokenizer')
