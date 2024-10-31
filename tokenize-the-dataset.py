from transformers import AutoTokenizer
from datasets import Dataset
import json

# Load the dataset
with open('erfurt-data.json', 'r') as f:
    data = json.load(f)

# Convert to a Dataset
dataset = Dataset.from_dict({
    "context": [item['context'] for item in data],
    "question": [item['question'] for item in data],
    "answer": [item['answer'] for item in data]
})

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Add a new padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Update the model with the new tokenizer (if needed)
# model.resize_token_embeddings(len(tokenizer))

# Define the tokenization function
# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize the inputs and outputs
    return tokenizer(examples['question'], padding='max_length', truncation=True, max_length=128)

# Apply the tokenization function
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Save the tokenized dataset
tokenized_datasets.save_to_disk('tokenized_datasets')

# Print a sample of the tokenized dataset
print(tokenized_datasets)
