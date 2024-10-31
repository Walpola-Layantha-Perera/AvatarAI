from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Check the vocabulary size
vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")

# Example to check if any token ID is out of range
sample_ids = [0, 1, vocab_size, vocab_size + 1]  # Include an out-of-range ID
for token_id in sample_ids:
    if token_id >= vocab_size:
        print(f"Token ID {token_id} is out of range")
