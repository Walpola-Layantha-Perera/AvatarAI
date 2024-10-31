from datasets import Dataset
import json

# Load the dataset from the JSON file
with open('erfurt-data.json', 'r') as f:
    data = json.load(f)

# Print the first item to verify structure
print(data[0])

# Convert the data into a Hugging Face Dataset
# Use 'question' as input and 'answer' as output
dataset = Dataset.from_dict({
    "context": [item['context'] for item in data],
    "question": [item['question'] for item in data],
    "answer": [item['answer'] for item in data]
})

# Print dataset to verify
print(dataset)
