from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Define your label mappings
label2id = {
    'positive': 0,
    'negative': 1,
    'neutral': 2
}

id2label = {v: k for k, v in label2id.items()}

# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


def preprocess_function(examples):
    contexts = [str(context) for context in examples['context']]
    questions = [str(question) for question in examples['question']]

    texts = [context + " " + question for context, question in zip(contexts, questions)]

    # Tokenize the texts
    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

    # Handle empty or invalid input_ids
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    token_type_ids = encodings.get('token_type_ids', [None] * len(input_ids))

    # Check for any empty input_ids
    if any(len(ids) == 0 for ids in input_ids):
        print("Warning: Some input_ids are empty")

    # Convert labels to numerical values
    labels = [label2id.get(answer, -1) for answer in examples['answer']]

    # Filter out invalid labels
    valid_indices = [i for i, label in enumerate(labels) if label != -1]
    if valid_indices:
        encodings = {k: [v[i] for i in valid_indices] for k, v in encodings.items()}
        encodings['labels'] = [labels[i] for i in valid_indices]
    else:
        encodings = {k: [] for k in encodings.keys()}
        encodings['labels'] = []

    return encodings


# Load and prepare the dataset
dataset = load_dataset('json', data_files={'train': 'erfurt-data.json'})  # Adjust the path as needed

# Tokenize the dataset
train_dataset = dataset['train'].map(preprocess_function, batched=True)
# val_dataset = dataset['validation'].map(preprocess_function, batched=True)  # If you have a validation set

# Initialize the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label2id))
model.config.label2id = label2id
model.config.id2label = id2label

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=val_dataset,  # If you have a validation dataset
    tokenizer=tokenizer,
)

# Start training
trainer.train()
