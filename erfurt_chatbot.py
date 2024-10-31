import streamlit as st
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

# Suppress warnings related to image extension and storage deprecation
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="Failed to load image Python extension")

# Load the dataset from the JSON file
dataset = load_dataset('json', data_files='erfurt-data.json')


# Extract information from the dataset
def get_erfurt_context():
    # Ensure the dataset is loaded and not empty
    if 'train' in dataset and len(dataset['train']) > 0:
        context = " ".join(dataset['train']['context'])
        return context
    else:
        return "No context available."


# Load the tokenizer and model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Streamlit application setup
st.title("Erfurt City Chatbot")
st.write("Ask me anything about the city of Erfurt!")

# Text input for user query
user_input = st.text_input("You:", "")

# Button to submit query
if st.button("Send"):
    if user_input:
        # Retrieve context from dataset and respond
        erfurt_context = get_erfurt_context()

        # Prepare the input
        input_text = f"{erfurt_context}\n\nUser: {user_input}\nChatbot:"

        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024, padding=True,
                           return_attention_mask=True)

        # Debugging: Print tokenized inputs
        st.write(f"Tokenized Inputs: {inputs}")

        # Generate response using the model
        response = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=1500,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # Decode the response and clean up the text
        chatbot_reply = tokenizer.decode(response[0], skip_special_tokens=True)
        chatbot_reply = chatbot_reply.replace(input_text, "").strip()  # Remove input text from the reply

        # Debugging: Print generated response
        st.write(f"Generated Response: {chatbot_reply}")

        # Display the chatbot's response
        st.text_area("Chatbot:", value=chatbot_reply, height=100)
    else:
        st.write("Please enter a question about Erfurt.")
