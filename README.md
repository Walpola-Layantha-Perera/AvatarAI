Erfurt City AI Avatar System
------------------------------

Multimodal LLM and ASR with NVIDIA NeMo
Welcome to the Erfurt City AI Avatar System! This repository is designed to showcase the power of NVIDIA NeMo, a conversational AI framework, to create a multimodal chatbot capable of understanding both text and voice inputs and generating responsive, dynamic visual avatars.

Project Overview
The Erfurt City Chatbot project is a comprehensive AI chatbot that integrates Large Language Models (LLM) and Automatic Speech Recognition (ASR) capabilities, leveraging NVIDIA NeMo to:

Train and fine-tune domain-specific language models for accurate, context-driven responses.
Enable high-quality ASR for voice input and real-time conversation.
Generate on-the-go avatar animations to provide a visual, interactive experience.
This bot specializes in providing information about the city of Erfurt, handling inquiries about historical landmarks, cultural sites, and other areas of interest.

Features
Text and Voice Interaction: Interact with the chatbot through text input or speech recognition.
Dynamic Avatar Animations: Visual responses using animated avatars that change based on the chatbot’s response.
Domain-Specific Understanding: Enhanced with custom-trained NeMo models on Erfurt-related datasets for contextually accurate responses.
Tech Stack
NVIDIA NeMo: For LLM fine-tuning and ASR model development.
Transformers: Leveraging Hugging Face models for foundational LLM capabilities.
Streamlit: A fast and easy interface for interacting with the chatbot.
Pygame & gTTS: For playing voice responses and generating spoken outputs.
Setup and Installation

1. Prerequisites: 
Python 3.8 or higher
NVIDIA GPU with CUDA support (recommended for NeMo)
NVIDIA NeMo: pip install nemo_toolkit['all']
2. Clone the repository: 
   git clone https://github.com/Walpola-Layantha-Perera/AvatarAI


4. Install required packages:
pip install -r requirements.txt

5. Setting Up NVIDIA NeMo models: 
To enable NeMo for both text and voice handling, set up pre-trained models in the following categories:
ASR Model: Download a NeMo ASR model or fine-tune on your domain-specific dataset if available.
Language Model: Download or fine-tune an existing model using domain-specific data.
Example setup:

python
from nemo.collections.asr.models import ASRModel
from nemo.collections.nlp.models.language_modeling import TransformerLMModel

asr_model = ASRModel.from_pretrained(model_name="stt_en_conformer_ctc_large")
nlp_model = TransformerLMModel.from_pretrained(model_name="nemo_gpt")

5. Dataset preparation: 
To customize the chatbot’s responses and ASR accuracy:

Text dataset: Curate text data relevant to the city of Erfurt for LLM fine-tuning.
Audio Dataset: Collect domain-specific audio samples to train or fine-tune the ASR model.
Use NeMo Curator to preprocess and manage your dataset for training.

6. Run the chatbot: 
Start the chatbot by running the main Streamlit app: 

streamlit run app.py
Usage
Select Input Mode: Choose between “Type your question” or “Use voice command.”
Ask a Question: If voice mode is selected, speak your question; in text mode, type it directly.
View Responses: The chatbot will respond via text and play an audio response. Visual avatar animations may play for specific responses.
Project Structure
app.py: Main application logic
data/: Dataset for training and fine-tuning
models/: Saved models and NeMo configurations
responses/: Audio responses generated by gTTS
avatars/: Visual assets and avatar animations
README.md: Project documentation
Challenges: 
Dataset Quality: 
Domain-specific text and audio data are crucial for high accuracy. Building a quality dataset for ASR and LLM training with NeMo can be time-intensive but is essential for optimal chatbot performance.

Infrastructure: 
Deploying NeMo-based models requires GPU support, especially for real-time ASR and LLM inference. Cloud GPU services like NVIDIA’s NGC are recommended for larger scale deployments.

Customization and Fine-Tuning: 
Tuning NeMo models for specific domain terminologies and accents can improve accuracy but requires a specialized dataset and training time.

Contributions: 
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request for review.

License: 
This project is licensed under the MIT License.

Acknowledgments: 
NVIDIA NeMo: For providing the AI framework and tools used in this project.
Hugging Face: For LLM resources.
City of Erfurt: For inspiring this project.

Enjoy building and interacting with the Erfurt AI Avatar System!
