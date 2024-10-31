import os
import streamlit as st
import webbrowser
import http.server
import socketserver
import threading
import time
import uuid  # To generate unique filenames

from transformers import pipeline
from gtts import gTTS
import pygame  # For audio playback
import speech_recognition as sr  # For voice commands

# NeMo Imports
import nemo
import nemo.collections.asr as nemo_asr  # Automatic Speech Recognition
import nemo.collections.nlp as nemo_nlp  # NLP capabilities
import nemo.collections.tts as nemo_tts  # Text-to-Speech

# Initialize NeMo Core
nemo_core = nemo.core.NeMoCore()


# Function to create an HTML file with an embedded video
def create_html_file(video_path, html_file_path):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chatbot Video</title>
        <style>
            body {{
                background-color: #f0f0f0;
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100vh;
            }}
            #video-container {{
                width: 50vw;
                height: 50vh;
                background-color: #fff;
                border: 2px solid #ccc;
            }}
            video {{
                width: 100%;
                height: 100%;
            }}
        </style>
    </head>
    <body>
        <div id="video-container">
            <video id="chatbot-video" autoplay loop muted playsinline>
                <source src="{video_path}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
    </body>
    </html>
    """
    with open(html_file_path, 'w') as file:
        file.write(html_content)


# NeMo Functions: ASR and TTS Model Initialization
def initialize_nemo_models():
    # ASR Model for speech recognition (transcribing voice input)
    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="stt_en_conformer_ctc_large")

    # TTS Model for generating audio from text
    tts_model = nemo_tts.models.Tacotron2Model.from_pretrained(model_name="tts_en_tacotron2")

    return asr_model, tts_model


# Function to capture voice input using NeMo ASR model
def capture_voice_input(asr_model):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening for your question...")
        audio = r.listen(source)
    try:
        # Use ASR model to transcribe the audio
        user_query = asr_model.transcribe([audio.get_wav_data()])[0]  # Get first result from batch
        st.write(f"Your question: {user_query}")
        return user_query
    except Exception as e:
        st.write(f"Error with ASR model: {e}")
    return ""


# Text-to-Speech function using NeMo TTS model
def speak_text(tts_model, text):
    # Generate a unique filename for the response audio
    response_file = os.path.join("responses", f"response_{uuid.uuid4()}.wav")
    if not os.path.exists("responses"):
        os.makedirs("responses")

    # Convert text to speech
    tts_audio = tts_model.infer_spectrogram(text)
    tts_model.save_audio(tts_audio, response_file)

    # Play the audio using pygame
    pygame.mixer.init()
    pygame.mixer.music.load(response_file)
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    try:
        os.remove(response_file)
    except Exception as e:
        print(f"Error deleting file: {e}")


# Load the model for question-answering
def initialize_qa_model():
    # Custom pipeline using Hugging Face Transformers
    return pipeline("question-answering", model="deepset/roberta-base-squad2")


# Streamlit Interface
st.title("Erfurt City Chatbot")
st.write("Ask me anything about Erfurt city!")
input_mode = st.radio("Choose your input method", ("Type your question", "Use voice command"))

# Load or initialize models
qa_model = initialize_qa_model()
asr_model, tts_model = initialize_nemo_models()

# Checkbox for audio control
play_audio = st.checkbox("Unmute Audio", value=True)

# Main Application Loop
while True:
    # Capture input
    if input_mode == "Type your question":
        user_question = st.text_input("You:", "")
    else:
        user_question = capture_voice_input(asr_model)

    if user_question:
        # Answer generation using QA model
        result = qa_model(question=user_question, context="Erfurt city is known for...")
        answer = result['answer']
        st.write("Chatbot:", answer)

        # Play the response as speech if unmuted
        if play_audio:
            speak_text(tts_model, answer)

        # Example video paths
        video_path = 'avatars/Avatar-1.mp4'
        html_file_path = 'video_player.html'

        # Play animation if answer contains specific keywords
        if answer in ["landmarks", "cultural scene", "church"]:
            create_html_file(video_path, html_file_path)

            # Start a local HTTP server
            server_thread = threading.Thread(target=start_server, args=(os.getcwd(),))
            server_thread.start()
            time.sleep(1)

            # Open HTML video in a browser
            open_browser(f'http://localhost:8000/{html_file_path}')
