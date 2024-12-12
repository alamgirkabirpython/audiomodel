import streamlit as st
from transformers import pipeline, VitsModel, AutoTokenizer
from pydub import AudioSegment
import torch
import numpy as np
import os
import time
import yt_dlp
import tempfile

# Load pipeline
@st.cache_resource
def load_pipeline():
    transcriber = pipeline(
        "automatic-speech-recognition", 
        model="BELLE-2/Belle-whisper-large-v3-turbo-zh"
    )
    transcriber.model.config.forced_decoder_ids = (
        transcriber.tokenizer.get_decoder_prompt_ids(
            language="zh", 
            task="transcribe"
        )
    )
    return transcriber

pipe = load_pipeline()

# Streamlit app UI
st.title("Speech-to-Text with Belle Whisper")

st.write("Upload an audio file, and the app will transcribe it using the Belle Whisper model.")

audio_file = st.file_uploader("Choose an audio file (e.g., .wav, .mp3)", type=["wav", "mp3"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    # Convert audio file to wav if necessary
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    audio = AudioSegment.from_file(audio_file)
    audio.export(audio_path, format="wav")

    # Read audio file
    audio_data = AudioSegment.from_wav(audio_path)
    samples = np.array(audio_data.get_array_of_samples())
    sample_rate = audio_data.frame_rate

    # Process audio file
    st.write("Processing the audio file...")
    result = pipe({"array": samples, "sampling_rate": sample_rate})

    # Display transcription
    st.subheader("Transcription:")
    st.write(result["text"])

# Optional: Process YouTube video audio
url = st.text_input("Enter a YouTube video URL to transcribe its audio:")
if url:
    with st.spinner("Downloading and processing audio..."):
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = ydl_opts['outtmpl']

        audio_data = AudioSegment.from_wav(audio_path)
        samples = np.array(audio_data.get_array_of_samples())
        sample_rate = audio_data.frame_rate

        result = pipe({"array": samples, "sampling_rate": sample_rate})

        st.subheader("Transcription for YouTube audio:")
        st.write(result["text"])
