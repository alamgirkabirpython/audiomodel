import streamlit as st
from transformers import pipeline
from datasets import load_dataset
from io import BytesIO
import soundfile as sf

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

    # Read audio file
    with BytesIO(audio_file.read()) as audio_buffer:
        audio_data, sample_rate = sf.read(audio_buffer)
        
    # Process audio file
    st.write("Processing the audio file...")
    result = pipe({"array": audio_data, "sampling_rate": sample_rate})

    # Display transcription
    st.subheader("Transcription:")
    st.write(result["text"])

# Optional sample processing
if st.button("Try a sample audio"):
    st.write("Using a sample from the LibriSpeech dataset...")
    dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    sample = dataset[0]["audio"]
    
    # Process sample
    result = pipe(sample)
    st.subheader("Transcription for sample audio:")
    st.write(result["text"])
