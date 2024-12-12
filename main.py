import torch
import streamlit as st
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Set up the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the model and processor
model_id = "openai/whisper-large"

@st.cache_resource
def load_model():
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        torch_dtype=torch_dtype,
    )
    return pipe

pipe = load_model()

# Streamlit UI elements
st.title("Speech-to-Text with Whisper")
st.write("Upload an audio file for transcription:")

# File uploader for audio files
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "flac"])

if uploaded_file is not None:
    # Display audio player for the uploaded file
    st.audio(uploaded_file)

    # Process the uploaded file
    audio = uploaded_file.read()
    result = pipe(audio)
    
    # Display the transcribed text
    st.subheader("Transcription:")
    st.write(result["text"])
