import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from io import BytesIO
import soundfile as sf

# Set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model and processor
model_id = "openai/whisper-large-v3"

@st.cache_resource
def load_model():
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

pipe = load_model()

# Streamlit app UI
st.title("Speech-to-Text with Whisper")

st.write("Upload an audio file, and the app will transcribe it using the Whisper model.")

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
