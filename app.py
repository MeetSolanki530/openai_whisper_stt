# streamlit_whisper.py
import streamlit as st
import whisper
import tempfile
import os

# --- Load Whisper model ---
st.title("Whisper STT with Streamlit")
model = whisper.load_model("base")  # or "tiny", "small", etc.

# --- Upload audio file ---
uploaded_file = st.file_uploader("Upload an audio file (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    st.audio(tmp_file_path, format='audio/wav')

    # --- Transcribe audio ---
    st.info("Transcribing audio...")
    result = model.transcribe(tmp_file_path)  # Force Gujarati

    st.success("Transcription complete!")
    st.text_area("Transcribed Text:", result["text"], height=300)

    # Remove temp file
    os.remove(tmp_file_path)
