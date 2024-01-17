# Install dependencies

# Linux
# sudo apt update && sudo apt install ffmpeg

# MacOS
# brew install ffmpeg

# Windows
# chco install ffmpeg

# Installing pytorch
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Installing Whisper
# pip install git+https://github.com/openai/whisper.git -q

# pip install streamlit
import streamlit as st
import whisper
import tempfile
import os

st.title("AI Audio Transcribe")

# upload audio file with streamlit
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

model = whisper.load_model("base")
st.text("Transcribe loading")

if st.sidebar.button("Transcribe Audio"):
    if audio_file is not None:
        st.sidebar.success("Transcribing Audio")

        # Save the uploaded audio file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
            temp_audio.write(audio_file.read())
            audio_path = temp_audio.name

        # Transcribe audio
        transcription = model.transcribe(audio_path)

        # Remove the temporary audio file
        os.remove(audio_path)

        st.sidebar.success("Transcription Complete")
        st.markdown(transcription["text"])
    else:
        st.sidebar.error("Please upload an audio file")

st.sidebar.header("Play Original Audio File")
st.sidebar.audio(audio_file)

