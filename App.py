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

# Specify the full path to ffmpeg executable
ffmpeg_path = "/opt/homebrew/bin/ffmpeg"  # Replace with the actual path to ffmpeg

# Set the path to ffmpeg in the model
whisper.FFMPEG_PATH = ffmpeg_path

# Load the Whisper model
model = whisper.load_model("base")

# Streamlit App Title
st.title("AI Audio Transcribe")

# Upload audio file with Streamlit
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

# Display a message while the transcription is loading
st.text("Transcription loading...")

# Sidebar button to trigger transcription
if st.sidebar.button("Transcribe Audio"):
    # Check if an audio file is uploaded
    if audio_file is not None:
        st.sidebar.success("Transcribing Audio")

        # Save the uploaded audio file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
            temp_audio.write(audio_file.read())
            audio_path = temp_audio.name

        try:
            # Transcribe audio
            transcription = model.transcribe(audio_path)

            # Display the transcription
            st.sidebar.success("Transcription Complete")
            st.markdown(transcription["text"])

        except Exception as e:
            st.sidebar.error(f"Error during transcription: {str(e)}")

        finally:
            # Remove the temporary audio file
            os.remove(audio_path)

    else:
        st.sidebar.error("Please upload an audio file")

# Sidebar to play the original audio file
st.sidebar.header("Play Original Audio File")
st.sidebar.audio(audio_file)

# Run the Streamlit app
# Run the Streamlit app
# Run the Streamlit app
if __name__ == "__main__":
    st.run()

