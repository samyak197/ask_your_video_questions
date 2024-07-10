import streamlit as st
import yt_dlp
import ffmpeg
import whisper
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Function to download a YouTube video
def download_youtube_video(url, output_path):
    ydl_opts = {
        "outtmpl": f"{output_path}/%(title)s.%(ext)s",
        "format": "bestvideo+bestaudio/best",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    st.success(f"Downloaded video from '{url}' successfully!")


# Function to convert video to audio
def convert_video_to_audio(video_path, audio_path):
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(
            stream, audio_path, format="wav", acodec="pcm_s16le", ac=1, ar="16000"
        )
        ffmpeg.run(stream, overwrite_output=True)
        st.success("Video converted to audio successfully!")
    except ffmpeg.Error as e:
        st.error("ffmpeg error:", e)


# Function to transcribe audio
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["segments"]


# Function to create DataFrame from transcription
def create_transcription_dataframe(transcription):
    data = [
        {
            "text": segment["text"],
            "start_time": segment["start"],
            "end_time": segment["end"],
        }
        for segment in transcription
    ]
    df = pd.DataFrame(data)
    return df


# Function to get timestamp for a query
def get_timestamp_for_query_bert(query, df, text_embeddings, model):
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], text_embeddings)
    idx = np.argmax(similarities)
    return df.iloc[idx]["start_time"], df.iloc[idx]["end_time"]


# Streamlit App
st.title("YouTube Video Transcription and Query")

# Input for YouTube video link or local video file
video_input = st.text_input("Enter YouTube video URL or local video file path:")

if st.button("Process Video"):
    if video_input:
        output_path = "./downloaded_video"
        if video_input.startswith("http"):
            download_youtube_video(video_input, output_path)
            video_path = output_path + "/video.mp4"
        else:
            video_path = video_input

        audio_path = "audio.wav"
        convert_video_to_audio(video_path, audio_path)
        transcription = transcribe_audio(audio_path)
        df = create_transcription_dataframe(transcription)
        st.dataframe(df)

        # Load pre-trained BERT model and create embeddings
        st.info("Creating embeddings...")
        model = SentenceTransformer("bert-base-nli-mean-tokens")
        text_embeddings = model.encode(df["text"].tolist())
        st.success("Embeddings created!")

        # Input for query
        query = st.text_input("Enter your query:")
        if st.button("Get Timestamp"):
            if query:
                start_time, end_time = get_timestamp_for_query_bert(
                    query, df, text_embeddings, model
                )
                st.success(
                    f"The tutorial for '{query}' starts at: {start_time} and ends at: {end_time}"
                )

# Run the app using: streamlit run your_script_name.py
