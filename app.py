import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import requests
from urllib.parse import urlparse, parse_qs
import streamlit as st
from googletrans import Translator
from pytube import YouTube
from pydub import AudioSegment
import os

# Initialize your components
openai.api_key = "AIzaSyCSOt-RM3M-SsEQObh5ZBe-XwDK36oD3lM"  # Set your OpenAI API key
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
translator = Translator()

# Function to extract the video ID from the YouTube URL
def extract_video_id(youtube_url: str) -> str:
    parsed_url = urlparse(youtube_url)
    video_id = parse_qs(parsed_url.query).get('v')
    if video_id:
        return video_id[0]
    else:
        return None

# Function to download YouTube video and convert to mp3
def download_youtube_audio(video_url: str) -> str:
    try:
        video = YouTube(video_url)
        stream = video.streams.filter(only_audio=True).first()
        output_file = stream.download()
        base, ext = os.path.splitext(output_file)
        audio_file = base + '.mp3'
        AudioSegment.from_file(output_file).export(audio_file, format="mp3")
        os.remove(output_file)
        return audio_file
    except Exception as e:
        return None, f"Error downloading or converting audio: {str(e)}"

# Function to transcribe the audio file using OpenAI Whisper API
def transcribe_audio_openai(audio_file_path: str):
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return transcription["text"], None
    except Exception as e:
        return None, f"Error during audio transcription: {str(e)}"

# Function to load and translate subtitles
def load_and_translate_subtitles(video_url: str):
    try:
        audio_file_path, error = download_youtube_audio(video_url)
        if not audio_file_path:
            return None, error

        transcript, error = transcribe_audio_openai(audio_file_path)
        if error:
            return None, error

        # Detect language if not English and translate if necessary
        detected_lang = "en"  # Whisper can detect non-English languages, but we assume English by default here
        if detected_lang != 'en':
            translated_transcript = translator.translate(transcript, src=detected_lang, dest='en').text
            return translated_transcript, None
        else:
            return transcript, None
    except Exception as e:
        return None, f"Error loading or translating subtitles: {str(e)}"

# Function to create a FAISS index from the transcript
def create_db_from_youtube_video_url(video_url: str):
    try:
        transcript, error = load_and_translate_subtitles(video_url)
        if error:
            return None, None, error
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_text(transcript)

        # Convert the resulting split text into the required format
        docs = [{"page_content": doc} for doc in docs]

        if not docs:
            return None, None, "Failed to split the transcript into documents."

        docs_content = [doc["page_content"] for doc in docs]
        embeddings = embedding_model.encode(docs_content)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return docs, index, None
    except Exception as e:
        return None, None, f"Error during database creation: {str(e)}"

# Function to query the index and get a response
def get_response_from_query(docs, index, query, k=4):
    try:
        query_embedding = embedding_model.encode([query])
        distances, indices = index.search(query_embedding, k)

        docs_page_content = " ".join([docs[idx]["page_content"] for idx in indices[0]])

        prompt = f"Question: {query}\nVideo Content: {docs_page_content}\nPlease respond as if discussing the video itself, without referencing transcripts or any such terms."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={openai.api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ]
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()

        if 'candidates' not in response_data:
            return f"Error: 'candidates' not found in response.", None

        if not response_data['candidates']:
            return f"Error: 'candidates' list is empty.", None

        if 'content' not in response_data['candidates'][0]:
            return f"Error: 'content' not found in the first candidate.", None

        if 'parts' not in response_data['candidates'][0]['content']:
            return f"Error: 'parts' not found in the content of the first candidate.", None

        if not response_data['candidates'][0]['content']['parts']:
            return f"Error: 'parts' list is empty.", None

        try:
            generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError) as e:
            return f"Error accessing response content: {e}", None

        return generated_text, docs
    except requests.exceptions.RequestException as e:
        return f"Request error: {e}", None
    except ValueError as e:
        return f"JSON decoding error: {e} - Response content: {response.text}", None
    except Exception as e:
        return f"Error during query processing: {str(e)}", None

# Streamlit interface
st.title("YouTube Query Assistant")

st.write("""
## Welcome to the YouTube Query Assistant!

This AI-powered tool is designed to save you time by providing precise answers to your queries about any YouTube video.

## How It Works:
1. **Enter the YouTube Video URL**: Provide the link to the YouTube video you want to query.
2. **Ask Your Question**: Type in the specific information you're looking for within the video.
3. **Get Instant Results**: The AI processes the video content and returns the most relevant information, helping you quickly determine if the video contains what you need.
""")

video_url = st.text_input("Enter YouTube video URL")
query = st.text_input("Enter your query")

if st.button("Get Response"):
    if video_url and query:
        docs, index, error = create_db_from_youtube_video_url(video_url)
        if error:
            st.error(error)
        else:
            response, docs = get_response_from_query(docs, index, query)
            if docs is None:
                st.error(response)
            else:
                st.write(response)
    else:
        st.warning("Please enter both the video URL and query.")
