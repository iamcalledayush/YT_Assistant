import pytube
import faiss
import requests
import json
from urllib.parse import urlparse, parse_qs
import streamlit as st
from googletrans import Translator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Initialize your components
gemini_api_key = "AIzaSyCSOt-RM3M-SsEQObh5ZBe-XwDK36oD3lM"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
translator = Translator()

def extract_video_id(youtube_url: str) -> str:
    parsed_url = urlparse(youtube_url)
    video_id = parse_qs(parsed_url.query).get('v')
    if video_id:
        return video_id[0]
    else:
        return None

def download_auto_generated_captions(video_url: str):
    try:
        youtube = pytube.YouTube(video_url)
        captions = youtube.captions
        
        # Find auto-generated English captions
        auto_generated_caption = None
        for caption in youtube.caption_tracks:
            if "auto-generated" in caption.name and "English" in caption.name:
                auto_generated_caption = caption
                break
        
        if not auto_generated_caption:
            return None, "No auto-generated English captions available for this video."
        
        # Download the auto-generated captions
        caption_text = auto_generated_caption.download()
        return caption_text, None
    except Exception as e:
        return None, f"Error downloading captions: {str(e)}"

def load_and_translate_subtitles(video_url: str):
    try:
        transcript, error = download_auto_generated_captions(video_url)
        if error:
            return None, error
        
        # Clean up the SRT format by removing time codes and line numbers
        transcript = "\n".join([line for line in transcript.splitlines() if not line.strip().isdigit() and "-->" not in line and line.strip() != ''])

        return transcript, None

    except Exception as e:
        return None, f"Error loading or translating subtitles: {str(e)}"

def create_db_from_youtube_video_url(video_url: str):
    try:
        transcript, error = load_and_translate_subtitles(video_url)
        if error:
            return None, None, error
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_text(transcript)  # Use split_text instead of split_documents

        # Convert the resulting split text into the required format
        docs = [{"page_content": doc} for doc in docs]

        if not docs:
            return None, None, "Failed to split the transcript into documents."

        docs_content = [doc["page_content"] for doc in docs]

        if not docs_content:
            return None, None, "Document content is empty after splitting the transcript."

        embeddings = embedding_model.encode(docs_content)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return docs, index, None
    except Exception as e:
        return None, None, f"Error during database creation: {str(e)}"

def get_response_from_query(docs, index, query, k=4):
    try:
        query_embedding = embedding_model.encode([query])
        distances, indices = index.search(query_embedding, k)

        docs_page_content = " ".join([docs[idx]["page_content"] for idx in indices[0]])

        prompt = f"Question: {query}\nVideo Content: {docs_page_content}\nPlease respond as if discussing the video itself, without referencing transcripts or any such terms."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={gemini_api_key}"
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

## Why Use This Tool?
- **Time-Saving**: No need to scrub through long videos. Get the answers you need in seconds.
- **Precision**: Target specific content within a video without watching it in full.
- **Informed Viewing**: Know in advance if the video covers the topic you’re interested in.

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
