import os
import requests
from googleapiclient.discovery import build
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from urllib.parse import urlparse, parse_qs


# Initialize your components
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Your YouTube Data API key
YOUTUBE_API_KEY = "AIzaSyD3iEuERg5qL2e2nAS3aO0k34wUHAHlNBQ"

def extract_video_id(youtube_url: str) -> str:
    parsed_url = urlparse(youtube_url)
    video_id = parse_qs(parsed_url.query).get('v')
    if video_id:
        return video_id[0]
    else:
        return None

def list_captions(video_id):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    request = youtube.captions().list(part="snippet", videoId=video_id)
    response = request.execute()
    
    captions = []
    for item in response.get('items', []):
        captions.append({
            'id': item['id'],
            'language': item['snippet']['language'],
            'name': item['snippet']['name'],
            'is_auto_generated': item['snippet']['trackKind'] == 'ASR',
        })
    return captions

def download_caption(caption_id):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    request = youtube.captions().download(id=caption_id, tfmt='srt')
    try:
        response = request.execute()
        return response.decode('utf-8')
    except Exception as e:
        return None, f"Error downloading captions: {str(e)}"

def load_and_translate_subtitles(video_url: str):
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            return None, "Invalid YouTube URL or video ID could not be extracted."

        captions = list_captions(video_id)
        auto_generated_caption = None
        for caption in captions:
            if caption['language'] == 'en' and caption['is_auto_generated']:
                auto_generated_caption = caption['id']
                break

        if not auto_generated_caption:
            return None, "No auto-generated English captions available for this video."

        transcript, error = download_caption(auto_generated_caption)
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
        docs = text_splitter.split_text(transcript)

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

        # Replace this with your API integration
        generated_text = f"Response based on the content: {docs_page_content[:500]}"

        return generated_text, docs
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
