import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import requests
import json

# Initialize your components
gemini_api_key = "AIzaSyCSOt-RM3M-SsEQObh5ZBe-XwDK36oD3lM"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def create_db_from_youtube_video_url(video_url: str):
    try:
        loader = YoutubeLoader.from_youtube_url(video_url)
        transcript = loader.load()
        
        if not transcript:
            return None, None, "Transcript could not be loaded or is empty."
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(transcript)

        if not docs:
            return None, None, "No documents were created from the transcript."

        docs_content = [doc.page_content for doc in docs]

        if not docs_content:
            return None, None, "The content of the documents is empty."

        embeddings = embedding_model.encode(docs_content)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return docs, index, None
    except Exception as e:
        return None, None, f"An error occurred while processing the video: {str(e)}"

def get_response_from_query(docs, index, query, k=4):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)

    docs_page_content = " ".join([docs[idx].page_content for idx in indices[0]])

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={gemini_api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {"parts": [{"text": f"Question: {query}\nDocs: {docs_page_content}"}]}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
    except requests.exceptions.RequestException as e:
        return f"Request error: {e}", None
    except ValueError as e:
        return f"JSON decoding error: {e} - Response content: {response.text}", None

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
