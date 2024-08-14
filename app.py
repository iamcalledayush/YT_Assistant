from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import requests
import json
from urllib.parse import urlparse, parse_qs
import streamlit as st
from googletrans import Translator
from bs4 import BeautifulSoup
import re

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

def fetch_captions_by_scraping(video_id: str, target_lang='en'):
    """Fetch auto-generated captions by scraping the YouTube page."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    response = requests.get(url)
    if response.status_code != 200:
        return None, f"Failed to load YouTube page for video ID: {video_id}"

    soup = BeautifulSoup(response.text, 'html.parser')
    scripts = soup.find_all('script')

    # Find the script that contains 'ytInitialPlayerResponse'
    for script in scripts:
        if 'ytInitialPlayerResponse' in script.text:
            try:
                # Use regex to extract the JSON part
                match = re.search(r'ytInitialPlayerResponse\s*=\s*({.*?});', script.text)
                if not match:
                    return None, "Could not find ytInitialPlayerResponse in the script."
                
                json_text = match.group(1)
                data = json.loads(json_text)
                
                # Ensure 'captions' key exists in the extracted data
                if 'captions' not in data:
                    return None, "'captions' key not found in the YouTube page's JSON data."
                
                captions = data['captions']['playerCaptionsTracklistRenderer']['captionTracks']

                # Look for auto-generated captions in the desired language
                for caption in captions:
                    if 'kind' in caption and caption['kind'] == 'asr' and caption['languageCode'] == target_lang:
                        subtitle_url = caption['baseUrl']
                        subtitle_response = requests.get(subtitle_url)
                        return subtitle_response.text, None
            except json.JSONDecodeError as e:
                return None, f"JSON decoding error: {str(e)}"
            except Exception as e:
                return None, f"Error extracting captions: {str(e)}"

    return None, "No auto-generated captions found or failed to extract."

def load_and_translate_subtitles(video_url: str):
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            return None, "Invalid YouTube URL or video ID could not be extracted."
        
        # Try to fetch a transcript using YouTubeTranscriptApi
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([item['text'] for item in transcript_list])
            language_code = YouTubeTranscriptApi.list_transcripts(video_id).find_transcript(['en', 'a.en']).language_code
        except (NoTranscriptFound, TranscriptsDisabled):
            # Fallback to scraping if transcripts are disabled or not found
            transcript, error = fetch_captions_by_scraping(video_id, target_lang='en')
            if error:
                return None, error
            transcript = translator.translate(transcript, dest='en').text if transcript else None
        
        if transcript is None:
            return None, "No captions or transcripts found for this video."
        
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
