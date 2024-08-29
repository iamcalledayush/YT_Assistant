import yt_dlp
import faiss
import streamlit as st
from urllib.parse import urlparse, parse_qs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Initialize your components
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_video_id(youtube_url: str) -> str:
    parsed_url = urlparse(youtube_url)
    video_id = parse_qs(parsed_url.query).get('v')
    if video_id:
        return video_id[0]
    else:
        return None

def download_auto_generated_captions(video_url: str):
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        subtitles = info_dict.get('subtitles')
        
        if subtitles and 'en' in subtitles:
            subtitle_url = subtitles['en'][0]['url']
            response = requests.get(subtitle_url)
            if response.status_code == 200:
                srt_data = response.content.decode('utf-8')
                return srt_data
        return None

def parse_srt(srt_data):
    lines = srt_data.splitlines()
    transcript = []
    for line in lines:
        if "-->" not in line and line.strip().isdigit() is False:
            transcript.append(line.strip())
    return " ".join(transcript)

def load_and_parse_captions(video_url: str):
    try:
        srt_data = download_auto_generated_captions(video_url)
        if not srt_data:
            return None, "Failed to download or find English captions."
        
        transcript = parse_srt(srt_data)
        return transcript, None

    except Exception as e:
        return None, f"Error loading captions: {str(e)}"

def create_db_from_youtube_video_url(video_url: str):
    try:
        transcript, error = load_and_parse_captions(video_url)
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
