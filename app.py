from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import requests
import json

app = Flask(__name__)

gemini_api_key = "AIzaSyCWu7t3NGc0mx06fHZqGlBKJc_h-I20ppk"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def create_db_from_youtube_video_url(video_url: str):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    docs_content = [doc.page_content for doc in docs]
    embeddings = embedding_model.encode(docs_content)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return docs, index

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    video_url = request.form['link']
    query = request.form['query']

    try:
        docs, index = create_db_from_youtube_video_url(video_url)
        response, docs = get_response_from_query(docs, index, query)
        if docs is None:
            return jsonify({'error': response})
        return jsonify({'result': response})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
