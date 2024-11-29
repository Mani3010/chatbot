from flask import Flask, request, jsonify
from query import query_faiss, query_whoosh, rank_with_tfidf
from preprocessor import preprocess_text
import json
import faiss
from whoosh import index

app = Flask(__name__)


# Load FAISS index and section mappings
def load_faiss_index(index_file="faiss_index.bin", sections_file="faiss_sections.json"):
    index = faiss.read_index(index_file)
    with open(sections_file, "r") as f:
        sections = json.load(f)
    return index, sections


# Load Whoosh index
def load_whoosh_index(index_dir="whoosh_index"):
    return index.open_dir(index_dir)


@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('query')
    if not user_input:
        return jsonify({"error": "No query provided"}), 400

    query_text = preprocess_text(user_input)

    # FAISS Query
    faiss_index, faiss_sections = load_faiss_index()
    faiss_result = query_faiss(query_text, faiss_index, faiss_sections)

    if faiss_result != "Sorry, I didn’t understand your question. Do you want to connect with a live agent?":
        return jsonify(faiss_result)

    # Whoosh Query
    whoosh_index = load_whoosh_index()
    whoosh_result = query_whoosh(query_text, whoosh_index)

    if whoosh_result != "Sorry, I didn’t understand your question. Do you want to connect with a live agent?":
        return jsonify(whoosh_result)

    # Fallback with TF-IDF
    documents = list(faiss_sections.values())
    tfidf_result = rank_with_tfidf(query_text, documents)
    return jsonify(tfidf_result)


if __name__ == '__main__':
    app.run(debug=True)
import json
import faiss
from sentence_transformers import SentenceTransformer
from whoosh.fields import Schema, TEXT, STORED
from whoosh import index
import os
import pdfplumber
import re


def extract_text_sections(pdf_path):
    heading_pattern = re.compile(r'^\s*[A-Z][A-Z\s]+$')
    structured_data = {}
    current_section = None
    current_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                for line in text.split("\n"):
                    if heading_pattern.match(line):
                        if current_section and current_text:
                            structured_data[current_section] = " ".join(current_text).strip()
                        current_section = line.strip()
                        current_text = []
                    else:
                        current_text.append(line.strip())
        if current_section and current_text:
            structured_data[current_section] = " ".join(current_text).strip()

    return structured_data


def index_with_faiss(structured_data, index_file="faiss_index.bin"):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sections = list(structured_data.keys())
    texts = list(structured_data.values())
    embeddings = model.encode(texts, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_file)
    with open("faiss_sections.json", "w") as f:
        json.dump(sections, f, indent=4)
    print(f"FAISS index saved to {index_file} with {len(sections)} sections.")


def index_with_whoosh(structured_data, index_dir="whoosh_index"):
    schema = Schema(title=TEXT(stored=True), content=TEXT)
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    idx = index.create_in(index_dir, schema)
    writer = idx.writer()
    for title, content in structured_data.items():
        writer.add_document(title=title, content=content)
    writer.commit()
    print(f"Whoosh index created in directory: {index_dir}")


# Example usage
pdf_path = "example.pdf"
data = extract_text_sections(pdf_path)
index_with_faiss(data)
index_with_whoosh(data)
def preprocess_text(text):
    stopwords = set(["a", "an", "the", "and", "but", "or", "for", "to", "in", "on", "at", "of"])
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stopwords])
    return text
import json
import faiss
from whoosh.qparser import QueryParser
import numpy as np
from sentence_transformers import SentenceTransformer


def query_faiss(query_text, index, sections, threshold=0.7):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query_text], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=5)

    best_match_idx = indices[0][0]
    best_match_distance = distances[0][0]

    if best_match_distance < threshold:
        return "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"

    return {"section": sections[best_match_idx], "confidence": best_match_distance}


def query_whoosh(query_text, idx, threshold=0.7):
    with idx.searcher() as searcher:
        query = QueryParser("content", idx.schema).parse(query_text)
        results = searcher.search(query, limit=5)

        if not results:
            return "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"

        best_result = results[0]
        return {"section": best_result["title"], "confidence": best_result.score}


def rank_with_tfidf(query_text, documents, threshold=0.7):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    query_vec = vectorizer.transform([query_text])
    cosine_similarities = np.dot(query_vec, tfidf_matrix.T).toarray().flatten()

    max_sim = max(cosine_similarities)
    if max_sim < threshold:
        return "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"

    best_match_idx = np.argmax(cosine_similarities)
    return {"section": documents[best_match_idx], "confidence": max_sim}
import streamlit as st
import requests

API_URL = "http://127.0.0.1:5000/ask"

def query_bot(query):
    response = requests.post(API_URL, json={"query": query})
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "There was an issue with the request"}

st.title("Chatbot Interface")
st.write("Ask me anything about the content!")

user_input = st.text_input("Enter your question:")

if user_input:
    with st.spinner("Fetching the answer..."):
        response = query_bot(user_input)

        if "section" in response:
            st.write(f"Answer: {response['section']}")
            st.write(f"Confidence: {response['confidence']}")
        else:
            st.write(response["error"])
