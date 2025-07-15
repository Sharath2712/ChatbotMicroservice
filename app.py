from flask import Flask, request
from ddtrace import patch_all, tracer
import time
import os
import rag_utils

# Automatically patch all supported libraries (Flask, requests, etc.)
patch_all()

app = Flask(__name__)

@app.route("/")
def index():
        return "Hello from Flask with RAG using AWS Bedrock!"

@app.route("/health")
def health():
    return "Health Check OK", 200

@app.route("/queryPrompt", methods=["POST"])
def query_prompt():
    data = request.get_json()
    query = data.get('query')
    query_response = rag_utils.perform_rag(query)
    return query_response, 200

@app.route("/train")
def training():
        rag_utils.text = rag_utils.read_pdf_from_s3("rag-pinecone-chatbot", "Embedded_Systems_RAG.pdf")
        rag_utils.get_embedding(rag_utils.text)
        rag_utils.upsert_embeddings()
        return "Training has been done!", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
