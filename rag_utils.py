import boto3
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from io import BytesIO
import json
from datetime import datetime, timedelta
import re
import logging

text = ""

# Initialize S3 and other clients
s3 = boto3.client("s3")
bedrock_runtime = boto3.client("bedrock-runtime", region_name="ap-south-1")
pinecone = Pinecone("pcsk_oHzKY_56NMTkH6K9vuqF882EpDbF2ssaVUjt6LSbpZBBXEx9ek6AkVCr5ZTQa3wLhi7JD")
index = pinecone.Index("rag-chatbot")
print(f"Connected to index: {index}")

# Function to read PDF from S3
def read_pdf_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    pdf_content = response["Body"].read()
    pdf_reader = PdfReader(BytesIO(pdf_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to create embeddings
def get_embedding(text):
    response = bedrock_runtime.invoke_model(
        modelId="amazon.titan-embed-text-v2:0", body=json.dumps({"inputText": text})
    )
    return json.loads(response["body"].read())["embedding"]

def upsert_embeddings():
    # Create and store embeddings
    chunks = [text[i : i + 1000] for i in range(0, len(text), 1000)]
    embeddings = [get_embedding(chunk) for chunk in chunks]

    for i, embedding in enumerate(embeddings):
        index.upsert([(str(i), embedding, {"text": chunks[i]})])


# Function to perform RAG
def perform_rag(query):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    
    context = " ".join([result["metadata"].get("text", "") for result in results.get("matches", [])])
    
    prompt = f"""
    You are a friendly and helpful AI assistant. Respond in a natural, human-like manner, and always offer a follow-up like "Can I help you with anything else?" when appropriate. Answer the question concisely and to the point, without unnecessary details. Keep your response within two sentences.
    Refer to the following document information to answer the question:
    
    {context}
    
    User: {query}
    Assistant:
    """
    payload = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0.5,
            "topP": 0.9,
        },
    }
    
    response = bedrock_runtime.invoke_model(
        modelId="amazon.titan-text-express-v1",
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json",
    )
    
    response_body = json.loads(response["body"].read())
    output_text = response_body["results"][0]["outputText"]
    return output_text

# Function to perform RAG
def delete_vector_embeddings():
    try:
        index.delete(delete_all=True)
        print("All vector embeddings deleted successfully.")
        return "All vector embeddings deleted successfully."
    except Exception as e:
        logging.error(f"Error deleting vector embeddings: {e}")
        return f"Error deleting vector embeddings: {e}"