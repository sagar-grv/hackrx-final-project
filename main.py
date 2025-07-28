# main.py - Final, Secure High-Speed Version
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
import requests
from io import BytesIO
import pypdf
import google.generativeai as genai
from pinecone import Pinecone
from groq import Groq
import time
import os
import hashlib

# --- Configuration ---
# This code is SAFE for GitHub. It reads the keys from a secure place.
# For local testing, it uses the fallback value you paste here.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "PASTE_YOUR_GOOGLE_API_KEY_HERE")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "PASTE_YOUR_PINECONE_API_KEY_HERE")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "PASTE_YOUR_GROQ_API_KEY_HERE")

# Configure all our services
genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

PINECONE_INDEX_NAME = "hackrx-library"
index = pc.Index(PINECONE_INDEX_NAME)

SECRET_PASSWORD = "1e83fbe10fa7c1be5ffa312d8b283e496b82c2470dee257fb48b82ad7e8ba562"
app = FastAPI(title="HackRx 6.0 API - Final Version")

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Helper Functions ---
def read_pdf_from_url(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        pdf_file = BytesIO(response.content)
        pdf_reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def get_text_chunks(text: str) -> List[str]:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunks.append(text[i:i + CHUNK_SIZE])
    return chunks

def get_embedding(text: str, model="models/embedding-001"):
    try:
        result = genai.embed_content(model=model, content=text)
        return result['embedding']
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None

# --- Main "Thinking" Function ---
def generate_answer(query: str, index: Pinecone.Index, namespace: str):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return "Sorry, I couldn't understand the question."

    results = index.query(vector=query_embedding, top_k=3, include_metadata=True, namespace=namespace)
    
    context = ""
    if results['matches']:
        for match in results['matches']:
            context += match['metadata']['text'] + "\n---\n"
    
    if not context:
        return "Sorry, I couldn't find any relevant information in the document."

    system_prompt = "Based ONLY on the context provided below, answer the question. Do not use any other information. If the answer is not in the context, say 'I could not find the answer in the document.'"
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{query}"
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, the Groq AI brain had an error: {e}"

# --- Main API Endpoint ---
@app.post("/hackrx/run")
def receive_a_letter(request: HackRxRequest, authorization: Optional[str] = Header(None)):
    if authorization is None or "Bearer " not in authorization or authorization.split()[1] != SECRET_PASSWORD:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong or missing password!")

    doc_url = request.documents
    questions = request.questions
    doc_id_hash = hashlib.sha256(doc_url.encode()).hexdigest()
    
    index_stats = index.describe_index_stats()
    if doc_id_hash not in index_stats.namespaces or index_stats.namespaces.get(doc_id_hash, {}).vector_count == 0:
        print(f"Document {doc_id_hash} is new. Reading and adding to the library...")
        document_text = read_pdf_from_url(doc_url)
        text_chunks = get_text_chunks(document_text)
        if text_chunks:
            vectors_to_upsert = []
            for i, chunk in enumerate(text_chunks):
                embedding = get_embedding(chunk)
                if embedding:
                    vectors_to_upsert.append({"id": f"chunk_{i}", "values": embedding, "metadata": {"text": chunk}})
            if vectors_to_upsert:
                index.upsert(vectors=vectors_to_upsert, namespace=doc_id_hash)
                # This is the corrected line:
                print(f"--- Successfully stored chunks in namespace {doc_id_hash}! ---")
                time.sleep(10)
    else:
        print(f"Document {doc_id_hash} is already in the library. Skipping the reading part.")

    final_answers = []
    for q in questions:
        answer = generate_answer(q, index, namespace=doc_id_hash)
        final_answers.append(answer)

    return HackRxResponse(answers=final_answers)
