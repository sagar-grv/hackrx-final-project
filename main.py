# main.py
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
import requests
from io import BytesIO
import pypdf
import google.generativeai as genai
from pinecone import Pinecone
import time
import os # <-- THE MISSING IMPORT IS NOW ADDED

# --- Configuration ---
# This will use the secure environment variable on Render,
# but will use the key you paste here as a fallback for local testing.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "PASTE_YOUR_GOOGLE_API_KEY_HERE")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "PASTE_YOUR_PINECONE_API_KEY_HERE")

genai.configure(api_key=GOOGLE_API_KEY)
PINECONE_INDEX_NAME = "hackrx-library"

# --- Pinecone Connection ---
pc = Pinecone(api_key=PINECONE_API_KEY)
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
    except Exception:
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
    except Exception:
        return None

# --- Main "Thinking" Function ---
def generate_answer(query: str, index: Pinecone.Index):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return "Sorry, I couldn't understand the question."

    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    
    context = ""
    if results['matches']:
        for match in results['matches']:
            context += match['metadata']['text'] + "\n---\n"
    
    if not context:
        return "Sorry, I couldn't find any relevant information in the document."

    prompt = f"Based ONLY on the context provided below, answer the question.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    
    try:
        generative_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        answer_response = generative_model.generate_content(prompt)
        return answer_response.text.strip()
    except Exception:
        return "Sorry, the AI brain had an error while generating the answer."

# --- Main API Endpoint ---
@app.post("/hackrx/run")
def receive_a_letter(request: HackRxRequest, authorization: Optional[str] = Header(None)):
    if authorization is None or "Bearer " not in authorization or authorization.split()[1] != SECRET_PASSWORD:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong or missing password!")

    document_text = read_pdf_from_url(request.documents)
    text_chunks = get_text_chunks(document_text)

    if text_chunks:
        index.delete(delete_all=True)
        vectors_to_upsert = []
        for i, chunk in enumerate(text_chunks):
            embedding = get_embedding(chunk)
            if embedding:
                vectors_to_upsert.append({"id": f"chunk_{i}", "values": embedding, "metadata": {"text": chunk}})
        
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
            time.sleep(10)

    questions = request.questions
    final_answers = []
    for q in questions:
        answer = generate_answer(q, index)
        final_answers.append(answer)

    return HackRxResponse(answers=final_answers)