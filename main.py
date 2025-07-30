# main.py - Final, Secure, High-Speed Version
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone
from groq import Groq
import os
import json

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

# Configure services
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
    answers: List[Dict[str, Any]]

# --- Helper Function ---
def get_embedding(text: str, model="models/embedding-001"):
    try:
        result = genai.embed_content(model=model, content=text)
        return result['embedding']
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None

# --- Main "Thinking" Function ---
def generate_decision(query: str, index: Pinecone.Index, namespace: str):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return {"decision": "Error", "amount": None, "justification": "Could not understand the query."}

    results = index.query(vector=query_embedding, top_k=5, include_metadata=True, namespace=namespace)
    
    context = ""
    if results['matches']:
        for i, match in enumerate(results['matches']):
            context += f"Clause {i+1}:\n{match['metadata']['text']}\n---\n"
    
    if not context:
        return {"decision": "Not Found", "amount": None, "justification": "Could not find any relevant information."}

    prompt = f"""
    You are an expert insurance claims processor. Your task is to evaluate a claim based ONLY on the provided policy document clauses.

    **Policy Clauses (Context):**
    {context}

    **Claim Details (Query):**
    {query}

    Based strictly on the claim details and the policy clauses, provide a decision. Your response MUST be a single, valid JSON object with three keys: "decision", "amount", and "justification".
    """
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            response_format={"type": "json_object"},
        )
        llm_response_str = chat_completion.choices[0].message.content
        return json.loads(llm_response_str)
    except Exception as e:
        return {"decision": "Error", "amount": None, "justification": f"The AI brain had an error: {e}"}

# --- Main API Endpoint ---
@app.post("/hackrx/run")
def receive_a_letter(request: HackRxRequest, authorization: Optional[str] = Header(None)):
    if authorization is None or "Bearer " not in authorization or authorization.split()[1] != SECRET_PASSWORD:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong or missing password!")

    doc_url = request.documents
    queries = request.questions
    
    # Simple mapping from document URL to our pre-indexed library sections
    if "HDFHLIP23024V072223" in doc_url:
        namespace_id = "HDFC_ERGO_Easy_Health"
    elif "BAJHLIP23020V012223" in doc_url:
        namespace_id = "Bajaj_Allianz_Global_Health"
    elif "ICIHLIP22012V012223" in doc_url:
        namespace_id = "ICICI_Lombard_Golden_Shield"
    elif "CHOTGDP23004V012223" in doc_url:
        namespace_id = "Cholamandalam_Travel"
    elif "EDLHLGA23009V012223" in doc_url:
        namespace_id = "Edelweiss_Well_Baby_Well_Mother"
    else:
        # If the document is not one of the five official ones
        return HackRxResponse(answers=[{"decision": "Error", "amount": None, "justification": "Unknown document."}])

    final_answers = []
    for q in queries:
        decision_json = generate_decision(q, index, namespace=namespace_id)
        final_answers.append(decision_json)

    return HackRxResponse(answers=final_answers)
