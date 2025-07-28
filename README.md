# HackRx 6.0 - Intelligent Query-Retrieval System

This project is a solution for the HackRx 6.0 hackathon. It's an AI-powered API that can read a document and answer questions about its content.

## 🤖 How It Works

The system uses a Retrieval-Augmented Generation (RAG) architecture:
1.  It receives a document URL and a list of questions.
2.  It reads and breaks the document into smaller "chunks".
3.  It converts these chunks into numerical "embeddings" using the Google Gemini API.
4.  It stores these embeddings in a Pinecone vector database.
5.  For each question, it searches the database for the most relevant chunks.
6.  Finally, it uses the Gemini LLM to generate an answer based on the retrieved chunks.

## 🚀 API Endpoint

### Welcome Message

- Sending a `GET` request to the root URL (`/`) will return a JSON object with a welcome message and basic information about the API.
- To use the API, send a `POST` request to the `/hackrx/run` endpoint.

### Request Body

```json
{
  "documents": "[https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=](https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=)...",
  "questions": [
    "What is the grace period for premium payment?",
    "Does this policy cover maternity expenses?"
  ]
}
```
## Authorization
You must include an Authorization header with the Bearer token provided in the hackathon documentation.
- Header: Authorization
- Value:  Bearer 1e83fbe10fa7c1be5ffa312d8b283e496b82c2470dee257fb48b82ad7e8ba562

## 🛠️ Tech Stack
- Backend: FastAPI
- LLM & Embeddings: Google Gemini API , Groq API
- Vector Database: Pinecone
