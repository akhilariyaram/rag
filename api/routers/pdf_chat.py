from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import io
import uuid
import os
import httpx # <--- MAKE SURE THIS IS HERE
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

router = APIRouter()

# Initialize Qdrant Client (Only for Uploads/Upsert)
qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_KEY")
)

# --- MANUAL SEARCH FUNCTION (Bypasses the library "search" command) ---
async def raw_qdrant_search(query_vector, session_id):
    url = os.environ.get("QDRANT_URL")
    key = os.environ.get("QDRANT_KEY")
    
    # 1. Clean the URL (Remove :6333 if present, as HTTP API uses standard ports often)
    # But Qdrant Cloud usually accepts the raw URL. 
    # Let's ensure we target the REST endpoint correctly.
    base_url = url.replace(":6333", "") 
    
    # 2. Define the Search Endpoint
    search_url = f"{base_url}/collections/pdf_chat/points/search"
    
    # 3. Build the Payload Manually
    payload = {
        "vector": query_vector,
        "limit": 4,
        "filter": {
            "must": [
                {
                    "key": "session_id",
                    "match": {
                        "value": session_id
                    }
                }
            ]
        },
        "with_payload": True
    }
    
    headers = {
        "api-key": key,
        "Content-Type": "application/json"
    }

    # 4. Send the Request
    async with httpx.AsyncClient() as client:
        response = await client.post(search_url, json=payload, headers=headers)
        if response.status_code != 200:
            # If this fails, we will see the REAL error from the server
            print(f"Qdrant Error: {response.text}") 
            raise Exception(f"Qdrant Search Failed: {response.status_code}")
        return response.json()["result"]
# ---------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    session_id: str
    api_key: str

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    api_key: str = Form(...)
):
    try:
        client = genai.Client(api_key=api_key)

        pdf_reader = PdfReader(io.BytesIO(await file.read()))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")

        chunk_size = 1000
        overlap = 100
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i+chunk_size])
            
        response = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=chunks,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        
        vectors = [e.values for e in response.embeddings]

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"session_id": session_id, "text": chunk}
            ))
        
        qdrant_client.upsert(collection_name="pdf_chat", points=points)
        return {"message": "PDF processed", "chunks": len(chunks)}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        client = genai.Client(api_key=request.api_key)

        # Embed Question
        response = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=request.question,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        query_vector = response.embeddings[0].values
        
        # --- EXECUTE MANUAL SEARCH ---
        # If the code reaches here, it CANNOT throw an "AttributeError" 
        # because we aren't calling .search() on the object anymore.
        search_results = await raw_qdrant_search(query_vector, request.session_id)
        # -----------------------------
        
        if not search_results:
            return {"answer": "I couldn't find any relevant info in the PDF."}

        context_text = "\n\n".join([hit['payload']['text'] for hit in search_results])
        
        prompt = f"""You are a helpful assistant. Answer based on the context provided.
        
Context:
{context_text}

Question: {request.question}

Answer:"""
        
        chat_response = client.models.generate_content(
            model="models/gemma-3-4b-it",
            contents=prompt
        )
        
        return {"answer": chat_response.text}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
async def cleanup(session_id: str = Form(...)):
    # Simple cleanup can stay as is, or we can wrap it in try/except pass
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qdrant_client.delete(
            collection_name="pdf_chat",
            points_selector=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            )
        )
        return {"status": "cleaned"}
    except Exception as e:
        print(f"Cleanup Error: {e}")
        return {"status": "error", "detail": str(e)}