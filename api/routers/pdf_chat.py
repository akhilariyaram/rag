from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import io
import uuid
import os
import httpx
import time  # <--- NEW: Needed for sleeping
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

router = APIRouter()

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_KEY")
)

# --- MANUAL SEARCH FUNCTION ---
async def raw_qdrant_search(query_vector, session_id):
    url = os.environ.get("QDRANT_URL")
    key = os.environ.get("QDRANT_KEY")
    base_url = url.replace(":6333", "") 
    search_url = f"{base_url}/collections/pdf_chat/points/search"
    
    payload = {
        "vector": query_vector,
        "limit": 4,
        "filter": {
            "must": [{"key": "session_id", "match": {"value": session_id}}]
        },
        "with_payload": True
    }
    
    headers = {"api-key": key, "Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        response = await client.post(search_url, json=payload, headers=headers)
        if response.status_code != 200:
            print(f"Qdrant Error: {response.text}") 
            raise Exception(f"Qdrant Search Failed: {response.status_code}")
        return response.json()["result"]
# -----------------------------

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
        await wipe_and_recreate_collection()
        client = genai.Client(api_key=api_key)

        # 1. Extract Text
        pdf_reader = PdfReader(io.BytesIO(await file.read()))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")

        # 2. Chunking
        chunk_size = 1000
        overlap = 100
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i+chunk_size])
            
        print(f"Total chunks to process: {len(chunks)}")

        # 3. Embedding with Rate Limit Handling (THE FIX)
        vectors = []
        batch_size = 50 # Smaller batch size to be safe
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            print(f"Processing batch {i} to {i+len(batch)}...")
            
            # Retry Loop
            retry_count = 0
            while True:
                try:
                    response = client.models.embed_content(
                        model="models/gemini-embedding-001",
                        contents=batch,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                    )
                    batch_vectors = [e.values for e in response.embeddings]
                    vectors.extend(batch_vectors)
                    
                    # Polite pause to avoid hitting limit immediately
                    time.sleep(1) 
                    break # Success, exit retry loop
                    
                except Exception as e:
                    # Check if it's a Rate Limit error (429)
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        print(f"⚠️ Quota hit! Sleeping for 30 seconds... (Attempt {retry_count+1})")
                        time.sleep(30) # Wait for quota to reset
                        retry_count += 1
                        if retry_count > 5:
                            raise HTTPException(status_code=429, detail="Google API Quota Exceeded. Please try again later.")
                    else:
                        raise e # If it's another error, crash

        # 4. Upsert to Qdrant
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

        response = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=request.question,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        query_vector = response.embeddings[0].values
        
        search_results = await raw_qdrant_search(query_vector, request.session_id)
        
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
        return {"status": "error", "detail": str(e)}
async def wipe_and_recreate_collection():
    print("🧹 Starting Database Wipe...")
    url = os.environ.get("QDRANT_URL")
    key = os.environ.get("QDRANT_KEY")
    base_url = url.replace(":6333", "")
    collection_url = f"{base_url}/collections/pdf_chat"
    
    headers = {"api-key": key, "Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        # 1. DELETE existing collection
        await client.delete(collection_url, headers=headers)
        
        # 2. CREATE new collection
        # CRITICAL FIX: Gemini models use 768 dimensions by default. 
        # Setting this to 3072 would cause a crash.
        payload = {
            "vectors": {
                "size": 3072,  
                "distance": "Cosine"
            }
        }
        res = await client.put(collection_url, json=payload, headers=headers)
        if res.status_code != 200:
            print(f"⚠️ Recreate Warning: {res.text}")
    print("✨ Database Cleaned & Recreated.")