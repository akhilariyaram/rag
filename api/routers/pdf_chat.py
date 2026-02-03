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
import time
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

router = APIRouter()

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_KEY")
)

# --- HELPER: WIPE & RESET DB ---
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
        payload = {
            "vectors": {
                "size": 3072,  # Kept as requested
                "distance": "Cosine"
            }
        }
        res = await client.put(collection_url, json=payload, headers=headers)
        if res.status_code != 200:
            print(f"⚠️ Recreate Warning: {res.text}")
    print("✨ Database Cleaned & Recreated.")

# --- MANUAL SEARCH FUNCTION ---
async def raw_qdrant_search(query_vector, session_id):
    url = os.environ.get("QDRANT_URL")
    key = os.environ.get("QDRANT_KEY")
    base_url = url.replace(":6333", "") 
    search_url = f"{base_url}/collections/pdf_chat/points/search"
    
    # --- LOGIC FIX HERE ---
    # Removed the "filter" block. 
    # Since we wipe the DB on upload, filtering is not required and caused the error.
    payload = {
        "vector": query_vector,
        "limit": 5,
        "with_payload": True
    }
    
    headers = {"api-key": key, "Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        response = await client.post(search_url, json=payload, headers=headers)
        if response.status_code != 200:
            print(f"Qdrant Error: {response.text}") 
            raise Exception(f"Qdrant Search Failed: {response.status_code}")
        return response.json()["result"]

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
        # 1. WIPE DB
        await wipe_and_recreate_collection()

        client = genai.Client(api_key=api_key)

        # 2. Extract Text
        pdf_reader = PdfReader(io.BytesIO(await file.read()))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")

        # 3. Chunking
        chunk_size = 1000
        overlap = 100
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i+chunk_size])
            
        print(f"Total chunks to process: {len(chunks)}")

        # 4. Embedding
        vectors = []
        batch_size = 50
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            print(f"Processing batch {i} to {i+len(batch)}...")
            
            retry_count = 0
            while True:
                try:
                    response = client.models.embed_content(
                        model="models/gemini-embedding-001", # Kept as requested
                        contents=batch,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                    )
                    batch_vectors = [e.values for e in response.embeddings]
                    vectors.extend(batch_vectors)
                    time.sleep(1) 
                    break 
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        print(f"⚠️ Quota hit! Sleeping for 30 seconds...")
                        time.sleep(30)
                        retry_count += 1
                        if retry_count > 5: raise e
                    else:
                        raise e

        # 5. Upsert to Qdrant
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"session_id": session_id, "text": chunk}
            ))
        
        qdrant_client.upsert(collection_name="pdf_chat", points=points)
        return {"message": "PDF processed & Database Reset", "chunks": len(chunks)}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        client = genai.Client(api_key=request.api_key)

        response = client.models.embed_content(
            model="models/gemini-embedding-001", # Kept as requested
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
    await wipe_and_recreate_collection()
    return {"status": "cleaned"}

@router.post("/reset")
async def reset_database():
    await wipe_and_recreate_collection()
    return {"status": "reset"}