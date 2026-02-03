from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
import io
import uuid
import os
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

router = APIRouter()

# 1. Initialize Clients
qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_KEY")
)

# --- DEBUG BLOCK (This will save us) ---
# This prints all available commands to your Vercel Logs
print("🔍 DEBUGGING QDRANT OBJECT:")
print(f"Type: {type(qdrant_client)}")
try:
    print(f"DIR: {dir(qdrant_client)}")
except:
    print("Could not print dir")
# ---------------------------------------

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

        response = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=request.question,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        query_vector = response.embeddings[0].values
        
        search_filter = Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=request.session_id))]
        )

        # --- EXTREME SAFETY BLOCK ---
        # We check if the method exists before calling it
        if hasattr(qdrant_client, 'search'):
            print("✅ Using .search()")
            search_result = qdrant_client.search(
                collection_name="pdf_chat",
                query_vector=query_vector,
                limit=4,
                query_filter=search_filter
            )
        elif hasattr(qdrant_client, 'search_points'):
            print("⚠️ Using .search_points()")
            search_result = qdrant_client.search_points(
                collection_name="pdf_chat",
                vector=query_vector,
                limit=4,
                filter=search_filter
            )
        else:
            # If both fail, we manually crash and print why
            print("❌ FATAL: Neither .search nor .search_points found!")
            print(f"Available methods: {dir(qdrant_client)}")
            raise HTTPException(status_code=500, detail="Qdrant Client Version Mismatch")
        # ----------------------------
        
        if not search_result:
            return {"answer": "I couldn't find any relevant info in the PDF."}

        context_text = "\n\n".join([hit.payload['text'] for hit in search_result])
        
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
        qdrant_client.delete(
            collection_name="pdf_chat",
            points_selector=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            )
        )
        return {"status": "cleaned"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}