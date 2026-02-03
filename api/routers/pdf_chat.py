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

# 1. Initialize Qdrant (Same as before)
qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_KEY")
)

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
        # --- NEW SDK SETUP ---
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
            
        # 3. Generate Embeddings (Your Specific Model)
        # We process chunks in batch for efficiency
        response = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=chunks,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        
        # New SDK returns a list of embedding objects. We extract .values from each.
        vectors = [e.values for e in response.embeddings]

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
        # --- NEW SDK SETUP ---
        client = genai.Client(api_key=request.api_key)

        # 1. Embed Question (Your Specific Model)
        response = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=request.question,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            )
        )
        query_vector = response.embeddings[0].values
        
        # 2. Search Qdrant
        search_result = qdrant_client.search(
            collection_name="pdf_chat",
            query_vector=query_vector,
            limit=4,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="session_id",
                        match=MatchValue(value=request.session_id)
                    )
                ]
            )
        )
        
        if not search_result:
            return {"answer": "I couldn't find any relevant info in the PDF."}

        # 3. Generate Answer (Your Specific Model)
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