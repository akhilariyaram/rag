# api/routers/pdf_chat.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
import io
import uuid
import os
from pypdf import PdfReader
from dotenv import load_dotenv  # <--- Add this
load_dotenv()
# 1. Initialize Router instead of FastAPI
router = APIRouter()

# 2. Shared Config (Qdrant/Gemini)
qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_KEY")
)

class ChatRequest(BaseModel):
    question: str
    session_id: str
    api_key: str

# 3. Routes (Notice we use @router, not @app)
@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    api_key: str = Form(...)
):
    # ... [PASTE YOUR EXISTING UPLOAD CODE HERE] ...
    # (Copy the exact logic from your current upload_pdf function)
    # Just ensure indentation is correct
    try:
        genai.configure(api_key=api_key)
        pdf_reader = PdfReader(io.BytesIO(await file.read()))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found")

        chunk_size = 1000
        overlap = 100
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i+chunk_size])
            
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=chunks,
            task_type="retrieval_document"
        )
        embeddings = result['embedding']

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"session_id": session_id, "text": chunk}
            ))
        
        qdrant_client.upsert(collection_name="pdf_chat", points=points)
        return {"message": "PDF processed"}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat(request: ChatRequest):
    # ... [PASTE YOUR EXISTING CHAT CODE HERE] ...
    try:
        genai.configure(api_key=request.api_key)
        embedding_res = genai.embed_content(
            model="models/gemini-embedding-001",
            content=request.question,
            task_type="retrieval_query"
        )
        query_vector = embedding_res['embedding']
        
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
            return {"answer": "No info found."}

        context_text = "\n\n".join([hit.payload['text'] for hit in search_result])
        prompt = f"Context:\n{context_text}\n\nQuestion: {request.question}\nAnswer:"
        
        model = genai.GenerativeModel('models/gemma-3-4b-it')
        response = model.generate_content(prompt)
        return {"answer": response.text}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
async def cleanup(session_id: str = Form(...)):
    # ... [PASTE YOUR EXISTING CLEANUP CODE HERE] ...
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