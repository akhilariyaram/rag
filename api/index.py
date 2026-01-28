# api/index.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import your routers
from routers import pdf_chat 
# from .routers import todo_app  <-- In the future, you add this!

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MOUNT THE PROJECTS ---

# Project 1: PDF Chat
# Endpoints become: /api/pdf/upload, /api/pdf/chat, etc.
app.include_router(pdf_chat.router, prefix="/api/pdf", tags=["PDF Chat"])

# Project 2: Todo App (Future)
# app.include_router(todo_app.router, prefix="/api/todo", tags=["Todo App"])

@app.get("/api/health")
def health():
    return {"status": "Active", "projects": ["pdf_chat"]}