from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from rag_pipeline import get_answer

app = FastAPI(title="Hybrid RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "Hybrid RAG API running"}

@app.post("/chat")
def chat(query: Query):
    return {
        "question": query.question,
        "answer": get_answer(query.question)
    }
