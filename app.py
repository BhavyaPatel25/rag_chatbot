from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uuid

from rag_pipeline import get_answer

app = FastAPI(title="Hybrid RAG API with Session Memory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://bhavyapatel25.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "Hybrid RAG API running (auto session enabled)"}

@app.post("/chat")
def chat(query: Query, request: Request, response: Response):
    # 1. Get session_id from cookie
    session_id = request.cookies.get("session_id")

    # 2. If missing, generate a new one
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            samesite="none",  
            secure=True       
        )


    # 3. Generate answer using session-specific memory
    answer = get_answer(
        question=query.question,
        session_id=session_id
    )

    return {
        "answer": answer
    }
