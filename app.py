import logging
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import get_answer

logger = logging.getLogger(__name__)

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
    session_id = request.cookies.get("session_id")

    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            samesite="none",  
            secure=True       
        )


    try:
        answer = get_answer(question=query.question, session_id=session_id)
    except Exception as exc:
        logger.exception("Error generating answer for session %s", session_id)
        raise HTTPException(status_code=500, detail="Failed to generate a response. Please try again.") from exc

    return {"answer": answer}
