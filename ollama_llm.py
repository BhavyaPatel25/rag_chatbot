import os
from ollama import Client
from dotenv import load_dotenv

load_dotenv()

client = Client(
    host="https://ollama.com",
    headers={
        "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
    }
)

MODEL = "gpt-oss:120b"

def generate_answer(question: str, context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer the question using ONLY the context below. "
                "If the answer is not in the context, say you do not know.\n\n"
                f"Context:\n{context}"
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    response = client.chat(
        model=MODEL,
        messages=messages,
        stream=False
    )

    return response["message"]["content"]
