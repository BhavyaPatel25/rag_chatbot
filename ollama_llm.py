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

MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")

def generate_answer(question: str, context: str, memory: list[dict]) -> str:
    messages = []

    # System instruction
    messages.append({
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Use the provided context to answer. "
            "Use conversation history if relevant. "
            "If the answer is not in the context, say you do not know.\n\n"
            "Gender is Male"
            f"Context:\n{context}"
            "Format the final output in professional way and don't use any styling like bold, italic, etc. "
        )
    })

    for msg in memory:
        messages.append(msg)

    messages.append({
        "role": "user",
        "content": question
    })

    response = client.chat(
        model=MODEL,
        messages=messages,
        stream=False
    )

    return response["message"]["content"]
