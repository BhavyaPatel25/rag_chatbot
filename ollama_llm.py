import os
from ollama import Client
from dotenv import load_dotenv

load_dotenv()

client = Client(
    host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    headers={
        "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY', '')}"
    }
)

MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")

_SYSTEM_PROMPT = (
    "You are a helpful assistant representing Bhavya Patel (Male).\n"
    "Use the provided context to answer questions accurately.\n"
    "Use conversation history if relevant to the current question.\n"
    "If the answer is not in the context, say you do not know.\n"
    "Format responses professionally without markdown styling (no bold, italic, or headers).\n\n"
)

def generate_answer(question: str, context: str, memory: list[dict]) -> str:
    system_content = _SYSTEM_PROMPT + f"Context:\n{context}"

    messages = [{"role": "system", "content": system_content}]
    messages.extend(memory)
    messages.append({"role": "user", "content": question})

    response = client.chat(model=MODEL, messages=messages, stream=False)

    # Support both modern Pydantic response (.message.content) and legacy dict response
    if hasattr(response, "message"):
        return response.message.content
    return response["message"]["content"]
