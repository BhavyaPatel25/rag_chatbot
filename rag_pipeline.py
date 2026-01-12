from retriever import retrieve_context
from ollama_llm import generate_answer

def get_answer(question: str) -> str:
    context = retrieve_context(question)
    return generate_answer(question, context)
