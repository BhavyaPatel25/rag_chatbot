from retriever import retrieve_context
from ollama_llm import generate_answer
from memory import get_memory, append_memory

def get_answer(question: str, session_id: str) -> str:
    memory = get_memory(session_id)
    context = retrieve_context(question)

    answer = generate_answer(
        question=question,
        context=context,
        memory=memory
    )

    append_memory(session_id, "user", question)
    append_memory(session_id, "assistant", answer)

    return answer
