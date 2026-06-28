from retriever import retrieve_context
from ollama_llm import condense_question, generate_answer
from memory import get_memory, append_memory


def get_answer(question: str, session_id: str) -> str:
    memory = get_memory(session_id)

    # Resolve pronouns / follow-ups into a self-contained question for retrieval.
    retrieval_query = condense_question(question, memory)
    context = retrieve_context(retrieval_query)

    answer = generate_answer(question=question, context=context, memory=memory)

    append_memory(session_id, "user", question)
    append_memory(session_id, "assistant", answer)

    return answer
