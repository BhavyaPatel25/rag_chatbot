"""Local terminal REPL to talk to the Bhavya chatbot.

Usage:
    python3 test_chat.py

Set OLLAMA_HOST / OLLAMA_API_KEY / OLLAMA_MODEL / OLLAMA_EMBED_MODEL first
(local: http://localhost:11434 and no key; Ollama Cloud: your cloud URL + key).
"""
import sys
import uuid

from rag_pipeline import get_answer

SESSION_ID = str(uuid.uuid4())


def main() -> None:
    print("Bhavya chatbot (local) - type 'quit' or 'exit' to leave.\n")
    while True:
        try:
            question = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", ":q"}:
            break
        try:
            answer = get_answer(question=question, session_id=SESSION_ID)
        except Exception as exc:
            print(f"(error: {exc})")
            continue
        print(f"\nBhavya > {answer}\n")


if __name__ == "__main__":
    main()
