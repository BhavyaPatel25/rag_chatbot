from collections import deque
from typing import Dict, Deque, List

# Stores last 10 messages per session
# message format: {"role": "user" | "assistant", "content": str}

MAX_MESSAGES = 10

_memory_store: Dict[str, Deque[dict]] = {}

def get_memory(session_id: str) -> List[dict]:
    if session_id not in _memory_store:
        _memory_store[session_id] = deque(maxlen=MAX_MESSAGES)
    return list(_memory_store[session_id])

def append_memory(session_id: str, role: str, content: str):
    if session_id not in _memory_store:
        _memory_store[session_id] = deque(maxlen=MAX_MESSAGES)

    _memory_store[session_id].append({
        "role": role,
        "content": content
    })
