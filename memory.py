from collections import deque
from typing import Dict, Deque, List

MAX_MESSAGES = 10

_memory_store: Dict[str, Deque[dict]] = {}

def _get_or_create(session_id: str) -> Deque[dict]:
    return _memory_store.setdefault(session_id, deque(maxlen=MAX_MESSAGES))

def get_memory(session_id: str) -> List[dict]:
    return list(_get_or_create(session_id))

def append_memory(session_id: str, role: str, content: str) -> None:
    _get_or_create(session_id).append({"role": role, "content": content})
