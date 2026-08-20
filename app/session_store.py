"""
app/session_store.py — Redis-backed conversation history and human feedback.

Shared by app.py (Gradio) and streamlit_app.py so both UIs read/write the
same per-document history and feedback records instead of drifting apart
with their own copies. Falls back to in-memory dicts when Redis is
unavailable (matches PersistentSemanticCache's fallback pattern in
rag_pipeline.py).
"""

import json
import threading
from datetime import datetime

from app.config import Config

_redis_client = None
_redis_lock = threading.Lock()
_HISTORY_TTL = 60 * 60 * 24 * 7   # 7 days
_history_mem: dict = {}            # {doc_name: history_text}
_feedback_mem: dict = {}           # {doc_name: [{"rating": ..., "q": ..., "a": ..., "ts": ...}]}


def _get_redis():
    global _redis_client
    if _redis_client is None:
        with _redis_lock:
            if _redis_client is None:
                try:
                    import redis
                    c = redis.from_url(Config.REDIS_URL, decode_responses=True)
                    c.ping()
                    _redis_client = c
                except Exception:
                    _redis_client = False   # mark as unavailable so we don't retry
    return _redis_client if _redis_client else None


def _history_key(doc_name: str) -> str:
    return f"rag:history:{doc_name}"


def save_history(doc_name: str, history_text: str) -> None:
    client = _get_redis()
    if client and doc_name:
        try:
            client.setex(_history_key(doc_name), _HISTORY_TTL, history_text)
            return
        except Exception:
            pass
    if doc_name:
        _history_mem[doc_name] = history_text


def load_history(doc_name: str) -> str:
    client = _get_redis()
    if client and doc_name:
        try:
            return client.get(_history_key(doc_name)) or ""
        except Exception:
            pass
    return _history_mem.get(doc_name, "")


def delete_history(doc_name: str) -> None:
    client = _get_redis()
    if client and doc_name:
        try:
            client.delete(_history_key(doc_name))
        except Exception:
            pass
    _history_mem.pop(doc_name, None)


def _feedback_key(doc_name: str) -> str:
    return f"rag:feedback:{doc_name}"


def save_feedback(doc_name: str, question: str, answer: str, rating: str) -> None:
    """Store one thumbs-up/down record. Redis list when available, else in-memory."""
    entry_dict = {
        "q": question[:120],
        "a": answer[:120],
        "rating": rating,
        "ts": datetime.now().isoformat(),
    }
    client = _get_redis()
    if client and doc_name:
        try:
            key = _feedback_key(doc_name)
            client.rpush(key, json.dumps(entry_dict))
            client.expire(key, 60 * 60 * 24 * 30)
            return
        except Exception:
            pass
    if doc_name:
        _feedback_mem.setdefault(doc_name, []).append(entry_dict)


def get_feedback_stats(doc_name: str) -> dict:
    if not doc_name:
        return {"total": 0, "up": 0, "down": 0, "rate": None}

    client = _get_redis()
    if client:
        try:
            entries = [json.loads(e) for e in client.lrange(_feedback_key(doc_name), 0, -1)]
        except Exception:
            entries = _feedback_mem.get(doc_name, [])
    else:
        entries = _feedback_mem.get(doc_name, [])

    total = len(entries)
    up = sum(1 for e in entries if e.get("rating") == "up")
    return {
        "total": total,
        "up": up,
        "down": total - up,
        "rate": round(up / total * 100, 1) if total else None,
    }
