"""In-memory job store.

Keeps each run's response and generated file path. Simple and dependency-free; on
an ephemeral host (e.g. Render) files live only for the process lifetime, which is
fine for this demo — a persistent store (S3/DB) would be the production swap.
"""
import threading
from typing import Dict, List, Optional

from .models import AgentResponse


class JobStore:
    def __init__(self) -> None:
        self._data: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def save(self, document_id: str, response: AgentResponse, path: str) -> None:
        with self._lock:
            self._data[document_id] = {"response": response, "path": path}

    def get(self, document_id: str) -> Optional[dict]:
        with self._lock:
            return self._data.get(document_id)

    def list_ids(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())
