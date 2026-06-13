"""In-memory task status store for background task polling."""
from __future__ import annotations

from typing import Optional

tasks: dict[str, dict] = {}


def set_task(
    task_id: str,
    status: str,
    result: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    """Create or update a task entry."""
    tasks[task_id] = {
        "task_id": task_id,
        "status": status,
        "result": result,
        "error": error,
    }


def get_task(task_id: str) -> Optional[dict]:
    """Return task dict or None if not found."""
    return tasks.get(task_id)