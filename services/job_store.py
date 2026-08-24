import json
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import redis


REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://127.0.0.1:6379/0",
)

JOB_TTL_SECONDS = int(
    os.getenv("JOB_TTL_SECONDS", "86400")
)
MAX_HISTORY_ITEMS = int(
    os.getenv("JOB_MAX_HISTORY_ITEMS", "200")
)


def get_client() -> redis.Redis:
    return redis.Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )


def now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _history_event(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": now_iso(),
        "status": job.get("status", "created"),
        "progress": int(job.get("progress", 0)),
        "stage": str(job.get("stage", "")),
    }


def append_history(
    job: dict[str, Any],
    *,
    force: bool = False,
) -> None:
    """Append a stage snapshot unless it duplicates the previous one."""
    history = job.setdefault("history", [])

    if not isinstance(history, list):
        history = []
        job["history"] = history

    event = _history_event(job)
    if not force and history:
        last = history[-1]
        same_stage = all(
            last.get(key) == event[key]
            for key in ("status", "progress", "stage")
        )
        if same_stage:
            return

    history.append(event)
    job["history"] = history[-MAX_HISTORY_ITEMS:]


def create_job(meeting_id: str) -> dict[str, Any]:
    job = {
        "job_id": f"job_{uuid4().hex}",
        "meeting_id": meeting_id,
        "status": "created",
        "progress": 0,
        "stage": "任务已创建",
        "error": None,
        "result": None,
        "created_at": now_iso(),
        "started_at": None,
        "finished_at": None,
        "history": [],
    }

    append_history(job, force=True)
    save_job(job)
    return job


def save_job(job: dict[str, Any]) -> None:
    key = f"meeting-agent:job:{job['job_id']}"

    get_client().set(
        key,
        json.dumps(
            job,
            ensure_ascii=False,
        ),
        ex=JOB_TTL_SECONDS,
    )


def get_job(job_id: str) -> dict[str, Any] | None:
    key = f"meeting-agent:job:{job_id}"
    raw = get_client().get(key)

    if not raw:
        return None

    return json.loads(raw)


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    progress: int | None = None,
    stage: str | None = None,
    error: str | None = None,
    result: dict[str, Any] | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    job = get_job(job_id)

    if not job:
        raise KeyError(
            f"job not found: {job_id}"
        )

    # Jobs created before history support do not have a timeline. Preserve
    # their last known snapshot before recording the next update.
    if not isinstance(job.get("history"), list):
        job["history"] = []
        append_history(job, force=True)

    previous = {
        "status": job.get("status"),
        "progress": job.get("progress"),
        "stage": job.get("stage"),
    }

    previous_progress = int(
        job.get("progress", 0)
    )

    if status is not None:
        job["status"] = status

    if progress is not None:
        job["progress"] = max(
            previous_progress,
            min(100, max(0, int(progress))),
        )

    if stage is not None:
        job["stage"] = stage

    if error is not None:
        job["error"] = str(error)[:1000]

    if result is not None:
        job["result"] = result

    if started_at is not None:
        job["started_at"] = started_at

    if finished_at is not None:
        job["finished_at"] = finished_at

    current = {
        "status": job.get("status"),
        "progress": job.get("progress"),
        "stage": job.get("stage"),
    }
    if current != previous:
        append_history(job)

    save_job(job)
    return job
