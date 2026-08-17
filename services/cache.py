import json
import os

import redis


REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
STATE_TTL_SECONDS = int(os.getenv("MEETING_STATE_TTL_SECONDS", "86400"))


def get_redis_client():
    return redis.Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )


def cache_meeting_state(
    meeting_id: str,
    state: dict,
    ttl_seconds: int = STATE_TTL_SECONDS,
) -> None:
    payload = json.dumps(
        state,
        ensure_ascii=False,
        default=str,
    )
    get_redis_client().set(
        f"meeting:{meeting_id}:state",
        payload,
        ex=ttl_seconds,
    )


def load_cached_meeting_state(meeting_id: str) -> dict | None:
    raw = get_redis_client().get(f"meeting:{meeting_id}:state")
    if not raw:
        return None
    return json.loads(raw)
