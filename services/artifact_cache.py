import hashlib
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / "outputs" / "cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def build_cache_key(
    path: str,
    namespace: str,
    config: dict,
) -> str:
    payload = {
        "audio_sha256": file_sha256(path),
        "namespace": namespace,
        "config": config,
    }

    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def build_agent_cache_key(
    meeting_id: str,
    evidence: list[dict],
    config: dict,
) -> str:
    """Build a stable key from the exact evidence and LLM configuration."""
    payload = {
        "meeting_id": meeting_id,
        "namespace": "agent",
        "evidence": evidence,
        "config": config,
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cache_path(namespace: str, key: str) -> Path:
    directory = CACHE_ROOT / namespace
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.json"


def load_cache(namespace: str, key: str) -> dict | None:
    path = cache_path(namespace, key)

    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return None


def save_cache(namespace: str, key: str, value: dict) -> None:
    path = cache_path(namespace, key)
    temp_path = path.with_suffix(".tmp")

    temp_path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(temp_path, path)
