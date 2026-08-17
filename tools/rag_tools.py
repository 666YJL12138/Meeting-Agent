import json
from pathlib import Path


def load_agent_result(meeting_id: str) -> dict:
    path = Path("outputs/agent") / f"{meeting_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Agent result not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def search_meeting_evidence(
    meeting_id: str,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    data = load_agent_result(meeting_id)
    evidence = data.get("evidence", [])

    hits = []
    for item in evidence:
        text = item.get("text", "")
        score = 0
        for token in query:
            if token in text:
                score += 1

        if score > 0:
            hits.append({
                "score": score,
                "evidence_id": item["evidence_id"],
                "speaker_id": item["speaker_id"],
                "text": text,
                "start_ms": item["start_ms"],
                "end_ms": item["end_ms"],
            })

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:top_k]
