from collections import defaultdict


def build_pdf_ready_result(state: dict) -> dict:
    """Convert the multi-agent state into the schema expected by the PDF service."""
    claims = []
    evidence_links = _build_evidence_links(state.get("evidence", []))

    for claim in _all_agent_claims(state):
        claims.append({
            "claim_id": _build_claim_id(claim),
            "speaker_id": claim.get("speaker_id", "speaker_unknown"),
            "claim_type": claim.get("claim_type", "瑙傜偣"),
            "statement": claim.get("summary", ""),
            "summary": claim.get("summary", ""),
            "quote": claim.get("quote", ""),
            "evidence_ids": claim.get("evidence_ids", []),
            "start_ms": claim.get("start_ms", 0),
            "end_ms": claim.get("end_ms", 0),
            "confidence": claim.get("confidence", 0),
            "review_status": claim.get("support_status", "supported"),
            "support_status": claim.get("support_status", "supported"),
        })

    return {
        "meeting_id": state["meeting_id"],
        "title": state.get("title", "浼氳绾"),
        "host": state.get("host", "-"),
        "language": state.get("language", "zh-CN"),
        "status": "agent_workflow_done",
        "audio_info": state.get("audio_info", {}),
        "claims": claims,
        "speaker_summaries": _build_speaker_summaries(claims),
        "evidence_links": evidence_links,
        "rag_supplements": state.get("rag_supplements", []),
        "errors": state.get("errors", []),
    }


def _all_agent_claims(state: dict) -> list[dict]:
    return (
        state.get("contributions", [])
        + state.get("action_items", [])
        + state.get("risks", [])
    )


def _build_claim_id(claim: dict) -> str:
    evidence_id = "_".join(claim.get("evidence_ids", [])) or "no_evidence"
    return f"{claim.get('claim_type', 'claim')}_{evidence_id}_{claim.get('start_ms', 0)}"


def _build_evidence_links(evidence: list[dict]) -> list[dict]:
    return [
        {
            "evidence_id": item.get("evidence_id"),
            "speaker_id": item.get("speaker_id"),
            "quote": item.get("text", ""),
            "text": item.get("text", ""),
            "start_ms": item.get("start_ms", 0),
            "end_ms": item.get("end_ms", 0),
        }
        for item in evidence
        if item.get("text")
    ]


def _build_speaker_summaries(claims: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for claim in claims:
        grouped[claim.get("speaker_id", "speaker_unknown")].append(claim)

    return [
        {
            "speaker_id": speaker_id,
            "display_name": speaker_id,
            "key_points": items,
        }
        for speaker_id, items in grouped.items()
    ]