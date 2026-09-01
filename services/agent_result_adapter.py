from collections import defaultdict


def _repair_mojibake(value):
    if isinstance(value, str):
        try:
            repaired = value.encode("latin1").decode("utf-8")
        except UnicodeError:
            return value
        return repaired if repaired else value
    if isinstance(value, list):
        return [_repair_mojibake(item) for item in value]
    if isinstance(value, dict):
        return {key: _repair_mojibake(item) for key, item in value.items()}
    return value


def _speaker_name_map(state: dict) -> dict[str, str]:
    speaker_mapping = state.get("speaker_mapping") or state.get("speaker_name_map")
    if isinstance(speaker_mapping, dict):
        return {
            str(key): str(value)
            for key, value in speaker_mapping.items()
            if str(key).strip() and str(value).strip()
        }
    return {}


def _display_name(item: dict, mapping: dict[str, str]) -> str:
    speaker_id = item.get("speaker_id", "speaker_unknown")
    return (
        item.get("speaker_name")
        or item.get("display_name")
        or mapping.get(speaker_id)
        or speaker_id
    )


def _speaker_ids(item: dict) -> list[str]:
    values = item.get("speaker_ids")
    if not isinstance(values, list):
        values = [item.get("speaker_id", "speaker_unknown")]

    speaker_ids = []
    for value in values:
        speaker_id = str(value or "").strip()
        if speaker_id and speaker_id not in speaker_ids:
            speaker_ids.append(speaker_id)
    return speaker_ids or ["speaker_unknown"]


def build_pdf_ready_result(state: dict) -> dict:
    """Convert the multi-agent state into the schema expected by the PDF service."""
    claims = []
    speaker_mapping = _speaker_name_map(state)
    evidence_links = _build_evidence_links(
        state.get("evidence", []),
        speaker_mapping,
    )

    for claim in _all_agent_claims(state):
        speaker_name = _display_name(claim, speaker_mapping)
        claims.append({
            "claim_id": _build_claim_id(claim),
            "speaker_id": claim.get("speaker_id", "speaker_unknown"),
            "speaker_name": speaker_name,
            "display_name": speaker_name,
            "claim_type": claim.get("claim_type", "瑙傜偣"),
            "statement": claim.get("summary", ""),
            "summary": claim.get("summary", ""),
            "quote": claim.get("quote", ""),
            "evidence_ids": claim.get("evidence_ids", []),
            "start_ms": claim.get("start_ms", 0),
            "end_ms": claim.get("end_ms", 0),
            "confidence": claim.get("confidence", 0),
            "confidence_breakdown": claim.get("confidence_breakdown", {}),
            "confidence_method": claim.get(
                "confidence_method",
                "evidence_weighted_v1",
            ),
            "review_status": claim.get("support_status", "supported"),
            "support_status": claim.get("support_status", "supported"),
            "review_reason": claim.get("review_reason"),
        })

    return {
        "meeting_id": state["meeting_id"],
        "title": _repair_mojibake(state.get("title", "会议纪要")),
        "host": state.get("host", "-"),
        "language": state.get("language", "zh-CN"),
        "status": "agent_workflow_done",
        "audio_info": state.get("audio_info", {}),
        "speaker_ids": state.get("speaker_ids", []),
        "speaker_segments": state.get("speaker_segments", []),
        "exclusive_speaker_segments": state.get(
            "exclusive_speaker_segments",
            [],
        ),
        "overlap_segments": state.get("overlap_segments", []),
        "diarization_metrics": state.get(
            "diarization_metrics",
            {},
        ),
        "diarization_evaluation": state.get(
            "diarization_evaluation",
            {},
        ),
        "speaker_mapping": speaker_mapping,
        "speaker_name_map": speaker_mapping,
        "claims": claims,
        "speaker_summaries": _build_speaker_summaries(claims),
        "evidence_links": evidence_links,
        "rag_supplements": state.get("rag_supplements", []),
        "review_required": state.get("review_required", False),
        "review_reasons": state.get("review_reasons", []),
        "review_queue": state.get("review_queue", []),
        "retry_count": state.get("retry_count", 0),
        "node_attempts": state.get("node_attempts", {}),
        "orchestration_trace": state.get("orchestration_trace", []),
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


def _build_evidence_links(
    evidence: list[dict],
    speaker_mapping: dict[str, str] | None = None,
) -> list[dict]:
    speaker_mapping = speaker_mapping or {}
    return [
        {
            "evidence_id": item.get("evidence_id"),
            "evidence_hash": item.get("evidence_hash"),
            "speaker_id": item.get("speaker_id"),
            "speaker_ids": _speaker_ids(item),
            "speaker_names": [
                speaker_mapping.get(speaker_id, speaker_id)
                for speaker_id in _speaker_ids(item)
            ],
            "speaker_confidences": item.get(
                "speaker_confidences",
                {},
            ),
            "speaker_candidates": item.get(
                "speaker_candidates",
                [],
            ),
            "overlap": bool(
                item.get("overlap", False)
                or len(_speaker_ids(item)) > 1
            ),
            "confidence_source": item.get(
                "confidence_source",
                "derived_alignment",
            ),
            "speaker_name": item.get("speaker_name")
            or item.get("display_name")
            or speaker_mapping.get(item.get("speaker_id"))
            or item.get("speaker_id"),
            "quote": item.get("text", ""),
            "text": item.get("text", ""),
            "start_ms": item.get("start_ms", 0),
            "end_ms": item.get("end_ms", 0),
            "asr_confidence": item.get("asr_confidence", 0.5),
            "speaker_confidence": item.get("speaker_confidence", 0.0),
            "speaker_source": item.get("speaker_source", "unknown"),
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
            "display_name": items[0].get("speaker_name")
            or items[0].get("display_name")
            or speaker_id,
            "key_points": [
                {
                    **item,
                    "confidence_breakdown": item.get(
                        "confidence_breakdown",
                        {},
                    ),
                    "confidence_method": item.get(
                        "confidence_method",
                        "evidence_weighted_v1",
                    ),
                }
                for item in items
            ],
        }
        for speaker_id, items in grouped.items()
    ]
