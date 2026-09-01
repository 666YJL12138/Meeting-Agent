from typing import TypedDict


class MeetingAgentState(TypedDict, total=False):
    meeting_id: str
    title: str
    host: str
    language: str
    audio_info: dict

    evidence: list[dict]
    speaker_ids: list[str]
    participants: list[str]
    speaker_mapping: dict[str, str]
    transcript_spans: list[dict]
    speaker_segments: list[dict]
    exclusive_speaker_segments: list[dict]
    overlap_segments: list[dict]
    diarization_metrics: dict
    diarization_evaluation: dict

    contributions: list[dict]
    action_items: list[dict]
    risks: list[dict]

    rag_supplements: list[dict]

    claims: list[dict]
    speaker_summaries: list[dict]
    evidence_links: list[dict]

    pdf_path: str

    send_email: bool
    to_email: str | None
    email_subject: str | None
    email_body: str | None
    email_delivery: dict | None

    compact_evidence_json: str
    agent_cache_hit: bool
    review_required: bool
    review_reasons: list[str]
    review_queue: list[dict]
    retry_count: int
    node_attempts: dict[str, int]
    orchestration_trace: list[dict]
    errors: list[str]
    timings: dict[str, float]
