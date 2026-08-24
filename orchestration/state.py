from typing import TypedDict


class MeetingAgentState(TypedDict, total=False):
    meeting_id: str
    title: str
    host: str
    language: str
    audio_info: dict

    evidence: list[dict]
    speaker_ids: list[str]

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

    errors: list[str]
    timings: dict[str, float]
