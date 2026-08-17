from typing import TypedDict


class MeetingAgentState(TypedDict):
    meeting_id: str
    title: str
    evidence: list[dict]
    speaker_ids: list[str]
    contributions: list[dict]
    action_items: list[dict]
    risks: list[dict]
    errors: list[str]
