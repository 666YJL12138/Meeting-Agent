from typing import TypedDict, List, Dict, Any, Optional

class MeetingState(TypedDict, total=False):
    meeting_id: str
    title: str
    audio_uri: str
    transcript_spans: List[Dict[str, Any]]
    speakers: List[Dict[str, Any]]
    claims: List[Dict[str, Any]]
    evidence_links: List[Dict[str, Any]]
    status: str
    progress: int
    errors: List[str]
