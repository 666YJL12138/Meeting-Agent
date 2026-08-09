from typing import TypedDict, List, Dict, Any


class MeetingState(TypedDict, total=False):
    meeting_id: str
    title: str
    audio_uri: str
    normalized_audio_uri: str

    audio_info: Dict[str, Any]
    voice_segments: List[Dict[str, Any]]
    transcript_spans: List[Dict[str, Any]]

    speakers: List[Dict[str, Any]]
    speaker_segments: List[Dict[str, Any]]
    speaker_mapping: Dict[str, str]

    claims: List[Dict[str, Any]]
    evidence_links: List[Dict[str, Any]]

    status: str
    progress: int
    errors: List[str]
