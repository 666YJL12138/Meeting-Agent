from typing import TypedDict, List, Dict, Any


class MeetingState(TypedDict, total=False):
    job_id: str | None
    meeting_id: str
    title: str
    audio_uri: str
    normalized_audio_uri: str

    audio_info: Dict[str, Any]
    voice_segments: List[Dict[str, Any]]
    transcript_spans: List[Dict[str, Any]]

    participants: List[str]
    speakers: List[Dict[str, Any]]
    speaker_segments: List[Dict[str, Any]]
    speaker_ids: List[str]
    exclusive_speaker_segments: List[Dict[str, Any]]
    overlap_segments: List[Dict[str, Any]]
    diarization_metrics: Dict[str, Any]
    diarization_evaluation: Dict[str, Any]
    speaker_mapping: Dict[str, str]
    speaker_name_map: Dict[str, str]

    claims: List[Dict[str, Any]]
    speaker_summaries: List[Dict[str, Any]]
    evidence_links: List[Dict[str, Any]]

    timings: Dict[str, float]
    status: str
    progress: int
    index_status: List[str]
    errors: List[str]
