from pydantic import BaseModel, Field
from typing import List, Optional, Any

class MeetingCreate(BaseModel):
    title: str
    host: str
    language: str = "zh-CN"
    participants: List[str] = []

class MeetingOut(BaseModel):
    meeting_id: str
    title: str
    host: str
    language: str
    status: str
    audio_uri: Optional[str] = None
    progress: int = 0

    normalized_audio_uri: Optional[str] = None
    audio_info: Optional[dict[str, Any]] = None
    voice_segments: list[dict[str, Any]] = []
    speakers: list[dict[str, Any]] = []
    speaker_segments: list[dict[str, Any]] = []
    speaker_mapping: dict[str, str] = {}
    transcript_spans: list[dict[str, Any]] = []
    evidence_links: list[dict[str, Any]] = []  
    claims: list[dict[str, Any]] = Field(default_factory=list)
    speaker_summaries: list[dict[str, Any]] = Field(default_factory=list)

class MeetingStatusOut(BaseModel):
    meeting_id: str
    status: str
    progress: int = 0
    error: Optional[str] = None


class SpeakerMappingIn(BaseModel):
    mapping: dict[str, str]


class DiarizationDebugOut(BaseModel):
    meeting_id: str
    speaker_source: str
    speakers: list[dict[str, Any]]
    speaker_segments: list[dict[str, Any]]


class ClaimReviewIn(BaseModel):
    review_status: str = "reviewed"
    review_note: str = ""
    reviewer: str = "human"
    
