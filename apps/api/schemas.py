from pydantic import BaseModel, Field
from typing import List, Optional, Any, Literal


JobStatus = Literal[
    "created",
    "uploaded",
    "queued",
    "audio_normalizing",
    "asr_processing",
    "diarization_processing",
    "evidence_building",
    "agent_processing",
    "quality_checking",
    "pdf_generating",
    "completed",
    "failed",
    "cancelled",
]


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


class JobHistoryItem(BaseModel):
    timestamp: str
    status: str
    progress: int = Field(default=0, ge=0, le=100)
    stage: str = ""


class JobStatusOut(BaseModel):
    job_id: str
    meeting_id: str
    status: JobStatus
    progress: int = Field(default=0, ge=0, le=100)
    stage: str = ""
    error: str | None = None
    result: dict[str, Any] | None = None
    history: list[JobHistoryItem] = Field(
        default_factory=list
    )


class JobHistoryOut(BaseModel):
    job_id: str
    meeting_id: str
    history: list[JobHistoryItem] = Field(
        default_factory=list
    )


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
    
