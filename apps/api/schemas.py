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
    title: str = "真实会议测试"
    host: str = "主持人"
    language: str = "zh-CN"
    participants: List[str] = Field(
        default_factory=lambda: ["张三", "李四", "王五"]
    )

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
    exclusive_speaker_segments: list[dict[str, Any]] = []
    overlap_segments: list[dict[str, Any]] = []
    diarization_metrics: dict[str, Any] = {}
    diarization_evaluation: dict[str, Any] = {}
    speaker_mapping: dict[str, str] = {}
    speaker_name_map: dict[str, str] = {}
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
    exclusive_speaker_segments: list[dict[str, Any]] = []
    overlap_segments: list[dict[str, Any]] = []
    diarization_metrics: dict[str, Any] = {}
    diarization_evaluation: dict[str, Any] = {}


class DiarizationSegmentIn(BaseModel):
    speaker_id: str
    start_ms: int
    end_ms: int
    confidence: float | None = None
    confidence_source: str | None = None


class DiarizationEvaluationIn(BaseModel):
    reference_speaker_segments: list[DiarizationSegmentIn]
    hypothesis_speaker_segments: list[DiarizationSegmentIn] | None = None
    reference_overlap_segments: list[dict[str, Any]] = Field(default_factory=list)
    hypothesis_overlap_segments: list[dict[str, Any]] = Field(default_factory=list)
    collar_seconds: float = Field(default=0.0, ge=0.0)
    skip_overlap: bool = False


class DiarizationEvaluationOut(BaseModel):
    meeting_id: str
    evaluation_status: str
    reference_speaker_count: int = 0
    hypothesis_speaker_count: int = 0
    speaker_label_mapping: dict[str, str] = Field(default_factory=dict)
    speaker_metrics: dict[str, Any] = Field(default_factory=dict)
    overlap_metrics: dict[str, Any] = Field(default_factory=dict)


class ClaimReviewIn(BaseModel):
    review_status: str = "reviewed"
    review_note: str = ""
    reviewer: str = "human"
    
