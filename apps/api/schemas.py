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
    transcript_spans: list[dict[str, Any]] = []
    evidence_links: list[dict[str, Any]] = []

class MeetingStatusOut(BaseModel):
    meeting_id: str
    status: str
    progress: int = 0
    error: Optional[str] = None
