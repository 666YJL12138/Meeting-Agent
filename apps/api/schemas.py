from pydantic import BaseModel, Field
from typing import List, Optional

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

class MeetingStatusOut(BaseModel):
    meeting_id: str
    status: str
    progress: int = 0
    error: Optional[str] = None
