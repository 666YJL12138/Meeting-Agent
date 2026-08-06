from sqlalchemy import Column, String, Text, Integer
from .db import Base

class Meeting(Base):
    __tablename__ = "meetings"

    meeting_id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    host = Column(String, nullable=False)
    language = Column(String, default="zh-CN")
    status = Column(String, default="created")
    audio_uri = Column(Text, nullable=True)
    progress = Column(Integer, default=0)
