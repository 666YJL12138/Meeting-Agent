from fastapi import FastAPI, UploadFile, File, HTTPException
from uuid import uuid4
from pathlib import Path
import shutil

from .schemas import MeetingCreate, MeetingOut, MeetingStatusOut
from .db import Base, engine
from .models import Meeting
from agents.graph import run_demo_graph

app = FastAPI(title="Meeting Agent API")

STORE = {}
AUDIO_DIR = Path("data/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/meetings", response_model=MeetingOut)
def create_meeting(payload: MeetingCreate):
    meeting_id = uuid4().hex
    item = {
        "meeting_id": meeting_id,
        "title": payload.title,
        "host": payload.host,
        "language": payload.language,
        "status": "created",
        "audio_uri": None,
        "progress": 0,
    }
    STORE[meeting_id] = item
    return item

@app.post("/meetings/{meeting_id}/audio", response_model=MeetingOut)
async def upload_audio(meeting_id: str, file: UploadFile = File(...)):
    if meeting_id not in STORE:
        raise HTTPException(404, "meeting not found")

    suffix = Path(file.filename).suffix or ".wav"
    out_path = AUDIO_DIR / f"{meeting_id}{suffix}"
    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    STORE[meeting_id]["audio_uri"] = str(out_path)
    STORE[meeting_id]["status"] = "audio_uploaded"
    STORE[meeting_id]["progress"] = 10
    return STORE[meeting_id]

@app.post("/meetings/{meeting_id}/run-demo", response_model=MeetingOut)
def run_demo(meeting_id: str):
    if meeting_id not in STORE:
        raise HTTPException(404, "meeting not found")

    state = run_demo_graph(STORE[meeting_id])
    STORE[meeting_id].update(state)
    return STORE[meeting_id]

@app.get("/meetings/{meeting_id}", response_model=MeetingOut)
def get_meeting(meeting_id: str):
    if meeting_id not in STORE:
        raise HTTPException(404, "meeting not found")
    return STORE[meeting_id]

@app.get("/meetings/{meeting_id}/status", response_model=MeetingStatusOut)
def get_status(meeting_id: str):
    if meeting_id not in STORE:
        raise HTTPException(404, "meeting not found")
    item = STORE[meeting_id]
    return {
        "meeting_id": meeting_id,
        "status": item["status"],
        "progress": item["progress"],
        "error": item.get("error"),
    }
