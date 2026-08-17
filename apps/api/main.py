from fastapi import FastAPI, UploadFile, File, HTTPException
from uuid import uuid4
from pathlib import Path
import json
import shutil
from fastapi.responses import FileResponse
from services.pdf_report import generate_meeting_pdf
from .schemas import (
    ClaimReviewIn,
    DiarizationDebugOut,
    MeetingCreate,
    MeetingOut,
    MeetingStatusOut,
    SpeakerMappingIn,
)
from .db import Base, engine
from .models import Meeting
from agents.graph import run_audio_asr_graph
from services.cache import cache_meeting_state, load_cached_meeting_state
from services.vector_store import search_meeting
from services.quality_gate import build_quality_report


app = FastAPI(title="Meeting Agent API")

STORE = {}
AUDIO_DIR = Path("data/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR = Path("outputs/asr")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def save_asr_artifacts(meeting_id: str, state: dict) -> dict:
    json_path = RESULT_DIR / f"{meeting_id}.json"
    txt_path = RESULT_DIR / f"{meeting_id}_transcript.txt"

    payload = {
        "meeting_id": state.get("meeting_id"),
        "status": state.get("status"),
        "audio_info": state.get("audio_info", {}),
        "voice_segments": state.get("voice_segments", []),
        "speakers": state.get("speakers", []),
        "speaker_segments": state.get("speaker_segments", []),
        "speaker_mapping": state.get("speaker_mapping", {}),
        "transcript_spans": state.get("transcript_spans", []),
        "evidence_links": state.get("evidence_links", []),
        "claims": state.get("claims", []),
        "speaker_summaries": state.get("speaker_summaries", []),
        "normalized_audio_uri": state.get("normalized_audio_uri"),
        "index_status": state.get("index_status", []),
        "errors": state.get("errors", []),
    }

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    transcript_lines = []
    for span in payload["transcript_spans"]:
        transcript_lines.append(
            f"[{span.get('start_ms', 0)}-{span.get('end_ms', 0)}] {span.get('text', '')}"
        )

    txt_path.write_text("\n".join(transcript_lines), encoding="utf-8")

    return {
        "result_json_path": str(json_path),
        "result_txt_path": str(txt_path),
    }


def restore_meeting_from_cache(meeting_id: str) -> dict | None:
    if meeting_id in STORE:
        return STORE[meeting_id]

    cached = load_cached_meeting_state(meeting_id)
    if cached:
        STORE[meeting_id] = cached
        return cached

    return None

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)


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
    if not restore_meeting_from_cache(meeting_id):
        raise HTTPException(404, "meeting not found")
    return STORE[meeting_id]

@app.get("/meetings/{meeting_id}/status", response_model=MeetingStatusOut)
def get_status(meeting_id: str):
    if not restore_meeting_from_cache(meeting_id):
        raise HTTPException(404, "meeting not found")
    item = STORE[meeting_id]
    return {
        "meeting_id": meeting_id,
        "status": item["status"],
        "progress": item["progress"],
        "error": item.get("error"),
    }


@app.post("/meetings/{meeting_id}/run-asr", response_model=MeetingOut)
def run_asr(meeting_id: str):
    if meeting_id not in STORE:
        raise HTTPException(404, "meeting not found")

    try:
        state = run_audio_asr_graph(STORE[meeting_id])
        STORE[meeting_id].update(state)

        saved_paths = save_asr_artifacts(meeting_id, STORE[meeting_id])
        STORE[meeting_id].update(saved_paths)

        return STORE[meeting_id]
    except Exception as exc:
        STORE[meeting_id]["status"] = "failed"
        STORE[meeting_id]["progress"] = 0
        STORE[meeting_id]["error"] = str(exc)
        raise HTTPException(500, str(exc))


@app.post("/meetings/{meeting_id}/speaker-mapping", response_model=MeetingOut)
def update_speaker_mapping(meeting_id: str, payload: SpeakerMappingIn):
    if meeting_id not in STORE:
        raise HTTPException(404, "meeting not found")

    mapping = payload.mapping
    STORE[meeting_id]["speaker_mapping"] = mapping

    for speaker in STORE[meeting_id].get("speakers", []):
        speaker_id = speaker["speaker_id"]
        if speaker_id in mapping:
            speaker["real_name"] = mapping[speaker_id]
            speaker["display_name"] = mapping[speaker_id]
            speaker["review_status"] = "confirmed"

    cache_meeting_state(meeting_id, STORE[meeting_id])
    return STORE[meeting_id]


@app.get("/meetings/{meeting_id}/diarization", response_model=DiarizationDebugOut)
def get_diarization_result(meeting_id: str):
    if meeting_id not in STORE:
        raise HTTPException(404, "meeting not found")

    item = STORE[meeting_id]
    speaker_segments = item.get("speaker_segments", [])

    source = "unknown"
    if speaker_segments:
        source = speaker_segments[0].get("source", "unknown")

    return {
        "meeting_id": meeting_id,
        "speaker_source": source,
        "speakers": item.get("speakers", []),
        "speaker_segments": speaker_segments,
    }


@app.get("/meetings/{meeting_id}/claims")
def get_claims(meeting_id: str):
    if meeting_id not in STORE:
        raise HTTPException(404, "meeting not found")

    return {
        "meeting_id": meeting_id,
        "claims": STORE[meeting_id].get("claims", []),
    }


@app.get("/meetings/{meeting_id}/speaker-summaries")
def get_speaker_summaries(meeting_id: str):
    if meeting_id not in STORE:
        raise HTTPException(404, "meeting not found")

    return {
        "meeting_id": meeting_id,
        "speaker_summaries": STORE[meeting_id].get("speaker_summaries", []),
    }


@app.get("/meetings/{meeting_id}/search")
def search_meeting_evidence(
    meeting_id: str,
    q: str,
    limit: int = 5,
):
    if not restore_meeting_from_cache(meeting_id):
        raise HTTPException(404, "meeting not found")

    query = q.strip()
    if not query:
        raise HTTPException(400, "query cannot be empty")

    safe_limit = max(1, min(limit, 20))

    try:
        hits = search_meeting(
            meeting_id=meeting_id,
            query=query,
            limit=safe_limit,
        )
    except Exception as exc:
        raise HTTPException(
            503,
            f"evidence search unavailable: {exc}",
        ) from exc

    return {
        "meeting_id": meeting_id,
        "query": query,
        "hits": hits,
    }


@app.post(
    "/meetings/{meeting_id}/claims/{claim_id}/review",
)
def review_claim(
    meeting_id: str,
    claim_id: str,
    payload: ClaimReviewIn,
):
    if not restore_meeting_from_cache(meeting_id):
        raise HTTPException(404, "meeting not found")

    for claim in STORE[meeting_id].get("claims", []):
        if claim.get("claim_id") != claim_id:
            continue

        claim["review_status"] = payload.review_status
        claim["review_note"] = payload.review_note
        claim["reviewer"] = payload.reviewer
        cache_meeting_state(meeting_id, STORE[meeting_id])
        return claim

    raise HTTPException(404, "claim not found")


def load_meeting_state_for_report(meeting_id: str) -> dict:
    if meeting_id in STORE:
        return STORE[meeting_id]

    json_path = RESULT_DIR / f"{meeting_id}.json"
    if not json_path.exists():
        raise HTTPException(404, "meeting not found")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    data.setdefault("title", "离线会议")
    data.setdefault("host", "-")
    data.setdefault("language", "zh-CN")

    return data


@app.get("/meetings/{meeting_id}/report/pdf")
def download_pdf_report(meeting_id: str):
    meeting = load_meeting_state_for_report(meeting_id)

    if not meeting.get("claims"):
        raise HTTPException(400, "claims not found, please run /run-asr first")

    pdf_path = generate_meeting_pdf(meeting)

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"{meeting_id}_trusted_minutes.pdf",
    )


@app.get("/meetings/{meeting_id}/quality-report")
def get_quality_report(meeting_id: str):
    result = restore_meeting_from_cache(meeting_id)

    if not result:
        raise HTTPException(status_code=404, detail="Meeting result not found")

    return build_quality_report(result)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "meeting-agent-api",
        "capabilities": {
            "asr": True,
            "diarization": True,
            "contribution_extraction": True,
            "pdf_report": True,
            "redis_cache": True,
            "qdrant_search": True,
            "local_bge": True,
        },
    }


