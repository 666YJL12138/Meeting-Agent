from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from uuid import uuid4
from pathlib import Path
import json
import shutil
from fastapi.responses import FileResponse
from services.pdf_report import generate_meeting_pdf
from .schemas import (
    ClaimReviewIn,
    DiarizationDebugOut,
    JobHistoryOut,
    JobStatusOut,
    MeetingCreate,
    MeetingOut,
    MeetingStatusOut,
    SpeakerMappingIn,
)
from services.job_store import create_job, get_job, update_job
from services.workflow_runner import run_full_workflow
from .db import Base, engine
from .models import Meeting
from agents.graph import run_audio_asr_graph
from services.cache import cache_meeting_state, load_cached_meeting_state
from services.vector_store import search_meeting
from services.quality_gate import build_quality_report
from orchestration.graph import build_meeting_graph
from pydantic import BaseModel, EmailStr
from tools.email_tools import send_report_email_tool
from services.email_sender import EmailSendError


app = FastAPI(title="Meeting Agent API")

STORE = {}
AUDIO_DIR = Path("data/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR = Path("outputs/asr")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


class SendReportEmailRequest(BaseModel):
    to_email: EmailStr
    subject: str | None = None
    body: str | None = None

class RunAgentWorkflowRequest(BaseModel):
    send_email: bool = False
    to_email: EmailStr | None = None
    email_subject: str | None = None
    email_body: str | None = None

def get_latest_report_pdf_path(meeting_id: str) -> str:
    report_dir = Path("outputs/reports")
    candidates = sorted(
        report_dir.glob(f"{meeting_id}*.pdf"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise HTTPException(status_code=404, detail="PDF report not found")
    return str(candidates[0])


def save_asr_artifacts(meeting_id: str, state: dict) -> dict:
    json_path = RESULT_DIR / f"{meeting_id}.json"
    txt_path = RESULT_DIR / f"{meeting_id}_transcript.txt"

    payload = {
        "meeting_id": state.get("meeting_id"),
        "title": state.get("title", "会议纪要"),
        "host": state.get("host", "-"),
        "language": state.get("language", "zh-CN"),
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


def load_asr_result_for_agent(meeting_id: str) -> dict:
    path = Path("outputs/asr") / f"{meeting_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"ASR result not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_agent_result(meeting_id: str, result: dict) -> str:
    output_dir = Path("outputs/agent")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{meeting_id}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


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
        "participants": payload.participants,
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


def build_evidence_from_asr(meeting_id: str, asr_result: dict) -> list[dict]:
    spans = asr_result.get("transcript_spans", [])
    evidence_links = asr_result.get("evidence_links", [])
    evidence_id_by_span_id = {
        item.get("span_id"): item.get("evidence_id")
        for item in evidence_links
        if item.get("span_id") and item.get("evidence_id")
    }

    evidence = []
    for index, span in enumerate(spans):
        text = str(span.get("text", "")).strip()
        if not text:
            continue

        span_id = span.get("span_id")
        if span_id:
            evidence_id = evidence_id_by_span_id.get(span_id) or f"ev_{span_id}"
        else:
            evidence_id = f"seg_{index:04d}"

        evidence.append({
            "evidence_id": evidence_id,
            "meeting_id": meeting_id,
            "speaker_id": span.get("speaker_id", "speaker_unknown"),
            "text": text,
            "start_ms": int(span.get("start_ms", 0)),
            "end_ms": int(span.get("end_ms", 0)),
            "asr_confidence": float(span.get("asr_confidence", 0.5)),
            "speaker_confidence": float(span.get("speaker_confidence", 0.5)),
            "speaker_source": span.get("speaker_source", "unknown"),
        })

    return evidence

@app.post("/meetings/{meeting_id}/run-agent")
def run_agent(meeting_id: str):
    asr_result = load_asr_result_for_agent(meeting_id)

    spans = asr_result.get("transcript_spans", [])
    evidence = []

    for index, span in enumerate(spans):
        text = span.get("text", "").strip()
        if not text:
            continue

        evidence.append({
            "evidence_id": f"seg_{index:04d}",
            "meeting_id": meeting_id,
            "speaker_id": span.get("speaker_id", "UNKNOWN"),
            "text": text,
            "start_ms": int(span.get("start_ms", 0)),
            "end_ms": int(span.get("end_ms", 0)),
        })

    state = {
        "meeting_id": meeting_id,
        "title": asr_result.get("title", "会议纪要"),
        "evidence": evidence,
        "speaker_ids": sorted({item["speaker_id"] for item in evidence}),
        "contributions": [],
        "action_items": [],
        "risks": [],
        "errors": [],
    }

    graph = build_meeting_graph()
    result = graph.invoke(state)
    output_path = save_agent_result(meeting_id, result)
    result["agent_result_path"] = output_path
    return result


@app.post("/meetings/{meeting_id}/run-agent-workflow")
def run_agent_workflow(
    meeting_id: str,
    request: RunAgentWorkflowRequest | None = None,
):
    request = request or RunAgentWorkflowRequest()
    asr_result = load_asr_result_for_agent(meeting_id)
    evidence = build_evidence_from_asr(meeting_id, asr_result)

    if not evidence:
        raise HTTPException(
            status_code=400,
            detail="No evidence found, please run /run-asr first",
        )

    state = {
        "meeting_id": meeting_id,
        "title": asr_result.get("title", "会议纪要"),
        "host": asr_result.get("host", "-"),
        "language": asr_result.get("language", "zh-CN"),
        "audio_info": asr_result.get("audio_info", {}),
        "evidence": evidence,
        "speaker_ids": sorted({item["speaker_id"] for item in evidence}),
        "contributions": [],
        "action_items": [],
        "risks": [],
        "rag_supplements": [],
        "claims": [],
        "speaker_summaries": [],
        "evidence_links": [],
        "send_email": request.send_email,
        "to_email": str(request.to_email) if request.to_email else None,
        "email_subject": request.email_subject,
        "email_body": request.email_body,
        "email_delivery": None,
        "errors": [],
    }

    result = build_meeting_graph().invoke(state)
    output_path = save_agent_result(meeting_id, result)
    result["agent_result_path"] = output_path
    return result

@app.post("/meetings/{meeting_id}/report/email")
def send_report_email_api(meeting_id: str, request: SendReportEmailRequest):
    pdf_path = get_latest_report_pdf_path(meeting_id)

    try:
        result = send_report_email_tool(
            meeting_id=meeting_id,
            pdf_path=pdf_path,
            to_email=request.to_email,
            subject=request.subject,
            body=request.body,
        )
    except EmailSendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return result


@app.post("/meetings/analyze")
async def analyze_meeting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    host: str = Form(...),
    language: str = Form("zh-CN"),
    participants: str = Form(""),
    send_email: bool = Form(False),
    to_email: str | None = Form(None),
):
    meeting_id = uuid4().hex

    suffix = (
        Path(file.filename or "").suffix
        or ".wav"
    )

    audio_path = (
        AUDIO_DIR / f"{meeting_id}{suffix}"
    )

    with audio_path.open("wb") as target:
        shutil.copyfileobj(
            file.file,
            target,
        )

    participant_list = [
        item.strip()
        for item in participants
        .replace(",", "\n")
        .splitlines()
        if item.strip()
    ]

    meeting = {
        "meeting_id": meeting_id,
        "title": title.strip()
        or "会议纪要",
        "host": host.strip()
        or "-",
        "language": language,
        "participants": participant_list,
        "status": "audio_uploaded",
        "audio_uri": str(audio_path),
        "progress": 10,
        "send_email": send_email,
        "to_email": to_email,
    }

    STORE[meeting_id] = meeting

    job = create_job(meeting_id)

    update_job(
        job["job_id"],
        status="queued",
        progress=10,
        stage="等待后台处理",
    )

    background_tasks.add_task(
        run_full_workflow,
        job["job_id"],
        meeting,
    )

    return {
        "job_id": job["job_id"],
        "meeting_id": meeting_id,
        "status": "queued",
        "progress": 10,
        "stage": "等待后台处理",
    }


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatusOut,
)
def get_job_status(job_id: str):
    job = get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="job not found",
        )

    return job


@app.get(
    "/jobs/{job_id}/history",
    response_model=JobHistoryOut,
)
def get_job_history(job_id: str):
    job = get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="job not found",
        )

    return {
        "job_id": job_id,
        "meeting_id": job["meeting_id"],
        "history": job.get("history", []),
    }


def load_agent_result_by_job(job: dict) -> dict:
    meeting_id = job["meeting_id"]

    result_info = job.get("result") or {}
    result_path = result_info.get("agent_result_path")

    if result_path:
        path = Path(result_path)
    else:
        path = (
            Path("outputs/agent")
            / f"{meeting_id}.json"
        )

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="agent result not found",
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


@app.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    job = get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="job not found",
        )

    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="job is not completed",
        )

    return load_agent_result_by_job(job)


@app.get("/jobs/{job_id}/report/pdf")
def download_job_report(job_id: str):
    job = get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="job not found",
        )

    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="job is not completed",
        )

    result_info = job.get("result") or {}
    pdf_path_value = result_info.get("pdf_path")

    if pdf_path_value:
        pdf_path = Path(pdf_path_value)
    else:
        pdf_path = (
            Path("outputs/reports")
            / f"{job['meeting_id']}_trusted_minutes.pdf"
        )

    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail="PDF report not found",
        )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=(
            f"{job['meeting_id']}_trusted_minutes.pdf"
        ),
    )


