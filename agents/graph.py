from langgraph.graph import END, START, StateGraph
from services.performance import measure_stage
from .state import MeetingState
from services.asr import transcribe_audio
from services.audio import (
    detect_voice_segments,
    inspect_wav,
    normalize_audio_to_wav,
)
from services.cache import cache_meeting_state
from services.contribution import (
    build_speaker_summaries,
    extract_contributions,
)
from services.diarization import (
    assign_speakers_to_spans,
    build_speakers,
    diarize_audio,
)
from services.speaker_identity import (
    apply_speaker_name_map,
    infer_speaker_name_map,
)
from services.vector_store import index_meeting


def update_job_progress(
    state: MeetingState,
    *,
    status: str,
    progress: int,
    stage: str,
) -> None:
    """Best-effort progress reporting for the asynchronous workflow."""
    job_id = state.get("job_id")
    if not job_id:
        return

    try:
        from services.job_store import update_job

        update_job(
            job_id,
            status=status,
            progress=progress,
            stage=stage,
        )
    except Exception as exc:
        # Redis 不可用时不能让 ASR 主流程失败。
        print(
            f"[job-progress] update failed for {job_id}: {exc}",
            flush=True,
        )


def audio_quality_node(state: MeetingState) -> MeetingState:
    update_job_progress(
        state,
        status="audio_normalizing",
        progress=10,
        stage="音频标准化与语音段检测",
    )

    normalized_path = normalize_audio_to_wav(
        input_path=state["audio_uri"],
        meeting_id=state["meeting_id"],
    )

    audio_info = inspect_wav(normalized_path)
    voice_segments = detect_voice_segments(normalized_path)

    update_job_progress(
        state,
        status="audio_normalizing",
        progress=25,
        stage="音频预处理完成",
    )

    return {
        "normalized_audio_uri": normalized_path,
        "audio_info": audio_info,
        "voice_segments": voice_segments,
        "progress": 35,
        "status": "audio_checked",
    }


def asr_node(state: MeetingState) -> MeetingState:
    update_job_progress(
        state,
        status="asr_processing",
        progress=30,
        stage="faster-whisper 语音识别",
    )

    with measure_stage(state, "asr"):
        spans = transcribe_audio(
            wav_path=state["normalized_audio_uri"],
            meeting_id=state["meeting_id"],
            participants=state.get("participants", []),
        )

    update_job_progress(
        state,
        status="asr_processing",
        progress=45,
        stage="语音识别完成",
    )

    return {
        "transcript_spans": spans,
        "timings": state.get("timings", {}),
        "progress": 70,
        "status": "asr_done",
    }


def diarization_node(state: MeetingState) -> MeetingState:
    update_job_progress(
        state,
        status="diarization_processing",
        progress=50,
        stage="pyannote 说话人分离",
    )

    participants = state.get("participants", [])
    expected_speakers = (
        len(participants)
        if len(participants) >= 2
        else None
    )

    with measure_stage(state, "diarization"):
        diarization_result = diarize_audio(
            wav_path=state["normalized_audio_uri"],
            voice_segments=state.get("voice_segments", []),
            min_speakers=expected_speakers,
            max_speakers=expected_speakers,
        )
        speaker_segments = diarization_result["speaker_segments"]

        assigned_spans = assign_speakers_to_spans(
            transcript_spans=state.get("transcript_spans", []),
            speaker_segments=speaker_segments,
            min_confidence=0.2,
        )

    speakers = build_speakers(speaker_segments)
    speaker_mapping = infer_speaker_name_map(
        assigned_spans,
        participants,
    )
    mapped = apply_speaker_name_map(
        {
            "transcript_spans": assigned_spans,
            "speakers": speakers,
            "speaker_segments": speaker_segments,
            "exclusive_speaker_segments": diarization_result[
                "exclusive_speaker_segments"
            ],
            "overlap_segments": diarization_result["overlap_segments"],
            "diarization_metrics": diarization_result[
                "diarization_metrics"
            ],
            "speaker_ids": [
                item.get("speaker_id")
                for item in speaker_segments
                if item.get("speaker_id")
            ],
        },
        speaker_mapping,
    )

    return {
        "speaker_segments": mapped["speaker_segments"],
        "speaker_ids": mapped["speaker_ids"],
        "exclusive_speaker_segments": mapped[
            "exclusive_speaker_segments"
        ],
        "overlap_segments": mapped["overlap_segments"],
        "diarization_metrics": mapped["diarization_metrics"],
        "transcript_spans": mapped["transcript_spans"],
        "speakers": mapped["speakers"],
        "speaker_mapping": speaker_mapping,
        "speaker_name_map": mapped.get(
            "speaker_name_map",
            {},
        ),
        "timings": state.get("timings", {}),
        "progress": 82,
        "status": "diarization_done",
    }


def evidence_stub_node(state: MeetingState) -> MeetingState:
    update_job_progress(
        state,
        status="evidence_building",
        progress=65,
        stage="构建可追溯证据链",
    )

    evidence_links = []

    for span in state.get("transcript_spans", []):
        evidence_links.append({
            "evidence_id": f"ev_{span['span_id']}",
            "span_id": span["span_id"],
            "quote": span["text"],
            "start_ms": span["start_ms"],
            "end_ms": span["end_ms"],
            "speaker_id": span.get("speaker_id"),
            "speaker_ids": span.get(
                "speaker_ids",
                [span.get("speaker_id")],
            ),
            "speaker_confidences": span.get(
                "speaker_confidences",
                {},
            ),
            "speaker_candidates": span.get(
                "speaker_candidates",
                [],
            ),
            "overlap": bool(span.get("overlap", False)),
            "confidence_source": span.get(
                "confidence_source",
                "derived_alignment",
            ),
            "speaker_name": span.get("speaker_name"),
        })

    update_job_progress(
        state,
        status="evidence_building",
        progress=72,
        stage="证据链构建完成",
    )

    return {
        "evidence_links": evidence_links,
        "progress": 92,
        "status": "evidence_done",
    }


def contribution_node(state: MeetingState) -> MeetingState:
    update_job_progress(
        state,
        status="agent_processing",
        progress=75,
        stage="抽取发言人关键贡献",
    )

    claims = extract_contributions(
        meeting_id=state["meeting_id"],
        transcript_spans=state.get("transcript_spans", []),
        evidence_links=state.get("evidence_links", []),
    )

    speaker_summaries = build_speaker_summaries(
        claims=claims,
        speakers=state.get("speakers", []),
    )

    update_job_progress(
        state,
        status="agent_processing",
        progress=82,
        stage="关键贡献抽取完成",
    )

    return {
        "claims": claims,
        "speaker_summaries": speaker_summaries,
        "progress": 96,
        "status": "contribution_done",
    }


def index_node(state: MeetingState) -> MeetingState:
    update_job_progress(
        state,
        status="agent_processing",
        progress=85,
        stage="写入 Qdrant 并缓存 Redis",
    )

    errors = list(state.get("errors", []))
    index_status = []

    try:
        count = index_meeting(
            meeting_id=state["meeting_id"],
            transcript_spans=state.get("transcript_spans", []),
            claims=state.get("claims", []),
        )
        index_status.append(f"qdrant:indexed:{count}")
    except Exception as exc:
        errors.append(f"qdrant indexing failed: {exc}")
        index_status.append("qdrant:failed")

    try:
        cache_meeting_state(
            meeting_id=state["meeting_id"],
            state=dict(state),
        )
        index_status.append("redis:cached")
    except Exception as exc:
        errors.append(f"redis caching failed: {exc}")
        index_status.append("redis:failed")

    update_job_progress(
        state,
        status="agent_processing",
        progress=95,
        stage="向量索引与缓存完成",
    )

    return {
        "progress": 98,
        "status": "indexed",
        "index_status": index_status,
        "errors": errors,
    }


def finish_node(state: MeetingState) -> MeetingState:
    update_job_progress(
        state,
        status="agent_processing",
        progress=98,
        stage="ASR 与证据阶段完成，准备生成报告",
    )

    return {
        "progress": 100,
        "status": "completed",
    }


def build_audio_asr_graph():
    graph = StateGraph(MeetingState)

    graph.add_node("audio_quality", audio_quality_node)
    graph.add_node("asr", asr_node)
    graph.add_node("diarization", diarization_node)
    graph.add_node("evidence_stub", evidence_stub_node)
    graph.add_node("contribution", contribution_node)
    graph.add_node("index", index_node)
    graph.add_node("finish", finish_node)

    graph.add_edge(START, "audio_quality")
    graph.add_edge("audio_quality", "asr")
    graph.add_edge("asr", "diarization")
    graph.add_edge("diarization", "evidence_stub")
    graph.add_edge("evidence_stub", "contribution")
    graph.add_edge("contribution", "index")
    graph.add_edge("index", "finish")
    graph.add_edge("finish", END)

    return graph.compile()


AUDIO_ASR_GRAPH = build_audio_asr_graph()


def run_audio_asr_graph(meeting: dict) -> dict:
    if not meeting.get("audio_uri"):
        raise ValueError(
            "audio_uri is required before running ASR graph"
        )

    return AUDIO_ASR_GRAPH.invoke({
        "job_id": meeting.get("job_id"),
        "meeting_id": meeting["meeting_id"],
        "title": meeting["title"],
        "audio_uri": meeting["audio_uri"],
        "participants": meeting.get("participants", []),
        "status": "processing",
        "progress": 10,
        "timings": {},
        "errors": [],
        "transcript_spans": [],
        "speakers": [],
        "speaker_segments": [],
        "exclusive_speaker_segments": [],
        "overlap_segments": [],
        "diarization_metrics": {},
        "diarization_evaluation": {},
        "speaker_mapping": {},
        "claims": [],
        "speaker_summaries": [],
        "evidence_links": [],
        "index_status": [],
    })
