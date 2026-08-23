from langgraph.graph import StateGraph, START, END

from .state import MeetingState
from services.audio import normalize_audio_to_wav, inspect_wav, detect_voice_segments
from services.asr import transcribe_audio
from services.diarization import diarize_audio, assign_speakers_to_spans, build_speakers
from services.contribution import extract_contributions, build_speaker_summaries
from services.cache import cache_meeting_state
from services.vector_store import index_meeting


def audio_quality_node(state: MeetingState) -> MeetingState:
    normalized_path = normalize_audio_to_wav(
        input_path=state["audio_uri"],
        meeting_id=state["meeting_id"],
    )

    audio_info = inspect_wav(normalized_path)
    voice_segments = detect_voice_segments(normalized_path)

    return {
        "normalized_audio_uri": normalized_path,
        "audio_info": audio_info,
        "voice_segments": voice_segments,
        "progress": 35,
        "status": "audio_checked",
    }


def asr_node(state: MeetingState) -> MeetingState:
    spans = transcribe_audio(
        wav_path=state["normalized_audio_uri"],
        meeting_id=state["meeting_id"],
    )

    return {
        "transcript_spans": spans,
        "progress": 70,
        "status": "asr_done",
    }


def diarization_node(state: MeetingState) -> MeetingState:
    participants = state.get("participants", [])
    expected_speakers = len(participants) if len(participants) >= 2 else None

    speaker_segments = diarize_audio(
        wav_path=state["normalized_audio_uri"],
        voice_segments=state.get("voice_segments", []),
        min_speakers=expected_speakers,
        max_speakers=expected_speakers,
    )

    assigned_spans = assign_speakers_to_spans(
        transcript_spans=state.get("transcript_spans", []),
        speaker_segments=speaker_segments,
        min_confidence=0.2,
    )

    speakers = build_speakers(speaker_segments)

    return {
        "speaker_segments": speaker_segments,
        "transcript_spans": assigned_spans,
        "speakers": speakers,
        "progress": 82,
        "status": "diarization_done",
    }


def evidence_stub_node(state: MeetingState) -> MeetingState:
    evidence_links = []

    for span in state.get("transcript_spans", []):
        evidence_links.append({
            "evidence_id": f"ev_{span['span_id']}",
            "span_id": span["span_id"],
            "quote": span["text"],
            "start_ms": span["start_ms"],
            "end_ms": span["end_ms"],
            "speaker_id": span.get("speaker_id"),
        })

    return {
        "evidence_links": evidence_links,
        "progress": 92,
        "status": "evidence_done",
    }


def contribution_node(state: MeetingState) -> MeetingState:
    claims = extract_contributions(
        meeting_id=state["meeting_id"],
        transcript_spans=state.get("transcript_spans", []),
        evidence_links=state.get("evidence_links", []),
    )

    speaker_summaries = build_speaker_summaries(
        claims=claims,
        speakers=state.get("speakers", []),
    )

    return {
        "claims": claims,
        "speaker_summaries": speaker_summaries,
        "progress": 96,
        "status": "contribution_done",
    }


def index_node(state: MeetingState) -> MeetingState:
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

    return {
        "progress": 98,
        "status": "indexed",
        "index_status": index_status,
        "errors": errors,
    }


def finish_node(state: MeetingState) -> MeetingState:
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
        raise ValueError("audio_uri is required before running ASR graph")

    return AUDIO_ASR_GRAPH.invoke({
        "meeting_id": meeting["meeting_id"],
        "title": meeting["title"],
        "audio_uri": meeting["audio_uri"],
        "participants": meeting.get("participants", []),
        "status": "processing",
        "progress": 10,
        "errors": [],
        "transcript_spans": [],
        "speakers": [],
        "speaker_segments": [],
        "speaker_mapping": {},
        "claims": [], 
        "speaker_summaries": [],
        "evidence_links": [],
        "index_status": [],
    })
