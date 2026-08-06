from langgraph.graph import StateGraph, START, END
from .state import MeetingState

def audio_quality_node(state: MeetingState) -> MeetingState:
    return {**state, "progress": 20, "status": "audio_checked"}

def asr_node(state: MeetingState) -> MeetingState:
    fake_spans = [
        {
            "span_id": "span_001",
            "speaker_id": "speaker_00",
            "start_ms": 12000,
            "end_ms": 18000,
            "text": "我们需要在下周完成接口联调。",
            "asr_confidence": 0.92,
        }
    ]
    return {**state, "progress": 40, "status": "asr_done", "transcript_spans": fake_spans}

def diarization_node(state: MeetingState) -> MeetingState:
    speakers = [{"speaker_id": "speaker_00", "name": "待确认"}]
    return {**state, "progress": 60, "status": "diarization_done", "speakers": speakers}

def contribution_node(state: MeetingState) -> MeetingState:
    claims = [
        {
            "claim_id": "claim_001",
            "speaker_id": "speaker_00",
            "claim_type": "commitment",
            "statement": "下周完成接口联调",
            "evidence_ids": ["span_001"],
            "confidence": 0.88,
            "review_status": "pending",
        }
    ]
    return {**state, "progress": 80, "status": "summary_done", "claims": claims}

def report_node(state: MeetingState) -> MeetingState:
    return {**state, "progress": 100, "status": "completed"}

def build_graph():
    g = StateGraph(MeetingState)
    g.add_node("audio_quality", audio_quality_node)
    g.add_node("asr", asr_node)
    g.add_node("diarization", diarization_node)
    g.add_node("contribution", contribution_node)
    g.add_node("report", report_node)

    g.add_edge(START, "audio_quality")
    g.add_edge("audio_quality", "asr")
    g.add_edge("asr", "diarization")
    g.add_edge("diarization", "contribution")
    g.add_edge("contribution", "report")
    g.add_edge("report", END)
    return g.compile()

GRAPH = build_graph()

def run_demo_graph(meeting: dict) -> dict:
    state = GRAPH.invoke(
        {
            "meeting_id": meeting["meeting_id"],
            "title": meeting["title"],
            "audio_uri": meeting.get("audio_uri", ""),
            "status": "processing",
            "progress": 10,
            "errors": [],
            "transcript_spans": [],
            "speakers": [],
            "claims": [],
            "evidence_links": [],
        }
    )
    return state
