import threading
import time

from agents.rag_agent import RAGAgent
from services.vector_store import evidence_hash, rerank_hits


def test_rag_enrich_claims_runs_searches_concurrently_and_keeps_order(monkeypatch):
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_search(meeting_id, query, limit):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return [{"score": 0.9, "payload": {"query": query}}]

    monkeypatch.setattr("agents.rag_agent.search_meeting", fake_search)

    claims = [
        {"summary": "结论一", "claim_type": "观点", "speaker_id": "speaker_00"},
        {"summary": "结论二", "claim_type": "行动项", "speaker_id": "speaker_01"},
        {"summary": "结论三", "claim_type": "风险", "speaker_id": "speaker_02"},
    ]

    result = RAGAgent(top_k=2, max_workers=3).enrich_claims(
        meeting_id="meeting_001",
        claims=claims,
        evidence=[],
    )

    assert [item["claim_summary"] for item in result] == [
        "结论一",
        "结论二",
        "结论三",
    ]
    assert all(item["hits"][0]["source"] == "qdrant" for item in result)
    assert max_active > 1


def test_rag_keeps_local_fallback_when_vector_search_fails(monkeypatch):
    def failing_search(*args, **kwargs):
        raise RuntimeError("qdrant unavailable")

    monkeypatch.setattr("agents.rag_agent.search_meeting", failing_search)

    result = RAGAgent(top_k=1, max_workers=1).enrich_claims(
        meeting_id="meeting_001",
        claims=[{"summary": "预算风险", "claim_type": "风险"}],
        evidence=[{
            "evidence_id": "ev_001",
            "speaker_id": "speaker_00",
            "text": "预算风险需要进一步确认",
            "start_ms": 0,
            "end_ms": 1000,
        }],
    )

    assert result[0]["hits"][0]["source"] == "local"
    assert result[0]["hits"][0]["payload"]["evidence_id"] == "ev_001"


def test_rag_local_fallback_preserves_speaker_and_time_filters(monkeypatch):
    monkeypatch.setattr(
        "agents.rag_agent.search_meeting",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("qdrant unavailable")
        ),
    )

    result = RAGAgent(top_k=5, max_workers=1).enrich_claims(
        meeting_id="meeting_001",
        claims=[{
            "summary": "预算风险",
            "claim_type": "风险",
            "speaker_id": "speaker_01",
            "start_ms": 1000,
            "end_ms": 2000,
        }],
        evidence=[
            {
                "evidence_id": "ev-wrong-speaker",
                "speaker_id": "speaker_00",
                "text": "预算风险需要确认",
                "start_ms": 1000,
                "end_ms": 2000,
            },
            {
                "evidence_id": "ev-wrong-time",
                "speaker_id": "speaker_01",
                "text": "预算风险需要确认",
                "start_ms": 3000,
                "end_ms": 4000,
            },
            {
                "evidence_id": "ev-match",
                "speaker_id": "speaker_01",
                "text": "预算风险需要确认",
                "start_ms": 1200,
                "end_ms": 1800,
            },
        ],
    )

    assert [
        hit["payload"]["evidence_id"]
        for hit in result[0]["hits"]
    ] == ["ev-match"]


def test_evidence_hash_is_stable_and_rerank_prefers_lexical_match():
    payload = {
        "speaker_id": "speaker_00",
        "start_ms": 0,
        "end_ms": 1000,
        "text": "预算风险需要确认",
    }
    assert evidence_hash("meeting_001", "transcript", "span_1", payload) == (
        evidence_hash("meeting_001", "transcript", "span_1", payload)
    )

    ranked = rerank_hits(
        "预算风险",
        [
            {"score": 0.95, "payload": {"evidence_id": "generic", "text": "项目进度很顺利"}},
            {"score": 0.20, "payload": {"evidence_id": "match", "text": "预算风险需要确认"}},
        ],
        limit=2,
    )
    assert ranked[0]["payload"]["evidence_id"] == "match"
    assert ranked[0]["rerank_score"] > ranked[1]["rerank_score"]
