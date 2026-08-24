import threading
import time

from agents.rag_agent import RAGAgent


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
