import threading

from llm.schemas import AgentClaim
from orchestration.graph import extract_all_claims


def test_claim_agents_run_in_parallel(monkeypatch):
    thread_names = set()
    barrier = threading.Barrier(3)

    class FakeClaimAgent:
        def __init__(self, task_type: str, claim_type: str):
            self.claim_type = claim_type

        def run(self, meeting_id: str, evidence: list[dict]):
            thread_names.add(threading.current_thread().name)
            barrier.wait(timeout=5)
            item = evidence[0]
            return [
                AgentClaim(
                    speaker_id=item["speaker_id"],
                    claim_type=self.claim_type,
                    summary=self.claim_type,
                    evidence_ids=[item["evidence_id"]],
                    quote=item["text"],
                    start_ms=item["start_ms"],
                    end_ms=item["end_ms"],
                )
            ]

    monkeypatch.setenv("LLM_CONCURRENCY", "3")
    monkeypatch.setenv("AGENT_EXTRACT_MODE", "parallel")
    monkeypatch.setenv("AGENT_CACHE_ENABLED", "0")
    monkeypatch.setattr(
        "orchestration.graph.ClaimAgent",
        FakeClaimAgent,
    )

    state = {
        "meeting_id": "parallel-test",
        "evidence": [
            {
                "evidence_id": "ev-1",
                "speaker_id": "speaker_00",
                "text": "会议测试原话",
                "start_ms": 0,
                "end_ms": 1000,
            }
        ],
        "errors": [],
        "timings": {},
    }

    result = extract_all_claims(state)

    assert len(result["contributions"]) == 1
    assert len(result["action_items"]) == 1
    assert len(result["risks"]) == 1
    assert result["errors"] == []
    assert "agent_claim_extraction" in result["timings"]
    assert len(thread_names) == 3
