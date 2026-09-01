import json

from agents.claim_agent import build_task_compact_evidence
from llm.schemas import AgentClaim
from orchestration.graph import extract_all_claims


def _state():
    return {
        "meeting_id": "agent-performance-test",
        "evidence": [
            {
                "evidence_id": "ev-1",
                "speaker_id": "speaker_00",
                "text": "李四负责整理测试表格，王五确认验收流程。",
                "start_ms": 0,
                "end_ms": 1000,
                "asr_confidence": 0.95,
                "speaker_confidence": 0.9,
            }
        ],
        "errors": [],
        "timings": {},
    }


def test_claim_agents_share_one_compact_context(monkeypatch, tmp_path):
    contexts = []

    class FakeClaimAgent:
        def __init__(self, task_type: str, claim_type: str):
            self.claim_type = claim_type

        def run(
            self,
            meeting_id: str,
            evidence: list[dict],
            compact_evidence_json: str | None = None,
        ):
            contexts.append(compact_evidence_json)
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
    monkeypatch.setattr("orchestration.graph.ClaimAgent", FakeClaimAgent)

    result = extract_all_claims(_state())

    assert len(contexts) == 3
    assert len(set(contexts)) == 1
    assert result["agent_cache_hit"] is False
    assert all(
        f"agent_{key}" in result["timings"]
        for key in ("contributions", "action_items", "risks")
    )


def test_claim_result_cache_skips_second_llm_execution(monkeypatch, tmp_path):
    calls = 0

    class FakeClaimAgent:
        def __init__(self, task_type: str, claim_type: str):
            self.claim_type = claim_type

        def run(
            self,
            meeting_id: str,
            evidence: list[dict],
            compact_evidence_json: str | None = None,
        ):
            nonlocal calls
            calls += 1
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
    monkeypatch.setenv("AGENT_CACHE_ENABLED", "1")
    monkeypatch.setattr("orchestration.graph.ClaimAgent", FakeClaimAgent)
    monkeypatch.setattr("services.artifact_cache.CACHE_ROOT", tmp_path)

    first = extract_all_claims(_state())
    second = extract_all_claims(_state())

    assert calls == 3
    assert first["agent_cache_hit"] is False
    assert second["agent_cache_hit"] is True
    assert second["contributions"] == first["contributions"]
    assert second["action_items"] == first["action_items"]
    assert second["risks"] == first["risks"]


def test_batch_claim_agent_runs_single_llm_call(monkeypatch):
    calls = 0

    class FakeBatchClaimAgent:
        def __init__(self):
            pass

        def run(self, meeting_id: str, evidence: list[dict]):
            nonlocal calls
            calls += 1
            item = evidence[0]
            claim = AgentClaim(
                speaker_id=item["speaker_id"],
                claim_type="观点",
                summary="观点",
                evidence_ids=[item["evidence_id"]],
                quote=item["text"],
                start_ms=item["start_ms"],
                end_ms=item["end_ms"],
            )
            return {
                "contributions": [claim],
                "action_items": [claim.model_copy(update={"claim_type": "行动项"})],
                "risks": [claim.model_copy(update={"claim_type": "风险"})],
            }

    monkeypatch.setenv("AGENT_EXTRACT_MODE", "batched")
    monkeypatch.setenv("AGENT_CACHE_ENABLED", "0")
    monkeypatch.setattr("orchestration.graph.BatchClaimAgent", FakeBatchClaimAgent)

    result = extract_all_claims(_state())

    assert calls == 1
    assert result["agent_cache_hit"] is False
    assert len(result["contributions"]) == 1
    assert len(result["action_items"]) == 1
    assert len(result["risks"]) == 1
    assert "agent_batch_extraction" in result["timings"]


def test_task_evidence_filter_prefers_task_specific_spans():
    evidence = [
        {
            "evidence_id": "ev-contrib",
            "speaker_id": "speaker_00",
            "text": "我们建议先统一口径，再确认发布方案。",
            "start_ms": 0,
            "end_ms": 1000,
            "asr_confidence": 0.92,
            "speaker_confidence": 0.9,
        },
        {
            "evidence_id": "ev-action",
            "speaker_id": "speaker_01",
            "text": "李四负责整理测试表格，王五下周完成验收。",
            "start_ms": 1000,
            "end_ms": 2000,
            "asr_confidence": 0.95,
            "speaker_confidence": 0.93,
        },
        {
            "evidence_id": "ev-risk",
            "speaker_id": "speaker_02",
            "text": "如果依赖接口还没稳定，后面可能会有阻塞风险。",
            "start_ms": 2000,
            "end_ms": 3000,
            "asr_confidence": 0.96,
            "speaker_confidence": 0.9,
        },
    ]

    action_items = json.loads(
        build_task_compact_evidence(
            evidence,
            "action_items",
            top_k=1,
            max_items=1,
            max_chars=120,
        )
    )
    risks = json.loads(
        build_task_compact_evidence(
            evidence,
            "risks",
            top_k=1,
            max_items=1,
            max_chars=120,
        )
    )

    assert action_items[0]["evidence_id"] == "ev-action"
    assert risks[0]["evidence_id"] == "ev-risk"
