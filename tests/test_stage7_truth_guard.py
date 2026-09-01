from agents.critic_agent import CriticAgent
from llm.schemas import AgentClaim
from orchestration.graph import verify_all_claims


def _evidence_item(**overrides):
    item = {
        "evidence_id": "ev_001",
        "speaker_id": "speaker_00",
        "speaker_ids": ["speaker_00"],
        "text": "李四负责整理测试表格。",
        "start_ms": 1000,
        "end_ms": 2000,
        "asr_confidence": 0.95,
        "speaker_confidence": 0.9,
        "overlap": False,
    }
    item.update(overrides)
    return item


def test_agent_claim_supports_fact_and_commitment():
    claim = AgentClaim(
        speaker_id="speaker_00",
        claim_type="事实",
        summary="已完成整理",
        evidence_ids=["ev_001"],
        quote="李四负责整理测试表格。",
        start_ms=1000,
        end_ms=2000,
    )

    assert claim.claim_type == "事实"

    commitment = claim.model_copy(update={"claim_type": "承诺"})
    assert commitment.claim_type == "承诺"


def test_critic_agent_marks_missing_evidence_for_review():
    claim = AgentClaim(
        speaker_id="speaker_00",
        claim_type="观点",
        summary="缺少证据",
        evidence_ids=["ev_999"],
        quote="缺少证据",
        start_ms=0,
        end_ms=0,
    )

    verified = CriticAgent().verify([claim], [_evidence_item()])

    assert verified[0].support_status == "unsupported"
    assert verified[0].review_reason == "missing_evidence"


def test_critic_agent_marks_overlap_and_conflict_for_review():
    claim = AgentClaim(
        speaker_id="speaker_01",
        claim_type="风险",
        summary="有风险",
        evidence_ids=["ev_001"],
        quote="李四负责整理测试表格。",
        start_ms=1000,
        end_ms=2000,
    )
    evidence = [_evidence_item(overlap=True, speaker_ids=["speaker_00", "speaker_01"])]

    verified = CriticAgent().verify([claim], evidence)

    assert verified[0].support_status == "unsupported"
    assert "overlap_attribution_ambiguity" in (verified[0].review_reason or "")


def test_verify_all_claims_puts_partial_claims_into_review_queue():
    state = {
        "meeting_id": "stage7-review",
        "contributions": [
            {
                "speaker_id": "speaker_00",
                "claim_type": "观点",
                "summary": "摘要",
                "evidence_ids": ["ev_001"],
                "quote": "李四负责整理测试表格。",
                "start_ms": 1000,
                "end_ms": 2000,
            },
            {
                "speaker_id": "speaker_00",
                "claim_type": "观点",
                "summary": "摘要2",
                "evidence_ids": ["ev_999"],
                "quote": "没有证据",
                "start_ms": 0,
                "end_ms": 0,
            },
        ],
        "action_items": [],
        "risks": [],
        "evidence": [_evidence_item()],
        "review_queue": [],
        "errors": [],
        "timings": {},
        "retry_count": 0,
        "node_attempts": {},
        "orchestration_trace": [],
    }

    result = verify_all_claims(state)

    assert len(result["contributions"]) == 1
    assert len(result["review_queue"]) == 1
    assert result["review_queue"][0]["reason"] == "missing_evidence"
