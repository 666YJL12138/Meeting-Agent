from services.quality_gate import build_quality_report


def test_quality_report_passes_when_claim_has_matching_evidence():
    result = {
        "meeting_id": "demo",
        "evidence_links": [
            {
                "evidence_id": "ev_001",
                "quote": "李四负责测试计划。",
            }
        ],
        "claims": [
            {
                "claim_id": "claim_001",
                "evidence_ids": ["ev_001"],
                "quote": "李四负责测试计划。",
            }
        ],
    }

    report = build_quality_report(result)

    assert report["status"] == "passed"
    assert report["evidence_coverage"] == 1.0


def test_quality_report_rejects_claim_without_evidence():
    result = {
        "meeting_id": "demo",
        "evidence_links": [],
        "claims": [
            {
                "claim_id": "claim_001",
                "evidence_ids": [],
                "quote": "这是没有证据的结论。",
            }
        ],
    }

    report = build_quality_report(result)

    assert report["status"] == "needs_review"
    assert report["invalid_claim_count"] == 1
