import json
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api import main as api_main


client = TestClient(api_main.app)


def test_health_exposes_capabilities():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["capabilities"]["local_bge"] is True


def test_quality_report_returns_verified_result(monkeypatch, tmp_path):
    meeting = {
        "meeting_id": "meeting_001",
        "evidence_links": [
            {"evidence_id": "ev_001", "quote": "李四负责测试计划。"}
        ],
        "claims": [
            {
                "claim_id": "claim_001",
                "evidence_ids": ["ev_001"],
                "quote": "李四负责测试计划。",
            }
        ],
    }
    agent_dir = tmp_path / "agent_results"
    agent_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        api_main,
        "AGENT_RESULT_DIR",
        agent_dir,
    )
    (agent_dir / "meeting_001.json").write_text(
        json.dumps(meeting, ensure_ascii=False),
        encoding="utf-8",
    )

    response = client.get("/meetings/meeting_001/quality-report")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "passed"
    assert body["evidence_coverage"] == 1.0


def test_quality_report_prefers_final_agent_result(monkeypatch, tmp_path):
    meeting_id = "meeting_002"
    final_result = {
        "meeting_id": meeting_id,
        "evidence_links": [
            {"evidence_id": "ev_001", "quote": "李四负责测试计划。"}
        ],
        "claims": [
            {
                "claim_id": "claim_001",
                "evidence_ids": ["ev_001"],
                "quote": "李四负责测试计划。",
            }
        ],
    }

    agent_dir = tmp_path / "outputs" / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / f"{meeting_id}.json").write_text(
        json.dumps(final_result, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        api_main,
        "AGENT_RESULT_DIR",
        agent_dir,
    )
    response = client.get(f"/meetings/{meeting_id}/quality-report")

    assert response.status_code == 200
    body = response.json()
    assert body["claim_count"] == 1
    assert body["status"] == "passed"


def test_quality_report_returns_404_for_missing_meeting(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "load_meeting_result_for_quality",
        lambda meeting_id: None,
    )

    response = client.get("/meetings/not-found/quality-report")

    assert response.status_code == 404
