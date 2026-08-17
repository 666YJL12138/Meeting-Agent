from fastapi.testclient import TestClient

from apps.api import main as api_main


client = TestClient(api_main.app)


def test_health_exposes_capabilities():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["capabilities"]["local_bge"] is True


def test_quality_report_returns_verified_result(monkeypatch):
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

    monkeypatch.setattr(
        api_main,
        "restore_meeting_from_cache",
        lambda meeting_id: meeting if meeting_id == "meeting_001" else None,
    )

    response = client.get("/meetings/meeting_001/quality-report")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "passed"
    assert body["evidence_coverage"] == 1.0


def test_quality_report_returns_404_for_missing_meeting(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "restore_meeting_from_cache",
        lambda meeting_id: None,
    )

    response = client.get("/meetings/not-found/quality-report")

    assert response.status_code == 404
