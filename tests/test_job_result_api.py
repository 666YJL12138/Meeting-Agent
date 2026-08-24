import json

from fastapi.testclient import TestClient

from apps.api import main as api_main


client = TestClient(api_main.app)


def test_job_result_returns_completed_result(
    monkeypatch,
    tmp_path,
):
    result_path = tmp_path / "agent_result.json"

    result_path.write_text(
        json.dumps(
            {
                "meeting_id": "meeting_result_test",
                "claims": [],
                "speaker_summaries": [],
                "errors": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    fake_job = {
        "job_id": "job_result_test",
        "meeting_id": "meeting_result_test",
        "status": "completed",
        "progress": 100,
        "stage": "分析完成",
        "result": {
            "agent_result_path": str(result_path),
        },
    }

    monkeypatch.setattr(
        api_main,
        "get_job",
        lambda job_id: fake_job,
    )

    response = client.get(
        "/jobs/job_result_test/result"
    )

    assert response.status_code == 200
    assert response.json()["meeting_id"] == (
        "meeting_result_test"
    )


def test_job_result_rejects_unfinished_job(
    monkeypatch,
):
    fake_job = {
        "job_id": "job_running_test",
        "meeting_id": "meeting_running_test",
        "status": "asr_processing",
        "progress": 30,
        "stage": "语音识别",
        "result": None,
    }

    monkeypatch.setattr(
        api_main,
        "get_job",
        lambda job_id: fake_job,
    )

    response = client.get(
        "/jobs/job_running_test/result"
    )

    assert response.status_code == 409
