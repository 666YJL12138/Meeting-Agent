from fastapi.testclient import TestClient

from apps.api import main as api_main


client = TestClient(api_main.app)


FAKE_JOB = {
    "job_id": "job_history_api_test",
    "meeting_id": "meeting_history_api_test",
    "status": "asr_processing",
    "progress": 30,
    "stage": "faster-whisper 语音识别",
    "error": None,
    "result": None,
    "history": [
        {
            "timestamp": (
                "2026-08-24T10:00:00+00:00"
            ),
            "status": "created",
            "progress": 0,
            "stage": "任务已创建",
        },
        {
            "timestamp": (
                "2026-08-24T10:00:01+00:00"
            ),
            "status": "queued",
            "progress": 10,
            "stage": "等待后台处理",
        },
        {
            "timestamp": (
                "2026-08-24T10:00:02+00:00"
            ),
            "status": "asr_processing",
            "progress": 30,
            "stage": "faster-whisper 语音识别",
        },
    ],
}


def test_get_job_status_returns_history(
    monkeypatch,
):
    monkeypatch.setattr(
        api_main,
        "get_job",
        lambda job_id: FAKE_JOB,
    )

    response = client.get(
        "/jobs/job_history_api_test"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["job_id"] == (
        "job_history_api_test"
    )
    assert len(body["history"]) == 3
    assert body["history"][-1]["progress"] == 30


def test_get_job_history_returns_only_history(
    monkeypatch,
):
    monkeypatch.setattr(
        api_main,
        "get_job",
        lambda job_id: FAKE_JOB,
    )

    response = client.get(
        "/jobs/job_history_api_test/history"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["job_id"] == (
        "job_history_api_test"
    )
    assert body["meeting_id"] == (
        "meeting_history_api_test"
    )
    assert len(body["history"]) == 3


def test_get_job_history_returns_404(
    monkeypatch,
):
    monkeypatch.setattr(
        api_main,
        "get_job",
        lambda job_id: None,
    )

    response = client.get(
        "/jobs/not_found/history"
    )

    assert response.status_code == 404
