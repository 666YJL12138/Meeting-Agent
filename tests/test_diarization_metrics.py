from fastapi.testclient import TestClient

from apps.api import main as api_main
from services.diarization_metrics import (
    evaluate_diarization,
    evaluate_diarization_metrics,
)


def _segments():
    return [
        {"speaker_id": "ref_a", "start_ms": 0, "end_ms": 2000},
        {"speaker_id": "ref_b", "start_ms": 2000, "end_ms": 4000},
    ]


def test_metrics_are_unavailable_without_reference_or_hypothesis():
    result = evaluate_diarization_metrics(
        reference_speaker_segments=[],
        hypothesis_speaker_segments=[],
    )

    assert result["evaluation_status"] == "unavailable"
    assert result["reason"] == "missing_reference_or_hypothesis"
    assert result["speaker_metrics"] == {}
    assert result["overlap_metrics"] == {}


def test_permuted_speaker_labels_get_zero_der_and_jer():
    result = evaluate_diarization_metrics(
        reference_speaker_segments=_segments(),
        hypothesis_speaker_segments=[
            {"speaker_id": "hyp_1", "start_ms": 0, "end_ms": 2000},
            {"speaker_id": "hyp_0", "start_ms": 2000, "end_ms": 4000},
        ],
    )

    assert result["evaluation_status"] == "available"
    assert result["speaker_metrics"]["der"] == 0.0
    assert result["speaker_metrics"]["jer"] == 0.0
    assert result["speaker_metrics"]["miss_rate"] == 0.0
    assert result["speaker_metrics"]["false_alarm_rate"] == 0.0
    assert result["speaker_metrics"]["confusion_rate"] == 0.0
    assert all(
        isinstance(key, str)
        and isinstance(value, str)
        for key, value in result["speaker_label_mapping"].items()
    )


def test_overlap_precision_recall_and_f1_are_reported():
    result = evaluate_diarization_metrics(
        reference_speaker_segments=_segments(),
        hypothesis_speaker_segments=_segments(),
        reference_overlap_segments=[
            {"start_ms": 1000, "end_ms": 2000},
        ],
        hypothesis_overlap_segments=[
            {"start_ms": 1500, "end_ms": 2500},
        ],
    )

    assert result["overlap_metrics"] == {
        "reference_overlap_ms": 1000,
        "hypothesis_overlap_ms": 1000,
        "intersection_overlap_ms": 500,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_evaluate_diarization_outputs_professional_metrics():
    reference = [
        {"speaker_id": "speaker_00", "start_ms": 0, "end_ms": 2000},
        {"speaker_id": "speaker_01", "start_ms": 1000, "end_ms": 3000},
    ]
    hypothesis = [
        {"speaker_id": "speaker_00", "start_ms": 0, "end_ms": 2000},
        {"speaker_id": "speaker_unknown", "start_ms": 1000, "end_ms": 3000},
    ]

    result = evaluate_diarization(reference, hypothesis)

    assert "der_proxy" in result
    assert "jer_proxy" in result
    assert "speaker_accuracy" in result
    assert "osd_precision" in result
    assert "osd_recall" in result
    assert "osd_f1" in result
    assert "unknown_rate" in result
    assert result["unknown_rate"] == 0.5


def test_api_evaluates_current_meeting_hypothesis(monkeypatch):
    meeting_id = "metric_meeting_001"
    api_main.STORE[meeting_id] = {
        "meeting_id": meeting_id,
        "speaker_segments": [
            {"speaker_id": "hyp_0", "start_ms": 0, "end_ms": 2000},
            {"speaker_id": "hyp_1", "start_ms": 2000, "end_ms": 4000},
        ],
    }
    monkeypatch.setattr(api_main, "cache_meeting_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_main, "save_asr_artifacts", lambda *args, **kwargs: {})

    response = TestClient(api_main.app).post(
        f"/meetings/{meeting_id}/diarization/evaluate",
        json={
            "reference_speaker_segments": [
                {"speaker_id": "ref_0", "start_ms": 0, "end_ms": 2000},
                {"speaker_id": "ref_1", "start_ms": 2000, "end_ms": 4000},
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["meeting_id"] == meeting_id
    assert body["evaluation_status"] == "available"
    assert body["speaker_metrics"]["der"] == 0.0
    assert (
        api_main.STORE[meeting_id]["diarization_evaluation"][
            "evaluation_status"
        ]
        == "available"
    )


def test_api_returns_stored_diarization_evaluation():
    meeting_id = "metric_meeting_stored"
    api_main.STORE[meeting_id] = {
        "meeting_id": meeting_id,
        "speakers": [],
        "speaker_segments": [],
        "diarization_evaluation": {
            "evaluation_status": "available",
            "speaker_metrics": {"der": 0.1},
            "overlap_metrics": {"f1": 0.9},
        },
    }

    response = TestClient(api_main.app).get(
        f"/meetings/{meeting_id}/diarization"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["diarization_evaluation"]["evaluation_status"] == "available"
    assert body["diarization_evaluation"]["speaker_metrics"]["der"] == 0.1
