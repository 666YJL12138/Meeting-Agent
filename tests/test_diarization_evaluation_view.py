from services.diarization_evaluation_view import (
    evaluation_metric_rows,
    evaluation_reason_text,
    evaluation_status_text,
)


def test_evaluation_view_formats_all_professional_metrics():
    rows = dict(evaluation_metric_rows({
        "evaluation_status": "available",
        "speaker_metrics": {
            "der": 0.125,
            "jer": 0.2,
            "speaker_accuracy": 0.875,
            "miss_rate": 0.05,
            "false_alarm_rate": 0.025,
            "confusion_rate": 0.05,
        },
        "overlap_metrics": {
            "precision": 0.8,
            "recall": 0.75,
            "f1": 0.7742,
        },
    }))

    assert rows == {
        "DER": "12.50%",
        "JER": "20.00%",
        "说话人准确率": "87.50%",
        "漏检率": "5.00%",
        "误报率": "2.50%",
        "混淆率": "5.00%",
        "重叠语音 Precision": "80.00%",
        "重叠语音 Recall": "75.00%",
        "重叠语音 F1": "77.42%",
    }


def test_evaluation_view_explains_unavailable_status():
    evaluation = {
        "evaluation_status": "unavailable",
        "reason": "missing_reference_or_hypothesis",
    }

    assert evaluation_status_text(evaluation) == "不可用"
    assert evaluation_reason_text(evaluation) == "缺少参考标注或待评估的说话人分段"
