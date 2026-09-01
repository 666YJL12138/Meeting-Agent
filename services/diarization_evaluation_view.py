from typing import Any


def _format_rate(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _format_number(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def evaluation_metric_rows(evaluation: dict | None) -> list[tuple[str, str]]:
    """Return the stable user-facing metric labels shared by every client."""
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    speaker = evaluation.get("speaker_metrics") or {}
    overlap = evaluation.get("overlap_metrics") or {}

    return [
        ("DER", _format_rate(speaker.get("der"))),
        ("JER", _format_rate(speaker.get("jer"))),
        ("说话人准确率", _format_rate(speaker.get("speaker_accuracy"))),
        ("漏检率", _format_rate(speaker.get("miss_rate"))),
        ("误报率", _format_rate(speaker.get("false_alarm_rate"))),
        ("混淆率", _format_rate(speaker.get("confusion_rate"))),
        ("重叠语音 Precision", _format_rate(overlap.get("precision"))),
        ("重叠语音 Recall", _format_rate(overlap.get("recall"))),
        ("重叠语音 F1", _format_rate(overlap.get("f1"))),
    ]


def evaluation_status_text(evaluation: dict | None) -> str:
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    status = evaluation.get("evaluation_status") or "unavailable"
    if status == "available":
        return "可用"
    if status == "unavailable":
        return "不可用"
    return str(status)


def evaluation_reason_text(evaluation: dict | None) -> str:
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    reason = evaluation.get("reason")
    if reason == "missing_reference_or_hypothesis":
        return "缺少参考标注或待评估的说话人分段"
    return str(reason or "")
