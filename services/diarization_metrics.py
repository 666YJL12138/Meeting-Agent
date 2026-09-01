from __future__ import annotations

from collections.abc import Iterable
from typing import Any

try:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import (
        DiarizationErrorRate,
        JaccardErrorRate,
    )

    PYANNOTE_AVAILABLE = True
except ModuleNotFoundError:
    Annotation = None
    Segment = None
    DiarizationErrorRate = None
    JaccardErrorRate = None
    PYANNOTE_AVAILABLE = False


def _normalize_segments(segments: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not segments:
        return []

    normalized = []
    for index, segment in enumerate(segments):
        try:
            start_ms = int(segment.get("start_ms", 0))
            end_ms = int(segment.get("end_ms", 0))
        except (TypeError, ValueError):
            continue

        if end_ms <= start_ms:
            continue

        speaker_id = str(segment.get("speaker_id") or "speaker_unknown").strip()
        if not speaker_id:
            speaker_id = "speaker_unknown"

        normalized.append(
            {
                "speaker_id": speaker_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "segment_index": index,
            }
        )

    return sorted(
        normalized,
        key=lambda item: (
            item["start_ms"],
            item["end_ms"],
            item["speaker_id"],
            item["segment_index"],
        ),
    )


def build_annotation(
    segments: list[dict[str, Any]] | None,
    *,
    uri: str,
) -> Annotation:
    if not PYANNOTE_AVAILABLE:
        raise RuntimeError("pyannote is not available in this environment")

    annotation = Annotation(uri=uri)

    for item in _normalize_segments(segments):
        start = item["start_ms"] / 1000.0
        end = item["end_ms"] / 1000.0
        annotation[(Segment(start, end), f"{item['speaker_id']}::{item['segment_index']}")] = item[
            "speaker_id"
        ]

    return annotation


def _build_overlap_intervals(
    segments: list[dict[str, Any]] | None,
) -> list[tuple[int, int]]:
    items = _normalize_segments(segments)
    if not items:
        return []

    boundaries = sorted(
        {
            point
            for segment in items
            for point in (segment["start_ms"], segment["end_ms"])
        }
    )

    overlaps: list[tuple[int, int]] = []
    for start_ms, end_ms in zip(boundaries, boundaries[1:]):
        if end_ms <= start_ms:
            continue

        active_speakers = {
            segment["speaker_id"]
            for segment in items
            if segment["start_ms"] < end_ms and segment["end_ms"] > start_ms
        }

        if len(active_speakers) >= 2:
            overlaps.append((start_ms, end_ms))

    return _merge_adjacent_intervals(overlaps)


def _merge_adjacent_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not intervals:
        return []

    ordered = sorted(intervals)
    merged = [ordered[0]]

    for start_ms, end_ms in ordered[1:]:
        last_start, last_end = merged[-1]
        if start_ms <= last_end:
            merged[-1] = (last_start, max(last_end, end_ms))
        elif start_ms == last_end:
            merged[-1] = (last_start, end_ms)
        else:
            merged.append((start_ms, end_ms))

    return merged


def _interval_duration_ms(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(max(0, end_ms - start_ms) for start_ms, end_ms in intervals)


def _interval_intersection_ms(
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
) -> int:
    total = 0
    i = j = 0
    left_sorted = sorted(left)
    right_sorted = sorted(right)

    while i < len(left_sorted) and j < len(right_sorted):
        left_start, left_end = left_sorted[i]
        right_start, right_end = right_sorted[j]

        start_ms = max(left_start, right_start)
        end_ms = min(left_end, right_end)
        if end_ms > start_ms:
            total += end_ms - start_ms

        if left_end < right_end:
            i += 1
        else:
            j += 1

    return total


def _rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(max(0.0, min(1.0, numerator / denominator)), 4)


def _segment_time_ms(segment: dict[str, Any]) -> tuple[int, int]:
    return int(segment["start_ms"]), int(segment["end_ms"])


def _pair_overlap_ms(
    reference: list[dict[str, Any]],
    hypothesis: list[dict[str, Any]],
) -> int:
    total = 0
    for ref in reference:
        ref_start, ref_end = _segment_time_ms(ref)
        for hyp in hypothesis:
            hyp_start, hyp_end = _segment_time_ms(hyp)
            start_ms = max(ref_start, hyp_start)
            end_ms = min(ref_end, hyp_end)
            if end_ms > start_ms:
                total += end_ms - start_ms
    return total


def _fallback_mapping(
    reference: list[dict[str, Any]],
    hypothesis: list[dict[str, Any]],
) -> dict[str, str]:
    reference_by_speaker: dict[str, list[dict[str, Any]]] = {}
    hypothesis_by_speaker: dict[str, list[dict[str, Any]]] = {}

    for item in reference:
        reference_by_speaker.setdefault(item["speaker_id"], []).append(item)
    for item in hypothesis:
        hypothesis_by_speaker.setdefault(item["speaker_id"], []).append(item)

    pairs = []
    for hyp_id, hyp_items in hypothesis_by_speaker.items():
        for ref_id, ref_items in reference_by_speaker.items():
            pairs.append(
                (
                    _pair_overlap_ms(ref_items, hyp_items),
                    hyp_id,
                    ref_id,
                )
            )

    mapping: dict[str, str] = {}
    used_ref: set[str] = set()
    for overlap_ms, hyp_id, ref_id in sorted(
        pairs,
        key=lambda item: item[0],
        reverse=True,
    ):
        if overlap_ms <= 0 or hyp_id in mapping or ref_id in used_ref:
            continue
        mapping[hyp_id] = ref_id
        used_ref.add(ref_id)

    for hyp_id in hypothesis_by_speaker:
        mapping.setdefault(hyp_id, hyp_id)

    return mapping


def _fallback_duration_ms(segments: list[dict[str, Any]]) -> int:
    return sum(item["end_ms"] - item["start_ms"] for item in segments)


def _unknown_duration_ms(segments: list[dict[str, Any]]) -> int:
    return sum(
        item["end_ms"] - item["start_ms"]
        for item in segments
        if item["speaker_id"] == "speaker_unknown"
    )


def _fallback_evaluate(
    reference: list[dict[str, Any]],
    hypothesis: list[dict[str, Any]],
    reference_overlap: list[dict[str, Any]],
    hypothesis_overlap: list[dict[str, Any]],
) -> dict[str, Any]:
    mapping = _fallback_mapping(reference, hypothesis)
    correct_ms = 0
    for hyp_item in hypothesis:
        ref_id = mapping.get(hyp_item["speaker_id"], hyp_item["speaker_id"])
        ref_items = [
            item for item in reference if item["speaker_id"] == ref_id
        ]
        correct_ms += _pair_overlap_ms(ref_items, [hyp_item])

    total_ref_ms = _fallback_duration_ms(reference)
    total_hyp_ms = _fallback_duration_ms(hypothesis)
    missed_ms = max(0, total_ref_ms - correct_ms)
    false_alarm_ms = max(0, total_hyp_ms - correct_ms)
    confusion_ms = 0
    total_ms = max(total_ref_ms, 1)
    union_ms = max(total_ref_ms + total_hyp_ms - correct_ms, 1)

    reference_overlap_intervals = [
        (item["start_ms"], item["end_ms"])
        for item in reference_overlap
    ]
    hypothesis_overlap_intervals = [
        (item["start_ms"], item["end_ms"])
        for item in hypothesis_overlap
    ]
    overlap_intersection = _interval_intersection_ms(
        reference_overlap_intervals,
        hypothesis_overlap_intervals,
    )
    reference_overlap_ms = _interval_duration_ms(reference_overlap_intervals)
    hypothesis_overlap_ms = _interval_duration_ms(hypothesis_overlap_intervals)

    overlap_precision = _rate(
        overlap_intersection,
        hypothesis_overlap_ms,
    )
    overlap_recall = _rate(
        overlap_intersection,
        reference_overlap_ms,
    )
    overlap_f1 = (
        round(
            0.0
            if overlap_precision + overlap_recall == 0
            else 2 * overlap_precision * overlap_recall
            / (overlap_precision + overlap_recall),
            4,
        )
    )

    speaker_accuracy = _rate(correct_ms, total_ref_ms)
    speaker_error_rate = round(1.0 - speaker_accuracy, 4)
    der = round((missed_ms + false_alarm_ms + confusion_ms) / total_ms, 4)
    jer = round(1.0 - _rate(correct_ms, union_ms), 4)

    return {
        "evaluation_status": "available",
        "reference_speaker_count": len({item["speaker_id"] for item in reference}),
        "hypothesis_speaker_count": len({item["speaker_id"] for item in hypothesis}),
        "speaker_label_mapping": {
            str(hyp_id): str(ref_id)
            for hyp_id, ref_id in mapping.items()
        },
        "speaker_metrics": {
            "der": der,
            "speaker_accuracy": speaker_accuracy,
            "speaker_error_rate": speaker_error_rate,
            "confusion_rate": _rate(confusion_ms, total_ms),
            "miss_rate": _rate(missed_ms, total_ms),
            "false_alarm_rate": _rate(false_alarm_ms, total_ms),
            "correct_ms": correct_ms,
            "total_ms": total_ms,
            "confusion_ms": confusion_ms,
            "missed_ms": missed_ms,
            "false_alarm_ms": false_alarm_ms,
            "jer": jer,
            "jer_speaker_error": speaker_error_rate,
            "jer_speaker_count": len({item["speaker_id"] for item in hypothesis}),
        },
        "overlap_metrics": {
            "reference_overlap_ms": reference_overlap_ms,
            "hypothesis_overlap_ms": hypothesis_overlap_ms,
            "intersection_overlap_ms": overlap_intersection,
            "precision": overlap_precision,
            "recall": overlap_recall,
            "f1": overlap_f1,
        },
    }


def evaluate_diarization(
    reference: list[dict[str, Any]] | None,
    hypothesis: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Return a flat professional metric summary for acceptance tests.

    The full API keeps detailed pyannote-compatible nested metrics. This
    wrapper exposes stable top-level keys that are convenient for dashboards
    and regression tests.
    """
    result = evaluate_diarization_metrics(
        reference_speaker_segments=reference,
        hypothesis_speaker_segments=hypothesis,
    )
    normalized_hypothesis = _normalize_segments(hypothesis)
    hypothesis_ms = _fallback_duration_ms(normalized_hypothesis)
    speaker_metrics = result.get("speaker_metrics", {})
    overlap_metrics = result.get("overlap_metrics", {})

    return {
        "evaluation_status": result.get("evaluation_status", "unavailable"),
        "der_proxy": speaker_metrics.get("der"),
        "jer_proxy": speaker_metrics.get("jer"),
        "speaker_accuracy": speaker_metrics.get("speaker_accuracy"),
        "speaker_error_rate": speaker_metrics.get("speaker_error_rate"),
        "miss_rate": speaker_metrics.get("miss_rate"),
        "false_alarm_rate": speaker_metrics.get("false_alarm_rate"),
        "confusion_rate": speaker_metrics.get("confusion_rate"),
        "osd_precision": overlap_metrics.get("precision"),
        "osd_recall": overlap_metrics.get("recall"),
        "osd_f1": overlap_metrics.get("f1"),
        "unknown_rate": _rate(
            _unknown_duration_ms(normalized_hypothesis),
            hypothesis_ms,
        ),
        "speaker_label_mapping": result.get("speaker_label_mapping", {}),
    }


def evaluate_diarization_metrics(
    *,
    reference_speaker_segments: list[dict[str, Any]] | None,
    hypothesis_speaker_segments: list[dict[str, Any]] | None,
    reference_overlap_segments: list[dict[str, Any]] | None = None,
    hypothesis_overlap_segments: list[dict[str, Any]] | None = None,
    uri: str = "meeting",
    collar_seconds: float = 0.0,
    skip_overlap: bool = False,
) -> dict[str, Any]:
    reference = _normalize_segments(reference_speaker_segments)
    hypothesis = _normalize_segments(hypothesis_speaker_segments)

    if not reference or not hypothesis:
        return {
            "evaluation_status": "unavailable",
            "reason": "missing_reference_or_hypothesis",
            "reference_speaker_count": len(
                {item["speaker_id"] for item in reference}
            ),
            "hypothesis_speaker_count": len(
                {item["speaker_id"] for item in hypothesis}
            ),
            "speaker_metrics": {},
            "overlap_metrics": {},
        }

    if not PYANNOTE_AVAILABLE:
        reference_overlap = _normalize_segments(reference_overlap_segments)
        hypothesis_overlap = _normalize_segments(hypothesis_overlap_segments)
        if not reference_overlap:
            reference_overlap = [
                {
                    "speaker_id": "overlap",
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                }
                for start_ms, end_ms in _build_overlap_intervals(reference)
            ]
        if not hypothesis_overlap:
            hypothesis_overlap = [
                {
                    "speaker_id": "overlap",
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                }
                for start_ms, end_ms in _build_overlap_intervals(hypothesis)
            ]
        return _fallback_evaluate(
            reference,
            hypothesis,
            reference_overlap,
            hypothesis_overlap,
        )

    reference_annotation = build_annotation(reference, uri=uri)
    hypothesis_annotation = build_annotation(hypothesis, uri=uri)

    der_metric = DiarizationErrorRate(
        collar=collar_seconds,
        skip_overlap=skip_overlap,
    )
    jer_metric = JaccardErrorRate(
        collar=collar_seconds,
        skip_overlap=skip_overlap,
    )

    der_detail = der_metric(
        reference_annotation,
        hypothesis_annotation,
        detailed=True,
        uri=uri,
    )
    jer_detail = jer_metric(
        reference_annotation,
        hypothesis_annotation,
        detailed=True,
        uri=uri,
    )
    mapping = der_metric.optimal_mapping(
        reference_annotation.rename_labels(generator="string"),
        hypothesis_annotation.rename_labels(generator="int"),
    )
    stable_mapping = {
        str(hypothesis_label): str(reference_label)
        for hypothesis_label, reference_label in mapping.items()
    }

    reference_overlap = _normalize_segments(reference_overlap_segments)
    hypothesis_overlap = _normalize_segments(hypothesis_overlap_segments)
    if not reference_overlap:
        reference_overlap = [
            {
                "speaker_id": "overlap",
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
            for start_ms, end_ms in _build_overlap_intervals(reference)
        ]
    if not hypothesis_overlap:
        hypothesis_overlap = [
            {
                "speaker_id": "overlap",
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
            for start_ms, end_ms in _build_overlap_intervals(hypothesis)
        ]

    reference_overlap_intervals = [
        (item["start_ms"], item["end_ms"])
        for item in reference_overlap
    ]
    hypothesis_overlap_intervals = [
        (item["start_ms"], item["end_ms"])
        for item in hypothesis_overlap
    ]
    overlap_intersection = _interval_intersection_ms(
        reference_overlap_intervals,
        hypothesis_overlap_intervals,
    )
    reference_overlap_ms = _interval_duration_ms(reference_overlap_intervals)
    hypothesis_overlap_ms = _interval_duration_ms(hypothesis_overlap_intervals)

    overlap_precision = _rate(
        overlap_intersection,
        hypothesis_overlap_ms,
    )
    overlap_recall = _rate(
        overlap_intersection,
        reference_overlap_ms,
    )
    overlap_f1 = (
        round(
            0.0
            if overlap_precision + overlap_recall == 0
            else 2 * overlap_precision * overlap_recall
            / (overlap_precision + overlap_recall),
            4,
        )
    )

    total = float(der_detail.get("total", 0.0))
    correct = float(der_detail.get("correct", 0.0))
    missed = float(der_detail.get("missed detection", 0.0))
    false_alarm = float(der_detail.get("false alarm", 0.0))
    confusion = float(der_detail.get("confusion", 0.0))

    speaker_accuracy = _rate(correct, total)

    return {
        "evaluation_status": "available",
        "reference_speaker_count": len(
            {item["speaker_id"] for item in reference}
        ),
        "hypothesis_speaker_count": len(
            {item["speaker_id"] for item in hypothesis}
        ),
        "speaker_label_mapping": stable_mapping,
        "speaker_metrics": {
            "der": round(float(der_detail.get("diarization error rate", 0.0)), 4),
            "speaker_accuracy": speaker_accuracy,
            "speaker_error_rate": round(1.0 - speaker_accuracy, 4),
            "confusion_rate": _rate(confusion, total),
            "miss_rate": _rate(missed, total),
            "false_alarm_rate": _rate(false_alarm, total),
            "correct_ms": int(round(correct * 1000)),
            "total_ms": int(round(total * 1000)),
            "confusion_ms": int(round(confusion * 1000)),
            "missed_ms": int(round(missed * 1000)),
            "false_alarm_ms": int(round(false_alarm * 1000)),
            "jer": round(float(jer_detail.get("jaccard error rate", 0.0)), 4),
            "jer_speaker_error": round(
                float(jer_detail.get("speaker error", 0.0)),
                4,
            ),
            "jer_speaker_count": int(jer_detail.get("speaker count", 0)),
        },
        "overlap_metrics": {
            "reference_overlap_ms": reference_overlap_ms,
            "hypothesis_overlap_ms": hypothesis_overlap_ms,
            "intersection_overlap_ms": overlap_intersection,
            "precision": overlap_precision,
            "recall": overlap_recall,
            "f1": overlap_f1,
        },
    }
