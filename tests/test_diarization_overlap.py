from services.diarization import (
    assign_speakers_to_spans,
    detect_overlap_segments,
    parse_pyannote_output,
)
from services.agent_result_adapter import build_pdf_ready_result
from services.workflow_runner import build_evidence


def test_detect_overlap_segments_returns_shared_time_regions():
    segments = [
        {
            "speaker_id": "speaker_00",
            "start_ms": 0,
            "end_ms": 2000,
        },
        {
            "speaker_id": "speaker_01",
            "start_ms": 1000,
            "end_ms": 3000,
        },
    ]

    result = detect_overlap_segments(segments)

    assert result == [
        {
            "start_ms": 1000,
            "end_ms": 2000,
            "duration_ms": 1000,
            "speaker_ids": ["speaker_00", "speaker_01"],
            "source": "pyannote_overlap",
            "confidence": None,
            "confidence_source": "unavailable",
        }
    ]


def test_assign_speakers_to_spans_keeps_multiple_speakers():
    spans = [
        {
            "span_id": "span_001",
            "start_ms": 0,
            "end_ms": 2000,
            "text": "多人同时说话",
        }
    ]
    speaker_segments = [
        {
            "speaker_id": "speaker_00",
            "start_ms": 0,
            "end_ms": 2000,
            "source": "pyannote",
            "confidence": None,
            "confidence_source": "unavailable",
        },
        {
            "speaker_id": "speaker_01",
            "start_ms": 500,
            "end_ms": 1500,
            "source": "pyannote",
            "confidence": None,
            "confidence_source": "unavailable",
        },
    ]

    result = assign_speakers_to_spans(spans, speaker_segments)

    assert result[0]["speaker_id"] == "speaker_00"
    assert result[0]["speaker_ids"] == [
        "speaker_00",
        "speaker_01",
    ]
    assert result[0]["overlap"] is True
    assert result[0]["speaker_confidences"] == {
        "speaker_00": 1.0,
        "speaker_01": 0.5,
    }
    assert result[0]["speaker_candidates"][1]["segment_confidence"] is None


def test_parse_pyannote_output_does_not_invent_confidence():
    class FakeTurn:
        start = 1.25
        end = 2.5

    class FakeAnnotation:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield FakeTurn(), "track_0", "SPEAKER_00"

    result = parse_pyannote_output(FakeAnnotation())

    assert result[0]["speaker_id"] == "speaker_00"
    assert result[0]["confidence"] is None
    assert result[0]["confidence_source"] == "unavailable"


def test_parse_pyannote_output_does_not_fake_confidence():
    class FakeTurn:
        start = 0.0
        end = 1.0

    class FakeAnnotation:
        def itertracks(self, yield_label=False):
            yield FakeTurn(), "track_0", "SPEAKER_00"

    segments = parse_pyannote_output(FakeAnnotation())

    assert segments[0]["confidence"] is None
    assert segments[0]["confidence_source"] == "unavailable"


def test_overlap_fields_survive_evidence_and_report_adapters():
    span = {
        "span_id": "span_001",
        "speaker_id": "speaker_00",
        "speaker_ids": ["speaker_00", "speaker_01"],
        "speaker_confidence": 0.75,
        "speaker_confidences": {
            "speaker_00": 1.0,
            "speaker_01": 0.5,
        },
        "speaker_candidates": [
            {"speaker_id": "speaker_00", "alignment_confidence": 1.0},
            {"speaker_id": "speaker_01", "alignment_confidence": 0.5},
        ],
        "overlap": True,
        "confidence_source": "derived_alignment",
        "speaker_source": "pyannote",
        "text": "多人同时说话",
        "start_ms": 0,
        "end_ms": 2000,
        "asr_confidence": 0.9,
    }

    evidence = build_evidence(
        "meeting_001",
        {"transcript_spans": [span]},
    )
    result = build_pdf_ready_result(
        {
            "meeting_id": "meeting_001",
            "evidence": evidence,
            "speaker_ids": ["speaker_00", "speaker_01"],
        }
    )

    assert evidence[0]["speaker_ids"] == ["speaker_00", "speaker_01"]
    assert evidence[0]["overlap"] is True
    assert result["evidence_links"][0]["speaker_ids"] == [
        "speaker_00",
        "speaker_01",
    ]
    assert result["evidence_links"][0]["overlap"] is True
