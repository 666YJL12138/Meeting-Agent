from functools import lru_cache
from pathlib import Path
import os
import wave

import numpy as np
import torch
from dotenv import load_dotenv
try:
    from pyannote.audio import Pipeline
except ModuleNotFoundError:
    Pipeline = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE, override=True)

DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


@lru_cache(maxsize=1)
def get_diarization_pipeline():
    if Pipeline is None:
        print("[diarization] pyannote.audio is not installed, fallback enabled")
        return None

    try:
        load_dotenv(ENV_FILE, override=True)

        model_value = os.getenv("PYANNOTE_MODEL", DEFAULT_PYANNOTE_MODEL).strip()
        hf_token = os.getenv("HF_TOKEN")
        model_path = Path(model_value).expanduser()

        if model_path.is_dir():
            config_path = model_path / "config.yaml"
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"Local pyannote model is incomplete; missing: {config_path}"
                )

            print(f"[diarization] loading local pyannote model: {model_path}")
            pipeline = Pipeline.from_pretrained(model_path)
        else:
            if not hf_token:
                raise RuntimeError("HF_TOKEN is not set for pyannote model download")

            pipeline = Pipeline.from_pretrained(
                model_value,
                token=hf_token,
            )

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        return pipeline

    except Exception as exc:
        print(f"[diarization] pyannote unavailable, fallback enabled: {exc}")
        return None


def diarize_audio(
    wav_path: str,
    voice_segments: list[dict],
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict:
    pipeline = get_diarization_pipeline()

    if pipeline is None:
        fallback_segments = fallback_diarization(voice_segments)
        return build_diarization_result(
            speaker_segments=fallback_segments,
            exclusive_speaker_segments=fallback_segments,
            source="fallback_round_robin",
        )

    kwargs = {}

    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers

    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    audio = load_waveform_for_pyannote(wav_path)
    output = pipeline(audio, **kwargs)

    # Keep both views: the regular annotation can contain overlapping
    # speakers, while the exclusive view remains useful for ASR alignment.
    annotation = getattr(output, "speaker_diarization", output)
    exclusive_annotation = getattr(
        output,
        "exclusive_speaker_diarization",
        annotation,
    )

    speaker_segments = parse_pyannote_output(annotation)
    exclusive_segments = parse_pyannote_output(exclusive_annotation)

    if not speaker_segments:
        fallback_segments = fallback_diarization(voice_segments)
        return build_diarization_result(
            speaker_segments=fallback_segments,
            exclusive_speaker_segments=fallback_segments,
            source="fallback_round_robin",
        )

    speaker_segments = merge_close_segments(speaker_segments)
    exclusive_segments = merge_close_segments(
        exclusive_segments or speaker_segments,
    )

    return build_diarization_result(
        speaker_segments=speaker_segments,
        exclusive_speaker_segments=exclusive_segments,
        source="pyannote",
    )


def load_waveform_for_pyannote(wav_path: str) -> dict:
    with wave.open(wav_path, "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM wav is supported, got sample_width={sample_width}")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        audio = audio.reshape(-1, channels).T
    else:
        audio = audio.reshape(1, -1)

    waveform = torch.from_numpy(audio)

    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


def parse_pyannote_output(annotation) -> list[dict]:
    speaker_segments = []

    if hasattr(annotation, "itertracks"):
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            speaker_segments.append({
                "speaker_id": normalize_speaker_id(str(speaker)),
                "start_ms": int(turn.start * 1000),
                "end_ms": int(turn.end * 1000),
                # Community diarization output exposes labels and turns,
                # not a calibrated per-turn probability.
                "confidence": None,
                "confidence_source": "unavailable",
                "source": "pyannote",
            })
    else:
        try:
            for turn, speaker in annotation:
                speaker_segments.append({
                    "speaker_id": normalize_speaker_id(str(speaker)),
                    "start_ms": int(turn.start * 1000),
                    "end_ms": int(turn.end * 1000),
                    "confidence": None,
                    "confidence_source": "unavailable",
                    "source": "pyannote",
                })
        except TypeError:
            pass

    return speaker_segments


def fallback_diarization(voice_segments: list[dict]) -> list[dict]:
    speaker_segments = []

    for index, segment in enumerate(voice_segments):
        speaker_id = f"speaker_{index % 2:02d}"
        speaker_segments.append({
            "speaker_id": speaker_id,
            "start_ms": segment["start_ms"],
            "end_ms": segment["end_ms"],
            "confidence": 0.3,
            "confidence_source": "heuristic_fallback",
            "source": "fallback_round_robin",
        })

    return speaker_segments


def normalize_speaker_id(raw: str) -> str:
    raw = raw.lower().replace("speaker_", "").replace("speaker", "")
    digits = "".join(ch for ch in raw if ch.isdigit())

    if digits:
        return f"speaker_{int(digits):02d}"

    return f"speaker_{abs(hash(raw)) % 100:02d}"


def merge_close_segments(segments: list[dict], gap_ms: int = 300) -> list[dict]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda x: (x["speaker_id"], x["start_ms"]))
    merged = [ordered[0]]

    for seg in ordered[1:]:
        last = merged[-1]
        same_speaker = seg["speaker_id"] == last["speaker_id"]
        close_enough = seg["start_ms"] - last["end_ms"] <= gap_ms

        if same_speaker and close_enough:
            last["end_ms"] = max(last["end_ms"], seg["end_ms"])
            last["confidence"] = merge_confidence(
                last.get("confidence"),
                seg.get("confidence"),
            )
        else:
            merged.append(seg)

    return sorted(merged, key=lambda x: x["start_ms"])


def merge_confidence(
    left: float | None,
    right: float | None,
) -> float | None:
    """Merge optional confidence values without inventing a probability."""
    if left is None or right is None:
        return None
    return min(left, right)


def build_diarization_result(
    *,
    speaker_segments: list[dict],
    exclusive_speaker_segments: list[dict],
    source: str,
) -> dict:
    overlap_segments = detect_overlap_segments(speaker_segments)
    return {
        "speaker_segments": speaker_segments,
        "exclusive_speaker_segments": exclusive_speaker_segments,
        "overlap_segments": overlap_segments,
        "diarization_metrics": {
            "source": source,
            "evaluation_status": "not_evaluated",
            "evaluation_reason": "reference_annotations_required",
            "speaker_segment_count": len(speaker_segments),
            "exclusive_segment_count": len(exclusive_speaker_segments),
            "overlap_segment_count": len(overlap_segments),
            "overlap_total_ms": sum(
                item["duration_ms"] for item in overlap_segments
            ),
        },
    }


def detect_overlap_segments(
    speaker_segments: list[dict],
) -> list[dict]:
    """Find time regions covered by at least two distinct speakers."""
    boundaries = sorted({
        point
        for segment in speaker_segments
        for point in (
            int(segment["start_ms"]),
            int(segment["end_ms"]),
        )
    })

    overlaps = []
    for start_ms, end_ms in zip(boundaries, boundaries[1:]):
        if end_ms <= start_ms:
            continue

        active_speakers = sorted({
            segment["speaker_id"]
            for segment in speaker_segments
            if (
                segment["start_ms"] < end_ms
                and segment["end_ms"] > start_ms
            )
        })

        if len(active_speakers) < 2:
            continue

        overlaps.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "speaker_ids": active_speakers,
            "source": "pyannote_overlap",
            "confidence": None,
            "confidence_source": "unavailable",
        })

    return merge_adjacent_overlap_segments(overlaps)


def merge_adjacent_overlap_segments(
    segments: list[dict],
) -> list[dict]:
    if not segments:
        return []

    ordered = sorted(
        segments,
        key=lambda item: (
            item["start_ms"],
            tuple(item["speaker_ids"]),
        ),
    )
    merged = [dict(ordered[0])]

    for segment in ordered[1:]:
        last = merged[-1]
        if (
            last["speaker_ids"] == segment["speaker_ids"]
            and last["end_ms"] == segment["start_ms"]
        ):
            last["end_ms"] = segment["end_ms"]
            last["duration_ms"] += segment["duration_ms"]
        else:
            merged.append(dict(segment))

    return merged


def assign_speakers_to_spans(
    transcript_spans: list[dict],
    speaker_segments: list[dict],
    min_confidence: float = 0.2,
) -> list[dict]:
    assigned = []

    for span in transcript_spans:
        span_duration = max(
            1,
            span["end_ms"] - span["start_ms"],
        )
        candidates = []

        for seg in speaker_segments:
            overlap = overlap_ms(
                span["start_ms"],
                span["end_ms"],
                seg["start_ms"],
                seg["end_ms"],
            )

            if overlap <= 0:
                continue

            alignment_confidence = round(
                overlap / span_duration,
                4,
            )
            if alignment_confidence < min_confidence:
                continue

            candidates.append({
                "speaker_id": seg["speaker_id"],
                "overlap_ms": overlap,
                "alignment_confidence": alignment_confidence,
                "segment_confidence": seg.get("confidence"),
                "confidence_source": seg.get(
                    "confidence_source",
                    "unavailable",
                ),
                "source": seg.get("source", "unknown"),
            })

        candidates.sort(
            key=lambda item: (
                -item["alignment_confidence"],
                item["speaker_id"],
            ),
        )

        if candidates:
            primary = candidates[0]
            speaker_ids = [
                item["speaker_id"] for item in candidates
            ]
            speaker_confidences = {
                item["speaker_id"]: item["alignment_confidence"]
                for item in candidates
            }
            best_speaker = primary["speaker_id"]
            speaker_confidence = primary["alignment_confidence"]
            best_source = primary["source"]
        else:
            speaker_ids = ["speaker_unknown"]
            speaker_confidences = {"speaker_unknown": 0.0}
            best_speaker = "speaker_unknown"
            speaker_confidence = 0.0
            best_source = "unknown"

        assigned.append({
            **span,
            "speaker_id": best_speaker,
            "speaker_ids": speaker_ids,
            "speaker_confidence": speaker_confidence,
            "speaker_confidences": speaker_confidences,
            "speaker_candidates": candidates,
            "overlap": len(speaker_ids) > 1,
            "speaker_source": best_source,
            "confidence_source": "derived_alignment",
        })

    return assigned


def overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def build_speakers(speaker_segments: list[dict]) -> list[dict]:
    speaker_ids = sorted({seg["speaker_id"] for seg in speaker_segments})

    return [
        {
            "speaker_id": speaker_id,
            "display_name": speaker_id,
            "real_name": None,
            "source": detect_source(speaker_id, speaker_segments),
            "review_status": "pending",
        }
        for speaker_id in speaker_ids
    ]


def detect_source(speaker_id: str, speaker_segments: list[dict]) -> str:
    for seg in speaker_segments:
        if seg["speaker_id"] == speaker_id:
            return seg.get("source", "unknown")
    return "unknown"
