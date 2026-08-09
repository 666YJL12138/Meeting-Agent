from functools import lru_cache
from pathlib import Path
import os


PYANNOTE_MODEL = os.getenv(
    "PYANNOTE_MODEL",
    "pyannote/speaker-diarization-community-1",
)
HF_TOKEN = os.getenv("HF_TOKEN")


@lru_cache(maxsize=1)
def get_diarization_pipeline():
    try:
        from pyannote.audio import Pipeline
        import torch

        if Path(PYANNOTE_MODEL).exists():
            pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL)
        else:
            pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, token=HF_TOKEN)

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        return pipeline
    except Exception as exc:
        print(f"[diarization] pyannote unavailable, fallback enabled: {exc}")
        return None


def diarize_audio(wav_path: str, voice_segments: list[dict]) -> list[dict]:
    pipeline = get_diarization_pipeline()

    if pipeline is None:
        return fallback_diarization(voice_segments)

    output = pipeline(wav_path)
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)

    speaker_segments = []

    if hasattr(annotation, "itertracks"):
        iterator = annotation.itertracks(yield_label=True)
        for turn, _, speaker in iterator:
            speaker_segments.append({
                "speaker_id": normalize_speaker_id(str(speaker)),
                "start_ms": int(turn.start * 1000),
                "end_ms": int(turn.end * 1000),
                "confidence": 0.8,
                "source": "pyannote",
            })

    return merge_close_segments(speaker_segments)


def fallback_diarization(voice_segments: list[dict]) -> list[dict]:
    speaker_segments = []

    for index, segment in enumerate(voice_segments):
        speaker_id = f"speaker_{index % 2:02d}"
        speaker_segments.append({
            "speaker_id": speaker_id,
            "start_ms": segment["start_ms"],
            "end_ms": segment["end_ms"],
            "confidence": 0.3,
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
            last["confidence"] = min(last["confidence"], seg["confidence"])
        else:
            merged.append(seg)

    return sorted(merged, key=lambda x: x["start_ms"])


def assign_speakers_to_spans(
    transcript_spans: list[dict],
    speaker_segments: list[dict],
) -> list[dict]:
    assigned = []

    for span in transcript_spans:
        best_speaker = "speaker_unknown"
        best_overlap = 0

        for seg in speaker_segments:
            overlap = overlap_ms(
                span["start_ms"],
                span["end_ms"],
                seg["start_ms"],
                seg["end_ms"],
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg["speaker_id"]

        span_duration = max(1, span["end_ms"] - span["start_ms"])
        attribution_confidence = round(best_overlap / span_duration, 2)

        assigned.append({
            **span,
            "speaker_id": best_speaker,
            "speaker_confidence": attribution_confidence,
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
            "source": "diarization",
            "review_status": "pending",
        }
        for speaker_id in speaker_ids
    ]
