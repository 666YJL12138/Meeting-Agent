from functools import lru_cache
from pathlib import Path
import os
import wave

import numpy as np
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE, override=True)

DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


@lru_cache(maxsize=1)
def get_diarization_pipeline():
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
) -> list[dict]:
    pipeline = get_diarization_pipeline()

    if pipeline is None:
        return fallback_diarization(voice_segments)

    kwargs = {}

    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers

    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    audio = load_waveform_for_pyannote(wav_path)
    output = pipeline(audio, **kwargs)

    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)

    speaker_segments = parse_pyannote_output(annotation)

    if not speaker_segments:
        return fallback_diarization(voice_segments)

    return merge_close_segments(speaker_segments)


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
                "confidence": 0.8,
                "source": "pyannote",
            })
    else:
        try:
            for turn, speaker in annotation:
                speaker_segments.append({
                    "speaker_id": normalize_speaker_id(str(speaker)),
                    "start_ms": int(turn.start * 1000),
                    "end_ms": int(turn.end * 1000),
                    "confidence": 0.8,
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
    min_confidence: float = 0.2,
) -> list[dict]:
    assigned = []

    for span in transcript_spans:
        best_speaker = "speaker_unknown"
        best_overlap = 0
        best_source = "unknown"

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
                best_source = seg.get("source", "unknown")

        span_duration = max(1, span["end_ms"] - span["start_ms"])
        speaker_confidence = round(best_overlap / span_duration, 2)

        if speaker_confidence < min_confidence:
            best_speaker = "speaker_unknown"

        assigned.append({
            **span,
            "speaker_id": best_speaker,
            "speaker_confidence": speaker_confidence,
            "speaker_source": best_source,
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
