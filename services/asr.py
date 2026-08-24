import os

from services.artifact_cache import (
    build_cache_key,
    load_cache,
    save_cache,
)
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel
from services.speaker_identity import (
    apply_speaker_name_map,
    infer_speaker_name_map,
)

try:
    from opencc import OpenCC
except ImportError:
    OpenCC = None


MODEL_PATH = r"D:\futurework\Meeting-Agent\Meeting-Agent\models\faster-whisper-tiny"

SIMPLIFIER = OpenCC("t2s") if OpenCC else None


def get_asr_runtime_config() -> dict:
    """Choose the fastest supported local runtime without changing ASR quality."""
    configured_device = os.getenv("ASR_DEVICE", "").strip().lower()

    if configured_device:
        device = configured_device
    else:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    configured_compute_type = os.getenv(
        "ASR_COMPUTE_TYPE",
        "",
    ).strip()

    if configured_compute_type:
        compute_type = configured_compute_type
    else:
        compute_type = "float16" if device == "cuda" else "int8"

    default_threads = max(
        1,
        (os.cpu_count() or 4) - 2,
    )
    cpu_threads_value = os.getenv(
        "ASR_CPU_THREADS",
        "",
    ).strip()
    num_workers_value = os.getenv(
        "ASR_NUM_WORKERS",
        "1",
    ).strip()

    cpu_threads = max(
        1,
        int(cpu_threads_value or default_threads),
    )
    num_workers = max(
        1,
        int(num_workers_value or "1"),
    )

    return {
        "device": device,
        "compute_type": compute_type,
        "cpu_threads": cpu_threads,
        "num_workers": num_workers,
    }


@lru_cache(maxsize=1)
def get_asr_model():
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(f"ASR model path does not exist: {model_path}")

    runtime = get_asr_runtime_config()

    print(
        "[asr] loading model "
        f"device={runtime['device']} "
        f"compute_type={runtime['compute_type']} "
        f"cpu_threads={runtime['cpu_threads']} "
        f"num_workers={runtime['num_workers']}",
        flush=True,
    )

    return WhisperModel(
        str(model_path),
        device=runtime["device"],
        compute_type=runtime["compute_type"],
        cpu_threads=runtime["cpu_threads"],
        num_workers=runtime["num_workers"],
    )


def transcribe_audio(
    wav_path: str,
    meeting_id: str,
    participants: list[str] | None = None,
) -> list[dict]:
    cache_config = {
        "model_path": MODEL_PATH,
        "language": "zh",
        "beam_size": 5,
        "vad_filter": True,
        **get_asr_runtime_config(),
    }

    cache_key = build_cache_key(
        wav_path,
        "asr",
        cache_config,
    )

    cached = load_cache("asr", cache_key)

    if cached:
        cached_spans = reindex_spans(
            [
                {
                    **item,
                    "meeting_id": meeting_id,
                }
                for item in cached.get("transcript_spans", [])
            ],
            meeting_id,
        )

        speaker_name_map = infer_speaker_name_map(
            cached_spans,
            participants,
        )

        return apply_speaker_name_map(
            {
                "transcript_spans": cached_spans,
                "speaker_ids": sorted({
                    item.get("speaker_id")
                    for item in cached_spans
                    if item.get("speaker_id")
                }),
            },
            speaker_name_map,
        )["transcript_spans"]

    model = get_asr_model()

    segments, info = model.transcribe(
        wav_path,
        language="zh",
        task="transcribe",
        vad_filter=True,

        # 保留原有精度配置
        beam_size=5,

        initial_prompt=(
            "以下是普通话会议录音，"
            "请使用简体中文输出，"
            "不要使用繁体字。"
        ),
    )

    spans = []

    for index, segment in enumerate(segments, start=1):
        text = normalize_text(segment.text.strip())

        if not text:
            continue

        span = {
            "span_id": f"{meeting_id}_span_{index:04d}",
            "meeting_id": meeting_id,
            "speaker_id": "speaker_unknown",
            "start_ms": int(segment.start * 1000),
            "end_ms": int(segment.end * 1000),
            "text": text,
            "asr_confidence": estimate_confidence(segment),
            "overlap": False,
        }

        spans.extend(split_long_span(span))

    result = reindex_spans(spans, meeting_id)
    speaker_name_map = infer_speaker_name_map(result, participants)
    result = apply_speaker_name_map(
        {
            "transcript_spans": result,
            "speaker_ids": sorted({
                item.get("speaker_id")
                for item in result
                if item.get("speaker_id")
            }),
        },
        speaker_name_map,
    )["transcript_spans"]

    save_cache(
        "asr",
        cache_key,
        {
            "transcript_spans": result,
            "speaker_name_map": speaker_name_map,
            "model_config": cache_config,
        },
    )

    return result


def normalize_text(text: str) -> str:
    if SIMPLIFIER:
        text = SIMPLIFIER.convert(text)

    return (
        text.replace("，", ",")
        .replace("。", "。")
        .replace("？", "?")
        .replace("！", "!")
        .strip()
    )


def split_long_span(span: dict, max_duration_ms: int = 8000) -> list[dict]:
    duration = span["end_ms"] - span["start_ms"]

    if duration <= max_duration_ms:
        return [span]

    text = span["text"]
    parts = split_by_punctuation(text)

    if len(parts) <= 1:
        return [span]

    total_chars = sum(len(part) for part in parts)
    current_start = span["start_ms"]
    output = []

    for part in parts:
        ratio = len(part) / max(1, total_chars)
        part_duration = int(duration * ratio)
        current_end = current_start + part_duration

        output.append({
            **span,
            "start_ms": current_start,
            "end_ms": current_end,
            "text": part,
        })

        current_start = current_end

    output[-1]["end_ms"] = span["end_ms"]
    return output


def split_by_punctuation(text: str) -> list[str]:
    separators = ["。", "!", "?", "；", ";"]
    parts = []
    current = ""

    for char in text:
        current += char
        if char in separators:
            clean = current.strip()
            if clean:
                parts.append(clean)
            current = ""

    if current.strip():
        parts.append(current.strip())

    return parts


def reindex_spans(spans: list[dict], meeting_id: str) -> list[dict]:
    ordered = sorted(spans, key=lambda x: x["start_ms"])

    for index, span in enumerate(ordered, start=1):
        span["span_id"] = f"{meeting_id}_span_{index:04d}"

    return ordered


def estimate_confidence(segment) -> float:
    no_speech_prob = getattr(segment, "no_speech_prob", 0.0) or 0.0
    confidence = 1.0 - no_speech_prob
    return round(max(0.0, min(1.0, confidence)), 2)
