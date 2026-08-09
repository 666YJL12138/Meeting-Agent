from functools import lru_cache
from faster_whisper import WhisperModel

MODEL_PATH = r"D:\futurework\Meeting-Agent\Meeting-Agent\models\faster-whisper-tiny"

@lru_cache(maxsize=1)
def get_asr_model():
    return WhisperModel(
        MODEL_PATH,
        device="cpu",
        compute_type="int8",
    )


def transcribe_audio(wav_path: str, meeting_id: str) -> list[dict]:
    model = get_asr_model()

    segments, info = model.transcribe(
        wav_path,
        language="zh", 
        task="transcribe",
        vad_filter=False,
        beam_size=5,
        initial_prompt="以下是普通话会议录音，请使用简体中文输出，不要使用繁体字。"
    )

    spans = []

    for index, segment in enumerate(segments, start=1):
        text = segment.text.strip()
        if not text:
            continue

        spans.append({
            "span_id": f"{meeting_id}_span_{index:04d}",
            "meeting_id": meeting_id,
            "speaker_id": "speaker_unknown",
            "start_ms": int(segment.start * 1000),
            "end_ms": int(segment.end * 1000),
            "text": text,
            "asr_confidence": estimate_confidence(segment),
            "overlap": False,
        })

    return spans


def estimate_confidence(segment) -> float:
    no_speech_prob = getattr(segment, "no_speech_prob", 0.0) or 0.0
    confidence = 1.0 - no_speech_prob
    return round(max(0.0, min(1.0, confidence)), 2)
