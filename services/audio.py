import audioop
import subprocess
import wave
from pathlib import Path


PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def normalize_audio_to_wav(input_path: str, meeting_id: str) -> str:
    output_path = PROCESSED_DIR / f"{meeting_id}_16k_mono.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    return str(output_path)


def inspect_wav(wav_path: str) -> dict:
    with wave.open(wav_path, "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()
        duration = frame_count / sample_rate

    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "sample_width": sample_width,
        "duration_seconds": round(duration, 2),
    }


def detect_voice_segments(wav_path: str, threshold: int = 350, frame_ms: int = 30) -> list[dict]:
    segments = []

    with wave.open(wav_path, "rb") as wav:
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames_per_chunk = int(sample_rate * frame_ms / 1000)

        current_start = None
        current_end = None
        index = 0

        while True:
            data = wav.readframes(frames_per_chunk)
            if not data:
                break

            rms = audioop.rms(data, sample_width)
            start_ms = index * frame_ms
            end_ms = start_ms + frame_ms

            if rms >= threshold:
                if current_start is None:
                    current_start = start_ms
                current_end = end_ms
            else:
                if current_start is not None:
                    if current_end - current_start >= 300:
                        segments.append({
                            "start_ms": current_start,
                            "end_ms": current_end,
                        })
                    current_start = None
                    current_end = None

            index += 1

        if current_start is not None and current_end - current_start >= 300:
            segments.append({
                "start_ms": current_start,
                "end_ms": current_end,
            })

    return segments
