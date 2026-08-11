from pathlib import Path
import json

RESULT_DIR = Path("outputs/asr")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def save_asr_artifacts(meeting_id: str, state: dict) -> dict:
    json_path = RESULT_DIR / f"{meeting_id}.json"
    txt_path = RESULT_DIR / f"{meeting_id}_transcript.txt"

    payload = {
        "meeting_id": state.get("meeting_id"),
        "status": state.get("status"),
        "audio_info": state.get("audio_info", {}),
        "voice_segments": state.get("voice_segments", []),
        "speakers": state.get("speakers", []),
        "speaker_segments": state.get("speaker_segments", []),
        "speaker_mapping": state.get("speaker_mapping", {}),
        "transcript_spans": state.get("transcript_spans", []),
        "evidence_links": state.get("evidence_links", []),
        "claims": state.get("claims", []),
        "speaker_summaries": state.get("speaker_summaries", []),
        "normalized_audio_uri": state.get("normalized_audio_uri"),
    }

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    transcript_lines = []
    for span in payload["transcript_spans"]:
        transcript_lines.append(
            f"[{span.get('start_ms', 0)}-{span.get('end_ms', 0)}] {span.get('text', '')}"
        )

    txt_path.write_text("\n".join(transcript_lines), encoding="utf-8")

    return {
        "result_json_path": str(json_path),
        "result_txt_path": str(txt_path),
    }
