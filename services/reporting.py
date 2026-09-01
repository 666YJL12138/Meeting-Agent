from pathlib import Path
import json

RESULT_DIR = Path("outputs/asr")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def save_asr_artifacts(meeting_id: str, state: dict) -> dict:
    json_path = RESULT_DIR / f"{meeting_id}.json"
    txt_path = RESULT_DIR / f"{meeting_id}_transcript.txt"

    payload = {
        "meeting_id": state.get("meeting_id"),
        "title": state.get("title", "会议纪要"),
        "host": state.get("host", "-"),
        "language": state.get("language", "zh-CN"),
        "status": state.get("status"),
        "audio_info": state.get("audio_info", {}),
        "participants": state.get("participants", []),
        "voice_segments": state.get("voice_segments", []),
        "speakers": state.get("speakers", []),
        "speaker_ids": state.get("speaker_ids", []),
        "speaker_segments": state.get("speaker_segments", []),
        "exclusive_speaker_segments": state.get(
            "exclusive_speaker_segments",
            [],
        ),
        "overlap_segments": state.get("overlap_segments", []),
        "diarization_metrics": state.get(
            "diarization_metrics",
            {},
        ),
        "diarization_evaluation": state.get(
            "diarization_evaluation",
            {},
        ),
        "speaker_mapping": state.get("speaker_mapping", {}),
        "speaker_name_map": state.get("speaker_name_map", {}),
        "transcript_spans": state.get("transcript_spans", []),
        "evidence_links": state.get("evidence_links", []),
        "claims": state.get("claims", []),
        "speaker_summaries": state.get("speaker_summaries", []),
        "review_required": state.get("review_required", False),
        "review_reasons": state.get("review_reasons", []),
        "review_queue": state.get("review_queue", []),
        "retry_count": state.get("retry_count", 0),
        "node_attempts": state.get("node_attempts", {}),
        "orchestration_trace": state.get("orchestration_trace", []),
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
