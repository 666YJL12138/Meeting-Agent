import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.graph import run_audio_asr_graph
from orchestration.graph import build_meeting_graph
from services.job_store import get_job, update_job


def now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def save_asr_result(
    meeting_id: str,
    result: dict[str, Any],
) -> str:
    output_dir = Path("outputs/asr")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / f"{meeting_id}.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return str(output_path)


def save_agent_result(
    meeting_id: str,
    result: dict[str, Any],
) -> str:
    output_dir = Path("outputs/agent")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / f"{meeting_id}.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return str(output_path)


def build_evidence(
    meeting_id: str,
    asr_result: dict[str, Any],
) -> list[dict]:
    evidence = []

    for index, span in enumerate(
        asr_result.get(
            "transcript_spans",
            [],
        )
    ):
        text = str(
            span.get("text", "")
        ).strip()

        if not text:
            continue

        evidence.append({
            "evidence_id": (
                f"ev_{span.get('span_id', index)}"
            ),
            "meeting_id": meeting_id,
            "speaker_id": span.get(
                "speaker_id",
                "speaker_unknown",
            ),
            "speaker_name": span.get(
                "speaker_name",
                span.get("display_name"),
            ),
            "text": text,
            "start_ms": int(
                span.get("start_ms", 0)
            ),
            "end_ms": int(
                span.get("end_ms", 0)
            ),
            "asr_confidence": float(
                span.get(
                    "asr_confidence",
                    0.5,
                )
            ),
            "speaker_confidence": float(
                span.get(
                    "speaker_confidence",
                    0.5,
                )
            ),
            "speaker_source": span.get(
                "speaker_source",
                "unknown",
            ),
        })

    return evidence


def run_full_workflow(
    job_id: str,
    meeting: dict[str, Any],
) -> None:
    meeting_id = meeting["meeting_id"]

    try:
        update_job(
            job_id,
            status="audio_normalizing",
            progress=10,
            stage="音频预处理",
            started_at=now_iso(),
        )

        asr_meeting = {
            **meeting,
            "job_id": job_id,
        }

        asr_state = run_audio_asr_graph(
            asr_meeting
        )

        asr_result = {
            **meeting,
            **asr_state,
        }

        asr_path = save_asr_result(
            meeting_id,
            asr_result,
        )

        evidence = build_evidence(
            meeting_id,
            asr_result,
        )

        update_job(
            job_id,
            status="agent_processing",
            progress=98,
            stage="ASR 与证据阶段完成，开始多 Agent 抽取",
        )

        state = {
            "meeting_id": meeting_id,
            "title": meeting.get(
                "title",
                "会议纪要",
            ),
            "host": meeting.get(
                "host",
                "-",
            ),
            "language": meeting.get(
                "language",
                "zh-CN",
            ),
            "audio_info": asr_state.get(
                "audio_info",
                {},
            ),
            "participants": meeting.get(
                "participants",
                [],
            ),
            "evidence": evidence,
            "speaker_ids": sorted({
                item["speaker_id"]
                for item in evidence
            }),
            "speaker_mapping": asr_state.get(
                "speaker_mapping",
                {},
            ),
            "contributions": [],
            "action_items": [],
            "risks": [],
            "rag_supplements": [],
            "claims": [],
            "speaker_summaries": [],
            "evidence_links": [],
            "send_email": meeting.get(
                "send_email",
                False,
            ),
            "to_email": meeting.get(
                "to_email"
            ),
            "timings": {},
            "errors": [],
        }

        result = build_meeting_graph().invoke(
            state
        )

        update_job(
            job_id,
            status="pdf_generating",
            progress=99,
            stage="生成可信 PDF",
        )

        agent_path = save_agent_result(
            meeting_id,
            result,
        )

        result["agent_result_path"] = agent_path

        summary = {
            "asr_result_path": asr_path,
            "agent_result_path": agent_path,
            "pdf_path": result.get(
                "pdf_path"
            ),
            "claims": len(
                result.get("claims", [])
            ),
            "timings": result.get("timings", {}),
            "agent_cache_hit": result.get("agent_cache_hit", False),
            "contributions": len(
                result.get(
                    "contributions",
                    [],
                )
            ),
            "action_items": len(
                result.get(
                    "action_items",
                    [],
                )
            ),
            "risks": len(
                result.get("risks", [])
            ),
            "speakers": len(
                result.get(
                    "speaker_ids",
                    [],
                )
            ),
            "errors": result.get(
                "errors",
                [],
            ),
        }

        update_job(
            job_id,
            status="completed",
            progress=100,
            stage="分析完成",
            result=summary,
            finished_at=now_iso(),
        )

    except Exception as exc:
        current_job = get_job(job_id) or {}

        current_progress = int(
            current_job.get("progress", 0)
        )

        error_message = (
            f"{type(exc).__name__}: {exc}"
        )[:1000]

        update_job(
            job_id,
            status="failed",
            progress=current_progress,
            stage="任务失败",
            error=error_message,
            result={
                "success": False,
                "error_type": type(exc).__name__,
                "error_message": error_message,
            },
            finished_at=now_iso(),
        )
        
