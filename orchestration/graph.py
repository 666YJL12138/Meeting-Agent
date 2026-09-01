import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import signature
from time import perf_counter

from langgraph.graph import StateGraph, START, END

from agents.claim_agent import (
    BatchClaimAgent,
    ClaimAgent,
    build_compact_evidence,
)
from agents.critic_agent import CriticAgent
from agents.notification_agent import NotificationAgent
from agents.rag_agent import RAGAgent
from agents.report_agent import ReportAgent
from llm.schemas import AgentClaim
from orchestration.state import MeetingAgentState
from services.artifact_cache import (
    build_agent_cache_key,
    load_cache,
    save_cache,
)
from services.performance import measure_stage


DEFAULT_NUM_PREDICT = 768
DEFAULT_MAX_RETRIES = 1


CLAIM_TASKS = {
    "contributions": (
        "\u63d0\u53d6\u6bcf\u4e2a\u53d1\u8a00\u4eba\u7684\u5173\u952e\u8d21\u732e",
        "\u89c2\u70b9",
    ),
    "action_items": (
        "\u63d0\u53d6\u4f1a\u8bae\u4e2d\u7684\u884c\u52a8\u9879\u3001\u8d1f\u8d23\u4eba\u548c\u4efb\u52a1",
        "\u884c\u52a8\u9879",
    ),
    "risks": (
        "\u63d0\u53d6\u4f1a\u8bae\u4e2d\u63d0\u5230\u7684\u98ce\u9669\u3001\u963b\u585e\u548c\u4e0d\u786e\u5b9a\u6027",
        "\u98ce\u9669",
    ),
}


def _trace(
    state: MeetingAgentState,
    node: str,
    status: str,
    detail: str | None = None,
) -> None:
    event = {
        "node": node,
        "status": status,
    }
    if detail:
        event["detail"] = detail
    state.setdefault("orchestration_trace", []).append(event)


def _attempt(
    state: MeetingAgentState,
    node: str,
) -> int:
    attempts = state.setdefault("node_attempts", {})
    attempts[node] = attempts.get(node, 0) + 1
    return attempts[node]


def _max_retries() -> int:
    try:
        return max(
            0,
            int(os.getenv("ORCHESTRATOR_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))),
        )
    except ValueError:
        return DEFAULT_MAX_RETRIES


def _fallback_claim(
    state: MeetingAgentState,
    claim_type: str,
    keywords: list[str],
    summary: str,
    exclude_keywords: list[str] | None = None,
) -> list[dict]:
    exclude_keywords = exclude_keywords or []
    claims = []

    for item in state["evidence"]:
        text = item.get("text", "")
        if any(keyword in text for keyword in exclude_keywords):
            continue
        if not any(keyword in text for keyword in keywords):
            continue

        claims.append(
            AgentClaim(
                speaker_id=item["speaker_id"],
                claim_type=claim_type,
                summary=summary,
                evidence_ids=[item["evidence_id"]],
                quote=text[:60],
                start_ms=item["start_ms"],
                end_ms=item["end_ms"],
                confidence=0.55,
                support_status="supported",
            ).model_dump()
        )

    return claims


def _fallback_for_task(
    state: MeetingAgentState,
    key: str,
) -> list[dict]:
    if key == "contributions":
        return _fallback_claim(
            state,
            "\u89c2\u70b9",
            ["\u5efa\u8bae", "\u51b3\u7b56", "\u786e\u8ba4", "\u603b\u7ed3"],
            "\u6839\u636e\u539f\u8bdd\u63d0\u53d6\u7684\u5173\u952e\u8d21\u732e",
        )

    if key == "action_items":
        return _fallback_claim(
            state,
            "\u884c\u52a8\u9879",
            [
                "\u5f85\u529e",
                "\u8d1f\u8d23",
                "\u5206\u5de5",
                "\u6574\u7406",
                "\u6d4b\u8bd5\u8868\u683c",
                "\u8bb0\u5f55",
                "\u9a8c\u6536",
                "\u9a8c\u6536\u6d41\u7a0b",
                "\u63d0\u4ea4",
                "\u786e\u8ba4",
                "\u5b8c\u6210",
                "\u8ddf\u8fdb",
                "\u5b89\u6392",
                "\u540c\u6b65",
            ],
            "\u6839\u636e\u539f\u8bdd\u63d0\u53d6\u7684\u884c\u52a8\u9879",
            exclude_keywords=["\u98ce\u9669\u70b9", "\u98ce\u9669"],
        )

    return _fallback_claim(
        state,
        "\u98ce\u9669",
        ["\u98ce\u9669", "\u566a\u58f0", "\u9519\u8bef", "\u4eba\u5de5", "\u95ee\u9898"],
        "\u6839\u636e\u539f\u8bdd\u63d0\u53d6\u7684\u98ce\u9669\u70b9",
    )


def _run_claim_task(
    state: MeetingAgentState,
    key: str,
) -> tuple[str, list[dict], str | None, float]:
    task_type, claim_type = CLAIM_TASKS[key]
    started_at = perf_counter()

    try:
        agent = ClaimAgent(task_type, claim_type)
        run_parameters = signature(agent.run).parameters
        if "compact_evidence_json" in run_parameters:
            claims = agent.run(
                state["meeting_id"],
                state["evidence"],
                compact_evidence_json=state.get("compact_evidence_json"),
            )
        else:
            # Keep compatibility with older test doubles and custom agents.
            claims = agent.run(state["meeting_id"], state["evidence"])

        return (
            key,
            [item.model_dump() for item in claims],
            None,
            perf_counter() - started_at,
        )
    except Exception as exc:
        return (
            key,
            _fallback_for_task(state, key),
            f"{key} failed: {exc}",
            perf_counter() - started_at,
        )


def _build_agent_cache_config(extract_mode: str) -> dict:
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1536"))
    return {
        "model": os.getenv("LLM_MODEL", "glm4"),
        "prompt_version": (
            "claim_extraction_batch_v1"
            if extract_mode == "batched"
            else "claim_extraction_v1"
        ),
        "extract_mode": extract_mode,
        "max_evidence_items": int(
            os.getenv("LLM_MAX_EVIDENCE_ITEMS", "12")
        ),
        "max_evidence_chars": int(
            os.getenv("LLM_MAX_EVIDENCE_CHARS", "160")
        ),
        "max_tokens": max_tokens,
        "num_ctx": int(os.getenv("LLM_NUM_CTX", "4096")),
        "num_predict": int(
            os.getenv(
                "LLM_NUM_PREDICT",
                str(min(max_tokens, DEFAULT_NUM_PREDICT)),
            )
        ),
    }


def _apply_cached_claims(
    state: MeetingAgentState,
    cached: dict,
    *,
    extract_mode: str,
) -> MeetingAgentState:
    for key in CLAIM_TASKS:
        state[key] = cached.get(key, [])

    timings = state.setdefault("timings", {})
    cached_timings = cached.get("timings", {})
    if extract_mode == "batched":
        batch_elapsed = cached_timings.get(
            "batch",
            cached_timings.get("agent_claim_extraction", 0.0),
        )
        timings["agent_batch_extraction"] = batch_elapsed
        timings["agent_claim_extraction"] = batch_elapsed
        for key in CLAIM_TASKS:
            timings[f"agent_{key}"] = batch_elapsed
    else:
        timings["agent_cache_lookup"] = 0.0
        timings["agent_claim_extraction"] = 0.0
        for key in CLAIM_TASKS:
            timings[f"agent_{key}"] = cached_timings.get(
                key,
                0.0,
            )

    state["agent_cache_hit"] = True
    return state


def _run_batched_claim_extraction(
    state: MeetingAgentState,
) -> tuple[dict[str, list[dict]], str | None, float]:
    started_at = perf_counter()

    try:
        agent = BatchClaimAgent()
        result = agent.run(state["meeting_id"], state["evidence"])
        elapsed = perf_counter() - started_at
        return (
            {
                "contributions": [item.model_dump() for item in result["contributions"]],
                "action_items": [item.model_dump() for item in result["action_items"]],
                "risks": [item.model_dump() for item in result["risks"]],
            },
            None,
            elapsed,
        )
    except Exception as exc:
        return ({}, f"batched extraction failed: {exc}", perf_counter() - started_at)


def _apply_claim_task(
    state: MeetingAgentState,
    key: str,
) -> MeetingAgentState:
    result_key, claims, error, elapsed = _run_claim_task(state, key)
    state[result_key] = claims
    state.setdefault("timings", {})[f"agent_{result_key}"] = round(
        elapsed,
        3,
    )
    if error:
        state.setdefault("errors", []).append(error)
    return state


def extract_contributions(state: MeetingAgentState) -> MeetingAgentState:
    return _apply_claim_task(state, "contributions")


def extract_action_items(state: MeetingAgentState) -> MeetingAgentState:
    return _apply_claim_task(state, "action_items")


def extract_risks(state: MeetingAgentState) -> MeetingAgentState:
    return _apply_claim_task(state, "risks")


def extract_all_claims(
    state: MeetingAgentState,
) -> MeetingAgentState:
    """Run independent claim agents concurrently and merge deterministically."""
    _attempt(state, "extract_all_claims")
    _trace(state, "extract_all_claims", "started")
    extract_mode = os.getenv("AGENT_EXTRACT_MODE", "batched").strip().lower()
    max_workers = int(os.getenv("LLM_CONCURRENCY", "3"))
    max_workers = max(1, min(len(CLAIM_TASKS), max_workers))
    results = {}
    errors = []
    timings = state.setdefault("timings", {})
    cache_enabled = os.getenv("AGENT_CACHE_ENABLED", "1") == "1"
    cache_config = _build_agent_cache_config(extract_mode)
    cache_key = build_agent_cache_key(
        state["meeting_id"],
        state.get("evidence", []),
        cache_config,
    )

    if cache_enabled:
        cached = load_cache("agent", cache_key)
        if cached:
            result = _apply_cached_claims(
                state,
                cached,
                extract_mode=extract_mode,
            )
            _trace(state, "extract_all_claims", "completed", "cache_hit")
            return result

    state["agent_cache_hit"] = False
    if extract_mode == "batched":
        with measure_stage(state, "agent_claim_extraction"):
            batched_result, error, elapsed = _run_batched_claim_extraction(state)
            if batched_result:
                results.update(batched_result)
                timings["agent_batch_extraction"] = round(elapsed, 3)
                timings["agent_claim_extraction"] = round(elapsed, 3)
                for key in CLAIM_TASKS:
                    timings[f"agent_{key}"] = round(elapsed, 3)
            else:
                errors.append(error or "batched extraction failed")
                extract_mode = "parallel"
                cache_config = _build_agent_cache_config(extract_mode)
                cache_key = build_agent_cache_key(
                    state["meeting_id"],
                    state.get("evidence", []),
                    cache_config,
                )
                state["compact_evidence_json"] = build_compact_evidence(
                    state.get("evidence", [])
                )
                with ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="claim-agent",
                ) as executor:
                    futures = {
                        executor.submit(
                            _run_claim_task,
                            state,
                            key,
                        ): key
                        for key in CLAIM_TASKS
                    }

                    for future in as_completed(futures):
                        key = futures[future]
                        try:
                            result_key, claims, error, elapsed = future.result()
                        except Exception as exc:
                            result_key = key
                            claims = _fallback_for_task(state, key)
                            error = f"{key} failed: {exc}"
                            elapsed = 0.0

                        results[result_key] = claims
                        timings[f"agent_{result_key}"] = round(elapsed, 3)
                        if error:
                            errors.append(error)
    else:
        state["compact_evidence_json"] = build_compact_evidence(
            state.get("evidence", [])
        )

        with measure_stage(state, "agent_claim_extraction"):
            with ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="claim-agent",
            ) as executor:
                futures = {
                    executor.submit(
                        _run_claim_task,
                        state,
                        key,
                    ): key
                    for key in CLAIM_TASKS
                }

                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        result_key, claims, error, elapsed = future.result()
                    except Exception as exc:
                        result_key = key
                        claims = _fallback_for_task(state, key)
                        error = f"{key} failed: {exc}"
                        elapsed = 0.0

                    results[result_key] = claims
                    timings[f"agent_{result_key}"] = round(elapsed, 3)
                    if error:
                        errors.append(error)

    for key in CLAIM_TASKS:
        state[key] = results.get(key, [])

    state.setdefault("errors", []).extend(errors)
    if cache_enabled and not errors:
        if extract_mode == "batched":
            cached_timings = {
                "batch": timings.get("agent_batch_extraction", 0.0),
            }
        else:
            cached_timings = {
                key: timings.get(f"agent_{key}", 0.0)
                for key in CLAIM_TASKS
            }
        save_cache(
            "agent",
            cache_key,
            {
                **{key: state[key] for key in CLAIM_TASKS},
                "timings": cached_timings,
            },
        )
    _trace(
        state,
        "extract_all_claims",
        "completed" if not errors else "completed_with_errors",
    )
    return state


def verify_all_claims(state: MeetingAgentState) -> MeetingAgentState:
    _attempt(state, "verify_all_claims")
    _trace(state, "verify_all_claims", "started")
    last_error = None
    for retry_index in range(_max_retries() + 1):
        try:
            critic = CriticAgent()
            review_queue = []
            for key in ["contributions", "action_items", "risks"]:
                claim_objects = [AgentClaim(**item) for item in state.get(key, [])]
                verified = critic.verify(claim_objects, state["evidence"])
                supported = []
                for item in verified:
                    payload = item.model_dump()
                    if item.support_status == "supported":
                        supported.append(payload)
                    else:
                        review_queue.append({
                            "source": key,
                            "claim": payload,
                            "reason": item.review_reason or "claim_not_supported_by_evidence",
                            "support_status": item.support_status,
                        })
                state[key] = supported
            state["review_queue"] = review_queue
            state["retry_count"] = state.get("retry_count", 0) + retry_index
            _trace(
                state,
                "verify_all_claims",
                "completed",
                f"review_queue={len(review_queue)}",
            )
            return state
        except Exception as exc:
            last_error = exc
            if retry_index < _max_retries():
                _attempt(state, "verify_all_claims")
                continue

    state.setdefault("errors", []).append(
        f"verify_all_claims failed after retry: {last_error}"
    )
    state["review_queue"] = state.get("review_queue", [])
    _trace(state, "verify_all_claims", "failed", str(last_error))
    return state


def review_gate(state: MeetingAgentState) -> MeetingAgentState:
    reasons = list(state.get("review_reasons", []))
    if state.get("review_queue"):
        reasons.append("存在未通过证据校验的结论")
    if state.get("errors"):
        reasons.append("编排过程中存在节点错误")

    for item in state.get("evidence", []):
        if item.get("speaker_id") == "speaker_unknown":
            reasons.append("存在未知说话人")
            break
        if item.get("overlap"):
            reasons.append("存在多人重叠语音")
            break

    state["review_reasons"] = list(dict.fromkeys(reasons))
    state["review_required"] = bool(state["review_reasons"])
    _trace(
        state,
        "review_gate",
        "completed",
        "human_review" if state["review_required"] else "automatic",
    )
    return state


def route_after_review_gate(state: MeetingAgentState) -> str:
    return "human_review" if state.get("review_required") else "rag_enrich_claims"


def human_review(state: MeetingAgentState) -> MeetingAgentState:
    queue = list(state.get("review_queue", []))
    reasons = state.get("review_reasons", [])
    if not queue:
        queue.append({
            "source": "orchestrator",
            "reason": "; ".join(reasons) or "需要人工确认",
        })
    state["review_queue"] = queue
    _trace(
        state,
        "human_review",
        "queued",
        f"items={len(queue)}",
    )
    return state


def rag_enrich_claims(state: MeetingAgentState) -> MeetingAgentState:
    _attempt(state, "rag_enrich_claims")
    _trace(state, "rag_enrich_claims", "started")
    last_error = None
    for retry_index in range(_max_retries() + 1):
        try:
            all_claims = (
                state.get("contributions", [])
                + state.get("action_items", [])
                + state.get("risks", [])
            )
            state["rag_supplements"] = RAGAgent(top_k=3).enrich_claims(
                meeting_id=state["meeting_id"],
                claims=all_claims,
                evidence=state.get("evidence", []),
            )
            state["retry_count"] = state.get("retry_count", 0) + retry_index
            _trace(state, "rag_enrich_claims", "completed")
            return state
        except Exception as exc:
            last_error = exc
            if retry_index < _max_retries():
                _attempt(state, "rag_enrich_claims")
                continue

    state.setdefault("errors", []).append(
        f"rag_enrich_claims failed after retry: {last_error}"
    )
    state["rag_supplements"] = []
    _trace(state, "rag_enrich_claims", "failed", str(last_error))
    return state


def generate_report(state: MeetingAgentState) -> MeetingAgentState:
    _attempt(state, "generate_report")
    _trace(state, "generate_report", "started")
    last_error = None
    for retry_index in range(_max_retries() + 1):
        try:
            report_result = ReportAgent().run(state)
            state["pdf_path"] = report_result["pdf_path"]
            state["claims"] = report_result["claims"]
            state["speaker_summaries"] = report_result["speaker_summaries"]
            state["evidence_links"] = report_result["evidence_links"]
            state["speaker_ids"] = report_result.get(
                "speaker_ids",
                state.get("speaker_ids", []),
            )
            state["speaker_segments"] = report_result.get(
                "speaker_segments",
                state.get("speaker_segments", []),
            )
            state["exclusive_speaker_segments"] = report_result.get(
                "exclusive_speaker_segments",
                state.get("exclusive_speaker_segments", []),
            )
            state["overlap_segments"] = report_result.get(
                "overlap_segments",
                state.get("overlap_segments", []),
            )
            state["diarization_metrics"] = report_result.get(
                "diarization_metrics",
                state.get("diarization_metrics", {}),
            )
            state["diarization_evaluation"] = report_result.get(
                "diarization_evaluation",
                state.get("diarization_evaluation", {}),
            )
            state["retry_count"] = state.get("retry_count", 0) + retry_index
            _trace(state, "generate_report", "completed")
            return state
        except Exception as exc:
            last_error = exc
            if retry_index < _max_retries():
                _attempt(state, "generate_report")
                continue

    state.setdefault("errors", []).append(
        f"generate_report failed after retry: {last_error}"
    )
    _trace(state, "generate_report", "failed", str(last_error))
    return state


def deliver_report_email(state: MeetingAgentState) -> MeetingAgentState:
    try:
        state["email_delivery"] = NotificationAgent().run(state)
    except Exception as exc:
        state["errors"].append(f"deliver_report_email failed: {exc}")
        state["email_delivery"] = None
    return state


def build_meeting_graph():
    graph = StateGraph(MeetingAgentState)
    graph.add_node("extract_all_claims", extract_all_claims)
    graph.add_node("verify_all_claims", verify_all_claims)
    graph.add_node("review_gate", review_gate)
    graph.add_node("human_review", human_review)
    graph.add_node("rag_enrich_claims", rag_enrich_claims)
    graph.add_node("generate_report", generate_report)
    graph.add_node("deliver_report_email", deliver_report_email)

    graph.add_edge(START, "extract_all_claims")
    graph.add_edge("extract_all_claims", "verify_all_claims")
    graph.add_edge("verify_all_claims", "review_gate")
    graph.add_conditional_edges(
        "review_gate",
        route_after_review_gate,
        {
            "human_review": "human_review",
            "rag_enrich_claims": "rag_enrich_claims",
        },
    )
    graph.add_edge("human_review", "rag_enrich_claims")
    graph.add_edge("rag_enrich_claims", "generate_report")
    graph.add_edge("generate_report", "deliver_report_email")
    graph.add_edge("deliver_report_email", END)
    return graph.compile()
