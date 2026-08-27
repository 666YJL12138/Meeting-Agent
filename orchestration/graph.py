import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import signature
from time import perf_counter

from langgraph.graph import StateGraph, START, END

from agents.claim_agent import ClaimAgent, build_compact_evidence
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
    max_workers = int(os.getenv("LLM_CONCURRENCY", "3"))
    max_workers = max(1, min(len(CLAIM_TASKS), max_workers))
    results = {}
    errors = []
    timings = state.setdefault("timings", {})
    cache_enabled = os.getenv("AGENT_CACHE_ENABLED", "1") == "1"
    cache_config = {
        "model": os.getenv("LLM_MODEL", "glm4"),
        "prompt_version": "claim_extraction_v1",
        "max_evidence_items": int(
            os.getenv("LLM_MAX_EVIDENCE_ITEMS", "12")
        ),
        "max_evidence_chars": int(
            os.getenv("LLM_MAX_EVIDENCE_CHARS", "160")
        ),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1536")),
        "num_ctx": int(os.getenv("LLM_NUM_CTX", "4096")),
        "num_predict": int(
            os.getenv(
                "LLM_NUM_PREDICT",
                os.getenv("LLM_MAX_TOKENS", "1536"),
            )
        ),
    }
    cache_key = build_agent_cache_key(
        state["meeting_id"],
        state.get("evidence", []),
        cache_config,
    )

    if cache_enabled:
        cached = load_cache("agent", cache_key)
        if cached:
            for key in CLAIM_TASKS:
                state[key] = cached.get(key, [])
            state["agent_cache_hit"] = True
            timings["agent_cache_lookup"] = 0.0
            timings["agent_claim_extraction"] = 0.0
            for key in CLAIM_TASKS:
                timings[f"agent_{key}"] = cached.get(
                    "timings",
                    {},
                ).get(key, 0.0)
            return state

    state["agent_cache_hit"] = False
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
        save_cache(
            "agent",
            cache_key,
            {
                **{key: state[key] for key in CLAIM_TASKS},
                "timings": {
                    key: timings.get(f"agent_{key}", 0.0)
                    for key in CLAIM_TASKS
                },
            },
        )
    return state


def verify_all_claims(state: MeetingAgentState) -> MeetingAgentState:
    try:
        critic = CriticAgent()
        for key in ["contributions", "action_items", "risks"]:
            claim_objects = [AgentClaim(**item) for item in state.get(key, [])]
            verified = critic.verify(claim_objects, state["evidence"])
            state[key] = [
                item.model_dump() for item in verified if item.support_status == "supported"
            ]
    except Exception as exc:
        state["errors"].append(f"verify_all_claims failed: {exc}")
    return state


def rag_enrich_claims(state: MeetingAgentState) -> MeetingAgentState:
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
    except Exception as exc:
        state["errors"].append(f"rag_enrich_claims failed: {exc}")
        state["rag_supplements"] = []
    return state


def generate_report(state: MeetingAgentState) -> MeetingAgentState:
    try:
        report_result = ReportAgent().run(state)
        state["pdf_path"] = report_result["pdf_path"]
        state["claims"] = report_result["claims"]
        state["speaker_summaries"] = report_result["speaker_summaries"]
        state["evidence_links"] = report_result["evidence_links"]
    except Exception as exc:
        state["errors"].append(f"generate_report failed: {exc}")
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
    graph.add_node("rag_enrich_claims", rag_enrich_claims)
    graph.add_node("generate_report", generate_report)
    graph.add_node("deliver_report_email", deliver_report_email)

    graph.add_edge(START, "extract_all_claims")
    graph.add_edge("extract_all_claims", "verify_all_claims")
    graph.add_edge("verify_all_claims", "rag_enrich_claims")
    graph.add_edge("rag_enrich_claims", "generate_report")
    graph.add_edge("generate_report", "deliver_report_email")
    graph.add_edge("deliver_report_email", END)
    return graph.compile()
