from langgraph.graph import StateGraph, START, END

from agents.claim_agent import ClaimAgent
from agents.critic_agent import CriticAgent
from agents.notification_agent import NotificationAgent
from agents.rag_agent import RAGAgent
from agents.report_agent import ReportAgent
from llm.schemas import AgentClaim
from orchestration.state import MeetingAgentState


def _fallback_claim(
    state: MeetingAgentState,
    claim_type: str,
    keywords: list[str],
    summary: str,
    exclude_keywords: list[str] | None = None,
) -> list[dict]:
    exclude_keywords = exclude_keywords or []
    for item in state["evidence"]:
        text = item.get("text", "")
        if any(keyword in text for keyword in exclude_keywords):
            continue
        if any(keyword in text for keyword in keywords):
            return [
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
            ]
    return []


def extract_contributions(state: MeetingAgentState) -> MeetingAgentState:
    try:
        agent = ClaimAgent("\u63d0\u53d6\u6bcf\u4e2a\u53d1\u8a00\u4eba\u7684\u5173\u952e\u8d21\u732e", "\u89c2\u70b9")
        claims = agent.run(state["meeting_id"], state["evidence"])
        state["contributions"] = [item.model_dump() for item in claims]
    except Exception as exc:
        state["errors"].append(f"extract_contributions failed: {exc}")
        state["contributions"] = _fallback_claim(
            state,
            "\u89c2\u70b9",
            ["\u5efa\u8bae", "\u51b3\u7b56", "\u786e\u8ba4", "\u603b\u7ed3"],
            "\u6839\u636e\u539f\u8bdd\u63d0\u53d6\u7684\u5173\u952e\u8d21\u732e",
        )
    return state


def extract_action_items(state: MeetingAgentState) -> MeetingAgentState:
    try:
        agent = ClaimAgent("\u63d0\u53d6\u4f1a\u8bae\u4e2d\u7684\u884c\u52a8\u9879\u3001\u8d1f\u8d23\u4eba\u548c\u4efb\u52a1", "\u884c\u52a8\u9879")
        claims = agent.run(state["meeting_id"], state["evidence"])
        state["action_items"] = [item.model_dump() for item in claims]
    except Exception as exc:
        state["errors"].append(f"extract_action_items failed: {exc}")
        state["action_items"] = _fallback_claim(
            state,
            "\u884c\u52a8\u9879",
            ["\u5f85\u529e", "\u6574\u7406", "\u6d4b\u8bd5\u8868\u683c", "\u8bb0\u5f55", "\u9a8c\u6536\u6d41\u7a0b", "\u5b8c\u6210"],
            "\u6839\u636e\u539f\u8bdd\u63d0\u53d6\u7684\u884c\u52a8\u9879",
            exclude_keywords=["\u98ce\u9669\u70b9", "\u98ce\u9669"],
        )
    return state


def extract_risks(state: MeetingAgentState) -> MeetingAgentState:
    try:
        agent = ClaimAgent("\u63d0\u53d6\u4f1a\u8bae\u4e2d\u63d0\u5230\u7684\u98ce\u9669\u3001\u963b\u585e\u548c\u4e0d\u786e\u5b9a\u6027", "\u98ce\u9669")
        claims = agent.run(state["meeting_id"], state["evidence"])
        state["risks"] = [item.model_dump() for item in claims]
    except Exception as exc:
        state["errors"].append(f"extract_risks failed: {exc}")
        state["risks"] = _fallback_claim(
            state,
            "\u98ce\u9669",
            ["\u98ce\u9669", "\u566a\u58f0", "\u9519\u8bef", "\u4eba\u5de5", "\u95ee\u9898"],
            "\u6839\u636e\u539f\u8bdd\u63d0\u53d6\u7684\u98ce\u9669\u70b9",
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
    graph.add_node("extract_contributions", extract_contributions)
    graph.add_node("extract_action_items", extract_action_items)
    graph.add_node("extract_risks", extract_risks)
    graph.add_node("verify_all_claims", verify_all_claims)
    graph.add_node("rag_enrich_claims", rag_enrich_claims)
    graph.add_node("generate_report", generate_report)
    graph.add_node("deliver_report_email", deliver_report_email)

    graph.add_edge(START, "extract_contributions")
    graph.add_edge("extract_contributions", "extract_action_items")
    graph.add_edge("extract_action_items", "extract_risks")
    graph.add_edge("extract_risks", "verify_all_claims")
    graph.add_edge("verify_all_claims", "rag_enrich_claims")
    graph.add_edge("rag_enrich_claims", "generate_report")
    graph.add_edge("generate_report", "deliver_report_email")
    graph.add_edge("deliver_report_email", END)
    return graph.compile()