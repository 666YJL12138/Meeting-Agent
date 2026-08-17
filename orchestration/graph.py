from langgraph.graph import StateGraph, START, END

from agents.claim_agent import ClaimAgent
from agents.critic_agent import CriticAgent
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
        agent = ClaimAgent("提取每个发言人的关键贡献", "观点")
        claims = agent.run(state["meeting_id"], state["evidence"])
        state["contributions"] = [item.model_dump() for item in claims]
    except Exception as exc:
        state["errors"].append(f"extract_contributions failed: {exc}")
        state["contributions"] = _fallback_claim(
            state,
            "观点",
            ["建议", "决策", "确认", "总结"],
            "根据原话提取的关键贡献",
        )
    return state


def extract_action_items(state: MeetingAgentState) -> MeetingAgentState:
    try:
        agent = ClaimAgent("提取会议中的行动项、负责人和任务", "行动项")
        claims = agent.run(state["meeting_id"], state["evidence"])
        state["action_items"] = [item.model_dump() for item in claims]
    except Exception as exc:
        state["errors"].append(f"extract_action_items failed: {exc}")
        state["action_items"] = _fallback_claim(
            state,
            "行动项",
            ["待办", "整理", "测试表格", "记录", "验收流程", "完成"],
            "根据原话提取的行动项",
            exclude_keywords=["风险点", "风险"],
        )
    return state


def extract_risks(state: MeetingAgentState) -> MeetingAgentState:
    try:
        agent = ClaimAgent("提取会议中提到的风险、阻塞和不确定性", "风险")
        claims = agent.run(state["meeting_id"], state["evidence"])
        state["risks"] = [item.model_dump() for item in claims]
    except Exception as exc:
        state["errors"].append(f"extract_risks failed: {exc}")
        state["risks"] = _fallback_claim(
            state,
            "风险",
            ["风险", "噪声", "错误", "人工", "问题"],
            "根据原话提取的风险点",
        )
    return state


def verify_all_claims(state: MeetingAgentState) -> MeetingAgentState:
    try:
        critic = CriticAgent()

        for key in ["contributions", "action_items", "risks"]:
            claim_objects = [AgentClaim(**item) for item in state[key]]
            verified = critic.verify(claim_objects, state["evidence"])
            state[key] = [
                item.model_dump()
                for item in verified
                if item.support_status == "supported"
            ]
    except Exception as exc:
        state["errors"].append(f"verify_all_claims failed: {exc}")

    return state


def build_meeting_graph():
    graph = StateGraph(MeetingAgentState)

    graph.add_node("extract_contributions", extract_contributions)
    graph.add_node("extract_action_items", extract_action_items)
    graph.add_node("extract_risks", extract_risks)
    graph.add_node("verify_all_claims", verify_all_claims)

    graph.add_edge(START, "extract_contributions")
    graph.add_edge("extract_contributions", "extract_action_items")
    graph.add_edge("extract_action_items", "extract_risks")
    graph.add_edge("extract_risks", "verify_all_claims")
    graph.add_edge("verify_all_claims", END)

    return graph.compile()
