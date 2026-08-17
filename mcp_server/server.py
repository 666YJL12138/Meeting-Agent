import sys
from pathlib import Path

from mcp.server import MCPServer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.rag_tools import load_agent_result, search_meeting_evidence
from tools.email_tools import send_report_email_tool

mcp = MCPServer("Meeting-Agent-MCP")


@mcp.tool()
def search_evidence(meeting_id: str, query: str, top_k: int = 5) -> list[dict]:
    """检索会议中的原话证据。"""
    return search_meeting_evidence(meeting_id, query, top_k)


@mcp.tool()
def send_report_email(
    meeting_id: str,
    pdf_path: str,
    to_email: str,
    subject: str = "",
    body: str = "",
) -> dict:
    """把会议 PDF 作为附件发送到指定邮箱。"""
    return send_report_email_tool(
        meeting_id=meeting_id,
        pdf_path=pdf_path,
        to_email=to_email,
        subject=subject or None,
        body=body or None,
    )


@mcp.resource("meeting://{meeting_id}/agent-result")
def meeting_agent_result(meeting_id: str) -> str:
    """读取会议 Agent 结构化结果。"""
    data = load_agent_result(meeting_id)
    return str(data)


@mcp.prompt()
def evidence_verification_prompt() -> str:
    return (
        "你是事实核查 Agent。只能根据 evidence 原话判断结论是否成立。"
        "如果 quote 不存在于原文，必须输出 unsupported。"
    )


if __name__ == "__main__":
    mcp.run()
