from typing import Literal
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    evidence_id: str
    meeting_id: str
    speaker_id: str
    text: str
    start_ms: int
    end_ms: int


class AgentClaim(BaseModel):
    speaker_id: str
    claim_type: Literal["观点", "建议", "决策", "风险", "行动项", "问题", "结论"]
    summary: str
    evidence_ids: list[str]
    quote: str
    start_ms: int
    end_ms: int
    confidence: float = Field(ge=0, le=1)
    support_status: Literal["supported", "partially_supported", "unsupported"] = "supported"


class AgentResult(BaseModel):
    meeting_id: str
    contributions: list[AgentClaim] = []
    action_items: list[AgentClaim] = []
    risks: list[AgentClaim] = []
