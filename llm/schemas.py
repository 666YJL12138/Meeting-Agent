from typing import Literal
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    evidence_id: str
    meeting_id: str
    speaker_id: str
    text: str
    start_ms: int
    end_ms: int
    asr_confidence: float = Field(default=0.5, ge=0, le=1)
    speaker_confidence: float = Field(default=0.5, ge=0, le=1)
    speaker_source: str = "unknown"


class AgentClaim(BaseModel):
    speaker_id: str
    claim_type: Literal["观点", "建议", "决策", "风险", "行动项", "问题", "结论"]
    summary: str
    evidence_ids: list[str]
    quote: str
    start_ms: int
    end_ms: int
    confidence: float = Field(default=0.0, ge=0, le=1)
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)
    confidence_method: str = "evidence_weighted_v1"
    support_status: Literal["supported", "partially_supported", "unsupported"] = "supported"


class AgentResult(BaseModel):
    meeting_id: str
    contributions: list[AgentClaim] = []
    action_items: list[AgentClaim] = []
    risks: list[AgentClaim] = []
