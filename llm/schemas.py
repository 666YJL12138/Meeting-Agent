from typing import Literal
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    evidence_id: str
    meeting_id: str
    speaker_id: str
    speaker_ids: list[str] = Field(default_factory=list)
    text: str
    start_ms: int
    end_ms: int
    asr_confidence: float = Field(default=0.5, ge=0, le=1)
    speaker_confidence: float | None = Field(default=None, ge=0, le=1)
    speaker_confidences: dict[str, float] = Field(default_factory=dict)
    speaker_candidates: list[dict] = Field(default_factory=list)
    overlap: bool = False
    confidence_source: str = "unknown"
    speaker_source: str = "unknown"


class AgentClaim(BaseModel):
    speaker_id: str
    claim_type: Literal["观点", "事实", "决策", "风险", "建议", "承诺", "行动项", "问题", "结论"]
    summary: str
    evidence_ids: list[str]
    quote: str
    start_ms: int
    end_ms: int
    confidence: float = Field(default=0.0, ge=0, le=1)
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)
    confidence_method: str = "evidence_weighted_v1"
    support_status: Literal["supported", "partially_supported", "unsupported"] = "supported"
    review_reason: str | None = None


class AgentResult(BaseModel):
    meeting_id: str
    contributions: list[AgentClaim] = []
    action_items: list[AgentClaim] = []
    risks: list[AgentClaim] = []
