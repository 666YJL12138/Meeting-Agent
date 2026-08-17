import json
import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from llm.gateway import LLMGateway
from llm.schemas import AgentClaim


def _repair_mojibake(value):
    if isinstance(value, str):
        try:
            repaired = value.encode("latin1").decode("utf-8")
        except UnicodeError:
            return value
        return repaired if repaired else value
    if isinstance(value, list):
        return [_repair_mojibake(item) for item in value]
    if isinstance(value, dict):
        return {key: _repair_mojibake(item) for key, item in value.items()}
    return value


class ClaimAgent:
    def __init__(self, task_type: str, claim_type: str):
        self.task_type = task_type
        self.claim_type = claim_type
        self.llm = LLMGateway()
        self.max_evidence_items = int(os.getenv("LLM_MAX_EVIDENCE_ITEMS", "8"))
        self.max_evidence_chars = int(os.getenv("LLM_MAX_EVIDENCE_CHARS", "180"))
        prompt_path = Path("prompts/claim_extraction.yaml")
        self.prompt = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))

    def run(self, meeting_id: str, evidence: list[dict]) -> list[AgentClaim]:
        compact_evidence = []
        for item in evidence[: self.max_evidence_items]:
            compact_item = dict(item)
            compact_item["text"] = compact_item.get("text", "")[: self.max_evidence_chars]
            compact_evidence.append(compact_item)

        user_prompt = self.prompt["user_template"].format(
            meeting_id=meeting_id,
            task_type=self.task_type,
            claim_type=self.claim_type,
            evidence_json=json.dumps(compact_evidence, ensure_ascii=False, indent=2),
        )
        result = self.llm.chat_json(self.prompt["system"], user_prompt)
        claims = []
        for item in result.get("claims", []):
            try:
                claims.append(AgentClaim(**_repair_mojibake(item)))
            except ValidationError:
                continue
        return claims
