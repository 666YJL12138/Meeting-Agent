import json
import logging
import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from llm.gateway import LLMGateway
from llm.schemas import AgentClaim


logger = logging.getLogger(__name__)


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
        self.max_evidence_items = int(os.getenv("LLM_MAX_EVIDENCE_ITEMS", "12"))
        self.max_evidence_chars = int(os.getenv("LLM_MAX_EVIDENCE_CHARS", "160"))
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
        raw_claims = result.get("claims", [])
        if not isinstance(raw_claims, list):
            raise ValueError("LLM claims field must be a list")

        claims = []
        for index, item in enumerate(raw_claims):
            if not isinstance(item, dict):
                logger.warning(
                    "Skipping non-object claim: task=%s index=%s value=%r",
                    self.task_type,
                    index,
                    item,
                )
                continue

            try:
                claims.append(AgentClaim(**_repair_mojibake(item)))
            except ValidationError as exc:
                logger.warning(
                    "Skipping invalid claim: task=%s index=%s errors=%s item=%s",
                    self.task_type,
                    index,
                    exc.errors(),
                    item,
                )
                continue

        return claims
