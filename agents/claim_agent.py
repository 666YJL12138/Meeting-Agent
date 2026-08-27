import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import ValidationError

from llm.gateway import LLMGateway
from llm.schemas import AgentClaim


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_prompt() -> dict:
    prompt_path = Path("prompts/claim_extraction.yaml")
    return yaml.safe_load(prompt_path.read_text(encoding="utf-8"))


def build_compact_evidence(
    evidence: list[dict],
    *,
    max_items: int | None = None,
    max_chars: int | None = None,
) -> str:
    """Prepare one bounded, reusable evidence JSON payload for all agents."""
    max_items = max_items or int(os.getenv("LLM_MAX_EVIDENCE_ITEMS", "12"))
    max_chars = max_chars or int(os.getenv("LLM_MAX_EVIDENCE_CHARS", "160"))

    compact_evidence = []
    for item in evidence[:max_items]:
        compact_item = dict(item)
        compact_item["text"] = compact_item.get("text", "")[:max_chars]
        compact_evidence.append(compact_item)

    return json.dumps(
        compact_evidence,
        ensure_ascii=False,
        separators=(",", ":"),
    )


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
        self.prompt = _load_prompt()

    def run(
        self,
        meeting_id: str,
        evidence: list[dict],
        compact_evidence_json: str | None = None,
    ) -> list[AgentClaim]:
        compact_evidence_json = compact_evidence_json or build_compact_evidence(
            evidence,
            max_items=self.max_evidence_items,
            max_chars=self.max_evidence_chars,
        )

        user_prompt = self.prompt["user_template"].format(
            meeting_id=meeting_id,
            task_type=self.task_type,
            claim_type=self.claim_type,
            evidence_json=compact_evidence_json,
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
