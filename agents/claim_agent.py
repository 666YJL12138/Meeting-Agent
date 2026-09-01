import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import ValidationError

from llm.gateway import LLMGateway
from llm.schemas import AgentClaim


logger = logging.getLogger(__name__)
_WHITESPACE_RE = re.compile(r"\s+")


@lru_cache(maxsize=4)
def _load_prompt(prompt_filename: str) -> dict:
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / prompt_filename
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


TASK_EVIDENCE_PROFILES = {
    "contributions": {
        "positive": (
            "建议",
            "决策",
            "确认",
            "总结",
            "方案",
            "结论",
            "共识",
            "方向",
            "认为",
            "定下来",
            "重点",
        ),
        "negative": (
            "待办",
            "负责",
            "分工",
            "跟进",
            "安排",
            "提交",
            "完成",
            "验收",
            "风险",
            "问题",
            "阻塞",
            "延期",
        ),
    },
    "action_items": {
        "positive": (
            "待办",
            "负责",
            "分工",
            "跟进",
            "安排",
            "提交",
            "完成",
            "同步",
            "验收",
            "记录",
            "尽快",
            "本周",
            "下周",
            "今天",
            "明天",
        ),
        "negative": (
            "建议",
            "决策",
            "总结",
            "共识",
            "结论",
            "方向",
            "风险",
            "问题",
            "阻塞",
            "延期",
        ),
    },
    "risks": {
        "positive": (
            "风险",
            "问题",
            "阻塞",
            "延期",
            "异常",
            "不确定",
            "错误",
            "失败",
            "卡住",
            "依赖",
            "超时",
            "可能",
        ),
        "negative": (
            "建议",
            "决策",
            "总结",
            "共识",
            "行动项",
            "负责",
            "提交",
            "完成",
        ),
    },
}


def _normalize_text(value: object) -> str:
    return _WHITESPACE_RE.sub("", str(value or "")).lower()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_task_evidence(
    item: dict,
    task_key: str,
    index: int,
) -> tuple[int, int, float, int, int] | None:
    text = _normalize_text(item.get("text", ""))
    if not text:
        return None

    profile = TASK_EVIDENCE_PROFILES.get(task_key, {})
    positive_keywords = profile.get("positive", ())
    negative_keywords = profile.get("negative", ())

    positive_hits = sum(1 for keyword in positive_keywords if keyword in text)
    negative_hits = sum(1 for keyword in negative_keywords if keyword in text)

    confidence = (
        _safe_float(item.get("asr_confidence"), 0.5) * 0.6
        + _safe_float(item.get("speaker_confidence"), 0.5) * 0.4
    )

    score = positive_hits * 6
    score -= negative_hits * 3
    score += int(round(confidence * 4))

    if task_key == "action_items":
        if any(token in text for token in ("负责", "跟进", "提交", "完成", "安排", "验收", "同步")):
            score += 2
        if any(token in text for token in ("尽快", "本周", "下周", "今天", "明天", "回头", "稍后")):
            score += 1

    if task_key == "risks" and any(
        token in text
        for token in ("可能", "如果", "需要", "还没", "无法", "难以")
    ):
        score += 1

    return score, positive_hits, confidence, len(text), index


def rank_evidence_for_task(
    evidence: list[dict],
    task_key: str,
    *,
    top_k: int | None = None,
) -> list[dict]:
    top_k = top_k or int(os.getenv("LLM_BATCH_TOP_K", "6"))
    scored = []
    for index, item in enumerate(evidence):
        scored_item = _score_task_evidence(item, task_key, index)
        if scored_item is None:
            continue
        score, positive_hits, confidence, text_len, evidence_index = scored_item
        scored.append((score, positive_hits, confidence, text_len, evidence_index, item))

    if not scored:
        return evidence[:top_k]

    positive_scored = [entry for entry in scored if entry[1] > 0]
    if positive_scored:
        scored = positive_scored
    else:
        scored.sort(key=lambda entry: (-entry[2], entry[3], entry[4]))

    scored.sort(key=lambda entry: (-entry[0], -entry[1], -entry[2], entry[3], entry[4]))

    selected = []
    seen_texts: set[str] = set()
    for _, _, _, _, _, item in scored:
        normalized_text = _normalize_text(item.get("text", ""))
        if normalized_text in seen_texts:
            continue
        selected.append(item)
        seen_texts.add(normalized_text)
        if len(selected) >= top_k:
            break

    return selected


def build_task_compact_evidence(
    evidence: list[dict],
    task_key: str,
    *,
    top_k: int | None = None,
    max_items: int | None = None,
    max_chars: int | None = None,
) -> str:
    selected_evidence = rank_evidence_for_task(
        evidence,
        task_key,
        top_k=top_k,
    )
    return build_compact_evidence(
        selected_evidence,
        max_items=max_items,
        max_chars=max_chars,
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
        self.prompt = _load_prompt("claim_extraction.yaml")

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


def _parse_claim_list(
    raw_claims: object,
    *,
    task_type: str,
    fallback_claim_type: str,
) -> list[AgentClaim]:
    if not isinstance(raw_claims, list):
        raise ValueError(f"{task_type} claims field must be a list")

    claims = []
    for index, item in enumerate(raw_claims):
        if not isinstance(item, dict):
            logger.warning(
                "Skipping non-object claim: task=%s index=%s value=%r",
                task_type,
                index,
                item,
            )
            continue

        payload = dict(_repair_mojibake(item))
        payload.setdefault("claim_type", fallback_claim_type)
        payload.setdefault("support_status", "supported")

        try:
            claims.append(AgentClaim(**payload))
        except ValidationError as exc:
            logger.warning(
                "Skipping invalid claim: task=%s index=%s errors=%s item=%s",
                task_type,
                index,
                exc.errors(),
                item,
            )
            continue

    return claims


class BatchClaimAgent:
    def __init__(self):
        self.llm = LLMGateway()
        self.max_evidence_items = int(os.getenv("LLM_MAX_EVIDENCE_ITEMS", "12"))
        self.max_evidence_chars = int(os.getenv("LLM_MAX_EVIDENCE_CHARS", "160"))
        self.top_k = int(os.getenv("LLM_BATCH_TOP_K", "6"))
        self.prompt = _load_prompt("claim_extraction_batch.yaml")

    def run(
        self,
        meeting_id: str,
        evidence: list[dict],
    ) -> dict[str, list[AgentClaim]]:
        user_prompt = self.prompt["user_template"].format(
            meeting_id=meeting_id,
            contributions_evidence_json=build_task_compact_evidence(
                evidence,
                "contributions",
                top_k=self.top_k,
                max_items=self.max_evidence_items,
                max_chars=self.max_evidence_chars,
            ),
            action_items_evidence_json=build_task_compact_evidence(
                evidence,
                "action_items",
                top_k=self.top_k,
                max_items=self.max_evidence_items,
                max_chars=self.max_evidence_chars,
            ),
            risks_evidence_json=build_task_compact_evidence(
                evidence,
                "risks",
                top_k=self.top_k,
                max_items=self.max_evidence_items,
                max_chars=self.max_evidence_chars,
            ),
        )
        result = self.llm.chat_json(self.prompt["system"], user_prompt)

        return {
            "contributions": _parse_claim_list(
                result.get("contributions", []),
                task_type="contributions",
                fallback_claim_type="观点",
            ),
            "action_items": _parse_claim_list(
                result.get("action_items", []),
                task_type="action_items",
                fallback_claim_type="行动项",
            ),
            "risks": _parse_claim_list(
                result.get("risks", []),
                task_type="risks",
                fallback_claim_type="风险",
            ),
        }
