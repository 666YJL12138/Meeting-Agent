from __future__ import annotations

import copy
import re
from collections import Counter, defaultdict
from collections.abc import Iterable


INTRO_PATTERNS = [
    r"(?:我|本人)(?:是|叫)\s*([一-龥]{2,4})",
    r"(?:大家好[,，]?)?\s*(?:我是|这里是)\s*([一-龥]{2,4})",
    r"([一-龥]{2,4})\s*(?:来负责|负责|主讲|汇报)",
]


def normalize_participants(value) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        raw_items: Iterable[str] = re.split(r"[,，、;\n]+", value)
    elif isinstance(value, Iterable):
        raw_items = value
    else:
        return []

    participants = []
    for item in raw_items:
        text = str(item).strip()
        if text:
            participants.append(text)

    return participants


def infer_speaker_name_map(
    transcript_spans: list[dict] | None,
    participants,
) -> dict[str, str]:
    participant_names = normalize_participants(participants)
    if not transcript_spans or not participant_names:
        return {}

    votes = defaultdict(Counter)

    for span in transcript_spans:
        speaker_id = span.get("speaker_id")
        text = str(span.get("text") or span.get("quote") or "").strip()

        if not speaker_id or not text:
            continue

        for name in detect_introduction_names(text, participant_names):
            votes[speaker_id][name] += 5

        for name in participant_names:
            if name and name in text:
                votes[speaker_id][name] += 1

    speaker_name_map = {}
    for speaker_id, counter in votes.items():
        if not counter:
            continue

        best_name, best_score = counter.most_common(1)[0]
        second_score = counter.most_common(2)[1][1] if len(counter) > 1 else 0

        if best_score >= 5 and best_score > second_score:
            speaker_name_map[speaker_id] = best_name

    return speaker_name_map


def detect_introduction_names(text: str, participant_names: list[str]) -> list[str]:
    detected = []
    for pattern in INTRO_PATTERNS:
        for match in re.findall(pattern, text):
            candidate = match.strip()
            if candidate in participant_names:
                detected.append(candidate)
    return detected


def speaker_display_name(
    speaker_id: str | None,
    speaker_name_map: dict[str, str] | None = None,
    fallback: str = "未识别说话人",
) -> str:
    speaker_id = speaker_id or ""
    if speaker_name_map and speaker_id in speaker_name_map:
        return speaker_name_map[speaker_id]
    return speaker_id or fallback


def apply_speaker_name_map(payload: dict, speaker_name_map: dict[str, str] | None) -> dict:
    speaker_name_map = speaker_name_map or {}
    result = copy.deepcopy(payload)
    result["speaker_mapping"] = dict(speaker_name_map)
    result["speaker_name_map"] = dict(speaker_name_map)

    for collection_name in (
        "transcript_spans",
        "evidence_links",
        "claims",
        "contributions",
        "action_items",
        "risks",
        "speaker_summaries",
        "speakers",
    ):
        collection = result.get(collection_name)
        if not isinstance(collection, list):
            continue

        for item in collection:
            if not isinstance(item, dict):
                continue

            speaker_id = item.get("speaker_id")
            display_name = speaker_display_name(
                speaker_id,
                speaker_name_map,
                item.get("display_name") or item.get("speaker_name") or speaker_id,
            )

            if speaker_id and speaker_id in speaker_name_map:
                item["speaker_name"] = speaker_name_map[speaker_id]
                item["display_name"] = speaker_name_map[speaker_id]
                item.setdefault("real_name", speaker_name_map[speaker_id])
            elif display_name:
                item.setdefault("speaker_name", display_name)
                item.setdefault("display_name", display_name)

            speaker_ids = item.get("speaker_ids")
            if isinstance(speaker_ids, list):
                item["speaker_names"] = [
                    speaker_display_name(
                        speaker_id,
                        speaker_name_map,
                    )
                    for speaker_id in speaker_ids
                    if speaker_id
                ]

            candidates = item.get("speaker_candidates")
            if isinstance(candidates, list):
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    candidate_id = candidate.get("speaker_id")
                    if candidate_id:
                        candidate["speaker_name"] = speaker_display_name(
                            candidate_id,
                            speaker_name_map,
                        )

    if isinstance(result.get("speaker_ids"), list):
        result["speaker_ids"] = list(
            dict.fromkeys(
                [
                    speaker_id
                    for speaker_id in result["speaker_ids"]
                    if speaker_id
                ]
                + list(speaker_name_map.keys())
            )
        )

    return result
