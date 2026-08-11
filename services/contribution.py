from collections import defaultdict


CLAIM_KEYWORDS = {
    "decision": [
        "决定", "确定", "通过", "结论", "最终", "统一", "确认",
    ],
    "action_item": [
        "需要", "负责", "完成", "提交", "跟进", "推进", "落地", "下周", "明天",
    ],
    "risk": [
        "风险", "问题", "担心", "可能", "不确定", "阻塞", "延迟", "困难",
    ],
    "suggestion": [
        "建议", "可以", "最好", "应该", "考虑", "优化", "改进",
    ],
    "commitment": [
        "我会", "我们会", "我来", "我们负责", "承诺", "保证",
    ],
    "fact": [
        "目前", "已经", "完成了", "结果", "数据", "显示", "现在",
    ],
}


def extract_contributions(
    meeting_id: str,
    transcript_spans: list[dict],
    evidence_links: list[dict],
) -> list[dict]:
    evidence_by_span = {
        item["span_id"]: item
        for item in evidence_links
    }

    claims = []

    for span in transcript_spans:
        text = normalize_text(span.get("text", ""))

        if not text:
            continue

        claim_type = classify_claim(text)

        if claim_type == "ignore":
            continue

        evidence = evidence_by_span.get(span["span_id"])

        if not evidence:
            continue

        claim_id = f"{meeting_id}_claim_{len(claims) + 1:04d}"

        claim = {
            "claim_id": claim_id,
            "meeting_id": meeting_id,
            "speaker_id": span.get("speaker_id", "speaker_unknown"),
            "claim_type": claim_type,
            "statement": build_statement(text, claim_type),
            "evidence_ids": [evidence["evidence_id"]],
            "quote": evidence["quote"],
            "start_ms": evidence["start_ms"],
            "end_ms": evidence["end_ms"],
            "confidence": estimate_claim_confidence(span),
            "review_status": build_review_status(span),
        }

        claims.append(claim)

    return validate_claims(claims)


def classify_claim(text: str) -> str:
    for claim_type, keywords in CLAIM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return claim_type

    if len(text) >= 18:
        return "viewpoint"

    return "ignore"


def build_statement(text: str, claim_type: str) -> str:
    text = text.strip()

    if claim_type == "action_item":
        return text

    if claim_type == "decision":
        return text

    if claim_type == "risk":
        return text

    if claim_type == "suggestion":
        return text

    if claim_type == "commitment":
        return text

    return text


def estimate_claim_confidence(span: dict) -> float:
    asr_confidence = float(span.get("asr_confidence", 0.5))
    speaker_confidence = float(span.get("speaker_confidence", 0.5))

    confidence = asr_confidence * 0.6 + speaker_confidence * 0.4

    return round(max(0.0, min(1.0, confidence)), 2)


def build_review_status(span: dict) -> str:
    if span.get("speaker_id") == "speaker_unknown":
        return "needs_review"

    if float(span.get("speaker_confidence", 0.0)) < 0.5:
        return "needs_review"

    return "auto"


def validate_claims(claims: list[dict]) -> list[dict]:
    valid = []

    for claim in claims:
        if not claim.get("evidence_ids"):
            continue

        if not claim.get("quote"):
            continue

        if not claim.get("speaker_id"):
            continue

        valid.append(claim)

    return valid


def normalize_text(text: str) -> str:
    return (
        text.replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


def build_speaker_summaries(
    claims: list[dict],
    speakers: list[dict],
) -> list[dict]:
    claims_by_speaker = defaultdict(list)

    for claim in claims:
        claims_by_speaker[claim["speaker_id"]].append(claim)

    speaker_name_map = {
        speaker["speaker_id"]: speaker.get("display_name") or speaker["speaker_id"]
        for speaker in speakers
    }

    summaries = []

    for speaker_id, speaker_claims in claims_by_speaker.items():
        claim_types = defaultdict(int)

        for claim in speaker_claims:
            claim_types[claim["claim_type"]] += 1

        summaries.append({
            "speaker_id": speaker_id,
            "display_name": speaker_name_map.get(speaker_id, speaker_id),
            "claim_count": len(speaker_claims),
            "claim_type_counts": dict(claim_types),
            "key_points": [
                {
                    "claim_id": claim["claim_id"],
                    "claim_type": claim["claim_type"],
                    "statement": claim["statement"],
                    "evidence_ids": claim["evidence_ids"],
                    "confidence": claim["confidence"],
                    "review_status": claim["review_status"],
                }
                for claim in speaker_claims[:5]
            ],
        })

    return summaries
