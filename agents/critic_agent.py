from statistics import fmean

from llm.schemas import AgentClaim


WEIGHTS = {
    "evidence_binding": 0.25,
    "quote_match": 0.20,
    "speaker_match": 0.15,
    "time_match": 0.10,
    "asr_quality": 0.15,
    "diarization_quality": 0.15,
}


def _safe_score(value, default: float = 0.5) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(1.0, score))


def _normalized_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _calculate_confidence(
    claim: AgentClaim,
    sources: list[dict],
    evidence_valid: bool,
    same_speaker: bool,
) -> tuple[float, dict[str, float]]:
    """Calculate an auditable evidence support score.

    The LLM-provided confidence is deliberately ignored. The score is based
    only on source evidence and ASR/diarization signals.
    """
    source_texts = [
        _normalized_text(item.get("text", ""))
        for item in sources
    ]
    draft_quote = _normalized_text(claim.quote)

    quote_exact = bool(draft_quote) and any(
        draft_quote == source_text
        for source_text in source_texts
    )
    quote_partial = bool(draft_quote) and any(
        draft_quote in source_text or source_text in draft_quote
        for source_text in source_texts
    )

    quote_score = 1.0 if quote_exact else 0.5 if quote_partial else 0.0

    first_source = sources[0] if sources else {}
    expected_start = min(
        (int(item.get("start_ms", 0)) for item in sources),
        default=0,
    )
    expected_end = max(
        (int(item.get("end_ms", 0)) for item in sources),
        default=0,
    )

    speaker_match = bool(sources) and claim.speaker_id == first_source.get(
        "speaker_id"
    )
    time_match = (
        bool(sources)
        and claim.start_ms == expected_start
        and claim.end_ms == expected_end
    )

    asr_quality = fmean(
        _safe_score(item.get("asr_confidence"))
        for item in sources
    ) if sources else 0.0
    diarization_quality = fmean(
        _safe_score(item.get("speaker_confidence"))
        for item in sources
    ) if sources else 0.0

    breakdown = {
        "evidence_binding": 1.0 if evidence_valid else 0.0,
        "quote_match": quote_score,
        "speaker_match": 1.0 if same_speaker and speaker_match else 0.0,
        "time_match": 1.0 if time_match else 0.0,
        "asr_quality": round(asr_quality, 3),
        "diarization_quality": round(diarization_quality, 3),
    }

    score = sum(
        breakdown[name] * weight
        for name, weight in WEIGHTS.items()
    )
    return round(max(0.0, min(1.0, score)), 2), breakdown


class CriticAgent:
    def verify(self, claims: list[AgentClaim], evidence: list[dict]) -> list[AgentClaim]:
        evidence_map = {item["evidence_id"]: item for item in evidence}
        verified = []

        for claim in claims:
            draft_speaker_id = claim.speaker_id
            sources = [
                evidence_map.get(evidence_id)
                for evidence_id in claim.evidence_ids
            ]
            evidence_valid = bool(sources) and all(sources)
            same_speaker = False

            if evidence_valid:
                source_speakers = {
                    item["speaker_id"]
                    for item in sources
                }
                same_speaker = len(source_speakers) == 1

            valid = evidence_valid and same_speaker
            confidence, confidence_breakdown = _calculate_confidence(
                claim=claim,
                sources=[item for item in sources if item],
                evidence_valid=evidence_valid,
                same_speaker=same_speaker,
            )

            if valid:
                # The evidence ID is the trust anchor. The LLM may normalize
                # ASR typos in its draft quote, so restore the exact source
                # fields before the final quality gate.
                first_source = sources[0]
                claim.speaker_id = first_source["speaker_id"]
                claim.quote = first_source["text"]
                claim.start_ms = min(
                    item["start_ms"] for item in sources
                )
                claim.end_ms = max(
                    item["end_ms"] for item in sources
                )

            claim.confidence = confidence
            claim.confidence_breakdown = confidence_breakdown
            claim.confidence_method = (
                "evidence_weighted_v1:"
                "0.25*evidence+0.20*quote+0.15*speaker+"
                "0.10*time+0.15*asr+0.15*diarization"
            )
            claim.support_status = "supported" if valid else "unsupported"
            verified.append(claim)

        return verified
