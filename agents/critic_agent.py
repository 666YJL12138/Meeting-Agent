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


def _source_speaker_ids(sources: list[dict]) -> list[str]:
    speaker_ids = []
    for item in sources:
        for speaker_id in item.get("speaker_ids") or [item.get("speaker_id")]:
            speaker_id = str(speaker_id or "").strip()
            if speaker_id and speaker_id not in speaker_ids:
                speaker_ids.append(speaker_id)
    return speaker_ids


def _build_review_reason(
    *,
    evidence_valid: bool,
    sources: list[dict],
    quote_exact: bool,
    speaker_match: bool,
    time_match: bool,
    overlap_ambiguous: bool,
) -> str | None:
    reasons = []
    if not evidence_valid:
        reasons.append("missing_evidence")
    if evidence_valid and not quote_exact:
        reasons.append("quote_mismatch")
    if evidence_valid and not speaker_match:
        reasons.append("speaker_conflict")
    if evidence_valid and not time_match:
        reasons.append("time_mismatch")
    if overlap_ambiguous:
        reasons.append("overlap_attribution_ambiguity")
    if not reasons:
        return None
    return ",".join(reasons)


def _calculate_confidence(
    claim: AgentClaim,
    sources: list[dict],
    evidence_valid: bool,
    same_speaker: bool,
) -> tuple[float, dict[str, float], dict[str, bool]]:
    """Calculate an auditable evidence support score.

    The LLM-provided confidence is deliberately ignored. The score is based
    only on source evidence and ASR/diarization signals.
    """
    source_texts = [_normalized_text(item.get("text", "")) for item in sources]
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

    speaker_match = bool(sources) and claim.speaker_id == first_source.get("speaker_id")
    time_match = (
        bool(sources)
        and claim.start_ms == expected_start
        and claim.end_ms == expected_end
    )

    asr_quality = fmean(_safe_score(item.get("asr_confidence")) for item in sources) if sources else 0.0
    diarization_quality = fmean(_safe_score(item.get("speaker_confidence")) for item in sources) if sources else 0.0
    overlap_ambiguous = any(bool(item.get("overlap")) or len(item.get("speaker_ids") or []) > 1 for item in sources)

    breakdown = {
        "evidence_binding": 1.0 if evidence_valid else 0.0,
        "quote_match": quote_score,
        "speaker_match": 1.0 if same_speaker and speaker_match else 0.0,
        "time_match": 1.0 if time_match else 0.0,
        "asr_quality": round(asr_quality, 3),
        "diarization_quality": round(diarization_quality, 3),
    }

    score = sum(breakdown[name] * weight for name, weight in WEIGHTS.items())
    details = {
        "quote_exact": quote_exact,
        "speaker_match": speaker_match,
        "time_match": time_match,
        "overlap_ambiguous": overlap_ambiguous,
    }
    return round(max(0.0, min(1.0, score)), 2), breakdown, details


class CriticAgent:
    def verify(self, claims: list[AgentClaim], evidence: list[dict]) -> list[AgentClaim]:
        evidence_map = {item["evidence_id"]: item for item in evidence}
        verified = []

        for claim in claims:
            sources = [evidence_map.get(evidence_id) for evidence_id in claim.evidence_ids]
            sources = [item for item in sources if item]
            evidence_valid = bool(claim.evidence_ids) and len(sources) == len(claim.evidence_ids)

            source_speakers = _source_speaker_ids(sources)
            same_speaker = len(set(source_speakers)) == 1 if source_speakers else False
            first_source = sources[0] if sources else {}
            expected_start = min((int(item.get("start_ms", 0)) for item in sources), default=0)
            expected_end = max((int(item.get("end_ms", 0)) for item in sources), default=0)

            quote_exact = bool(_normalized_text(claim.quote)) and any(
                _normalized_text(claim.quote) == _normalized_text(item.get("text", ""))
                for item in sources
            )
            speaker_match = bool(sources) and claim.speaker_id == first_source.get("speaker_id")
            time_match = (
                bool(sources)
                and claim.start_ms == expected_start
                and claim.end_ms == expected_end
            )
            overlap_ambiguous = any(
                bool(item.get("overlap")) or len(item.get("speaker_ids") or []) > 1
                for item in sources
            )

            confidence, confidence_breakdown, details = _calculate_confidence(
                claim=claim,
                sources=sources,
                evidence_valid=evidence_valid,
                same_speaker=same_speaker,
            )
            review_reason = _build_review_reason(
                evidence_valid=evidence_valid,
                sources=sources,
                quote_exact=details["quote_exact"],
                speaker_match=details["speaker_match"],
                time_match=details["time_match"],
                overlap_ambiguous=details["overlap_ambiguous"],
            )

            supported = (
                evidence_valid
                and same_speaker
                and details["quote_exact"]
                and speaker_match
                and time_match
                and not overlap_ambiguous
            )
            partially_supported = (
                evidence_valid
                and not supported
                and not overlap_ambiguous
                and (
                    details["quote_exact"]
                    or speaker_match
                    or time_match
                )
            )

            if supported or partially_supported:
                if sources:
                    first_source = sources[0]
                    claim.speaker_id = first_source["speaker_id"]
                    claim.quote = first_source["text"]
                    claim.start_ms = min(item["start_ms"] for item in sources)
                    claim.end_ms = max(item["end_ms"] for item in sources)

            claim.confidence = confidence
            claim.confidence_breakdown = confidence_breakdown
            claim.confidence_method = (
                "evidence_weighted_v1:"
                "0.25*evidence+0.20*quote+0.15*speaker+"
                "0.10*time+0.15*asr+0.15*diarization"
            )
            claim.review_reason = review_reason
            if supported:
                claim.support_status = "supported"
            elif partially_supported:
                claim.support_status = "partially_supported"
            else:
                claim.support_status = "unsupported"
            verified.append(claim)

        return verified
