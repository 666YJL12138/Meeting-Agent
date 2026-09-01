from collections import Counter


def build_quality_report(result: dict) -> dict:
    evidence_by_id = {
        item["evidence_id"]: item
        for item in result.get("evidence_links", [])
    }

    claims = result.get("claims", [])
    invalid_claims = []

    for claim in claims:
        evidence_ids = claim.get("evidence_ids", [])
        sources = [evidence_by_id.get(item_id) for item_id in evidence_ids]

        has_evidence = bool(evidence_ids) and all(sources)
        quote_matches = any(
            source and claim.get("quote", "").strip() == source.get("quote", "").strip()
            for source in sources
        )

        support_status = claim.get("support_status", "supported")
        review_reason = claim.get("review_reason")
        if not has_evidence or not quote_matches or support_status != "supported":
            invalid_claims.append({
                "claim_id": claim.get("claim_id"),
                "reason": (
                    review_reason
                    or ("missing_evidence" if not has_evidence else "quote_mismatch")
                ),
                "support_status": support_status,
            })

    total = len(claims)
    valid = total - len(invalid_claims)

    return {
        "meeting_id": result["meeting_id"],
        "claim_count": total,
        "valid_claim_count": valid,
        "invalid_claim_count": len(invalid_claims),
        "evidence_coverage": round(valid / total, 4) if total else 1.0,
        "status": "passed" if not invalid_claims else "needs_review",
        "invalid_claims": invalid_claims,
    }
