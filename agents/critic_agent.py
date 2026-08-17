from llm.schemas import AgentClaim


class CriticAgent:
    def verify(self, claims: list[AgentClaim], evidence: list[dict]) -> list[AgentClaim]:
        evidence_map = {item["evidence_id"]: item for item in evidence}
        verified = []

        for claim in claims:
            valid = bool(claim.evidence_ids)

            for evidence_id in claim.evidence_ids:
                item = evidence_map.get(evidence_id)
                if not item:
                    valid = False
                    break
                if claim.speaker_id != item["speaker_id"]:
                    valid = False
                    break
                if claim.quote not in item["text"]:
                    valid = False
                    break
                if claim.start_ms < item["start_ms"] or claim.end_ms > item["end_ms"]:
                    valid = False
                    break

            claim.support_status = "supported" if valid else "unsupported"
            verified.append(claim)

        return verified
