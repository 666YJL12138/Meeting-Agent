import os
from concurrent.futures import ThreadPoolExecutor

from services.vector_store import search_meeting


class RAGAgent:
    def __init__(
        self,
        top_k: int = 3,
        max_workers: int | None = None,
    ):
        self.top_k = top_k
        self.max_workers = max(
            1,
            int(
                max_workers
                or os.getenv("RAG_MAX_WORKERS", "4")
            ),
        )

    def enrich_claims(
        self,
        meeting_id: str,
        claims: list[dict],
        evidence: list[dict],
    ) -> list[dict]:
        if not claims:
            return []

        def enrich_one(claim: dict) -> dict:
            query = self._build_query(claim)
            hits = self._search_with_fallback(
                meeting_id=meeting_id,
                query=query,
                evidence=evidence,
            )

            return {
                "claim_summary": claim.get("summary", ""),
                "claim_type": claim.get("claim_type", ""),
                "speaker_id": claim.get("speaker_id", ""),
                "query": query,
                "hits": hits,
            }

        # executor.map preserves input order while independent searches run
        # concurrently.
        worker_count = min(self.max_workers, len(claims))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(enrich_one, claims))

    def _build_query(self, claim: dict) -> str:
        return " ".join([
            str(claim.get("summary", "")),
            str(claim.get("quote", "")),
            str(claim.get("claim_type", "")),
        ]).strip()

    def _search_with_fallback(
        self,
        meeting_id: str,
        query: str,
        evidence: list[dict],
    ) -> list[dict]:
        try:
            vector_hits = search_meeting(
                meeting_id=meeting_id,
                query=query,
                limit=self.top_k,
            )
            if vector_hits:
                return [
                    {
                        "source": "qdrant",
                        "score": item.get("score"),
                        "payload": item.get("payload", {}),
                    }
                    for item in vector_hits
                ]
        except Exception:
            pass

        return self._local_keyword_search(query=query, evidence=evidence)

    def _local_keyword_search(
        self,
        query: str,
        evidence: list[dict],
    ) -> list[dict]:
        tokens = [char for char in query if char.strip()]
        hits = []

        for item in evidence:
            text = item.get("text", "")
            score = sum(1 for token in tokens if token in text)

            if score > 0:
                hits.append({
                    "source": "local",
                    "score": score,
                    "payload": {
                        "evidence_id": item.get("evidence_id"),
                        "speaker_id": item.get("speaker_id"),
                        "text": text,
                        "start_ms": item.get("start_ms"),
                        "end_ms": item.get("end_ms"),
                    },
                })

        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[: self.top_k]
