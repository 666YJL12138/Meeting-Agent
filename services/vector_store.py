import os
import hashlib
import re
from uuid import NAMESPACE_URL, uuid4, uuid5

from qdrant_client import QdrantClient, models

from services.embeddings import embed_texts


QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "meeting_evidence",
)


def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient) -> None:
    collections = client.get_collections().collections
    names = {item.name for item in collections}

    if COLLECTION_NAME in names:
        return

    vector_size = len(embed_texts(["meeting evidence"])[0])
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def _records(
    meeting_id: str,
    transcript_spans: list[dict],
    claims: list[dict],
) -> list[dict]:
    records = []

    for span in transcript_spans:
        text = str(span.get("text", "")).strip()
        if text:
            source_id = span.get("span_id") or text
            records.append({
                "kind": "transcript",
                "text": text,
                "payload": {
                    "meeting_id": meeting_id,
                    "evidence_hash": evidence_hash(
                        meeting_id,
                        "transcript",
                        source_id,
                        span,
                    ),
                    **span,
                },
            })

    for claim in claims:
        text = str(claim.get("statement", "")).strip()
        if text:
            source_id = claim.get("claim_id") or text
            records.append({
                "kind": "claim",
                "text": text,
                "payload": {
                    "meeting_id": meeting_id,
                    "evidence_hash": evidence_hash(
                        meeting_id,
                        "claim",
                        source_id,
                        claim,
                    ),
                    **claim,
                },
            })

    return records


def evidence_hash(
    meeting_id: str,
    kind: str,
    source_id: str,
    payload: dict,
) -> str:
    raw = "|".join([
        str(meeting_id),
        str(kind),
        str(source_id),
        str(payload.get("speaker_id", "")),
        str(payload.get("start_ms", "")),
        str(payload.get("end_ms", "")),
        str(payload.get("text") or payload.get("statement") or ""),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _query_tokens(query: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[\w\u4e00-\u9fff]+", str(query))
        if token.strip()
    }


def rerank_hits(
    query: str,
    hits: list[dict],
    *,
    limit: int,
) -> list[dict]:
    """Apply a deterministic lexical rerank on top of vector similarity."""
    query_text = re.sub(r"\s+", "", str(query))
    query_tokens = _query_tokens(query)
    ranked = []
    for item in hits:
        payload = item.get("payload") or {}
        text = str(payload.get("text") or payload.get("statement") or "")
        text_compact = re.sub(r"\s+", "", text)
        text_tokens = _query_tokens(text)
        if query_text and query_text in text_compact:
            lexical_score = 1.0
        elif text_compact and text_compact in query_text:
            lexical_score = round(len(text_compact) / max(len(query_text), 1), 4)
        else:
            lexical_score = (
                len(query_tokens & text_tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )
        vector_score = float(item.get("score") or 0.0)
        rerank_score = round(0.55 * vector_score + 0.45 * lexical_score, 4)
        ranked.append({
            **item,
            "rerank_score": rerank_score,
        })

    ranked.sort(
        key=lambda item: (
            -float(item.get("rerank_score") or 0.0),
            -float(item.get("score") or 0.0),
            str((item.get("payload") or {}).get("evidence_id", "")),
        )
    )
    return ranked[:limit]


def index_meeting(
    meeting_id: str,
    transcript_spans: list[dict],
    claims: list[dict],
) -> int:
    records = _records(meeting_id, transcript_spans, claims)
    if not records:
        return 0

    vectors = embed_texts([item["text"] for item in records])
    client = get_qdrant_client()
    ensure_collection(client)

    points = []
    for item, vector in zip(records, vectors):
        source_id = (
            item["payload"].get("span_id")
            or item["payload"].get("claim_id")
            or uuid4().hex
        )
        point_id = uuid5(
            NAMESPACE_URL,
            f"{meeting_id}:{item['kind']}:{source_id}",
        ).hex

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "kind": item["kind"],
                    "text": item["text"],
                    **item["payload"],
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )
    return len(points)


def search_meeting(
    meeting_id: str,
    query: str,
    limit: int = 5,
    speaker_id: str | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    topic: str | None = None,
) -> list[dict]:
    query_vector = embed_texts([query])[0]
    client = get_qdrant_client()
    ensure_collection(client)

    conditions = [
        models.FieldCondition(
            key="meeting_id",
            match=models.MatchValue(value=meeting_id),
        )
    ]
    if speaker_id:
        conditions.append(
            models.FieldCondition(
                key="speaker_id",
                match=models.MatchValue(value=speaker_id),
            )
        )
    if start_ms is not None:
        conditions.append(
            models.FieldCondition(
                key="end_ms",
                range=models.Range(gte=start_ms),
            )
        )
    if end_ms is not None:
        conditions.append(
            models.FieldCondition(
                key="start_ms",
                range=models.Range(lte=end_ms),
            )
        )
    if topic:
        conditions.append(
            models.FieldCondition(
                key="topic",
                match=models.MatchValue(value=topic),
            )
        )

    meeting_filter = models.Filter(must=conditions)

    if hasattr(client, "query_points"):
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=meeting_filter,
            limit=limit,
            with_payload=True,
        )
        hits = result.points
    else:
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=meeting_filter,
            limit=limit,
            with_payload=True,
        )

    formatted = [
        {
            "score": round(float(hit.score), 4),
            "payload": hit.payload or {},
        }
        for hit in hits
    ]
    return rerank_hits(query, formatted, limit=limit)
