import os
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
            records.append({
                "kind": "transcript",
                "text": text,
                "payload": {
                    "meeting_id": meeting_id,
                    **span,
                },
            })

    for claim in claims:
        text = str(claim.get("statement", "")).strip()
        if text:
            records.append({
                "kind": "claim",
                "text": text,
                "payload": {
                    "meeting_id": meeting_id,
                    **claim,
                },
            })

    return records


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
) -> list[dict]:
    query_vector = embed_texts([query])[0]
    client = get_qdrant_client()
    ensure_collection(client)

    meeting_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="meeting_id",
                match=models.MatchValue(value=meeting_id),
            )
        ]
    )

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

    return [
        {
            "score": round(float(hit.score), 4),
            "payload": hit.payload or {},
        }
        for hit in hits
    ]
