from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "bge-small-zh-v1.5"
MODEL_NAME = str(LOCAL_MODEL_PATH)


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    if not LOCAL_MODEL_PATH.is_dir():
        raise RuntimeError(
            f"Real local BGE model was not found: {LOCAL_MODEL_PATH}. "
            "Run setup_real_bge_modelscope_day9.ps1 first."
        )

    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_texts(texts: list[str]) -> list[list[float]]:
    clean_texts = [str(text).strip() for text in texts if str(text).strip()]
    if not clean_texts:
        return []

    vectors = get_embedder().encode(
        clean_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()
