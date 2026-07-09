import os
import logging
import numpy as np
from typing import Callable, Optional

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local").lower()
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

_encode_fn: Optional[Callable] = None


def get_provider_id() -> str:
    """Identificador único del proveedor+modelo activo. Se usa para detectar cambios de caché."""
    return f"{EMBEDDING_PROVIDER}::{EMBEDDING_MODEL}"


def _make_local_encoder() -> Callable:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from sentence_transformers import SentenceTransformer

    logging.info(f"[Embeddings] Cargando modelo local: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logging.info("[Embeddings] Modelo local listo")

    def encode(texts: list) -> np.ndarray:
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(emb, dtype=np.float32)

    return encode


def _make_openai_encoder() -> Callable:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    logging.info(f"[Embeddings] Usando OpenAI: {EMBEDDING_MODEL}")

    def encode(texts: list) -> np.ndarray:
        batch_size = 512
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
            all_vectors.extend(item.embedding for item in response.data)
        arr = np.array(all_vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    return encode


def get_encode_fn() -> Callable:
    """Retorna la función encode() del proveedor activo (lazy singleton)."""
    global _encode_fn
    if _encode_fn is None:
        if EMBEDDING_PROVIDER == "openai":
            _encode_fn = _make_openai_encoder()
        else:
            _encode_fn = _make_local_encoder()
    return _encode_fn
