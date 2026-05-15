import os
import logging
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logging.info(f"Cargando modelo de embeddings: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)
        logging.info("Modelo de embeddings listo")
    return _model
