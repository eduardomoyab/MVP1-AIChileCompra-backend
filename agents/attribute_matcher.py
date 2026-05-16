"""
attribute_matcher.py

Normalización de valores de atributos usando FAISS + sentence-transformers.
Lee attribute_dictionary.csv y attribute_complement.csv.

API pública:
  AttributeMatcher.warm()                                     → precarga índices FAISS en disco
  AttributeMatcher.normalize(categoria, atributo, valor)      → (valor_normalizado, score, candidatos)
  AttributeMatcher.get_valid_values(categoria, atributo)      → lista de valores canónicos
  AttributeMatcher.get_dict_attributes(categoria)             → atributos del diccionario
  AttributeMatcher.get_complements(categoria, atributo, valor)→ {atributo_comp: valor_comp}
"""

import os
import logging
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import faiss
import pandas as pd
from dotenv import load_dotenv

from agents.get_embeddings import get_encode_fn, get_provider_id

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICT_PATH = os.path.join(ROOT, "diccionarios", "attribute_dictionary.csv")
COMP_PATH = os.path.join(ROOT, "diccionarios", "attribute_complement.csv")
CACHE_DIR = os.path.join(ROOT, os.getenv("FAISS_CACHE_DIR", "cache/faiss_dict").lstrip("./"))

_dict_df: Optional[pd.DataFrame] = None
_comp_df: Optional[pd.DataFrame] = None
# (categoria, atributo) -> {"index": faiss.Index, "values": [str]}
_indices: Dict[Tuple[str, str], Dict] = {}

_PROVIDER_FILE = os.path.join(CACHE_DIR, "_provider.txt")


def _load_csvs() -> None:
    global _dict_df, _comp_df
    if _dict_df is None:
        _dict_df = pd.read_csv(DICT_PATH) if os.path.exists(DICT_PATH) else pd.DataFrame(
            columns=["categoria", "atributo", "valor", "fuente"]
        )
        logging.info(f"attribute_dictionary.csv: {len(_dict_df)} filas")
    if _comp_df is None:
        _comp_df = pd.read_csv(COMP_PATH) if os.path.exists(COMP_PATH) else pd.DataFrame(
            columns=["categoria", "atributo", "valor", "atributo_complementario", "complemento"]
        )
        logging.info(f"attribute_complement.csv: {len(_comp_df)} filas")


def _cache_path(categoria: str, atributo: str) -> str:
    safe = f"{categoria}__{atributo}".replace(" ", "_").replace("/", "-")
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def _invalidate_cache_if_provider_changed() -> None:
    import shutil
    current = get_provider_id()
    if os.path.exists(_PROVIDER_FILE):
        with open(_PROVIDER_FILE, "r", encoding="utf-8") as f:
            cached = f.read().strip()
        if cached == current:
            return
        logging.info(f"[FAISS] Proveedor cambió ({cached} → {current}). Limpiando caché...")
        for fname in os.listdir(CACHE_DIR):
            fpath = os.path.join(CACHE_DIR, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
        _indices.clear()
        logging.info("[FAISS] Caché eliminado")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_PROVIDER_FILE, "w", encoding="utf-8") as f:
        f.write(current)


def _build_index(categoria: str, atributo: str) -> Optional[Dict]:
    _load_csvs()
    mask = (_dict_df["categoria"] == categoria) & (_dict_df["atributo"] == atributo)
    valores = _dict_df[mask]["valor"].dropna().unique().tolist()
    if not valores:
        return None

    encode = get_encode_fn()
    embeddings = encode(valores)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    data = {"index": index, "values": valores, "embeddings": embeddings}

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(categoria, atributo), "wb") as f:
        pickle.dump({"values": valores, "embeddings": embeddings}, f)

    return data


def _load_or_build(categoria: str, atributo: str) -> Optional[Dict]:
    key = (categoria, atributo)
    if key in _indices:
        return _indices[key]

    cache = _cache_path(categoria, atributo)
    if os.path.exists(cache):
        try:
            with open(cache, "rb") as f:
                stored = pickle.load(f)
            emb = np.array(stored["embeddings"], dtype=np.float32)
            idx = faiss.IndexFlatIP(emb.shape[1])
            idx.add(emb)
            data = {"index": idx, "values": stored["values"], "embeddings": emb}
            _indices[key] = data
            logging.info(f"[FAISS] '{categoria}::{atributo}' cargado desde disco ({len(stored['values'])} valores)")
            return data
        except Exception as e:
            logging.warning(f"[FAISS] Error cargando caché para '{categoria}::{atributo}': {e}")

    data = _build_index(categoria, atributo)
    if data:
        _indices[key] = data
        logging.info(f"[FAISS] '{categoria}::{atributo}' construido ({len(data['values'])} valores)")
    return data


class AttributeMatcher:
    def __init__(self):
        _load_csvs()

    def warm(self) -> None:
        _load_csvs()
        _invalidate_cache_if_provider_changed()
        grupos = _dict_df.groupby(["categoria", "atributo"])
        total = 0
        for (cat, atr), _ in grupos:
            try:
                if _load_or_build(cat, atr):
                    total += 1
            except Exception as e:
                logging.error(f"Error calentando '{cat}::{atr}': {e}")
        logging.info(f"warm() completado — {total} índices listos")

    def normalize(
        self,
        categoria: str,
        atributo: str,
        valor: str,
        threshold: Optional[float] = None,
        k: int = 3,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Devuelve (valor_normalizado, score, candidatos).
        Si score >= threshold devuelve el mejor match; si no, devuelve el valor original.
        """
        from dotenv import load_dotenv
        load_dotenv()
        if threshold is None:
            threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))

        data = _load_or_build(categoria, atributo)
        if data is None:
            return valor, 0.0, []

        encode = get_encode_fn()
        q_emb = encode([valor])

        actual_k = min(k, len(data["values"]))
        scores, idxs = data["index"].search(q_emb, actual_k)

        candidates = [
            (data["values"][i], float(scores[0][j]))
            for j, i in enumerate(idxs[0])
            if i != -1
        ]

        if not candidates:
            return valor, 0.0, []

        best_value, best_score = candidates[0]
        if best_score >= threshold:
            return best_value, best_score, candidates
        return valor, best_score, candidates

    def get_valid_values(self, categoria: str, atributo: str) -> List[str]:
        _load_csvs()
        mask = (_dict_df["categoria"] == categoria) & (_dict_df["atributo"] == atributo)
        return _dict_df[mask]["valor"].dropna().unique().tolist()

    def get_dict_attributes(self, categoria: str) -> List[str]:
        _load_csvs()
        mask = _dict_df["categoria"] == categoria
        return _dict_df[mask]["atributo"].unique().tolist()

    def get_complements(
        self, categoria: str, atributo: str, valor: str
    ) -> Dict[str, str]:
        _load_csvs()
        if _comp_df is None or _comp_df.empty:
            return {}
        mask = (
            (_comp_df["categoria"] == categoria)
            & (_comp_df["atributo"] == atributo)
            & (_comp_df["valor"] == valor)
        )
        rows = _comp_df[mask]
        result = {}
        for _, row in rows.iterrows():
            raw = row["complemento"]
            try:
                fval = float(raw)
                result[str(row["atributo_complementario"])] = (
                    str(int(fval)) if fval == int(fval) else str(fval)
                )
            except (ValueError, TypeError):
                result[str(row["atributo_complementario"])] = str(raw)
        return result

    def get_all_categories(self) -> List[str]:
        _load_csvs()
        return _dict_df["categoria"].unique().tolist()
