"""
gama_service.py

Clasifica líneas de procesador (linea_procesador) en 4 gamas de calidad
(Básica/Media/Alta/Premium), sin distinción de marca -- Intel Core i5, AMD
Ryzen 5 y Qualcomm Snapdragon X Plus son todos "Media". Se usa para
expandir un procesador puntual a su grupo de equivalentes cross-marca,
para que la búsqueda/recomendación no favorezca sistemáticamente una sola
marca cuando el usuario no pide un procesador específico.

Fuente de los datos: diccionarios/gama_procesador.json (config estática,
sin DB ni red -- se carga una vez en memoria).
"""

import json
import logging
import os
from typing import Dict, List, Optional

_DICT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "diccionarios", "gama_procesador.json")

_data: Optional[Dict] = None


def _load() -> Dict:
    global _data
    if _data is None:
        try:
            with open(_DICT_PATH, "r", encoding="utf-8") as f:
                _data = json.load(f)
        except Exception as e:
            logging.warning(f"[gama] No se pudo cargar {_DICT_PATH}: {e}")
            _data = {"linea_to_gama": {}, "gama_order": []}
    return _data


def get_gama(linea_procesador: Optional[str]) -> Optional[str]:
    """Gama (Básica/Media/Alta/Premium) de una línea de procesador, o None si no está mapeada."""
    if not linea_procesador:
        return None
    entry = _load()["linea_to_gama"].get(linea_procesador)
    return entry["gama"] if entry else None


def get_passmark_median(linea_procesador: Optional[str]) -> Optional[float]:
    if not linea_procesador:
        return None
    entry = _load()["linea_to_gama"].get(linea_procesador)
    return entry.get("passmark_median") if entry else None


def get_lineas_by_gama(gama: str) -> List[str]:
    """Todas las líneas (cualquier marca) que pertenecen a una gama -- el grupo de equivalentes."""
    linea_to_gama = _load()["linea_to_gama"]
    return sorted(linea for linea, entry in linea_to_gama.items() if entry["gama"] == gama)


def expand_to_group(lineas: List[str]) -> List[str]:
    """Dada una línea (o lista de líneas), devuelve el grupo completo cross-marca de
    su(s) gama(s) -- ej. ["Intel Core i5"] -> ["Intel Core i5", "AMD Ryzen 5",
    "AMD Ryzen AI 5", ...] (todas las líneas "Media"). Preserva las líneas
    originales aunque no tengan gama mapeada (no se pierde la intención del
    usuario por una línea que no esté en el diccionario)."""
    gamas = set()
    result: List[str] = []
    for linea in lineas:
        if linea not in result:
            result.append(linea)
        gama = get_gama(linea)
        if gama:
            gamas.add(gama)
    for gama in gamas:
        for linea in get_lineas_by_gama(gama):
            if linea not in result:
                result.append(linea)
    return result


def gama_order() -> List[str]:
    return _load().get("gama_order", [])
