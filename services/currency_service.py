"""
currency_service.py

Tipo de cambio USD → CLP vía mindicador.cl (API pública, sin key).
Los precios de Convenio Marco (PrecioCM) vienen en USD; se usa este valor
para mostrarlos en CLP, consistente con PrecioCA.

Cacheado en memoria (TTL) para no golpear la API externa en cada mensaje
del chat. Si la llamada falla, se reutiliza el último valor conocido
(aunque haya vencido el TTL); si nunca hubo un valor exitoso, se usa un
fallback estático, dejando constancia en el log de que es una estimación.
"""

import os
import json
import time
import logging
import urllib.request
from typing import Optional, Tuple

_URL = "https://mindicador.cl/api/dolar"
_TTL_SECONDS = int(os.getenv("FX_CACHE_TTL_SECONDS", str(6 * 3600)))
_TIMEOUT = 8

# Fallback usado solo si mindicador.cl nunca respondió exitosamente.
_FALLBACK_RATE = float(os.getenv("FX_FALLBACK_USD_CLP", "950"))

_cache: Optional[Tuple[float, str]] = None  # (valor, fecha)
_cache_ts: float = 0.0


def _fetch() -> Optional[Tuple[float, str]]:
    try:
        req = urllib.request.Request(_URL, headers={"User-Agent": "AsistenteCompraAgil/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        serie = data.get("serie") or []
        if not serie:
            return None
        valor = float(serie[0]["valor"])
        fecha = str(serie[0].get("fecha", ""))[:10]
        return valor, fecha
    except Exception as e:
        logging.warning(f"[currency] Error consultando mindicador.cl: {e}")
        return None


def get_usd_clp() -> Tuple[float, str, bool]:
    """
    Devuelve (valor_clp_por_usd, fecha, is_fallback).
    Cachea en memoria por FX_CACHE_TTL_SECONDS.
    """
    global _cache, _cache_ts

    now = time.time()
    if _cache is not None and (now - _cache_ts) < _TTL_SECONDS:
        return _cache[0], _cache[1], False

    fresh = _fetch()
    if fresh is not None:
        _cache = fresh
        _cache_ts = now
        return fresh[0], fresh[1], False

    if _cache is not None:
        logging.warning("[currency] Usando último valor cacheado (mindicador.cl no disponible)")
        return _cache[0], _cache[1], False

    logging.warning(f"[currency] Sin valor previo — usando fallback estático {_FALLBACK_RATE} CLP/USD")
    return _FALLBACK_RATE, "", True


def warmup() -> None:
    get_usd_clp()
