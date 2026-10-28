"""
currency_service.py

Tipo de cambio USD → CLP vía mindicador.cl (API pública, sin key), con
respaldo opcional en la API oficial del Banco Central de Chile (serie del
dólar observado, F073.TCO.PRE.Z.D) si mindicador.cl no responde y hay un
BCCH_API_TOKEN configurado (gratuito, se obtiene registrándose en
si3.bcentral.cl -- ver _fetch_bcch). Sin ese token, el respaldo se omite en
silencio y el comportamiento es el mismo de siempre (fallback estático).

Los precios de Convenio Marco (PrecioCM) vienen en USD; se usa este valor
para mostrarlos en CLP, consistente con PrecioCA.

Cacheado en memoria (TTL) para no golpear las APIs externas en cada mensaje
del chat. Si ambas fuentes fallan, se reutiliza el último valor conocido
(aunque haya vencido el TTL); si nunca hubo un valor exitoso, se usa un
fallback estático, dejando constancia en el log de que es una estimación.
"""

import os
import json
import time
import logging
import urllib.parse
import urllib.request
from datetime import date, timedelta
from typing import Optional, Tuple

_URL = "https://mindicador.cl/api/dolar"
_TTL_SECONDS = int(os.getenv("FX_CACHE_TTL_SECONDS", str(6 * 3600)))
_TIMEOUT = 8

# Si un intento falla (ej. el EOF/SSL real que se ve en los logs --
# mindicador.cl cerrando la conexión a mitad del handshake, típico de un
# corte transitorio), no reintentar la API en CADA llamada -- estimate()
# de precio llama esto por cada ficha calculada, y sin este freno cada una
# pagaba el timeout completo (hasta 8s) y repetía el mismo warning.
_FAIL_RETRY_SECONDS = int(os.getenv("FX_RETRY_AFTER_FAIL_SECONDS", str(5 * 60)))

# Fallback usado solo si NINGUNA fuente (mindicador.cl ni Banco Central)
# respondió exitosamente.
_FALLBACK_RATE = float(os.getenv("FX_FALLBACK_USD_CLP", "950"))

# ── Respaldo: Banco Central de Chile ────────────────────────────────────
_BCCH_URL = "https://si3.bcentral.cl/SieteRestWS/SieteRestWS.ashx"
_BCCH_SERIES = "F073.TCO.PRE.Z.D"  # dólar observado, serie diaria oficial
_BCCH_TOKEN = os.getenv("BCCH_API_TOKEN", "").strip()

_cache: Optional[Tuple[float, str]] = None  # (valor, fecha)
_cache_ts: float = 0.0
_last_attempt_ts: float = 0.0


def _fetch_mindicador() -> Optional[Tuple[float, str]]:
    # Un reintento único y silencioso antes de darse por vencido -- el error
    # real observado (SSL EOF a mitad de handshake) suele ser un corte de
    # conexión transitorio, no un bloqueo permanente, y resolverse solo en
    # el segundo intento es mucho más común que una falla real de la API.
    for attempt in range(2):
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
            if attempt == 0:
                continue
            logging.warning(f"[currency] Error consultando mindicador.cl: {e}")
            return None


def _fetch_bcch() -> Optional[Tuple[float, str]]:
    """Respaldo oficial si mindicador.cl no responde. Requiere BCCH_API_TOKEN
    (token gratuito -- registrarse en si3.bcentral.cl, sección "Acceso API",
    para conseguirlo). Sin el token configurado, se omite en silencio: no es
    una falla, mindicador.cl sigue siendo la fuente primaria y este respaldo
    es opcional.

    Pide los últimos 10 días de la serie (no solo "hoy") porque el dólar
    observado es un dato de día hábil -- en fin de semana/feriado no hay
    observación para la fecha exacta, y se toma la más reciente con
    statusCode "OK"."""
    if not _BCCH_TOKEN:
        return None
    try:
        last = date.today()
        first = last - timedelta(days=10)
        params = {
            "token": _BCCH_TOKEN,
            "function": "GetSeries",
            "timeseries": _BCCH_SERIES,
            "firstdate": first.isoformat(),
            "lastdate": last.isoformat(),
        }
        url = f"{_BCCH_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "AsistenteCompraAgil/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read()
        # El .ashx del Banco Central no declara UTF-8 (es un handler .NET
        # clásico) -- responde en Latin-1/CP1252, así que un campo con
        # tilde (ej. "Descripción") rompe un decode("utf-8") directo (error
        # real observado: byte 0xf3, que es "ó" en Latin-1, no UTF-8 válido).
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        data = json.loads(text)
        if data.get("Codigo") != 0:
            return None
        obs = (data.get("Series") or {}).get("Obs") or []
        for o in reversed(obs):
            if o.get("statusCode") != "OK":
                continue
            try:
                valor = float(o["value"])
            except (TypeError, ValueError):
                continue
            d, m, y = str(o.get("indexDateString", "")).split("-")
            return valor, f"{y}-{m}-{d}"
        return None
    except Exception as e:
        logging.warning(f"[currency] Error consultando Banco Central (respaldo): {e}")
        return None


def _fetch() -> Optional[Tuple[float, str]]:
    """mindicador.cl primero (fuente primaria, sin key); si falla, Banco
    Central de Chile como respaldo (si hay token configurado, ver
    _fetch_bcch). Si ninguna responde, None -- get_usd_clp() decide ahí si
    usar el último valor cacheado o el fallback estático."""
    result = _fetch_mindicador()
    if result is not None:
        return result
    result = _fetch_bcch()
    if result is not None:
        logging.info("[currency] mindicador.cl no disponible -- usando respaldo Banco Central de Chile")
        return result
    return None


def get_usd_clp() -> Tuple[float, str, bool]:
    """
    Devuelve (valor_clp_por_usd, fecha, is_fallback).
    Cachea en memoria por FX_CACHE_TTL_SECONDS. Si el último intento de
    refrescar falló hace poco, no vuelve a intentar hasta pasado
    FX_RETRY_AFTER_FAIL_SECONDS -- ver nota en _FAIL_RETRY_SECONDS.
    """
    global _cache, _cache_ts, _last_attempt_ts

    now = time.time()
    if _cache is not None and (now - _cache_ts) < _TTL_SECONDS:
        return _cache[0], _cache[1], False

    if (now - _last_attempt_ts) < _FAIL_RETRY_SECONDS:
        if _cache is not None:
            return _cache[0], _cache[1], False
        return _FALLBACK_RATE, "", True

    _last_attempt_ts = now
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
