"""
usage_service.py

Límite diario de tokens de OpenAI por persona (correo logueado), para
controlar el gasto real de la API — no un tope global compartido, cada
correo tiene su propio balde. Reinicia a medianoche hora Chile
(America/Santiago), no UTC ni la hora del servidor: el "día" es el del
usuario, no el de la máquina donde corre esto.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from services import access_service

load_dotenv()

_DB_URL = os.getenv("DATABASE_URL")
_TZ = ZoneInfo("America/Santiago")
DAILY_TOKEN_LIMIT = int(os.getenv("DAILY_TOKEN_LIMIT", "100000"))

_engine = None

# Tope por persona configurado en panel_admin.application_users (ver
# access_service.get_daily_limit) — se cachea unos minutos para no
# consultar esa tabla en cada mensaje de chat, solo cambia cuando un
# admin edita el límite desde db-admin-panel.
_LIMIT_CACHE_TTL = 300  # 5 min
_limit_cache: dict = {}


def _get_engine():
    global _engine
    if _engine is None and _DB_URL:
        _engine = create_engine(_DB_URL, pool_pre_ping=True, connect_args={"connect_timeout": 10})
    return _engine


def ensure_table() -> None:
    engine = _get_engine()
    if not engine:
        logging.warning("[usage] Sin conexión DB — límite diario deshabilitado")
        return
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS daily_usage (
                    email       TEXT NOT NULL,
                    day         DATE NOT NULL,
                    tokens_used INT NOT NULL DEFAULT 0,
                    PRIMARY KEY (email, day)
                )
            """))
            conn.commit()
        logging.info("[usage] Tabla daily_usage lista")
    except Exception as e:
        logging.warning(f"[usage] No se pudo crear tabla daily_usage: {e}")


def _today_santiago():
    return datetime.now(_TZ).date()


def _next_reset_utc() -> datetime:
    """Próxima medianoche en hora Chile, en UTC — el frontend la recibe sin
    ambigüedad de huso y la formatea con Date() del lado del navegador."""
    now_cl = datetime.now(_TZ)
    next_midnight_cl = (now_cl + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return next_midnight_cl.astimezone(ZoneInfo("UTC"))


def _resolve_limit(email: str) -> Optional[int]:
    """None = sin límite para esta persona. Prioridad: valor propio en
    application_users (0=ilimitado, N=tope propio) sobre DAILY_TOKEN_LIMIT
    global, que solo aplica cuando esa fila no tiene nada asignado (NULL)."""
    now = time.monotonic()
    cached = _limit_cache.get(email)
    if cached and (now - cached[1]) < _LIMIT_CACHE_TTL:
        return cached[0]

    custom = access_service.get_daily_limit(email)
    if custom is None:
        resolved = DAILY_TOKEN_LIMIT
    elif custom == 0:
        resolved = None  # ilimitado
    else:
        resolved = custom

    _limit_cache[email] = (resolved, now)
    return resolved


def get_usage(email: str) -> dict:
    tokens_used = 0
    engine = _get_engine()
    if engine and email:
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT tokens_used FROM daily_usage WHERE email = :email AND day = :day"),
                    {"email": email, "day": _today_santiago()},
                ).fetchone()
                tokens_used = row[0] if row else 0
        except Exception as e:
            logging.warning(f"[usage] Error leyendo uso diario de '{email}': {e}")

    limit = _resolve_limit(email)
    if limit is None:
        return {
            "tokens_used": tokens_used,
            "daily_limit": None,
            "remaining": None,
            "percent_used": 0.0,
            "resets_at": _next_reset_utc().isoformat(),
            "blocked": False,
        }

    remaining = max(limit - tokens_used, 0)
    percent = round(min(tokens_used / limit * 100, 100), 1)
    return {
        "tokens_used": tokens_used,
        "daily_limit": limit,
        "remaining": remaining,
        "percent_used": percent,
        "resets_at": _next_reset_utc().isoformat(),
        "blocked": tokens_used >= limit,
    }


def add_usage(email: str, tokens: Optional[int]) -> None:
    if not email or not tokens:
        return
    engine = _get_engine()
    if not engine:
        return
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO daily_usage (email, day, tokens_used)
                    VALUES (:email, :day, :tokens)
                    ON CONFLICT (email, day) DO UPDATE
                    SET tokens_used = daily_usage.tokens_used + EXCLUDED.tokens_used
                """),
                {"email": email, "day": _today_santiago(), "tokens": int(tokens)},
            )
            conn.commit()
    except Exception as e:
        logging.warning(f"[usage] Error guardando uso diario de '{email}': {e}")


def is_blocked(email: str) -> bool:
    if not email:
        return False
    return get_usage(email)["blocked"]
