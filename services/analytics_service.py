"""
analytics_service.py
Registra métricas de uso en la tabla `metricas` del mismo PostgreSQL de Railway.
Todas las escrituras son fire-and-forget: nunca bloquean la respuesta al usuario.
"""

import asyncio
import logging
import os
from typing import Optional

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

_DB_URL = os.getenv("DATABASE_URL")
_engine = None


def _get_engine():
    global _engine
    if _engine is None and _DB_URL:
        _engine = create_engine(
            _DB_URL,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10},
        )
    return _engine


def ensure_table() -> None:
    """Crea la tabla metricas si no existe. Llamar al startup."""
    engine = _get_engine()
    if not engine:
        logging.warning("[analytics] Sin conexión DB — métricas deshabilitadas")
        return
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS metricas (
                    id           SERIAL PRIMARY KEY,
                    ts           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    session_id   TEXT,
                    tipo         TEXT,         -- 'chat' | 'blocked' | 'manual_update'
                    user_msg     TEXT,
                    ai_msg       TEXT,
                    tokens_in    INT,
                    tokens_out   INT,
                    tokens_tot   INT,
                    duration_ms  INT,
                    n_updates    INT,
                    price_found  BOOLEAN,
                    blocked      BOOLEAN NOT NULL DEFAULT FALSE,
                    block_reason TEXT,
                    model        TEXT
                )
            """))
            conn.commit()
        logging.info("[analytics] Tabla metricas lista")
    except Exception as e:
        logging.warning(f"[analytics] No se pudo crear tabla metricas: {e}")


def _write_sync(row: dict) -> None:
    engine = _get_engine()
    if not engine:
        return
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO metricas
                    (session_id, tipo, user_msg, ai_msg,
                     tokens_in, tokens_out, tokens_tot,
                     duration_ms, n_updates, price_found,
                     blocked, block_reason, model)
                VALUES
                    (:session_id, :tipo, :user_msg, :ai_msg,
                     :tokens_in, :tokens_out, :tokens_tot,
                     :duration_ms, :n_updates, :price_found,
                     :blocked, :block_reason, :model)
            """), row)
            conn.commit()
    except Exception as e:
        logging.warning(f"[analytics] Error guardando métrica: {e}")


def log(
    session_id: str,
    tipo: str,
    user_msg: str = "",
    ai_msg: str = "",
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    duration_ms: Optional[int] = None,
    n_updates: Optional[int] = None,
    price_found: Optional[bool] = None,
    blocked: bool = False,
    block_reason: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """
    Fire-and-forget: registra una métrica en background sin bloquear.
    Seguro llamar desde cualquier contexto async de FastAPI.
    """
    row = {
        "session_id": session_id,
        "tipo":       tipo,
        "user_msg":   (user_msg or "")[:2000],
        "ai_msg":     (ai_msg  or "")[:4000],
        "tokens_in":  tokens_in,
        "tokens_out": tokens_out,
        "tokens_tot": (tokens_in or 0) + (tokens_out or 0) if (tokens_in or tokens_out) else None,
        "duration_ms":  duration_ms,
        "n_updates":    n_updates,
        "price_found":  price_found,
        "blocked":      blocked,
        "block_reason": (block_reason or "")[:500] if block_reason else None,
        "model":        model,
    }
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(asyncio.to_thread(_write_sync, row))
    except RuntimeError:
        _write_sync(row)
