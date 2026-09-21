"""
chat_session_service.py

Persistencia de conversaciones completas por usuario (Computadores):
ficha técnica, mensajes del chat (para poder mostrarlos de nuevo) y carrito
de comparación -- todo por conversación, no por usuario a secas, para poder
tener varias conversaciones en paralelo con un sidebar tipo ChatGPT en vez
de una sola sesión que se pisa con el botón "Nueva sesión".

Reemplaza a session_service.py (ficha por sesión) y compare_service.py
(comparador por usuario, global) -- ambos quedan retirados. Mismo patrón que
los dos: SQL crudo vía SQLAlchemy (sin ORM), una tabla, blobs JSONB en vez de
fila por atributo/mensaje/item.

Lo que NO se guarda acá, a propósito: la estimación de precio (Compra Ágil y
Convenio Marco). Es 100% derivable de la ficha (price_service.estimate() /
cm_service.estimate(), las mismas funciones que ya corren tras cada turno de
chat) -- guardar un snapshot quedaría desactualizado si cambia el catálogo o
PrecioCA. Se recalcula en caliente cada vez que se abre una conversación
(ver GET /api/sessions/{id} en main.py).
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

_DB_URL = os.getenv("DATABASE_URL")
_MAX_COMPARE_ITEMS = 20
_TITLE_MAX_LEN = 48

_engine = None


def _get_engine():
    global _engine
    if _engine is None and _DB_URL:
        _engine = create_engine(_DB_URL, pool_pre_ping=True, connect_args={"connect_timeout": 10})
    return _engine


def ensure_table() -> None:
    engine = _get_engine()
    if not engine:
        logging.warning("[chat_session] Sin conexión DB — persistencia de conversaciones deshabilitada")
        return
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id      TEXT PRIMARY KEY,
                    email           TEXT NOT NULL,
                    title           TEXT,
                    title_is_custom BOOLEAN NOT NULL DEFAULT false,
                    ficha           JSONB NOT NULL DEFAULT '{}'::jsonb,
                    messages        JSONB NOT NULL DEFAULT '[]'::jsonb,
                    compare_items   JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_chat_sessions_email ON chat_sessions (email, updated_at DESC)"
            ))
            conn.commit()
        logging.info("[chat_session] Tabla chat_sessions lista")
    except Exception as e:
        logging.warning(f"[chat_session] No se pudo crear tabla chat_sessions: {e}")


def _derive_title(text_value: str) -> str:
    text_value = " ".join((text_value or "").split())
    if len(text_value) <= _TITLE_MAX_LEN:
        return text_value
    return text_value[:_TITLE_MAX_LEN].rstrip() + "…"


def _row_exists(conn, session_id: str) -> bool:
    row = conn.execute(
        text("SELECT 1 FROM chat_sessions WHERE session_id = :sid"), {"sid": session_id}
    ).fetchone()
    return row is not None


def get_agent_state(session_id: str) -> Dict[str, Any]:
    """Ficha + historial plano (para re-sembrar el contexto del LLM al
    retomar una conversación que este proceso todavía no tenía en memoria)."""
    engine = _get_engine()
    if not engine or not session_id:
        return {"ficha": {}, "messages": []}
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT ficha, messages FROM chat_sessions WHERE session_id = :sid"),
                {"sid": session_id},
            ).fetchone()
            if not row:
                return {"ficha": {}, "messages": []}
            return {"ficha": row[0] or {}, "messages": row[1] or []}
    except Exception as e:
        logging.warning(f"[chat_session] Error leyendo estado de '{session_id}': {e}")
        return {"ficha": {}, "messages": []}


def save_ficha(session_id: str, email: str, ficha: Dict[str, Any]) -> None:
    engine = _get_engine()
    if not engine or not session_id or not email:
        return
    try:
        with engine.connect() as conn:
            existed = _row_exists(conn, session_id)
            conn.execute(
                text("""
                    INSERT INTO chat_sessions (session_id, email, ficha, updated_at)
                    VALUES (:sid, :email, CAST(:ficha AS JSONB), now())
                    ON CONFLICT (session_id) DO UPDATE
                    SET ficha = EXCLUDED.ficha,
                        updated_at = now()
                """),
                {"sid": session_id, "email": email, "ficha": json.dumps(ficha)},
            )
            if not existed:
                conn.execute(
                    text("""
                        UPDATE chat_sessions SET title = :title
                        WHERE session_id = :sid AND title IS NULL AND NOT title_is_custom
                    """),
                    {"sid": session_id, "title": _derive_title(f"Ficha: {ficha.get('tipo_equipo', 'equipo')}")},
                )
            conn.commit()
    except Exception as e:
        logging.warning(f"[chat_session] Error guardando ficha de '{session_id}': {e}")


def append_messages(session_id: str, email: str, new_messages: List[Dict[str, Any]]) -> None:
    """Agrega turnos al transcript limpio (para mostrar en pantalla al
    restaurar la conversación) y deriva el título la primera vez, a partir
    del primer mensaje del usuario."""
    engine = _get_engine()
    if not engine or not session_id or not email or not new_messages:
        return
    first_user_msg = next((m["content"] for m in new_messages if m.get("role") == "user"), None)
    try:
        with engine.connect() as conn:
            existed = _row_exists(conn, session_id)
            conn.execute(
                text("""
                    INSERT INTO chat_sessions (session_id, email, messages, updated_at)
                    VALUES (:sid, :email, CAST(:msgs AS JSONB), now())
                    ON CONFLICT (session_id) DO UPDATE
                    SET messages = chat_sessions.messages || EXCLUDED.messages,
                        updated_at = now()
                """),
                {"sid": session_id, "email": email, "msgs": json.dumps(new_messages)},
            )
            if not existed and first_user_msg:
                conn.execute(
                    text("""
                        UPDATE chat_sessions SET title = :title
                        WHERE session_id = :sid AND title IS NULL AND NOT title_is_custom
                    """),
                    {"sid": session_id, "title": _derive_title(first_user_msg)},
                )
            conn.commit()
    except Exception as e:
        logging.warning(f"[chat_session] Error guardando mensajes de '{session_id}': {e}")


def save_compare_items(session_id: str, email: str, items: List[Dict[str, Any]]) -> None:
    engine = _get_engine()
    if not engine or not session_id or not email:
        return
    items = (items or [])[:_MAX_COMPARE_ITEMS]
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO chat_sessions (session_id, email, compare_items, updated_at)
                    VALUES (:sid, :email, CAST(:items AS JSONB), now())
                    ON CONFLICT (session_id) DO UPDATE
                    SET compare_items = EXCLUDED.compare_items,
                        updated_at = now()
                """),
                {"sid": session_id, "email": email, "items": json.dumps(items)},
            )
            conn.commit()
    except Exception as e:
        logging.warning(f"[chat_session] Error guardando comparador de '{session_id}': {e}")


def list_sessions(email: str) -> List[Dict[str, Any]]:
    engine = _get_engine()
    if not engine or not email:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT session_id, title, ficha->>'tipo_equipo' AS tipo_equipo, updated_at
                    FROM chat_sessions
                    WHERE email = :email
                    ORDER BY updated_at DESC
                """),
                {"email": email},
            ).mappings().fetchall()
            return [
                {
                    "session_id": r["session_id"],
                    "title": r["title"] or "Nueva conversación",
                    "tipo_equipo": r["tipo_equipo"],
                    "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                }
                for r in rows
            ]
    except Exception as e:
        logging.warning(f"[chat_session] Error listando conversaciones de '{email}': {e}")
        return []


def get_full_session(session_id: str, email: str) -> Optional[Dict[str, Any]]:
    """Ownership-checked: nunca devuelve una conversación de otro usuario."""
    engine = _get_engine()
    if not engine or not session_id or not email:
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT title, ficha, messages, compare_items
                    FROM chat_sessions WHERE session_id = :sid AND email = :email
                """),
                {"sid": session_id, "email": email},
            ).mappings().fetchone()
            if not row:
                return None
            return {
                "title": row["title"] or "Nueva conversación",
                "ficha": row["ficha"] or {},
                "messages": row["messages"] or [],
                "compare_items": row["compare_items"] or [],
            }
    except Exception as e:
        logging.warning(f"[chat_session] Error leyendo conversación '{session_id}': {e}")
        return None


def rename(session_id: str, email: str, title: str) -> bool:
    engine = _get_engine()
    if not engine or not session_id or not email:
        return False
    title = (title or "").strip()[:200] or "Nueva conversación"
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    UPDATE chat_sessions SET title = :title, title_is_custom = true, updated_at = now()
                    WHERE session_id = :sid AND email = :email
                """),
                {"sid": session_id, "email": email, "title": title},
            )
            conn.commit()
            return result.rowcount > 0
    except Exception as e:
        logging.warning(f"[chat_session] Error renombrando '{session_id}': {e}")
        return False


def delete(session_id: str, email: str) -> bool:
    engine = _get_engine()
    if not engine or not session_id or not email:
        return False
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("DELETE FROM chat_sessions WHERE session_id = :sid AND email = :email"),
                {"sid": session_id, "email": email},
            )
            conn.commit()
            return result.rowcount > 0
    except Exception as e:
        logging.warning(f"[chat_session] Error borrando '{session_id}': {e}")
        return False
