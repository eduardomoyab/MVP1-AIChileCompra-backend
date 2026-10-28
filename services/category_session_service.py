"""
category_session_service.py

Persistencia de "formularios" (fichas) del wizard genérico de categorías --
mismo patrón que chat_session_service.py (el comparador/historial del chat
legacy de Computadores), pero genérico por categoría (columna `slug`) en vez
de hardcodeado a Computadores, para no tener que duplicar esto en una fase 2
cuando se migre Medicamentos al mismo motor.

Pendiente real reportado: "ya no tengo el historial de formularios que he
rellenado... antes tenía como un historial de conversaciones, ahora nada".
El wizard nuevo no tenía NINGÚN mecanismo de persistencia -- cada carga de
página generaba un SESSION_ID nuevo que se perdía al refrescar. Esto agrega
guardado incremental (cada vez que se completa un atributo, ver
save_ficha() llamado desde los endpoints de analizar/manual_update en
main.py) y una forma de listar/retomar/renombrar/borrar esos formularios.

Lo que NO se guarda acá, a propósito (mismo criterio que chat_session_service):
el precio estimado -- se recalcula en caliente al retomar un formulario
(ver GET /api/categories/{slug}/sessions/{session_id} en main.py), nunca un
snapshot que podría quedar desactualizado.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

_DB_URL = os.getenv("DATABASE_URL")
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
        logging.warning("[category_session] Sin conexión DB — persistencia de formularios deshabilitada")
        return
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS category_sessions (
                    session_id      TEXT PRIMARY KEY,
                    slug            TEXT NOT NULL,
                    email           TEXT NOT NULL,
                    title           TEXT,
                    title_is_custom BOOLEAN NOT NULL DEFAULT false,
                    ficha           JSONB NOT NULL DEFAULT '{}'::jsonb,
                    texto_original  TEXT,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_category_sessions_slug_email "
                "ON category_sessions (slug, email, updated_at DESC)"
            ))
            conn.commit()
        logging.info("[category_session] Tabla category_sessions lista")
    except Exception as e:
        logging.warning(f"[category_session] No se pudo crear tabla category_sessions: {e}")


def _derive_title(ficha: Dict[str, Any], texto_original: str) -> str:
    text_value = " ".join((texto_original or "").split())
    if text_value:
        return text_value[:_TITLE_MAX_LEN].rstrip() + ("…" if len(text_value) > _TITLE_MAX_LEN else "")
    # Sin texto libre (el usuario "completó los datos directamente"): el
    # título se arma con lo que ya haya en la ficha, mismo criterio que
    # chat_session_service pero sin asumir un atributo fijo como tipo_equipo
    # (acá puede ser cualquier categoría futura).
    partes = [str(v) for v in ficha.values() if isinstance(v, (str, int, float)) and str(v).strip()][:3]
    return (" · ".join(partes))[:_TITLE_MAX_LEN] or "Nuevo formulario"


def _row_exists(conn, session_id: str) -> bool:
    row = conn.execute(
        text("SELECT 1 FROM category_sessions WHERE session_id = :sid"), {"sid": session_id}
    ).fetchone()
    return row is not None


def save_ficha(
    session_id: str,
    slug: str,
    email: str,
    ficha: Dict[str, Any],
    texto_original: Optional[str] = None,
    title_hint: Optional[str] = None,
) -> None:
    """title_hint: título ya armado (ej. generado por la IA en la misma
    pasada de extracción, ver titulo en ComputadoresWizardAgent.analyze) --
    si viene, se usa tal cual en vez de derivarlo truncando texto_original.
    Como ChatGPT con el título de cada conversación: más corto y
    descriptivo que el texto crudo del usuario."""
    engine = _get_engine()
    if not engine or not session_id or not email:
        return
    try:
        with engine.connect() as conn:
            existed = _row_exists(conn, session_id)
            if texto_original is not None:
                conn.execute(
                    text("""
                        INSERT INTO category_sessions (session_id, slug, email, ficha, texto_original, updated_at)
                        VALUES (:sid, :slug, :email, CAST(:ficha AS JSONB), :texto, now())
                        ON CONFLICT (session_id) DO UPDATE
                        SET ficha = EXCLUDED.ficha,
                            texto_original = COALESCE(category_sessions.texto_original, EXCLUDED.texto_original),
                            updated_at = now()
                    """),
                    {"sid": session_id, "slug": slug, "email": email, "ficha": json.dumps(ficha), "texto": texto_original},
                )
            else:
                conn.execute(
                    text("""
                        INSERT INTO category_sessions (session_id, slug, email, ficha, updated_at)
                        VALUES (:sid, :slug, :email, CAST(:ficha AS JSONB), now())
                        ON CONFLICT (session_id) DO UPDATE
                        SET ficha = EXCLUDED.ficha,
                            updated_at = now()
                    """),
                    {"sid": session_id, "slug": slug, "email": email, "ficha": json.dumps(ficha)},
                )
            if not existed:
                title = (title_hint or "").strip() or _derive_title(ficha, texto_original or "")
                conn.execute(
                    text("""
                        UPDATE category_sessions SET title = :title
                        WHERE session_id = :sid AND title IS NULL AND NOT title_is_custom
                    """),
                    {"sid": session_id, "title": title},
                )
            conn.commit()
    except Exception as e:
        logging.warning(f"[category_session] Error guardando ficha de '{session_id}': {e}")


def list_sessions(slug: str, email: str) -> List[Dict[str, Any]]:
    engine = _get_engine()
    if not engine or not email:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT session_id, title, updated_at
                    FROM category_sessions
                    WHERE slug = :slug AND email = :email
                    ORDER BY updated_at DESC
                    LIMIT 50
                """),
                {"slug": slug, "email": email},
            ).mappings().fetchall()
            return [
                {
                    "session_id": r["session_id"],
                    "title": r["title"] or "Nuevo formulario",
                    "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                }
                for r in rows
            ]
    except Exception as e:
        logging.warning(f"[category_session] Error listando formularios de '{email}': {e}")
        return []


def get_full_session(session_id: str, slug: str, email: str) -> Optional[Dict[str, Any]]:
    """Ownership-checked: nunca devuelve un formulario de otro usuario ni de
    otra categoría."""
    engine = _get_engine()
    if not engine or not session_id or not email:
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT title, ficha, texto_original
                    FROM category_sessions WHERE session_id = :sid AND slug = :slug AND email = :email
                """),
                {"sid": session_id, "slug": slug, "email": email},
            ).mappings().fetchone()
            if not row:
                return None
            return {
                "title": row["title"] or "Nuevo formulario",
                "ficha": row["ficha"] or {},
                "texto_original": row["texto_original"] or "",
            }
    except Exception as e:
        logging.warning(f"[category_session] Error leyendo formulario '{session_id}': {e}")
        return None


def rename(session_id: str, slug: str, email: str, title: str) -> bool:
    engine = _get_engine()
    if not engine or not session_id or not email:
        return False
    title = (title or "").strip()[:200] or "Nuevo formulario"
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    UPDATE category_sessions SET title = :title, title_is_custom = true, updated_at = now()
                    WHERE session_id = :sid AND slug = :slug AND email = :email
                """),
                {"sid": session_id, "slug": slug, "email": email, "title": title},
            )
            conn.commit()
            return result.rowcount > 0
    except Exception as e:
        logging.warning(f"[category_session] Error renombrando '{session_id}': {e}")
        return False


def delete(session_id: str, slug: str, email: str) -> bool:
    engine = _get_engine()
    if not engine or not session_id or not email:
        return False
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("DELETE FROM category_sessions WHERE session_id = :sid AND slug = :slug AND email = :email"),
                {"sid": session_id, "slug": slug, "email": email},
            )
            conn.commit()
            return result.rowcount > 0
    except Exception as e:
        logging.warning(f"[category_session] Error borrando '{session_id}': {e}")
        return False
