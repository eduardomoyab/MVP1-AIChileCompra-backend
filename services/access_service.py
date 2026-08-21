"""
access_service.py

Consulta la lista blanca de acceso a MVP1 (panel_admin.application_users),
administrada desde el módulo "Aplicaciones" de db-admin-panel — mismo
Postgres, no hay API entre ambos paneles. El backend es la única pieza de
MVP1 con credenciales de Postgres; el frontend le pregunta a este servicio
vía GET /api/auth/check_access en vez de conectarse él mismo a la base.
"""

import os
import logging
from typing import List, Optional

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

_DB_URL = os.getenv("DATABASE_URL")
_APPLICATION_SLUG = "mvp1-compra-agil"

_engine = None


def _get_engine():
    global _engine
    if _engine is None and _DB_URL:
        _engine = create_engine(_DB_URL, pool_pre_ping=True, connect_args={"connect_timeout": 10})
    return _engine


def is_email_allowed(email: str) -> bool:
    engine = _get_engine()
    if not engine or not email:
        return False
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT 1 FROM panel_admin.application_users au
                    JOIN panel_admin.applications a ON a.id = au.application_id
                    WHERE a.slug = :slug AND au.email = :email
                    """
                ),
                {"slug": _APPLICATION_SLUG, "email": email.strip().lower()},
            ).fetchone()
            return row is not None
    except Exception as e:
        logging.warning(f"[access] Error consultando lista blanca de acceso: {e}")
        return False


def get_allowed_sections(email: str) -> Optional[List[str]]:
    """Secciones (slugs) que este correo puede ver dentro de la app,
    administradas desde "Aplicaciones" > detalle de una app, en
    db-admin-panel. Devuelve None si la app no tiene ninguna sección
    definida todavía (sin restricción -- comportamiento previo a este
    mecanismo, para no romper apps que nunca configuren secciones), o la
    lista de slugs asignados a este correo (puede ser vacía: tiene acceso
    a la app pero ningún admin le asignó una sección todavía)."""
    engine = _get_engine()
    if not engine or not email:
        return []
    try:
        with engine.connect() as conn:
            total = conn.execute(
                text(
                    """
                    SELECT COUNT(*) FROM panel_admin.application_sections s
                    JOIN panel_admin.applications a ON a.id = s.application_id
                    WHERE a.slug = :slug
                    """
                ),
                {"slug": _APPLICATION_SLUG},
            ).scalar()
            if not total:
                return None
            rows = conn.execute(
                text(
                    """
                    SELECT s.slug
                    FROM panel_admin.application_sections s
                    JOIN panel_admin.applications a ON a.id = s.application_id
                    JOIN panel_admin.application_user_sections us ON us.section_id = s.id
                    JOIN panel_admin.application_users au ON au.id = us.application_user_id
                    WHERE a.slug = :slug AND au.email = :email
                    """
                ),
                {"slug": _APPLICATION_SLUG, "email": email.strip().lower()},
            ).fetchall()
            return [r[0] for r in rows]
    except Exception as e:
        logging.warning(f"[access] Error consultando secciones permitidas de '{email}': {e}")
        return []


def get_daily_limit(email: str) -> Optional[int]:
    """Tope diario de tokens configurado para este correo en
    panel_admin.application_users, administrado desde "Aplicaciones" en
    db-admin-panel. Convención de la columna: None = fila sin valor propio
    (usa el default global) o correo no encontrado; 0 = ilimitado; N =
    tope propio de esa persona."""
    engine = _get_engine()
    if not engine or not email:
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT au.daily_token_limit FROM panel_admin.application_users au
                    JOIN panel_admin.applications a ON a.id = au.application_id
                    WHERE a.slug = :slug AND au.email = :email
                    """
                ),
                {"slug": _APPLICATION_SLUG, "email": email.strip().lower()},
            ).fetchone()
            return row[0] if row else None
    except Exception as e:
        logging.warning(f"[access] Error consultando límite diario de '{email}': {e}")
        return None
