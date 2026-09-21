"""
admin_service.py

Panel de administrador de MVP1 (grupos, usuarios, tokens, accesos a
módulos) — antes solo administrable desde db-admin-panel. Escribe/lee las
MISMAS tablas que ya administra ese panel (esquema panel_admin, ver
access_service.py) — no hay tabla nueva, es una interfaz distinta sobre los
mismos datos, así que un cambio hecho desde acá se ve reflejado allá y
viceversa (misma fuente de verdad).

Puerto de db-admin-panel/app/db/applications.py, pero escopeado a UNA sola
aplicación (mvp1-compra-agil) -- sin selector de app, sin gestión de otras
aplicaciones. SQL crudo vía SQLAlchemy (mismo patrón que el resto de
services/, a diferencia del psycopg2 crudo del otro repo).
"""

import os
import re
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

_DB_URL = os.getenv("DATABASE_URL")
_APPLICATION_SLUG = "mvp1-compra-agil"
_SLUG_RE_MSG = "El slug solo puede tener minúsculas, números y guiones."
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_engine = None
_app_id_cache: Optional[int] = None


class AdminServiceError(Exception):
    pass


def _get_engine():
    global _engine
    if _engine is None and _DB_URL:
        _engine = create_engine(_DB_URL, pool_pre_ping=True, connect_args={"connect_timeout": 10})
    return _engine


def _app_id(conn) -> int:
    global _app_id_cache
    if _app_id_cache is not None:
        return _app_id_cache
    row = conn.execute(
        text("SELECT id FROM panel_admin.applications WHERE slug = :slug"),
        {"slug": _APPLICATION_SLUG},
    ).fetchone()
    if not row:
        raise AdminServiceError(f"Aplicación '{_APPLICATION_SLUG}' no encontrada en panel_admin.applications.")
    _app_id_cache = row[0]
    return _app_id_cache


def _parse_limit(raw_limit, unlimited: bool) -> Optional[int]:
    """None = default global, 0 = ilimitado, N = tope propio -- misma
    convención que panel_admin.applications.daily_token_limit."""
    if unlimited:
        return 0
    raw_limit = (raw_limit or "").strip() if isinstance(raw_limit, str) else raw_limit
    if raw_limit in (None, ""):
        return None
    try:
        value = int(raw_limit)
    except (TypeError, ValueError):
        raise AdminServiceError("El límite diario debe ser un número entero.")
    if value < 0:
        raise AdminServiceError("El límite diario no puede ser negativo.")
    return value


def _extract_emails(raw_text: str) -> List[str]:
    seen = set()
    emails = []
    for match in _EMAIL_RE.findall(raw_text or ""):
        email = match.strip().lower()
        if email and email not in seen:
            seen.add(email)
            emails.append(email)
    return emails


# ─── Secciones (módulos) ───────────────────────────────────────────────────

def list_sections() -> List[Dict[str, Any]]:
    engine = _get_engine()
    with engine.connect() as conn:
        app_id = _app_id(conn)
        rows = conn.execute(
            text(
                "SELECT id, slug, nombre FROM panel_admin.application_sections "
                "WHERE application_id = :app_id ORDER BY nombre"
            ),
            {"app_id": app_id},
        ).mappings().fetchall()
        return [dict(r) for r in rows]


def create_section(slug: str, nombre: str) -> None:
    slug = (slug or "").strip().lower()
    nombre = (nombre or "").strip()
    if not slug or not all(c.isalnum() or c == "-" for c in slug):
        raise AdminServiceError(_SLUG_RE_MSG)
    if not nombre:
        raise AdminServiceError("El nombre de la sección es obligatorio.")
    engine = _get_engine()
    with engine.connect() as conn:
        app_id = _app_id(conn)
        exists = conn.execute(
            text(
                "SELECT 1 FROM panel_admin.application_sections "
                "WHERE application_id = :app_id AND slug = :slug"
            ),
            {"app_id": app_id, "slug": slug},
        ).fetchone()
        if exists:
            raise AdminServiceError(f"La sección '{slug}' ya existe.")
        conn.execute(
            text(
                "INSERT INTO panel_admin.application_sections (application_id, slug, nombre) "
                "VALUES (:app_id, :slug, :nombre)"
            ),
            {"app_id": app_id, "slug": slug, "nombre": nombre},
        )
        conn.commit()


def delete_section(section_id: int) -> None:
    engine = _get_engine()
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM panel_admin.application_sections WHERE id = :id"), {"id": section_id})
        conn.commit()


# ─── Usuarios ────────────────────────────────────────────────────────────

def list_users_with_sections() -> List[Dict[str, Any]]:
    engine = _get_engine()
    with engine.connect() as conn:
        app_id = _app_id(conn)
        rows = conn.execute(
            text(
                """
                SELECT au.id, au.email, au.daily_token_limit, au.created_at, au.created_by,
                       au.group_id, g.nombre AS group_nombre,
                       COALESCE(
                           ARRAY_AGG(aus.section_id) FILTER (WHERE aus.section_id IS NOT NULL),
                           '{}'
                       ) AS section_ids
                FROM panel_admin.application_users au
                LEFT JOIN panel_admin.application_groups g ON g.id = au.group_id
                LEFT JOIN panel_admin.application_user_sections aus ON aus.application_user_id = au.id
                WHERE au.application_id = :app_id
                GROUP BY au.id, g.nombre
                ORDER BY au.email
                """
            ),
            {"app_id": app_id},
        ).mappings().fetchall()
        return [dict(r) for r in rows]


def add_user(email: str, daily_token_limit=None, unlimited: bool = False, created_by: Optional[str] = None) -> None:
    email = (email or "").strip().lower()
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise AdminServiceError("Correo inválido.")
    limit_value = _parse_limit(daily_token_limit, unlimited)
    engine = _get_engine()
    with engine.connect() as conn:
        app_id = _app_id(conn)
        exists = conn.execute(
            text(
                "SELECT 1 FROM panel_admin.application_users "
                "WHERE application_id = :app_id AND email = :email"
            ),
            {"app_id": app_id, "email": email},
        ).fetchone()
        if exists:
            raise AdminServiceError(f"El correo '{email}' ya tiene acceso.")
        conn.execute(
            text(
                "INSERT INTO panel_admin.application_users (application_id, email, daily_token_limit, created_by) "
                "VALUES (:app_id, :email, :limit, :created_by)"
            ),
            {"app_id": app_id, "email": email, "limit": limit_value, "created_by": created_by},
        )
        conn.commit()


def update_user_limit(user_id: int, daily_token_limit=None, unlimited: bool = False) -> None:
    limit_value = _parse_limit(daily_token_limit, unlimited)
    engine = _get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("UPDATE panel_admin.application_users SET daily_token_limit = :limit WHERE id = :id"),
            {"limit": limit_value, "id": user_id},
        )
        conn.commit()


def remove_user(user_id: int) -> None:
    engine = _get_engine()
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM panel_admin.application_users WHERE id = :id"), {"id": user_id})
        conn.commit()


def set_user_section(user_id: int, section_id: int, enabled: bool) -> None:
    engine = _get_engine()
    with engine.connect() as conn:
        if enabled:
            conn.execute(
                text(
                    "INSERT INTO panel_admin.application_user_sections (application_user_id, section_id) "
                    "VALUES (:uid, :sid) ON CONFLICT DO NOTHING"
                ),
                {"uid": user_id, "sid": section_id},
            )
        else:
            conn.execute(
                text(
                    "DELETE FROM panel_admin.application_user_sections "
                    "WHERE application_user_id = :uid AND section_id = :sid"
                ),
                {"uid": user_id, "sid": section_id},
            )
        conn.commit()


# ─── Grupos ─────────────────────────────────────────────────────────────
# Editar un grupo (límite o secciones) empuja el cambio de inmediato a cada
# miembro actual -- no es solo una plantilla al momento de sumarlos, es la
# fuente de verdad mientras estén agrupados (mismo comportamiento que
# db-admin-panel). Quitar a alguien del grupo no revoca nada: conserva tal
# cual el límite/secciones que tenía heredados, ahora como propios.

def list_groups() -> List[Dict[str, Any]]:
    engine = _get_engine()
    with engine.connect() as conn:
        app_id = _app_id(conn)
        rows = conn.execute(
            text(
                """
                SELECT g.id, g.nombre, g.daily_token_limit, g.created_at, g.created_by,
                       COUNT(DISTINCT au.id) AS n_miembros,
                       COALESCE(
                           ARRAY_AGG(DISTINCT ags.section_id) FILTER (WHERE ags.section_id IS NOT NULL),
                           '{}'
                       ) AS section_ids
                FROM panel_admin.application_groups g
                LEFT JOIN panel_admin.application_users au ON au.group_id = g.id
                LEFT JOIN panel_admin.application_group_sections ags ON ags.group_id = g.id
                WHERE g.application_id = :app_id
                GROUP BY g.id
                ORDER BY g.nombre
                """
            ),
            {"app_id": app_id},
        ).mappings().fetchall()
        return [dict(r) for r in rows]


def list_group_members(group_id: int) -> List[Dict[str, Any]]:
    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, email FROM panel_admin.application_users "
                "WHERE group_id = :gid ORDER BY email"
            ),
            {"gid": group_id},
        ).mappings().fetchall()
        return [dict(r) for r in rows]


def _get_group(conn, group_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        text(
            "SELECT id, application_id, nombre, daily_token_limit "
            "FROM panel_admin.application_groups WHERE id = :id"
        ),
        {"id": group_id},
    ).mappings().fetchone()
    return dict(row) if row else None


def create_group(nombre: str, daily_token_limit=None, unlimited: bool = False, created_by: Optional[str] = None) -> int:
    nombre = (nombre or "").strip()
    if not nombre:
        raise AdminServiceError("El nombre del grupo es obligatorio.")
    limit_value = _parse_limit(daily_token_limit, unlimited)
    engine = _get_engine()
    with engine.connect() as conn:
        app_id = _app_id(conn)
        exists = conn.execute(
            text(
                "SELECT 1 FROM panel_admin.application_groups "
                "WHERE application_id = :app_id AND nombre = :nombre"
            ),
            {"app_id": app_id, "nombre": nombre},
        ).fetchone()
        if exists:
            raise AdminServiceError(f"Ya existe un grupo llamado '{nombre}'.")
        row = conn.execute(
            text(
                "INSERT INTO panel_admin.application_groups (application_id, nombre, daily_token_limit, created_by) "
                "VALUES (:app_id, :nombre, :limit, :created_by) RETURNING id"
            ),
            {"app_id": app_id, "nombre": nombre, "limit": limit_value, "created_by": created_by},
        ).fetchone()
        conn.commit()
        return row[0]


def delete_group(group_id: int) -> None:
    engine = _get_engine()
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM panel_admin.application_groups WHERE id = :id"), {"id": group_id})
        conn.commit()


def update_group_limit(group_id: int, daily_token_limit=None, unlimited: bool = False) -> None:
    limit_value = _parse_limit(daily_token_limit, unlimited)
    engine = _get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("UPDATE panel_admin.application_groups SET daily_token_limit = :limit WHERE id = :id"),
            {"limit": limit_value, "id": group_id},
        )
        conn.execute(
            text("UPDATE panel_admin.application_users SET daily_token_limit = :limit WHERE group_id = :id"),
            {"limit": limit_value, "id": group_id},
        )
        conn.commit()


def set_group_section(group_id: int, section_id: int, enabled: bool) -> None:
    engine = _get_engine()
    with engine.connect() as conn:
        if enabled:
            conn.execute(
                text(
                    "INSERT INTO panel_admin.application_group_sections (group_id, section_id) "
                    "VALUES (:gid, :sid) ON CONFLICT DO NOTHING"
                ),
                {"gid": group_id, "sid": section_id},
            )
            conn.execute(
                text(
                    """
                    INSERT INTO panel_admin.application_user_sections (application_user_id, section_id)
                    SELECT au.id, :sid FROM panel_admin.application_users au
                    WHERE au.group_id = :gid
                    ON CONFLICT DO NOTHING
                    """
                ),
                {"gid": group_id, "sid": section_id},
            )
        else:
            conn.execute(
                text(
                    "DELETE FROM panel_admin.application_group_sections "
                    "WHERE group_id = :gid AND section_id = :sid"
                ),
                {"gid": group_id, "sid": section_id},
            )
            conn.execute(
                text(
                    """
                    DELETE FROM panel_admin.application_user_sections aus
                    USING panel_admin.application_users au
                    WHERE aus.application_user_id = au.id
                      AND au.group_id = :gid
                      AND aus.section_id = :sid
                    """
                ),
                {"gid": group_id, "sid": section_id},
            )
        conn.commit()


def remove_member_from_group(user_id: int) -> None:
    engine = _get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("UPDATE panel_admin.application_users SET group_id = NULL WHERE id = :id"),
            {"id": user_id},
        )
        conn.commit()


def bulk_add_emails_to_group(group_id: int, raw_text: str, created_by: Optional[str] = None) -> Dict[str, List[str]]:
    """Parsea correos pegados en cualquier formato y los suma al grupo: si
    el correo ya tenía acceso (suelto o en otro grupo) lo reasigna a este
    grupo; si es nuevo, crea el acceso directo con el límite/secciones del
    grupo. Las secciones del nuevo miembro pasan a ser EXACTAMENTE las del
    grupo, no se suman a las que ya tuviera sueltas."""
    emails = _extract_emails(raw_text)
    if not emails:
        raise AdminServiceError("No se encontró ningún correo válido en el texto pegado.")

    engine = _get_engine()
    with engine.connect() as conn:
        app_id = _app_id(conn)
        group = _get_group(conn, group_id)
        if not group or group["application_id"] != app_id:
            raise AdminServiceError("Grupo no encontrado.")

        section_ids = [
            r[0] for r in conn.execute(
                text("SELECT section_id FROM panel_admin.application_group_sections WHERE group_id = :gid"),
                {"gid": group_id},
            ).fetchall()
        ]

        created, moved, already = [], [], []
        for email in emails:
            existing = conn.execute(
                text(
                    "SELECT id, group_id FROM panel_admin.application_users "
                    "WHERE application_id = :app_id AND email = :email"
                ),
                {"app_id": app_id, "email": email},
            ).mappings().fetchone()

            if existing is None:
                user_id = conn.execute(
                    text(
                        "INSERT INTO panel_admin.application_users "
                        "(application_id, email, group_id, daily_token_limit, created_by) "
                        "VALUES (:app_id, :email, :gid, :limit, :created_by) RETURNING id"
                    ),
                    {"app_id": app_id, "email": email, "gid": group_id,
                     "limit": group["daily_token_limit"], "created_by": created_by},
                ).fetchone()[0]
                created.append(email)
            elif existing["group_id"] == group_id:
                already.append(email)
                continue
            else:
                user_id = existing["id"]
                conn.execute(
                    text(
                        "UPDATE panel_admin.application_users "
                        "SET group_id = :gid, daily_token_limit = :limit WHERE id = :id"
                    ),
                    {"gid": group_id, "limit": group["daily_token_limit"], "id": user_id},
                )
                moved.append(email)

            conn.execute(
                text("DELETE FROM panel_admin.application_user_sections WHERE application_user_id = :uid"),
                {"uid": user_id},
            )
            for section_id in section_ids:
                conn.execute(
                    text(
                        "INSERT INTO panel_admin.application_user_sections "
                        "(application_user_id, section_id) VALUES (:uid, :sid) ON CONFLICT DO NOTHING"
                    ),
                    {"uid": user_id, "sid": section_id},
                )

        conn.commit()
    return {"created": created, "moved": moved, "already": already}
