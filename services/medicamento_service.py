"""
medicamento_service.py

Buscador de medicamentos contra public.extractor_medicamentos (historial de
compras Compra Ágil, no un catálogo único). A diferencia de Computadores,
acá no hay ficha ni recomendación -- es búsqueda libre: el usuario escribe
texto (nombre de producto, principio activo, laboratorio, concentración,
todo junto) y se hace match por substring contra un blob de todos los
campos relevantes, sin depender de en qué columna cayó cada dato -- varios
de los campos estructurados vienen vacíos en 40-70% de las filas, así que
apoyarse solo en ellos dejaría afuera la mayoría de las búsquedas.
nombre_producto (texto libre del comprador, importado desde el CSV fuente)
es el único campo casi siempre presente.

Como la tabla es historial de compras (el mismo producto se repite muchas
veces, una fila por orden de compra), los resultados se agrupan por
producto normalizado con un conteo de compras (n_compras) en vez de
devolver filas crudas -- si no, buscar "eutirox" mostraría cientos de
filas casi idénticas.
"""

import os
import re
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

load_dotenv()

_DB_URL = os.getenv("DATABASE_URL")
_TABLE_NAME = "extractor_medicamentos"
_engine = None

_DB_MAX_RETRIES = 3
_DB_RETRY_DELAY = 2

# Campos sobre los que se arma el blob de búsqueda -- "eutirox 10 mg" debe
# calzar sin importar si "eutirox" está en nombre_producto y "10 mg" en
# concentracion_1, o si todo vino junto en un solo campo.
_SEARCH_COLS = [
    "nombre_producto", "laboratorio", "principio_activo_1", "principio_activo_2",
    "forma_farmaceutica", "concentracion_1", "concentracion_2",
]

_RESULT_COLS = _SEARCH_COLS + ["cantidad", "unidad_cantidad"]

_TOKEN_RE = re.compile(r"\S+")
_MAX_TOKENS = 8  # tope defensivo, ninguna búsqueda real necesita más palabras para acotar


def _get_engine():
    global _engine
    if _engine is None and _DB_URL:
        _engine = create_engine(_DB_URL, pool_pre_ping=True, connect_args={"connect_timeout": 10})
    return _engine


def _reset_engine():
    global _engine
    if _engine:
        try:
            _engine.dispose()
        except Exception:
            pass
    _engine = None


class MedicamentoService:
    def search(
        self,
        query: str,
        laboratorio: Optional[str] = None,
        forma_farmaceutica: Optional[str] = None,
        limit: int = 30,
    ) -> Dict[str, Any]:
        engine = _get_engine()
        if not engine:
            return {"results": [], "count": 0}

        query = (query or "").strip()
        tokens = _TOKEN_RE.findall(query)[:_MAX_TOKENS]

        blob = " || ' ' || ".join(f"COALESCE({c}, '')" for c in _SEARCH_COLS)
        where = ["nombre_producto IS NOT NULL"]
        params: Dict[str, Any] = {}
        for i, tok in enumerate(tokens):
            key = f"tok{i}"
            where.append(f"({blob}) ILIKE :{key}")
            params[key] = f"%{tok}%"

        if laboratorio:
            where.append("laboratorio ILIKE :laboratorio")
            params["laboratorio"] = f"%{laboratorio}%"
        if forma_farmaceutica:
            where.append("forma_farmaceutica ILIKE :forma_farmaceutica")
            params["forma_farmaceutica"] = f"%{forma_farmaceutica}%"

        where_sql = " AND ".join(where)
        # Se agrupa por la forma normalizada (mayúsculas + sin espacios extra)
        # de cada campo -- funde variantes que son el mismo dato con distinto
        # casing, sin juntar productos genuinamente distintos. "No
        # especificado" se trata igual que NULL/vacío acá (es el mismo "sin
        # dato" que dejó el proceso de extracción, no un valor real) -- si
        # no, una fila con NULL y otra con el literal "No especificado" para
        # el mismo producto quedaban en grupos distintos.
        def _norm(col: str) -> str:
            return f"NULLIF(UPPER(TRIM(COALESCE({col}, ''))), 'NO ESPECIFICADO')"

        group_cols = ", ".join(_norm(c) for c in _RESULT_COLS)

        sql = text(f"""
            SELECT
                MAX(nombre_producto)    AS nombre_producto,
                MAX(laboratorio)        AS laboratorio,
                MAX(principio_activo_1) AS principio_activo_1,
                MAX(principio_activo_2) AS principio_activo_2,
                MAX(forma_farmaceutica) AS forma_farmaceutica,
                MAX(concentracion_1)    AS concentracion_1,
                MAX(concentracion_2)    AS concentracion_2,
                MAX(cantidad)           AS cantidad,
                MAX(unidad_cantidad)    AS unidad_cantidad,
                COUNT(*)                AS n_compras
            FROM "{_TABLE_NAME}"
            WHERE {where_sql}
            GROUP BY {group_cols}
            ORDER BY n_compras DESC, MAX(nombre_producto) ASC
            LIMIT :limit
        """)
        params["limit"] = limit

        for attempt in range(_DB_MAX_RETRIES):
            try:
                with engine.connect() as conn:
                    rows = conn.execute(sql, params).mappings().all()
                results = [dict(r) for r in rows]
                return {"results": results, "count": len(results), "query": query}
            except OperationalError as e:
                logging.warning(f"[Medicamentos] Error de conexión (intento {attempt + 1}/{_DB_MAX_RETRIES}): {e}")
                _reset_engine()
                engine = _get_engine()
                if not engine:
                    break
            except Exception as e:
                logging.warning(f"[Medicamentos] Error en búsqueda: {e}")
                return {"results": [], "count": 0}

        return {"results": [], "count": 0}

    def get_dropdown_values(self) -> Dict[str, List[str]]:
        engine = _get_engine()
        if not engine:
            return {}
        result: Dict[str, List[str]] = {}
        for col in ["laboratorio", "forma_farmaceutica"]:
            try:
                with engine.connect() as conn:
                    rows = conn.execute(text(f"""
                        SELECT DISTINCT {col}
                        FROM "{_TABLE_NAME}"
                        WHERE {col} IS NOT NULL AND TRIM({col}) NOT IN ('', 'No especificado')
                        ORDER BY {col}
                    """)).fetchall()
                result[col] = [r[0] for r in rows]
            except Exception as e:
                logging.warning(f"[Medicamentos] Error en dropdown '{col}': {e}")
                result[col] = []
        return result
