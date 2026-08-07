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

Filtros: en vez de dropdowns globales (laboratorio tiene >3000 valores
distintos en la tabla, forma_farmaceutica >2000 -- inservible como <select>
fijo), search() devuelve "facets": los valores de laboratorio/forma
farmacéutica/concentración que realmente aparecen DENTRO de los resultados
de esa búsqueda, con su conteo -- lo mismo que un buscador de e-commerce.
Cada faceta se calcula excluyendo su propio filtro (para no auto-limitarse
a la única opción ya elegida).
"""

import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

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

# (nombre del filtro/faceta, columna real)
_FACET_COLS = [
    ("laboratorio", "laboratorio"),
    ("forma_farmaceutica", "forma_farmaceutica"),
    ("concentracion", "concentracion_1"),
]
_FACET_LIMIT = 12

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


def _build_where(
    tokens: List[str],
    filters: Dict[str, Optional[str]],
    exclude: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """filters: {"laboratorio": ..., "forma_farmaceutica": ..., "concentracion": ...}.
    exclude: nombre de filtro a NO aplicar (para calcular la faceta de ESE
    mismo filtro sin que su propia selección la reduzca a una sola opción)."""
    blob = " || ' ' || ".join(f"COALESCE({c}, '')" for c in _SEARCH_COLS)
    clauses = ["nombre_producto IS NOT NULL"]
    params: Dict[str, Any] = {}
    for i, tok in enumerate(tokens):
        key = f"tok{i}"
        clauses.append(f"({blob}) ILIKE :{key}")
        params[key] = f"%{tok}%"

    _FILTER_COL = {"laboratorio": "laboratorio", "forma_farmaceutica": "forma_farmaceutica", "concentracion": "concentracion_1"}
    for name, col in _FILTER_COL.items():
        if name == exclude:
            continue
        value = filters.get(name)
        if value:
            key = f"f_{name}"
            clauses.append(f"{col} ILIKE :{key}")
            params[key] = f"%{value}%"

    return " AND ".join(clauses), params


class MedicamentoService:
    def search(
        self,
        query: str,
        laboratorio: Optional[str] = None,
        forma_farmaceutica: Optional[str] = None,
        concentracion: Optional[str] = None,
        limit: int = 30,
    ) -> Dict[str, Any]:
        engine = _get_engine()
        if not engine:
            return {"results": [], "count": 0, "facets": {}}

        query = (query or "").strip()
        tokens = _TOKEN_RE.findall(query)[:_MAX_TOKENS]
        filters = {"laboratorio": laboratorio, "forma_farmaceutica": forma_farmaceutica, "concentracion": concentracion}

        where_sql, params = _build_where(tokens, filters)

        # Se agrupa por la forma normalizada (mayúsculas + sin espacios extra)
        # de cada campo -- funde variantes que son el mismo dato con distinto
        # casing, sin juntar productos genuinamente distintos. "No
        # especificado" se trata igual que NULL/vacío acá (es el mismo "sin
        # dato" que dejó el proceso de extracción, no un valor real) -- si
        # no, una fila con NULL y otra con el literal "No especificado" para
        # el mismo producto quedaban en grupos distintos.
        def _norm(col: str) -> str:
            return f"NULLIF(UPPER(TRIM(COALESCE({col}, ''))), 'NO ESPECIFICADO')"

        # "cantidad" también existe en OfertaProducto (op.cantidad) -- hay
        # que calificarla con la tabla o el JOIN de precio la deja ambigua.
        def _qualify(c: str) -> str:
            return f"{_TABLE_NAME}.cantidad" if c == "cantidad" else c

        group_cols = ", ".join(_norm(_qualify(c)) for c in _RESULT_COLS)

        # Precio: join contra OfertaProducto (misma tabla que ya usa PrecioCA
        # para Computadores) por id_oferta_producto_aquiles -- ese match es
        # 1:1 y cubre el 100% de las filas de extractor_medicamentos
        # (verificado contra la BD real). Se usa precio_unitario_iva (CLP con
        # IVA) para ser consistente con lo que ya expone PrecioCA. Como cada
        # grupo junta muchas compras históricas del "mismo" producto con
        # precios que varían fuerte según el tamaño del lote/envase
        # cotizado, se reporta un rango de percentiles (p25/mediana/p75) en
        # vez de un promedio o mínimo crudo -- la mediana no se distorsiona
        # con los ítems que algún proveedor cotizó en $0 o $1 por error.
        sql = text(f"""
            SELECT
                MAX(nombre_producto)    AS nombre_producto,
                MAX(laboratorio)        AS laboratorio,
                MAX(principio_activo_1) AS principio_activo_1,
                MAX(principio_activo_2) AS principio_activo_2,
                MAX(forma_farmaceutica) AS forma_farmaceutica,
                MAX(concentracion_1)    AS concentracion_1,
                MAX(concentracion_2)    AS concentracion_2,
                MAX({_TABLE_NAME}.cantidad) AS cantidad,
                MAX(unidad_cantidad)    AS unidad_cantidad,
                COUNT(*)                AS n_compras,
                COUNT(op.precio_unitario_iva) AS n_con_precio,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY op.precio_unitario_iva) AS precio_p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY op.precio_unitario_iva) AS precio_mediana,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY op.precio_unitario_iva) AS precio_p75
            FROM "{_TABLE_NAME}"
            LEFT JOIN "OfertaProducto" op ON op._id_aquiles = {_TABLE_NAME}.id_oferta_producto_aquiles::integer
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
                    results = [self._format_result(r) for r in rows]
                    facets = self._compute_facets(conn, tokens, filters)
                return {"results": results, "count": len(results), "query": query, "facets": facets, "currency": "CLP"}
            except OperationalError as e:
                logging.warning(f"[Medicamentos] Error de conexión (intento {attempt + 1}/{_DB_MAX_RETRIES}): {e}")
                _reset_engine()
                engine = _get_engine()
                if not engine:
                    break
            except Exception as e:
                logging.warning(f"[Medicamentos] Error en búsqueda: {e}")
                return {"results": [], "count": 0, "facets": {}}

        return {"results": [], "count": 0, "facets": {}}

    @staticmethod
    def _format_result(row) -> Dict[str, Any]:
        d = dict(row)
        for key in ("precio_p25", "precio_mediana", "precio_p75"):
            d[key] = int(round(d[key])) if d.get(key) is not None else None
        d["n_con_precio"] = int(d["n_con_precio"])
        d["n_compras"] = int(d["n_compras"])
        return d

    def _compute_facets(self, conn, tokens: List[str], filters: Dict[str, Optional[str]]) -> Dict[str, List[Dict]]:
        facets: Dict[str, List[Dict]] = {}
        for name, col in _FACET_COLS:
            f_where, f_params = _build_where(tokens, filters, exclude=name)
            f_sql = text(f"""
                SELECT {col} AS value, COUNT(*) AS n
                FROM "{_TABLE_NAME}"
                WHERE {f_where} AND {col} IS NOT NULL AND TRIM({col}) NOT IN ('', 'No especificado')
                GROUP BY {col}
                ORDER BY n DESC
                LIMIT :flimit
            """)
            f_params["flimit"] = _FACET_LIMIT
            try:
                rows = conn.execute(f_sql, f_params).mappings().all()
                facets[name] = [{"value": r["value"], "count": r["n"]} for r in rows]
            except Exception as e:
                logging.warning(f"[Medicamentos] Error calculando faceta '{name}': {e}")
                facets[name] = []
        return facets
