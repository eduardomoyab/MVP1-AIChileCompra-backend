"""
medicamento_service.py

Servicio de datos para el selector guiado de medicamentos, contra la vista
public."PrecioMedCA" (extractor_medicamentos LEFT JOIN OfertaProducto LEFT
JOIN Oferta, ver vista_MedCA.sql -- join 1:1, 212.302 filas, 100% con
precio y fecha).

A diferencia de la versión anterior (buscador de texto libre sobre el
historial), acá el reconocimiento de texto libre lo hace medicamento_agent.py
(LLM) -- este servicio ya no recibe tokens de búsqueda, solo **filtros
estructurados** ya resueltos (principio_activo, forma_farmaceutica,
concentracion, laboratorio), y expone tres cosas:

  - estimate_price(filters): un solo rango de percentiles (p25/mediana/p75)
    sobre TODO lo que calza con los filtros actuales -- el panel de precio
    en vivo del selector guiado.
  - get_facet_values(attr, filters, exclude): valores reales + conteo de UN
    atributo, acotados por lo que ya está filtrado -- alimenta tanto las
    "sugerencias inteligentes" (ej. concentraciones típicas de un principio
    activo) como el dropdown del editor manual de cada atributo.
  - get_historial(filters, sort, limit): compras anteriores agrupadas por
    producto (mismo criterio de agrupación que antes: normalizado por
    TRIM+UPPER, con conteo de compras), acotadas por los filtros ya
    elegidos -- es el acordeón opcional "Ver compras anteriores", ya no la
    pantalla principal.

`principio_activo` y `concentracion` viven en dos columnas de la BD
(principio_activo_1/2, concentracion_1/2 -- medicamentos combinados) y se
tratan como una sola dimensión lógica: cualquier filtro por esos atributos
calza contra CUALQUIERA de las dos columnas.
"""

import os
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

load_dotenv()

_DB_URL = os.getenv("DATABASE_URL")
_TABLE_NAME = "PrecioMedCA"
_engine = None

_DB_MAX_RETRIES = 3
_DB_RETRY_DELAY = 2

_RESULT_COLS = [
    "nombre_producto", "laboratorio", "principio_activo_1", "principio_activo_2",
    "forma_farmaceutica", "concentracion_1", "concentracion_2",
    "cantidad", "unidad_venta",
]

_FACET_LIMIT = 12
_MAX_CODIGOS_REQUERIMIENTO = 5

_SORT_OPTIONS = {
    "fecha_desc":  "fecha_reciente DESC NULLS LAST, n_compras DESC",
    "precio_asc":  "precio_mediana ASC NULLS LAST, n_compras DESC",
    "precio_desc": "precio_mediana DESC NULLS LAST, n_compras DESC",
}
_DEFAULT_SORT = "fecha_desc"


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


def _norm_sql(col: str) -> str:
    return f"NULLIF(UPPER(TRIM(COALESCE({col}, ''))), 'NO ESPECIFICADO')"


_equivalentes_unidad_cache: Optional[set] = None


def _get_equivalentes_unidad() -> set:
    """Nombres de unidad_venta estadísticamente equivalentes a "Unidad"
    (ver public.unidad_medida_catalogo.equivalente_a_unidad -- calculado
    comparando medianas de precio por combinación principio_activo+forma_farmaceutica;
    ej. Comprimido sí, Caja no -- ratio ~23x). Cacheado en memoria porque el
    catálogo es chico (87 filas) y casi no cambia -- evita una consulta
    extra en cada filtro."""
    global _equivalentes_unidad_cache
    if _equivalentes_unidad_cache is None:
        engine = _get_engine()
        if not engine:
            return set()
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT nombre FROM public.unidad_medida_catalogo WHERE equivalente_a_unidad"
                )).fetchall()
            _equivalentes_unidad_cache = {r[0] for r in rows}
        except Exception as e:
            logging.warning(f"[Medicamentos] Error cargando unidad_medida_catalogo: {e}")
            return set()
    return _equivalentes_unidad_cache


_GROUP_COLS_SQL = ", ".join(_norm_sql(c) for c in _RESULT_COLS)


def _filter_clauses(filters: Dict[str, Any], exclude: Optional[str] = None) -> Tuple[List[str], Dict[str, Any]]:
    """filters: {"principio_activo": str|list|None, "forma_farmaceutica": str|None,
    "concentracion": str|None, "laboratorio": str|None}.
    exclude: nombre de atributo a NO filtrar (para calcular la faceta de ESE
    mismo atributo sin que su propia selección la reduzca a una sola opción)."""
    clauses: List[str] = []
    params: Dict[str, Any] = {}

    if exclude != "principio_activo" and filters.get("principio_activo"):
        # AND entre los principios seleccionados (deben estar TODOS en la
        # misma fila -- ej. "Amoxicilina + Ácido clavulánico" es un producto
        # combinado real, no dos productos distintos). Antes era OR, lo que
        # diluía precio/historial con compras de un solo principio activo
        # (ej. "Paracetamol solo" contaminando la estimación de un combo de
        # 3 principios donde ninguna fila real los tiene juntos).
        vals = filters["principio_activo"]
        vals = vals if isinstance(vals, list) else [vals]
        for i, v in enumerate(vals):
            if not v:
                continue
            k = f"pa_{i}"
            clauses.append(f"(principio_activo_1 ILIKE :{k} OR principio_activo_2 ILIKE :{k})")
            params[k] = f"%{v}%"

    if exclude != "forma_farmaceutica" and filters.get("forma_farmaceutica"):
        clauses.append("forma_farmaceutica ILIKE :ff")
        params["ff"] = f"%{filters['forma_farmaceutica']}%"

    if exclude != "concentracion" and filters.get("concentracion"):
        clauses.append("(concentracion_1 ILIKE :conc OR concentracion_2 ILIKE :conc)")
        params["conc"] = f"%{filters['concentracion']}%"

    if exclude != "laboratorio" and filters.get("laboratorio"):
        clauses.append("laboratorio ILIKE :lab")
        params["lab"] = f"%{filters['laboratorio']}%"

    if exclude != "unidad_venta" and filters.get("unidad_venta"):
        # OR entre los valores que terminen aplicando (a diferencia de
        # principio_activo, que es AND) -- acá significa "cualquiera de
        # estas etiquetas es aceptable para la misma unidad real". Además
        # de lo que el usuario eligió, se agrega automáticamente "Unidad"
        # cuando la unidad elegida es estadísticamente equivalente (ver
        # unidad_medida_catalogo.equivalente_a_unidad -- ej. Comprimido
        # sí se expande a incluir Unidad, Caja NO). Exacto, no ILIKE
        # substring -- unidad_venta ya viene normalizada contra un
        # catálogo cerrado (public.unidad_medida_catalogo, ver
        # vista_MedCA.sql), y varios pares chocan por substring si no
        # (ej. "Caja" es substring de "Cajetilla"; "Frasco" de "Frasco
        # Ampolla").
        vals = filters["unidad_venta"]
        vals = vals if isinstance(vals, list) else [vals]
        equivalentes = _get_equivalentes_unidad()
        expanded = set()
        for v in vals:
            if not v:
                continue
            expanded.add(v)
            if v in equivalentes:
                expanded.add("Unidad")
        or_parts = []
        for i, v in enumerate(expanded):
            k = f"uv_{i}"
            or_parts.append(f"UPPER(unidad_venta) = UPPER(:{k})")
            params[k] = v
        if or_parts:
            clauses.append("(" + " OR ".join(or_parts) + ")")

    return clauses, params


def _row_id(row: Dict[str, Any]) -> str:
    key = "|".join(str(row.get(c) or "").strip().upper() for c in _RESULT_COLS)
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]


class MedicamentoService:
    def estimate_price(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Un solo rango de percentiles sobre todo lo que calza con los
        filtros actuales -- requiere principio_activo Y unidad_venta (el
        precio no es comparable entre unidades: Metformina mediana $74 por
        Comprimido vs $23.205 por Caja, ~314x -- sin unidad_venta el
        percentil mezclaría cosas no comparables)."""
        if not filters.get("principio_activo") or not filters.get("unidad_venta"):
            return None

        engine = _get_engine()
        if not engine:
            return None

        clauses, params = _filter_clauses(filters)
        where_sql = " AND ".join(["nombre_producto IS NOT NULL"] + clauses)

        sql = text(f"""
            SELECT
                COUNT(*) AS n,
                COUNT(precio_unitario_iva) AS n_con_precio,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY precio_unitario_iva) AS p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY precio_unitario_iva) AS mediana,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY precio_unitario_iva) AS p75,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY precio_unitario) AS p25_neto,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY precio_unitario) AS mediana_neto,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY precio_unitario) AS p75_neto
            FROM "{_TABLE_NAME}"
            WHERE {where_sql}
        """)

        for attempt in range(_DB_MAX_RETRIES):
            try:
                with engine.connect() as conn:
                    row = conn.execute(sql, params).mappings().first()
                break
            except OperationalError as e:
                logging.warning(f"[Medicamentos] Error de conexión estimate_price (intento {attempt + 1}): {e}")
                _reset_engine()
                engine = _get_engine()
                if not engine:
                    return None
            except Exception as e:
                logging.warning(f"[Medicamentos] Error en estimate_price: {e}")
                return None
        else:
            return None

        if not row or not row["n_con_precio"]:
            return None

        return {
            "count": int(row["n"]),
            "n_con_precio": int(row["n_con_precio"]),
            "p25": int(round(row["p25"])),
            "mediana": int(round(row["mediana"])),
            "p75": int(round(row["p75"])),
            "p25_neto": int(round(row["p25_neto"])) if row["p25_neto"] is not None else None,
            "mediana_neto": int(round(row["mediana_neto"])) if row["mediana_neto"] is not None else None,
            "p75_neto": int(round(row["p75_neto"])) if row["p75_neto"] is not None else None,
            "currency": "CLP",
        }

    def get_facet_values(
        self,
        attr: str,
        filters: Dict[str, Any],
        exclude: Optional[str] = None,
        limit: int = _FACET_LIMIT,
    ) -> List[Dict[str, Any]]:
        """Valores reales + conteo de UN atributo, acotados por los demás
        filtros ya elegidos -- alimenta sugerencias y el editor manual."""
        engine = _get_engine()
        if not engine:
            return []

        clauses, params = _filter_clauses(filters, exclude=exclude if exclude is not None else attr)
        where_sql = " AND ".join(["nombre_producto IS NOT NULL"] + clauses)
        params["flimit"] = limit

        if attr == "principio_activo":
            sql = text(f"""
                SELECT (MODE() WITHIN GROUP (ORDER BY valor)) AS valor, COUNT(*) AS n FROM (
                    SELECT principio_activo_1 AS valor FROM "{_TABLE_NAME}" WHERE {where_sql}
                    UNION ALL
                    SELECT principio_activo_2 AS valor FROM "{_TABLE_NAME}" WHERE {where_sql}
                ) u
                WHERE valor IS NOT NULL AND TRIM(valor) NOT IN ('', 'No especificado')
                GROUP BY UPPER(unaccent(TRIM(valor))) ORDER BY n DESC LIMIT :flimit
            """)
        elif attr == "concentracion":
            sql = text(f"""
                SELECT (MODE() WITHIN GROUP (ORDER BY valor)) AS valor, COUNT(*) AS n FROM (
                    SELECT concentracion_1 AS valor FROM "{_TABLE_NAME}" WHERE {where_sql}
                    UNION ALL
                    SELECT concentracion_2 AS valor FROM "{_TABLE_NAME}" WHERE {where_sql}
                ) u
                WHERE valor IS NOT NULL AND TRIM(valor) NOT IN ('', 'No especificado')
                GROUP BY UPPER(unaccent(TRIM(valor))) ORDER BY n DESC LIMIT :flimit
            """)
        elif attr in ("forma_farmaceutica", "laboratorio", "unidad_venta", "cantidad"):
            # laboratorio tiene muchísimas variantes de un mismo fabricante
            # escritas de forma distinta más allá de mayúsculas/acentos (ej.
            # "Ethon" vs "Ethon Pharmaceuticals S.p.A.") -- probé normalizar
            # agrupando por sufijo corporativo (S.A., Ltda., Pharmaceutical,
            # etc.) pero contra la BD real quedaban grupos peor formados
            # (puntos sueltos, fusiones parciales) que el problema original,
            # así que se descartó. En su lugar, para laboratorio se exige
            # apariciones >= 2 -- saca el ruido de variantes que aparecen
            # una sola vez, sin arriesgar una fusión incorrecta de entidades.
            having_sql = "HAVING COUNT(*) >= 2" if attr == "laboratorio" else ""
            sql = text(f"""
                SELECT (MODE() WITHIN GROUP (ORDER BY {attr})) AS valor, COUNT(*) AS n
                FROM "{_TABLE_NAME}"
                WHERE {where_sql} AND {attr} IS NOT NULL AND TRIM({attr}) NOT IN ('', 'No especificado')
                GROUP BY UPPER(unaccent(TRIM({attr})))
                {having_sql}
                ORDER BY n DESC LIMIT :flimit
            """)
        else:
            return []

        try:
            with engine.connect() as conn:
                rows = conn.execute(sql, params).mappings().all()
            return [{"value": r["valor"], "count": int(r["n"])} for r in rows]
        except Exception as e:
            logging.warning(f"[Medicamentos] Error calculando faceta '{attr}': {e}")
            return []

    def get_price_by_unidad(
        self,
        filters: Dict[str, Any],
        limit: int = _FACET_LIMIT,
    ) -> List[Dict[str, Any]]:
        """Precio (p25/mediana/p75) agrupado por unidad_venta, para el paso
        de elegir unidad -- ahí es donde tiene sentido comparar (ej.
        "Comprimido $56 c/u" vs "Caja $15.417 c/u"), a diferencia de
        estimate_price() que exige una unidad ya elegida para no mezclar
        cosas no comparables. No aplica el merge de "Unidad" equivalente
        (unidad_medida_catalogo.equivalente_a_unidad) -- acá "Unidad" se
        muestra como su propia fila; ese merge solo tiene sentido una vez
        que ya se sabe a qué forma concreta corresponde (ver _filter_clauses)."""
        if not filters.get("principio_activo"):
            return []

        engine = _get_engine()
        if not engine:
            return []

        clauses, params = _filter_clauses(filters, exclude="unidad_venta")
        where_sql = " AND ".join(
            ["nombre_producto IS NOT NULL", "unidad_venta IS NOT NULL",
             "TRIM(unidad_venta) NOT IN ('', '000')"] + clauses
        )
        params["flimit"] = limit

        sql = text(f"""
            SELECT unidad_venta AS valor,
                   COUNT(*) AS n,
                   PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY precio_unitario_iva) AS p25,
                   PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY precio_unitario_iva) AS mediana,
                   PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY precio_unitario_iva) AS p75
            FROM "{_TABLE_NAME}"
            WHERE {where_sql}
            GROUP BY unidad_venta
            HAVING COUNT(precio_unitario_iva) >= 3
            ORDER BY n DESC
            LIMIT :flimit
        """)

        try:
            with engine.connect() as conn:
                rows = conn.execute(sql, params).mappings().all()
            return [{
                "value": r["valor"],
                "count": int(r["n"]),
                "p25": int(round(r["p25"])) if r["p25"] is not None else None,
                "mediana": int(round(r["mediana"])) if r["mediana"] is not None else None,
                "p75": int(round(r["p75"])) if r["p75"] is not None else None,
            } for r in rows]
        except Exception as e:
            logging.warning(f"[Medicamentos] Error en get_price_by_unidad: {e}")
            return []

    def get_companion_values(
        self,
        filters: Dict[str, Any],
        limit: int = _FACET_LIMIT,
    ) -> List[Dict[str, Any]]:
        """Solo para principio_activo (list-capable -- medicamentos
        combinados como Amoxicilina + Ácido clavulánico): dado el/los
        principio(s) activo(s) ya elegido(s), sugiere OTROS principios
        activos que aparecen en la MISMA fila de compra -- a diferencia de
        get_facet_values (que EXCLUYE el propio filtro para no
        autolimitarse a lo ya elegido), acá el filtro de principio_activo
        SÍ se aplica -- lo que se excluye es el/los valor(es) ya elegidos
        de la lista de sugerencias, para no sugerir "agregar Levotiroxina"
        a una búsqueda que ya tiene Levotiroxina."""
        if not filters.get("principio_activo"):
            return []

        engine = _get_engine()
        if not engine:
            return []

        clauses, params = _filter_clauses(filters)  # sin exclude: el filtro de principio_activo aplica
        where_sql = " AND ".join(["nombre_producto IS NOT NULL"] + clauses)
        params["flimit"] = limit

        selected = filters["principio_activo"]
        selected_vals = selected if isinstance(selected, list) else [selected]
        exclude_clauses = []
        for i, v in enumerate(selected_vals):
            if not v:
                continue
            k = f"excl_{i}"
            exclude_clauses.append(f"valor NOT ILIKE :{k}")
            params[k] = f"%{v}%"
        exclude_sql = " AND ".join(exclude_clauses) if exclude_clauses else "TRUE"

        sql = text(f"""
            SELECT (MODE() WITHIN GROUP (ORDER BY valor)) AS valor, COUNT(*) AS n FROM (
                SELECT principio_activo_1 AS valor FROM "{_TABLE_NAME}" WHERE {where_sql}
                UNION ALL
                SELECT principio_activo_2 AS valor FROM "{_TABLE_NAME}" WHERE {where_sql}
            ) u
            WHERE valor IS NOT NULL AND TRIM(valor) NOT IN ('', 'No especificado') AND {exclude_sql}
            GROUP BY UPPER(unaccent(TRIM(valor))) ORDER BY n DESC LIMIT :flimit
        """)

        try:
            with engine.connect() as conn:
                rows = conn.execute(sql, params).mappings().all()
            return [{"value": r["valor"], "count": int(r["n"])} for r in rows]
        except Exception as e:
            logging.warning(f"[Medicamentos] Error calculando principios activos combinados: {e}")
            return []

    def get_historial(
        self,
        filters: Dict[str, Any],
        sort: str = _DEFAULT_SORT,
        limit: int = 30,
    ) -> Dict[str, Any]:
        """Compras anteriores agrupadas por producto, acotadas por los
        filtros ya elegidos en el selector -- el acordeón opcional, ya no
        la búsqueda principal. Exige unidad_venta igual que estimate_price
        (mismo motivo: el precio no es comparable entre unidades)."""
        engine = _get_engine()
        if not engine or not filters.get("principio_activo") or not filters.get("unidad_venta"):
            return {"results": [], "count": 0}

        clauses, params = _filter_clauses(filters)
        where_sql = " AND ".join(["nombre_producto IS NOT NULL"] + clauses)
        order_sql = _SORT_OPTIONS.get(sort, _SORT_OPTIONS[_DEFAULT_SORT])

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
                MAX(unidad_venta)       AS unidad_venta,
                COUNT(*)                AS n_compras,
                COUNT(precio_unitario_iva) AS n_con_precio,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY precio_unitario_iva) AS precio_p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY precio_unitario_iva) AS precio_mediana,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY precio_unitario_iva) AS precio_p75,
                MAX(fecha_oferta)      AS fecha_reciente,
                (ARRAY_AGG(DISTINCT codigo_requerimiento)
                    FILTER (WHERE codigo_requerimiento IS NOT NULL))[1:{_MAX_CODIGOS_REQUERIMIENTO}] AS codigos_requerimiento
            FROM "{_TABLE_NAME}"
            WHERE {where_sql}
            GROUP BY {_GROUP_COLS_SQL}
            ORDER BY {order_sql}
            LIMIT :limit
        """)

        try:
            with engine.connect() as conn:
                rows = conn.execute(sql, {**params, "limit": limit}).mappings().all()
            results = [self._format_historial_row(r) for r in rows]
            return {"results": results, "count": len(results), "sort": sort}
        except Exception as e:
            logging.warning(f"[Medicamentos] Error en get_historial: {e}")
            return {"results": [], "count": 0}

    @staticmethod
    def _format_historial_row(row) -> Dict[str, Any]:
        d = dict(row)
        for key in ("precio_p25", "precio_mediana", "precio_p75"):
            d[key] = int(round(d[key])) if d.get(key) is not None else None
        d["n_con_precio"] = int(d["n_con_precio"])
        d["n_compras"] = int(d["n_compras"])
        d["codigos_requerimiento"] = list(d.get("codigos_requerimiento") or [])
        d["anio"] = d["fecha_reciente"].year if d.get("fecha_reciente") else None
        d["fecha_reciente"] = str(d["fecha_reciente"]) if d.get("fecha_reciente") else None
        d["id"] = _row_id(d)
        return d
