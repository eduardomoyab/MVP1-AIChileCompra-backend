"""
price_service.py

Estima el precio de un computador según la ficha técnica consultando
la tabla PrecioCA en PostgreSQL.

Estrategia: búsqueda exacta con todos los atributos disponibles.
Mínimo 5 registros para reportar estimación.
"""

import os
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

load_dotenv()

_DB_URL = os.getenv("DATABASE_URL")
_engine = None
_TABLE_NAME: Optional[str] = None

_DB_MAX_RETRIES = 3
_DB_RETRY_DELAY = 2

# Columnas de PrecioCA usadas para filtrar
PRICE_QUERY_COLS = [
    "tipo_equipo",
    "procesador_principal",
    "linea_procesador",
    "generacion_procesador",
    "nucleos_procesador",
    "total_ram_gb",
    "tecnologia_ram",
    "total_almacenamiento_gb",
    "tecnologia_disco_principal",
    "tipo_configuracion_discos",
    "tiene_gpu_dedicada",
    "marca",
    "sistema_operativo",
]

# Columnas disponibles para dropdowns: (nombre_ficha, columna_db)
DROPDOWN_DB_COLS = [
    ("marca",                  "marca"),
    ("procesador_principal",   "procesador_principal"),
    ("linea_procesador",       "linea_procesador"),
    ("total_ram_gb",           "total_ram_gb"),
    ("total_almacenamiento_gb","total_almacenamiento_gb"),
    ("sistema_operativo",      "sistema_operativo"),
    ("pantalla_pulgadas",      "pantalla_pulgadas"),
]


def _get_engine():
    global _engine
    if _engine is None and _DB_URL:
        _engine = create_engine(_DB_URL, pool_pre_ping=True, connect_args={"connect_timeout": 10})
    return _engine


def _reset_engine():
    global _engine, _TABLE_NAME
    if _engine:
        try:
            _engine.dispose()
        except Exception:
            pass
    _engine = None
    _TABLE_NAME = None


def _resolve_table_name() -> Optional[str]:
    global _TABLE_NAME
    if _TABLE_NAME is not None:
        return _TABLE_NAME

    candidates = ["PrecioCA", "precioca", "precio_ca", "PRECIOCA"]

    for attempt in range(_DB_MAX_RETRIES):
        engine = _get_engine()
        if not engine:
            return None
        try:
            with engine.connect() as conn:
                for name in candidates:
                    try:
                        conn.execute(text(f'SELECT 1 FROM "{name}" LIMIT 1'))
                        _TABLE_NAME = name
                        logging.info(f"Tabla PrecioCA encontrada: '{name}'")
                        return name
                    except Exception as e:
                        logging.warning(f"[DB] Candidato '{name}' no accesible: {e}")
                        continue
            break
        except OperationalError as e:
            logging.warning(f"[DB] Intento {attempt + 1}/{_DB_MAX_RETRIES} fallido: {e}")
            _reset_engine()
            if attempt < _DB_MAX_RETRIES - 1:
                time.sleep(_DB_RETRY_DELAY)
        except Exception as e:
            logging.warning(f"No se pudo conectar a la base de datos: {e}")
            break

    return None


_DB_ERROR = object()


def _percentile_query(table: str, where_clauses: list, params: dict):
    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
    sql = text(f"""
        SELECT
            COUNT(*)                                                                                        AS n,
            MIN(precio_unitario::numeric)                                                                   AS precio_min,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY precio_unitario::numeric)                         AS p25,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY precio_unitario::numeric)                         AS mediana,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY precio_unitario::numeric)                         AS p75,
            MAX(precio_unitario::numeric)                                                                   AS precio_max,
            ROUND(AVG(precio_unitario::numeric), 0)                                                        AS promedio,
            ROUND(AVG(precio_unitario_iva::numeric), 0)                                                    AS promedio_iva,
            MIN(precio_unitario_iva::numeric)                                                               AS precio_min_iva,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY precio_unitario_iva::numeric)                     AS p25_iva,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY precio_unitario_iva::numeric)                     AS mediana_iva,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY precio_unitario_iva::numeric)                     AS p75_iva,
            MAX(precio_unitario_iva::numeric)                                                               AS precio_max_iva
        FROM "{table}"
        WHERE
            LOWER(COALESCE(es_accesorio::text, 'false')) != 'true'
            AND precio_unitario::numeric > 200000
            AND precio_unitario::numeric < 5000000
            AND {where_sql}
    """)

    for attempt in range(_DB_MAX_RETRIES):
        try:
            engine = _get_engine()
            with engine.connect() as conn:
                row = conn.execute(sql, params).fetchone()
            count = int(row[0]) if row and row[0] else 0
            logging.info(f"[PrecioCA] {json.dumps(params, ensure_ascii=False)} → {count} registros")
            if count >= 1:
                return {
                    "count":      int(row[0]),
                    "min":        int(row[1])  if row[1]  else None,
                    "p25":        int(row[2])  if row[2]  else None,
                    "median":     int(row[3])  if row[3]  else None,
                    "p75":        int(row[4])  if row[4]  else None,
                    "max":        int(row[5])  if row[5]  else None,
                    "mean":       int(row[6])  if row[6]  else None,
                    "mean_iva":   int(row[7])  if row[7]  else None,
                    "min_iva":    int(row[8])  if row[8]  else None,
                    "p25_iva":    int(row[9])  if row[9]  else None,
                    "median_iva": int(row[10]) if row[10] else None,
                    "p75_iva":    int(row[11]) if row[11] else None,
                    "max_iva":    int(row[12]) if row[12] else None,
                }
            return None
        except OperationalError as e:
            logging.warning(f"[DB] Error de conexión (intento {attempt + 1}/{_DB_MAX_RETRIES}): {e}")
            _reset_engine()
            if attempt < _DB_MAX_RETRIES - 1:
                time.sleep(_DB_RETRY_DELAY)
        except Exception as e:
            logging.warning(f"Error en consulta de precio: {e}")
            return None

    logging.warning("[DB] Todos los reintentos fallaron por conexión")
    return _DB_ERROR


def _build_where(attrs: dict) -> Tuple[list, dict]:
    where_clauses = []
    params = {}

    for col, value in attrs.items():
        if value is None:
            continue

        is_range = isinstance(value, dict) and ("min" in value or "max" in value)
        is_list  = isinstance(value, list)

        if col == "tiene_gpu_dedicada":
            if is_list or is_range:
                continue
            where_clauses.append(f"{col} = :{col}")
            params[col] = "true" if value is True else "false" if value is False else str(value).lower()

        elif col in ("total_ram_gb", "total_almacenamiento_gb"):
            sp = f"SPLIT_PART({col}::text, ' ', 1)"
            if is_range:
                mn, mx = value.get("min"), value.get("max")
                parts = [f"{sp} ~ '^[0-9]+$'"]
                try:
                    if mn not in (None, ""):
                        parts.append(f"{sp}::numeric >= :{col}_min")
                        params[f"{col}_min"] = int(float(mn))
                    if mx not in (None, ""):
                        parts.append(f"{sp}::numeric <= :{col}_max")
                        params[f"{col}_max"] = int(float(mx))
                except (ValueError, TypeError):
                    continue
                if len(parts) > 1:
                    where_clauses.append(f"({' AND '.join(parts)})")
            elif is_list:
                try:
                    num_vals = [str(int(float(v))) for v in value if v not in (None, "")]
                    if not num_vals:
                        continue
                    in_params = {f"{col}_{i}": v for i, v in enumerate(num_vals)}
                    in_keys = ", ".join(f":{k}" for k in in_params)
                    where_clauses.append(f"({sp} ~ '^[0-9]+$' AND {sp} IN ({in_keys}))")
                    params.update(in_params)
                except (ValueError, TypeError):
                    continue
            else:
                try:
                    num_val = str(int(float(value)))
                    where_clauses.append(f"({sp} ~ '^[0-9]+$' AND {sp} = :{col})")
                    params[col] = num_val
                except (ValueError, TypeError):
                    continue

        elif col in ("linea_procesador", "generacion_procesador"):
            if is_list:
                if not value:
                    continue
                or_parts = []
                for i, v in enumerate(value):
                    k = f"{col}_{i}"
                    or_parts.append(f"{col} ILIKE :{k}")
                    params[k] = f"%{v}%"
                where_clauses.append(f"({' OR '.join(or_parts)})")
            elif is_range:
                continue
            else:
                where_clauses.append(f"{col} ILIKE :{col}")
                params[col] = f"%{value}%"

        else:
            if is_list:
                if not value:
                    continue
                in_params = {f"{col}_{i}": str(v) for i, v in enumerate(value)}
                in_keys = ", ".join(f":{k}" for k in in_params)
                where_clauses.append(f"{col} IN ({in_keys})")
                params.update(in_params)
            elif is_range:
                continue
            else:
                where_clauses.append(f"{col} = :{col}")
                params[col] = str(value)

    return where_clauses, params


class PriceService:
    def estimate(self, ficha: Dict[str, Any]) -> Optional[Dict]:
        table = _resolve_table_name()
        if not table:
            logging.warning("PriceService: tabla no disponible")
            return None

        if not ficha.get("tipo_equipo"):
            return None

        current_attrs = {
            col: ficha[col]
            for col in PRICE_QUERY_COLS
            if ficha.get(col) is not None
        }

        where_clauses, params = _build_where(current_attrs)
        if not where_clauses:
            return None

        result = _percentile_query(table, where_clauses, params)

        if result is _DB_ERROR:
            logging.warning("[PrecioCA] Error de conexión al estimar precio")
            return None

        if isinstance(result, dict):
            result["match_attrs"] = list(current_attrs.keys())
            result["match_level"] = 1
            result["match_description"] = _describe_attrs(current_attrs)
            result["currency"] = "CLP"
            return result

        return None

    def get_dropdown_values(self) -> Dict[str, List]:
        """Retorna valores distintos de PrecioCA para usar en dropdowns del frontend."""
        table = _resolve_table_name()
        if not table:
            return {}

        result = {}
        engine = _get_engine()
        if not engine:
            return {}

        _NUMERIC_COLS = {"total_ram_gb", "total_almacenamiento_gb", "pantalla_pulgadas"}

        def _numeric_key(v: str):
            try:
                return float(v.split()[0])
            except Exception:
                return float("inf")

        for field, col in DROPDOWN_DB_COLS:
            try:
                with engine.connect() as conn:
                    rows = conn.execute(text(f"""
                        SELECT DISTINCT {col}::text
                        FROM "{table}"
                        WHERE {col} IS NOT NULL
                          AND TRIM({col}::text) != ''
                          AND LOWER(COALESCE(es_accesorio::text,'false')) != 'true'
                    """)).fetchall()
                values = [r[0] for r in rows if r[0]]
                if col in _NUMERIC_COLS:
                    values.sort(key=_numeric_key)
                else:
                    values.sort()
                result[field] = values
            except Exception as e:
                logging.warning(f"[Dropdowns] Error en columna '{col}': {e}")
                result[field] = []

        return result

    def get_token(self) -> Optional[str]:
        engine = _get_engine()
        if not engine:
            return None
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT access_token FROM token_store ORDER BY id DESC LIMIT 1")
                ).fetchone()
            if not row:
                logging.warning("[token_store] No hay registros — no se pueden obtener OC codes")
                return None
            return row[0]
        except Exception as e:
            logging.warning(f"[token_store] No se pudo leer el token: {e}")
            return None

    def get_offer_rows(self, ficha: Dict[str, Any], limit: int = 30) -> List[Dict]:
        table = _resolve_table_name()
        if not table or not ficha.get("tipo_equipo"):
            return []
        current_attrs = {
            col: ficha[col]
            for col in PRICE_QUERY_COLS
            if ficha.get(col) is not None
        }
        where_clauses, params = _build_where(current_attrs)
        if not where_clauses:
            return []
        where_sql = " AND ".join(where_clauses)
        sql = text(f"""
            SELECT
                codigo_requerimiento,
                precio_unitario::numeric        AS precio_unitario,
                precio_unitario_iva::numeric    AS precio_unitario_iva,
                descripcion,
                fecha_modificacion::text
            FROM "{table}"
            WHERE
                LOWER(COALESCE(es_accesorio::text,'false')) != 'true'
                AND precio_unitario::numeric > 200000
                AND precio_unitario::numeric < 5000000
                AND {where_sql}
            ORDER BY fecha_modificacion DESC NULLS LAST
            LIMIT {limit}
        """)
        try:
            engine = _get_engine()
            with engine.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
            return [
                {
                    "codigo_requerimiento": row[0],
                    "precio_unitario":      int(row[1]) if row[1] else None,
                    "precio_unitario_iva":  int(row[2]) if row[2] else None,
                    "descripcion":          row[3],
                    "fecha_modificacion":   str(row[4])[:10] if row[4] else None,
                }
                for row in rows
            ]
        except Exception as e:
            logging.warning(f"[get_offer_rows] Error: {e}")
            return []


def _describe_attrs(attrs: dict) -> str:
    labels = {
        "tipo_equipo": "tipo",
        "procesador_principal": "procesador exacto",
        "linea_procesador": "línea proc.",
        "generacion_procesador": "generación proc.",
        "nucleos_procesador": "núcleos",
        "hilos_procesador": "hilos",
        "total_ram_gb": "RAM",
        "tecnologia_ram": "tecnología RAM",
        "total_almacenamiento_gb": "almacenamiento",
        "tecnologia_disco_principal": "tecnología disco",
        "tipo_configuracion_discos": "config. discos",
        "tiene_gpu_dedicada": "GPU dedicada",
        "marca": "marca",
        "sistema_operativo": "SO",
    }
    return ", ".join(labels.get(k, k) for k in attrs)


def diagnostico():
    table = _resolve_table_name()
    if not table:
        logging.warning("[Diagnóstico] Tabla no encontrada")
        return
    engine = _get_engine()
    try:
        with engine.connect() as conn:
            total = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
            logging.info(f"[Diagnóstico] Total registros en {table}: {total}")
    except Exception as e:
        logging.error(f"[Diagnóstico] Error: {e}")
