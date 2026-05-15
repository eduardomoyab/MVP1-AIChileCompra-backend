"""
price_service.py

Estima el precio de un computador según la ficha técnica consultando
la tabla PrecioCA en PostgreSQL.

Estrategia LLM-driven:
  1. Query inicial con todos los atributos disponibles en la ficha
     (incluye linea_procesador, generacion_procesador del complemento)
  2. Si < 5 registros → pide al LLM una variación inteligente:
       - puede reemplazar procesador_principal por linea/generacion/nucleos
       - puede saltar al tier estándar correcto de RAM/almacenamiento
       - puede quitar 1, 2 o más atributos
       - puede combinar cualquiera de lo anterior
  3. Repite hasta 5 iteraciones antes de retornar None

Mínimo 5 registros para reportar estimación.
"""

import os
import re
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import openai
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

load_dotenv()

_DB_URL = os.getenv("DATABASE_URL")
_engine = None
_TABLE_NAME: Optional[str] = None
_openai_client = None

_DB_MAX_RETRIES = 3
_DB_RETRY_DELAY = 2  # segundos entre reintentos

# Columnas de PrecioCA que usaremos para filtrar (en orden de especificidad)
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


def _get_engine():
    global _engine
    if _engine is None and _DB_URL:
        _engine = create_engine(_DB_URL, pool_pre_ping=True, connect_args={"connect_timeout": 10})
    return _engine


def _reset_engine():
    """Descarta el pool de conexiones para forzar reconexión limpia."""
    global _engine, _TABLE_NAME
    if _engine:
        try:
            _engine.dispose()
        except Exception:
            pass
    _engine = None
    _TABLE_NAME = None  # también resetea la tabla para que revalide


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


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
                    except Exception:
                        continue
            break  # conectó pero no encontró la tabla
        except OperationalError as e:
            logging.warning(
                f"[DB] Intento {attempt + 1}/{_DB_MAX_RETRIES} fallido: {e}"
            )
            _reset_engine()
            if attempt < _DB_MAX_RETRIES - 1:
                time.sleep(_DB_RETRY_DELAY)
        except Exception as e:
            logging.warning(f"No se pudo conectar a la base de datos: {e}")
            break

    return None


_DB_ERROR = object()  # Sentinel: la query no corrió por error de conexión


def _percentile_query(table: str, where_clauses: list, params: dict):
    """
    Ejecuta la consulta de estadísticas de precio con retry ante caídas.
    Retorna:
      dict  — resultado con >= 5 registros
      None  — query ejecutada correctamente pero < 5 registros
      _DB_ERROR — no se pudo conectar tras todos los reintentos
    """
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
            if count >= 5:
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
            return None  # query ok pero count < 5
        except OperationalError as e:
            logging.warning(
                f"[DB] Error de conexión (intento {attempt + 1}/{_DB_MAX_RETRIES}): {e}"
            )
            _reset_engine()
            if attempt < _DB_MAX_RETRIES - 1:
                time.sleep(_DB_RETRY_DELAY)
        except Exception as e:
            logging.warning(f"Error en consulta de precio: {e}")
            return None  # error de SQL (sintaxis, tipos, etc.) — no reintentar

    logging.warning("[DB] Todos los reintentos fallaron por conexión")
    return _DB_ERROR


def _build_where(attrs: dict) -> Tuple[list, dict]:
    """Construye WHERE clauses y params a partir del dict de atributos."""
    where_clauses = []
    params = {}

    for col, value in attrs.items():
        if value is None:
            continue

        if col == "tiene_gpu_dedicada":
            where_clauses.append(f"{col} = :{col}")
            if isinstance(value, bool):
                params[col] = "true" if value else "false"
            else:
                params[col] = str(value).lower()

        elif col in ("total_ram_gb", "total_almacenamiento_gb"):
            # Almacenado como '8 GB', '256 GB', etc. Match exacto al número entero.
            # Algunas filas tienen valores no numéricos ('No', etc.), guard con regex.
            try:
                num_val = str(int(float(value)))
                clause = (
                    f"(SPLIT_PART({col}::text, ' ', 1) ~ '^[0-9]+$' "
                    f"AND SPLIT_PART({col}::text, ' ', 1) = :{col})"
                )
                where_clauses.append(clause)
                params[col] = num_val
            except (ValueError, TypeError):
                continue

        elif col in ("linea_procesador", "generacion_procesador"):
            # ILIKE con wildcards: 'Core i7' matchea 'Intel Core i7-13700H', 'core i7', etc.
            where_clauses.append(f"{col} ILIKE :{col}")
            params[col] = f"%{value}%"

        else:
            where_clauses.append(f"{col} = :{col}")
            params[col] = str(value)

    return where_clauses, params


_LINEA_PATTERNS = [
    # Intel
    (r"Core\s+(i\d)", lambda m: f"Core {m.group(1)}"),   # Core i5, Core i7, Core i9
    (r"Pentium",      lambda m: "Pentium"),
    (r"Celeron",      lambda m: "Celeron"),
    (r"Xeon",         lambda m: "Xeon"),
    # AMD
    (r"Ryzen\s+(\d)",  lambda m: f"Ryzen {m.group(1)}"),  # Ryzen 5, Ryzen 7, Ryzen 9
    (r"Athlon",        lambda m: "Athlon"),
    # Apple
    (r"Apple\s+(M\d)", lambda m: f"Apple {m.group(1)}"),
]


def _extract_linea(procesador: str) -> Optional[str]:
    """Extrae la línea del procesador desde su nombre completo."""
    for pattern, formatter in _LINEA_PATTERNS:
        m = re.search(pattern, procesador, re.IGNORECASE)
        if m:
            return formatter(m)
    return None


def _llm_next_query(ficha_full: dict, failed_queries: List[dict]) -> Optional[dict]:
    """
    El LLM elige QUÉ atributos incluir, no qué valores.
    Los valores SIEMPRE vienen de la ficha — así no puede inventar nada.
    """
    # Solo los atributos que tenemos disponibles en la ficha
    available = {k: ficha_full[k] for k in PRICE_QUERY_COLS if ficha_full.get(k) is not None}

    # Si linea_procesador no está en la ficha pero sí procesador_principal, extraerla
    if "linea_procesador" not in available and available.get("procesador_principal"):
        linea = _extract_linea(str(available["procesador_principal"]))
        if linea:
            available["linea_procesador"] = linea
            logging.info(f"[PrecioCA] linea_procesador extraída del nombre: '{linea}'")

    failed_sets = [
        sorted(q.keys()) for q in failed_queries
    ]
    failed_str = "\n".join(
        f"{i + 1}. {sorted(q.keys())} → < 5 registros"
        for i, q in enumerate(failed_queries)
    )

    prompt = f"""Eres un experto en bases de datos de precios de computadores.
Tienes que elegir qué atributos usar en la próxima consulta SQL para encontrar precios.

Atributos disponibles con sus valores (úsalos EXACTAMENTE así, no cambies los valores):
{json.dumps(available, ensure_ascii=False, indent=2)}

Combinaciones ya probadas que fallaron (< 5 registros):
{failed_str}

JERARQUÍA de especificidad del procesador (de mayor a menor):
  procesador_principal > linea_procesador + generacion_procesador > nucleos_procesador + hilos_procesador > (ninguno)

REGLAS:
1. tipo_equipo SIEMPRE debe estar incluido.
2. Si procesador_principal está en los fallidos, el SIGUIENTE paso es probar con linea_procesador y/o generacion_procesador (si existen en los atributos disponibles). No saltes directo a quitar todo el procesador.
3. Solo omite toda info de procesador si linea_procesador y generacion_procesador también fallaron.
4. tecnologia_ram es restrictiva — quítala antes que total_ram_gb.
5. No repitas ninguna combinación ya probada.
6. Elige el subconjunto más específico posible que probablemente tenga registros.

Devuelve SOLO JSON con la lista de atributos a incluir:
{{"include": ["tipo_equipo", ...], "razon": "por qué este subconjunto"}}"""

    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=300,
        )
        data = json.loads(response.choices[0].message.content)
        include_keys = data.get("include", [])
        razon = data.get("razon", "")

        if "tipo_equipo" not in include_keys:
            return None

        # Construir attrs usando SOLO valores de la ficha (no valores del LLM)
        next_attrs = {k: available[k] for k in include_keys if k in available}
        if not next_attrs.get("tipo_equipo"):
            return None

        # Verificar que no repita un set ya fallido
        next_keys_sorted = sorted(next_attrs.keys())
        if next_keys_sorted in failed_sets:
            logging.info(f"[PrecioCA] LLM repitió combinación fallida: {next_keys_sorted}")
            return None

        logging.info(f"[PrecioCA] LLM sugiere: {next_keys_sorted} — {razon}")
        return next_attrs
    except Exception as e:
        logging.warning(f"[PrecioCA] LLM reformulation failed: {e}")
        return None


def _attrs_equal(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


class PriceService:
    def estimate(self, ficha: Dict[str, Any]) -> Optional[Dict]:
        """
        Estima precio para la ficha dada usando estrategia LLM-driven.
        Retorna dict con estadísticas o None si no hay suficientes datos.
        """
        table = _resolve_table_name()
        if not table:
            logging.warning("PriceService: tabla no disponible")
            return None

        if not ficha.get("tipo_equipo"):
            return None

        # Atributos iniciales: todos los relevantes en la ficha (incluye complementos)
        current_attrs = {
            col: ficha[col]
            for col in PRICE_QUERY_COLS
            if ficha.get(col) is not None
        }

        failed_queries: List[dict] = []

        for iteration in range(6):
            where_clauses, params = _build_where(current_attrs)
            if not where_clauses:
                break

            result = _percentile_query(table, where_clauses, params)

            if result is _DB_ERROR:
                # La BD no respondió — no tiene sentido reformular la query
                logging.warning("[PrecioCA] Abortando estimación por error de conexión")
                return None

            if isinstance(result, dict):
                result["match_attrs"] = list(current_attrs.keys())
                result["match_level"] = iteration + 1
                result["match_description"] = _describe_attrs(current_attrs)
                result["currency"] = "CLP"
                return result

            # result is None → query corrió pero < 5 registros → pedir variación al LLM
            failed_queries.append(dict(current_attrs))
            logging.info(f"[PrecioCA] Iteración {iteration + 1}: < 5 registros, consultando LLM")

            if len(current_attrs) <= 1:
                break

            next_attrs = _llm_next_query(ficha, failed_queries)
            if not next_attrs:
                break

            current_attrs = next_attrs

        return None


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
    """Llama esto una vez en startup para verificar la BD."""
    table = _resolve_table_name()
    if not table:
        logging.warning("[Diagnóstico] Tabla no encontrada")
        return
    engine = _get_engine()
    try:
        with engine.connect() as conn:
            total = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
            logging.info(f"[Diagnóstico] Total registros en {table}: {total}")

            vals = conn.execute(text(f'SELECT DISTINCT es_accesorio FROM "{table}" LIMIT 10')).fetchall()
            logging.info(f"[Diagnóstico] Valores de es_accesorio: {[r[0] for r in vals]}")

            tipos = conn.execute(text(f'SELECT DISTINCT tipo_equipo FROM "{table}" LIMIT 10')).fetchall()
            logging.info(f"[Diagnóstico] Valores de tipo_equipo: {[r[0] for r in tipos]}")

            rams = conn.execute(text(f'SELECT DISTINCT total_ram_gb FROM "{table}" LIMIT 15')).fetchall()
            logging.info(f"[Diagnóstico] Valores de total_ram_gb: {[r[0] for r in rams]}")

            discos = conn.execute(text(f'SELECT DISTINCT total_almacenamiento_gb FROM "{table}" LIMIT 15')).fetchall()
            logging.info(f"[Diagnóstico] Valores de total_almacenamiento_gb: {[r[0] for r in discos]}")

    except Exception as e:
        logging.error(f"[Diagnóstico] Error: {e}")
