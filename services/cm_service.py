"""
cm_service.py

Estima precios contra PrecioCM (vista en Postgres/public que expone el
catálogo VIGENTE de Convenio Marco, precios en USD). A diferencia de
PrecioCA, acá:

  - Es un catálogo chico (decenas de productos), no un histórico de
    transacciones: se cachea completo en memoria (TTL) y el matching se
    hace en Python, no con SQL dinámico.
  - PrecioCM se arma (ver vista_CM.sql) sobre cm_computadores.
    productos_extraccion, extraída con el mismo esquema de atributos que
    PrecioCA -- linea_procesador, tecnologia_ram, tiene_gpu_dedicada, etc.
    ya vienen razonablemente normalizados. RAM/almacenamiento/núcleos igual
    se parsean con regex acá porque en la vista son texto libre ("8 GB");
    para el procesador se prefiere el linea_procesador de la vista y, si
    faltara, se usa una familia por reglas (processor_family.py) de respaldo.
    Se probó un fallback semántico (embeddings) para cuando no hay match
    de familia, pero contra el catálogo real los scores de similitud
    (0.37–0.68) no separan por tier de CPU — ej. "Intel Core i5 1250P"
    (premium) queda más "similar" a "Pentium Gold 7505" que "Celeron
    N150" (gama real equivalente). Un modelo de embeddings genérico no
    entiende jerarquía de rendimiento de procesadores, solo similitud
    textual superficial. Por eso, sin match de familia se relaja primero
    al grupo de líneas equivalentes en la misma gama (gama_service.py:
    Básica/Media/Alta/Premium, sin distinción de marca -- ej. pedido
    "AMD Ryzen 5" sin match relaja a también aceptar "Intel Core i5",
    "Apple M1 Pro", etc., todas gama Media). Esto evita mostrar solo lo
    que más abunda en el catálogo por marca cuando no hay match exacto.
    Si tampoco hay nada en esa gama, recién ahí se suelta el filtro de
    CPU por completo y se muestran todos los candidatos estructurales
    (tipo/RAM/almacenamiento) ordenados por precio — con ~93 productos
    en el catálogo, es fácil de escanear a simple vista.
  - Los precios están en USD: se convierten a CLP con currency_service.
"""

import os
import re
import time
import logging
import statistics
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

from agents.processor_family import extract_linea_procesador
from services import gama_service
from services.currency_service import get_usd_clp

load_dotenv()

_DB_URL = os.getenv("DATABASE_URL")
_TABLE_NAME = "PrecioCM"
_CACHE_TTL_SECONDS = int(os.getenv("CM_CACHE_TTL_SECONDS", str(30 * 60)))

_DB_MAX_RETRIES = 3
_DB_RETRY_DELAY = 2

_engine = None
_rows_cache: Optional[List[Dict]] = None
_rows_cache_ts: float = 0.0


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


def _parse_ram_gb(raw: Any) -> Optional[float]:
    if not raw:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(raw))
    return float(m.group(1)) if m else None


def _parse_storage_gb(raw: Any) -> Optional[float]:
    if not raw:
        return None
    s = str(raw).upper()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(GB|TB)?", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2) or "GB"
    return val * 1000 if unit == "TB" else val


def _clean_enum(raw: Any) -> Optional[str]:
    # productos_extraccion deja "No especificado" (mismo criterio que
    # extractor_combined/PrecioCA) cuando el dato no vino en la ficha del
    # producto -- se trata igual que NULL/vacío para filtrar y para
    # detectar cobertura real.
    if not raw:
        return None
    s = str(raw).strip()
    return s if s and s.lower() != "no especificado" else None


def _gpu_bool(raw: Any) -> Optional[bool]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("true", "si", "sí", "yes", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    return None  # "no especificado" u otro valor no interpretable


def _fetch_rows() -> List[Dict]:
    engine = _get_engine()
    if not engine:
        return []

    for attempt in range(_DB_MAX_RETRIES):
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(f'SELECT * FROM "{_TABLE_NAME}"')).mappings().all()
            result = []
            for r in rows:
                row = dict(r)
                row["_tipo_lower"] = (row.get("tipo_equipo") or "").strip().lower()
                row["_ram_gb"] = _parse_ram_gb(row.get("total_ram_gb"))
                row["_storage_gb"] = _parse_storage_gb(row.get("total_almacenamiento_gb"))
                row["_nucleos"] = _parse_ram_gb(row.get("nucleos_procesador"))
                row["_tecnologia_ram"] = _clean_enum(row.get("tecnologia_ram"))
                row["_tecnologia_disco"] = _clean_enum(row.get("tecnologia_disco_principal"))
                row["_config_discos"] = _clean_enum(row.get("tipo_configuracion_discos"))
                row["_generacion"] = _clean_enum(row.get("generacion_procesador"))
                row["_gpu_dedicada"] = _gpu_bool(row.get("tiene_gpu_dedicada"))
                # linea_procesador ahora viene de productos_extraccion (misma
                # extracción que PrecioCA, ~100% cobertura) -- se prefiere
                # sobre la familia derivada por regex, que queda solo como
                # respaldo si algún producto no trajera el dato.
                row["_familia"] = _clean_enum(row.get("linea_procesador")) or extract_linea_procesador(row.get("procesador_principal"))
                result.append(row)
            logging.info(f"[PrecioCM] Catálogo cargado: {len(result)} productos")
            return result
        except OperationalError as e:
            logging.warning(f"[PrecioCM] Error de conexión (intento {attempt + 1}/{_DB_MAX_RETRIES}): {e}")
            _reset_engine()
            engine = _get_engine()
            if attempt < _DB_MAX_RETRIES - 1:
                time.sleep(_DB_RETRY_DELAY)
        except Exception as e:
            logging.warning(f"[PrecioCM] Error consultando catálogo: {e}")
            return []

    return []


def _get_rows(force: bool = False) -> List[Dict]:
    global _rows_cache, _rows_cache_ts
    now = time.time()
    if force or _rows_cache is None or (now - _rows_cache_ts) >= _CACHE_TTL_SECONDS:
        fresh = _fetch_rows()
        if fresh or _rows_cache is None:
            _rows_cache = fresh
            _rows_cache_ts = now
    return _rows_cache or []


def _numeric_match(requested: Any, actual: Optional[float]) -> bool:
    if actual is None:
        return False
    if isinstance(requested, dict):
        mn, mx = requested.get("min"), requested.get("max")
        try:
            if mn not in (None, "") and actual < float(mn) - 1e-6:
                return False
            if mx not in (None, "") and actual > float(mx) + 1e-6:
                return False
        except (TypeError, ValueError):
            pass
        return True
    if isinstance(requested, list):
        for v in requested:
            if v in (None, ""):
                continue
            try:
                if abs(actual - float(v)) < 0.5:
                    return True
            except (TypeError, ValueError):
                continue
        return False
    try:
        return abs(actual - float(requested)) < 0.5
    except (TypeError, ValueError):
        return True


def _text_match(requested: Any, actual: Optional[str]) -> bool:
    if not actual:
        return False
    actual_l = actual.lower()
    if isinstance(requested, list):
        return any(str(v).lower() in actual_l for v in requested if v)
    return str(requested).lower() in actual_l


def _exact_match(requested: Any, actual: Optional[str]) -> bool:
    # A diferencia de _text_match (substring): las tecnologías de RAM son
    # tokens tipo enum donde el substring da falsos positivos -- "DDR5" es
    # substring de "LPDDR5", pero son tecnologías distintas (LPDDR es RAM
    # de bajo consumo para notebooks, no equivalente a DDR de escritorio).
    if not actual:
        return False
    actual_l = actual.strip().lower()
    if isinstance(requested, list):
        return any(str(v).strip().lower() == actual_l for v in requested if v)
    return str(requested).strip().lower() == actual_l


def _os_family(text: Any) -> Optional[str]:
    if not text:
        return None
    t = str(text).lower()
    if "windows" in t:
        return "windows"
    if "macos" in t or "mac os" in t or "os x" in t:
        return "macos"
    if "chrome" in t:
        return "chromeos"
    if "linux" in t:
        return "linux"
    return t.strip()


def _os_match(requested: Any, actual: Optional[str]) -> bool:
    # PrecioCM trae "WINDOWS 11 PRO" crudo mientras la ficha usa el diccionario
    # de PrecioCA ("Microsoft Windows 11 Home"/"...Pro") — comparar por familia
    # de SO (Windows/macOS/Chrome OS), no por string exacto ni por edición,
    # porque Convenio Marco suele ofrecer una sola edición por equipo.
    actual_fam = _os_family(actual)
    if actual_fam is None:
        return False
    if isinstance(requested, list):
        return any(_os_family(v) == actual_fam for v in requested if v)
    return _os_family(requested) == actual_fam


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]


def _wifi_token(text: Any) -> Optional[str]:
    # PrecioCM: "WIFI 6E" / ficha: "Wi-Fi 6E" — mismo dato, formato distinto.
    if not text:
        return None
    m = re.search(r"(\d+)\s*(E)?", str(text).upper())
    if not m:
        return None
    return m.group(1) + (m.group(2) or "")


def _wifi_match(requested: Any, actual: Optional[str]) -> bool:
    actual_tok = _wifi_token(actual)
    if actual_tok is None:
        return False
    if isinstance(requested, list):
        return any(_wifi_token(v) == actual_tok for v in requested if v)
    return _wifi_token(requested) == actual_tok


def _soft_filter(candidates, applied, unverified, *, requested, getter, matcher, label):
    """Filtra por un atributo con cobertura PARCIAL en el catálogo CM (a
    diferencia de tipo/marca/RAM, que casi siempre vienen informados): si
    NINGÚN candidato actual trae el dato, no se filtra -- se marca "no
    verificable" en vez de vaciar el panel solo porque ese campo no viene
    informado para nada en el subconjunto actual. Si algunos sí lo traen y
    calzan, se acota a esos; si no calzan, no se reduce (evita vaciar los
    resultados por un atributo secundario)."""
    with_data = [r for r in candidates if getter(r) is not None]
    if not with_data:
        unverified.append(label)
        return candidates
    matched = [r for r in with_data if matcher(requested, getter(r))]
    if matched:
        applied.append(label)
        return matched
    return candidates


# Atributos que la ficha puede pedir pero que PrecioCM no captura como columna
# estructurada — no es un problema de formato, el dato simplemente no existe
# para ningún producto del catálogo. Se avisa en vez de ignorarlo en silencio.
# (tecnologia_ram/tecnologia_disco_principal/tipo_configuracion_discos/
# tiene_gpu_dedicada salieron de acá: ahora sí son columnas reales de
# productos_extraccion y se filtran dinámicamente en _find_candidates --
# quedan solo los que PrecioCA tampoco filtra realmente, ver PRICE_QUERY_COLS
# en price_service.py.)
UNVERIFIABLE_LABELS = {
    "gpu_dedicada_nombre": "GPU dedicada",
    "pantalla_pulgadas": "tamaño de pantalla",
}


def _unverified_attrs(ficha: Dict[str, Any]) -> List[str]:
    labels = []
    for attr, label in UNVERIFIABLE_LABELS.items():
        if ficha.get(attr) is not None and label not in labels:
            labels.append(label)
    return labels


def _merge_unverified(static: List[str], dynamic: List[str]) -> List[str]:
    return static + [a for a in dynamic if a not in static]


class CMService:
    def warmup(self) -> None:
        _get_rows(force=True)

    def _find_candidates(self, ficha: Dict[str, Any]) -> Tuple[List[Dict], str, bool, List[str], List[str]]:
        """
        Devuelve (candidatos_finales, match_type_procesador, relajado, filtros_aplicados, no_verificables).
        match_type_procesador: "family" | "gama" | "none"
        """
        tipo = ficha.get("tipo_equipo")
        if not tipo:
            return [], "none", False, [], []

        rows = _get_rows()
        applied = ["tipo"]
        unverified: List[str] = []
        candidates = [r for r in rows if r["_tipo_lower"] == str(tipo).strip().lower()]

        if ficha.get("marca"):
            candidates = [r for r in candidates if _text_match(ficha["marca"], r.get("marca"))]
            applied.append("marca")

        if ficha.get("sistema_operativo"):
            candidates = [r for r in candidates if _os_match(ficha["sistema_operativo"], r.get("sistema_operativo"))]
            applied.append("SO")

        if ficha.get("wifi_generacion"):
            # ~0% de los laptops del catálogo traen wifi_generacion informado
            # (AIO/Desktop sí, 100%) — filtrar directo dejaría el panel vacío
            # cada vez que se pide Wi-Fi en un laptop. Ver _soft_filter.
            candidates = _soft_filter(
                candidates, applied, unverified,
                requested=ficha["wifi_generacion"], getter=lambda r: r.get("wifi_generacion") or None,
                matcher=_wifi_match, label="Wi-Fi",
            )

        if ficha.get("total_ram_gb") is not None:
            candidates = [r for r in candidates if _numeric_match(ficha["total_ram_gb"], r["_ram_gb"])]
            applied.append("RAM")

        if ficha.get("total_almacenamiento_gb") is not None:
            candidates = [r for r in candidates if _numeric_match(ficha["total_almacenamiento_gb"], r["_storage_gb"])]
            applied.append("almacenamiento")

        # ── Atributos con cobertura parcial en productos_extraccion (misma
        # lista que PriceService.PRICE_QUERY_COLS filtra para PrecioCA) ──
        if ficha.get("tecnologia_ram"):
            candidates = _soft_filter(
                candidates, applied, unverified,
                requested=ficha["tecnologia_ram"], getter=lambda r: r["_tecnologia_ram"],
                matcher=_exact_match, label="tecnología RAM",
            )

        if ficha.get("tecnologia_disco_principal"):
            candidates = _soft_filter(
                candidates, applied, unverified,
                requested=ficha["tecnologia_disco_principal"], getter=lambda r: r["_tecnologia_disco"],
                matcher=_text_match, label="tecnología disco",
            )

        if ficha.get("tipo_configuracion_discos"):
            candidates = _soft_filter(
                candidates, applied, unverified,
                requested=ficha["tipo_configuracion_discos"], getter=lambda r: r["_config_discos"],
                matcher=_text_match, label="config. discos",
            )

        if ficha.get("tiene_gpu_dedicada") is not None:
            candidates = _soft_filter(
                candidates, applied, unverified,
                requested=ficha["tiene_gpu_dedicada"], getter=lambda r: r["_gpu_dedicada"],
                matcher=lambda req, act: bool(req) == bool(act), label="GPU dedicada",
            )

        if ficha.get("generacion_procesador"):
            candidates = _soft_filter(
                candidates, applied, unverified,
                requested=ficha["generacion_procesador"], getter=lambda r: r["_generacion"],
                matcher=_text_match, label="generación proc.",
            )

        if ficha.get("nucleos_procesador") is not None:
            candidates = _soft_filter(
                candidates, applied, unverified,
                requested=ficha["nucleos_procesador"], getter=lambda r: r["_nucleos"],
                matcher=_numeric_match, label="núcleos",
            )

        if not candidates:
            return [], "none", False, applied, unverified

        # ── Procesador: familia por reglas; sin match se relaja (ver docstring) ──
        ficha_lineas = _as_list(ficha.get("linea_procesador"))
        if not ficha_lineas and ficha.get("procesador_principal"):
            derived = extract_linea_procesador(str(ficha["procesador_principal"]))
            if derived:
                ficha_lineas = [derived]

        if ficha_lineas:
            fl_lower = {f.lower() for f in ficha_lineas}
            family_matches = [r for r in candidates if r["_familia"] and r["_familia"].lower() in fl_lower]
            if family_matches:
                return family_matches, "family", False, applied + ["línea proc."], unverified

            # Sin match exacto de línea: relajar primero al grupo de líneas
            # equivalentes en la misma gama (cualquier marca) antes de soltar
            # el filtro de CPU por completo -- ver nota en el docstring del
            # módulo. Evita que la relajación total termine mostrando solo lo
            # que más abunda en el catálogo por marca.
            group = gama_service.expand_to_group(ficha_lineas)
            group_lower = {g.lower() for g in group} - fl_lower
            if group_lower:
                gama_matches = [r for r in candidates if r["_familia"] and r["_familia"].lower() in group_lower]
                if gama_matches:
                    return gama_matches, "gama", True, applied + ["gama equivalente"], unverified

            # Tampoco hay nada en la misma gama — se relaja del todo (ver nota
            # en el docstring del módulo sobre por qué no se usa similitud
            # semántica acá) para no dejar el panel vacío solo por el CPU.
            return candidates, "none", True, applied, unverified

        return candidates, "none", False, applied, unverified

    def _enrich(self, row: Dict, rate: float) -> Dict:
        precio_min_usd = row.get("precio_min")
        precio_max_usd = row.get("precio_max")
        precio_min_clp = int(round(precio_min_usd * rate)) if precio_min_usd is not None else None
        precio_max_clp = int(round(precio_max_usd * rate)) if precio_max_usd is not None else None
        return {
            "id_producto": row.get("id_producto"),
            "url": row.get("url"),
            "nombre": row.get("nombre"),
            "marca": row.get("marca"),
            "modelo": row.get("modelo"),
            "tipo_equipo": row.get("tipo_equipo"),
            "procesador_principal": row.get("procesador_principal"),
            "linea_procesador": row.get("linea_procesador"),
            "generacion_procesador": row["_generacion"],
            "puntaje_passmark_cpu": row.get("puntaje_passmark_cpu"),
            "total_ram_gb": row.get("total_ram_gb"),
            "tecnologia_ram": row["_tecnologia_ram"],
            "total_almacenamiento_gb": row.get("total_almacenamiento_gb"),
            "tecnologia_disco_principal": row["_tecnologia_disco"],
            "tipo_configuracion_discos": row["_config_discos"],
            "tiene_gpu_dedicada": row["_gpu_dedicada"],
            "gpu_dedicada_nombre": row.get("gpu_dedicada_nombre"),
            "sistema_operativo": row.get("sistema_operativo"),
            "wifi_generacion": row.get("wifi_generacion"),
            "peso_equipo": row.get("peso_equipo"),
            "monitor_si_no": row.get("monitor_si_no"),
            "precio_min_usd": precio_min_usd,
            "precio_max_usd": precio_max_usd,
            "precio_min_clp": precio_min_clp,
            "precio_max_clp": precio_max_clp,
        }

    def estimate(self, ficha: Dict[str, Any], top_n: int = 20) -> Optional[Dict]:
        candidates, match_type, relaxed, applied, dyn_unverified = self._find_candidates(ficha)
        if not candidates:
            return None

        rate, fx_date, fx_fallback = get_usd_clp()
        enriched = [self._enrich(r, rate) for r in candidates]
        enriched.sort(key=lambda r: (r["precio_min_clp"] is None, r["precio_min_clp"]))

        mins = [r["precio_min_clp"] for r in enriched if r["precio_min_clp"] is not None]
        maxs = [r["precio_max_clp"] for r in enriched if r["precio_max_clp"] is not None]
        mids = [
            (r["precio_min_clp"] + r["precio_max_clp"]) / 2
            for r in enriched
            if r["precio_min_clp"] is not None and r["precio_max_clp"] is not None
        ]
        mins_usd = [r["precio_min_usd"] for r in enriched if r["precio_min_usd"] is not None]
        maxs_usd = [r["precio_max_usd"] for r in enriched if r["precio_max_usd"] is not None]

        return {
            "count": len(enriched),
            "min": min(mins) if mins else None,
            "max": max(maxs) if maxs else None,
            "median": int(round(statistics.median(mids))) if mids else None,
            "min_usd": min(mins_usd) if mins_usd else None,
            "max_usd": max(maxs_usd) if maxs_usd else None,
            "currency": "CLP",
            "fx_rate": rate,
            "fx_date": fx_date,
            "fx_fallback": fx_fallback,
            "fx_source": "Banco Central de Chile (dólar observado) · vía mindicador.cl",
            "processor_match": match_type,
            "processor_relaxed": relaxed,
            # Con catálogo chico (~90 productos) no tiene sentido un umbral de
            # conteo como en PrecioCA (1000+) — "amplio" acá es simplemente
            # que no se filtró por nada más que el tipo de equipo.
            "broad_warning": len(applied) <= 1,
            "match_description": ", ".join(applied) if applied else "sin filtros",
            "unverified_attrs": _merge_unverified(_unverified_attrs(ficha), dyn_unverified),
            "products": enriched[:top_n],
        }

    def get_offer_rows(self, ficha: Dict[str, Any], limit: int = 30) -> List[Dict]:
        candidates, _match_type, _relaxed, _applied, _dyn_unverified = self._find_candidates(ficha)
        if not candidates:
            return []
        rate, _fx_date, _fx_fallback = get_usd_clp()
        enriched = [self._enrich(r, rate) for r in candidates]
        enriched.sort(key=lambda r: (r["precio_min_clp"] is None, r["precio_min_clp"]))
        return enriched[:limit]
