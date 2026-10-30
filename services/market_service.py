"""
services/market_service.py

Datos de mercado por ítem para el resumen ejecutivo: evolución del precio,
proveedores con más compras adjudicadas y distribución de precios, a partir
de las transacciones históricas de Compra Ágil que ya respaldan la
estimación (ver price_service.get_offer_rows).

No escribe SQL propio: reutiliza price_service.get_offer_rows /
count_offer_rows, que arman el filtro solo con columnas de
PRICE_QUERY_COLS y enlazan todos los valores como parámetros. Como la
ficha llega desde el cliente (cada ítem del carrito trae sus atributos),
acá se valida y se acota todo antes de tocar la base: solo atributos del
esquema, tipos simples, largos máximos. Devuelve únicamente lo necesario
para los gráficos (fecha, precio, nombre del proveedor): nunca las filas
completas.
"""

import hashlib
import json
import logging
import math
import time
from collections import Counter
from threading import Lock
from typing import Any, Dict, List, Optional

from services.price_service import PRICE_QUERY_COLS

MAX_ROWS = 400            # transacciones más recientes que se leen por ítem
MAX_SERIES = 400          # = MAX_ROWS: no se submuestrea, el detalle de cada compra se puede ver en pantalla
MAX_DESC = 140            # largo máximo de la descripción de cada compra
MAX_STR = 120             # largo máximo de un valor de texto de la ficha
MAX_LIST = 20             # opciones máximas por atributo
N_BINS = 10
TUKEY_K = 3.0           # límites a 3 IQR: solo recorta valores muy fuera de lo normal (con 1,5 caían compras legítimas cuando los precios son muy parejos)
TOP_PROVEEDORES = 5

CACHE_TTL_S = 600         # los datos históricos casi no cambian: 10 min bastan
CACHE_MAX = 256

_cache: Dict[str, tuple] = {}
_lock = Lock()


# ─── Validación de lo que llega del cliente ─────────────────────────────────

def _clean_scalar(v: Any) -> Any:
    """str acotado, número finito o bool; cualquier otra cosa se descarta."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v if math.isfinite(v) and abs(v) < 1e9 else None
    if isinstance(v, str):
        v = v.strip()
        return v if 0 < len(v) <= MAX_STR else None
    return None


def sanitize_ficha(schema, atributos: Any) -> Dict[str, Any]:
    """Deja solo atributos del esquema con valores simples. Lo que no calza se
    ignora (no se rechaza la petición completa por un campo raro)."""
    clean: Dict[str, Any] = {}
    if not isinstance(atributos, dict):
        return clean
    for key in schema.attributes:
        if key not in atributos:
            continue
        v = atributos[key]
        if isinstance(v, (list, tuple)):
            items = [x for x in (_clean_scalar(i) for i in v[:MAX_LIST]) if x is not None]
            if items:
                clean[key] = items
        elif isinstance(v, dict):
            rng = {k: _clean_scalar(v.get(k)) for k in ("min", "max") if v.get(k) is not None}
            rng = {k: x for k, x in rng.items() if isinstance(x, (int, float)) and not isinstance(x, bool)}
            if rng:
                clean[key] = rng
        else:
            x = _clean_scalar(v)
            if x is not None:
                clean[key] = x
    return clean


def sanitize_relaxed(relaxed: Any) -> List[str]:
    """Los atributos "relajados" solo pueden ser columnas de la consulta de precio."""
    if not isinstance(relaxed, (list, tuple)):
        return []
    return [c for c in relaxed[:MAX_LIST] if isinstance(c, str) and c in PRICE_QUERY_COLS]


# ─── Cálculo ────────────────────────────────────────────────────────────────

def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(sorted_vals[int(k)])
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _fences(prices: List[int]) -> Optional[tuple]:
    """Límites de Tukey (Q1 - K*IQR, Q3 + K*IQR, K = TUKEY_K). La consulta de precio
    acepta todo entre $200 mil y $5 millones, así que una sola compra
    atípica estira el eje y aplasta al resto del gráfico; la mediana que
    muestra la estimación no se ve afectada, pero un gráfico sí."""
    if len(prices) < 8:
        return None
    vals = sorted(prices)
    q1, q3 = _percentile(vals, 0.25), _percentile(vals, 0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return None
    return q1 - TUKEY_K * iqr, q3 + TUKEY_K * iqr


def _histogram(prices: List[int]) -> Optional[Dict[str, Any]]:
    """Histograma entre el mínimo y el máximo de `prices` (ya sin atípicos).
    La cantidad de tramos se adapta a cuántas compras hay (raíz cuadrada,
    entre 3 y N_BINS): con pocas compras, 10 tramos dejaban casi todos en
    cero. Con menos de 5 compras no hay distribución que mostrar."""
    n = len(prices)
    if n < 5:
        return None
    n_bins = max(3, min(N_BINS, round(math.sqrt(n))))
    lo, hi = int(min(prices)), int(math.ceil(max(prices)))
    if hi <= lo:
        hi = lo + 1
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for p in prices:
        counts[min(int((p - lo) / width), n_bins - 1)] += 1
    return {"lo": lo, "hi": hi, "counts": counts}


def _build(rows: List[Dict[str, Any]], total: int) -> Dict[str, Any]:
    all_prices = [int(r["precio_iva"]) for r in rows if r.get("precio_iva")]
    fences = _fences(all_prices)
    in_range = (lambda p: fences[0] <= p <= fences[1]) if fences else (lambda p: True)

    points = []
    prices = []
    providers: Counter = Counter()
    for r in rows:
        name = (r.get("proveedor") or "").strip() or "Proveedor no informado"
        p = r.get("precio_iva")
        if not p or not in_range(int(p)):
            continue
        # Los proveedores se cuentan sobre las mismas compras que se dibujan: así
        # lo que muestra la barra coincide con lo que se ve al tocarla.
        providers[name] += 1
        prices.append(int(p))
        f = r.get("fecha")
        if f:
            points.append({
                "f": str(f)[:10], "p": int(p), "v": name[:MAX_STR],
                "d": (r.get("descripcion") or "").strip()[:MAX_DESC],
                "c": str(r.get("codigo_requerimiento") or "")[:64],
            })

    points.sort(key=lambda d: d["f"])
    if len(points) > MAX_SERIES:
        step = len(points) / MAX_SERIES
        points = [points[int(i * step)] for i in range(MAX_SERIES)]

    top = providers.most_common(TOP_PROVEEDORES)
    return {
        "n_total": int(total),
        "n_muestra": len(rows),
        "n_atipicos": len(all_prices) - len(prices),   # fuera de los límites: no se dibujan
        "fecha_min": points[0]["f"] if points else None,
        "fecha_max": points[-1]["f"] if points else None,
        "serie": points,
        "proveedores": [{"name": n[:MAX_STR], "count": c} for n, c in top],
        "proveedores_total": sum(providers.values()),
        "histograma": _histogram(prices),
    }


EMPTY = {
    "n_total": 0, "n_muestra": 0, "n_atipicos": 0, "fecha_min": None, "fecha_max": None,
    "serie": [], "proveedores": [], "proveedores_total": 0, "histograma": None,
}


def _analyze_cached(namespace: str, key_payload: Dict[str, Any], fetch) -> Dict[str, Any]:
    """Núcleo compartido por todas las categorías: `fetch()` devuelve
    (filas, total) con las claves que espera _build. Resultado en caché por
    contenido (namespace + filtros) -- los datos históricos casi no cambian."""
    key = hashlib.sha256(json.dumps({"ns": namespace, "k": key_payload}, sort_keys=True, default=str).encode()).hexdigest()
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL_S:
            return hit[1]

    try:
        rows, total = fetch()
        result = _build(rows, total)
    except Exception as e:
        logging.warning(f"[market_service.analyze:{namespace}] Error: {e}")
        return dict(EMPTY)

    with _lock:
        if len(_cache) >= CACHE_MAX:
            oldest = sorted(_cache.items(), key=lambda kv: kv[1][0])[: CACHE_MAX // 4]
            for k, _ in oldest:
                _cache.pop(k, None)
        _cache[key] = (now, result)
    return result


def analyze(ficha: Dict[str, Any], relaxed_attrs: List[str], price_service) -> Dict[str, Any]:
    """Datos de mercado de UN ítem de Computadores. `ficha` y `relaxed_attrs`
    ya vienen saneados (ver sanitize_*)."""
    if not ficha.get("tipo_equipo"):
        return dict(EMPTY)
    return _analyze_cached(
        "computadores", {"f": ficha, "r": sorted(relaxed_attrs)},
        lambda: price_service.get_market_rows(ficha, MAX_ROWS, relaxed_attrs or None),
    )


def analyze_medicamentos(ficha: Dict[str, Any], medicamento_service) -> Dict[str, Any]:
    """Datos de mercado de UN ítem de Medicamentos (ficha ya saneada)."""
    if not ficha.get("principio_activo") or not ficha.get("unidad_venta"):
        return dict(EMPTY)
    filters = {k: ficha.get(k) for k in ("principio_activo", "forma_farmaceutica", "concentracion", "laboratorio", "unidad_venta")}
    return _analyze_cached(
        "medicamentos", {"f": filters},
        lambda: medicamento_service.get_market_rows(filters, MAX_ROWS),
    )
