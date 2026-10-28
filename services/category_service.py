"""
services/category_service.py

Capa delgada del wizard genérico de categorías (ver plan de implementación,
Fase 1 solo Computadores): traduce "atributo de un CategorySchema" a las
funciones que ya existen en AttributeMatcher/PriceService/CMService -- no
reimplementa ninguna consulta SQL ni normalización.

El punto central de este módulo es `estimate_price()`: decide, mirando
`schema.supports_convenio_marco`, si corresponde siquiera consultar
Convenio Marco -- ese booleano es el flag genérico pedido en el plan,
reemplazando el chequeo ad hoc que hoy vive en main.py sobre
`ficha.get("tipo_equipo")` (ver /api/cm_offers/{session_id}). El algoritmo
de MATCHING de Convenio Marco en sí (gamas, familia de procesador) sigue
siendo responsabilidad exclusiva de CMService -- acá solo se decide si se
le pregunta o no.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from agents.attribute_matcher import AttributeMatcher
from categories.schema import CategorySchema
from services.cm_service import CMService
from services.price_service import PriceService


def get_facet_values(
    schema: CategorySchema,
    attr: str,
    ficha: Dict[str, Any],
    matcher: AttributeMatcher,
    price_service: PriceService,
) -> List[str]:
    """Valores válidos para un atributo de esta categoría, según cómo esté
    declarado en el esquema:
    - "fixed": la lista fija del AttributeSchema, tal cual (sin conteo real).
    - "dictionary": intenta primero acotar por lo ya elegido (marca/
      tipo_equipo) vía price_service.get_filtered_dropdown_values -- mismo
      criterio que ya usan los editores de atributo del chat legacy. Si no
      hay nada acotado (el campo no es una columna de PrecioCA, o no hay
      resultados), cae a AttributeMatcher.get_valid_values(categoria_db, attr),
      el diccionario canónico completo.
    """
    meta = schema.attributes.get(attr)
    if meta is None:
        return []
    if meta.values_source == "fixed":
        return list(meta.values or [])

    try:
        acotado = price_service.get_filtered_dropdown_values(
            attr, marca=ficha.get("marca"), tipo_equipo=ficha.get("tipo_equipo"),
        )
        if acotado:
            return acotado
    except Exception as e:
        logging.warning(f"category_service.get_facet_values: fallback a diccionario completo para '{attr}': {e}")

    return matcher.get_valid_values(schema.categoria_db, attr)


def estimate_price(
    schema: CategorySchema,
    ficha: Dict[str, Any],
    price_service: PriceService,
    cm_service: CMService,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """(price_data, cm_price_data) -- cm_price_data siempre None si
    `not schema.supports_convenio_marco`, sin siquiera llamar a CMService."""
    price_data = price_service.estimate(ficha)
    # top_n alto a propósito -- el catálogo de Convenio Marco es chico
    # (~90 productos tras filtrar por gama/procesador), así que devolver
    # "todo" en vez del default (20) es barato y permite paginar la lista
    # completa en el cliente sin un segundo viaje al backend.
    cm_price_data = cm_service.estimate(ficha, top_n=200) if schema.supports_convenio_marco else None
    return price_data, cm_price_data


def get_offer_rows(
    ficha: Dict[str, Any],
    price_service: PriceService,
    limit: int = 30,
    price_min: Optional[int] = None,
    price_max: Optional[int] = None,
    relaxed_attrs: Optional[List[str]] = None,
    offset: int = 0,
) -> List[Dict]:
    """Transacciones individuales de Compra Ágil detrás de la estimación
    (price_service.estimate() solo da estadísticos agregados). `relaxed_attrs`
    debe ser el mismo que estimate() relajó para esta ficha (price_data
    cacheado por sesión en main.py) -- si no se pasa el mismo relajo acá, la
    consulta filtra con la ficha completa sin relajar y puede devolver 0
    filas mientras el panel de precio, arriba, sigue mostrando un conteo
    mayor (mismo mismatch real ya documentado en price_service.get_offer_rows).
    No hay equivalente para Convenio Marco acá -- CMService.estimate() ya
    devuelve las filas de producto en su propio resultado ("products")."""
    return price_service.get_offer_rows(ficha, limit, price_min, price_max, relaxed_attrs, offset)


def count_offer_rows(
    ficha: Dict[str, Any],
    price_service: PriceService,
    price_min: Optional[int] = None,
    price_max: Optional[int] = None,
    relaxed_attrs: Optional[List[str]] = None,
) -> int:
    """Total de transacciones que matchean el filtro actual -- para que el
    paginado de /api/categories/{slug}/offers sepa cuántas páginas hay."""
    return price_service.count_offer_rows(ficha, price_min, price_max, relaxed_attrs)
