"""
categories/schema.py

Contrato genérico de "categoría de producto" para el wizard de ficha
técnica. Antes de esto, cada categoría (Computadores, Medicamentos) era un
árbol de código paralelo completo (su propio agente, sus propios
endpoints, su propio JS) sin ninguna pieza común más que AttributeMatcher.
Este módulo declara el esquema del que el wizard genérico (backend y
frontend) puede leer para renderizarse, sin tener que escribir código
nuevo por cada categoría que se agregue a futuro -- agregar una categoría
pasa a ser "declarar un CategorySchema", no "duplicar un árbol de código".

Deliberadamente sin dependencia de FastAPI/Pydantic: lo importan
`agents/`, `services/` y `main.py` por igual, y una dependencia hacia
FastAPI acá crearía un acoplamiento innecesario (y, en los agentes, un
ciclo de imports con `main.py`).

Fase 1 (ver plan): solo Computadores se registra acá. Medicamentos sigue
100% en su propio código (`agents/medicamento_agent.py`, endpoints
`/api/medicamentos/*`) -- no se migra en esta fase, cero riesgo para esa
feature ya en producción.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

AttrType = Literal["enum", "enum_multi", "numeric_range", "dict", "boolean", "free"]

ValuesSource = Literal["fixed", "dictionary"]


@dataclass
class AttributeSchema:
    key: str
    label: str
    type: AttrType
    description: str = ""  # para el prompt del agente de extracción
    core: bool = False  # bloquea el precio completo hasta estar lleno (ver price_service._RELAX_ORDER: tipo_equipo/total_ram_gb/total_almacenamiento_gb nunca se relajan -- son los "core" naturales de Computadores)
    optional_detail: bool = False  # aparece en el paso "detalles opcionales" del wizard, no en los pasos core

    # enum / enum_multi / dict: de dónde salen los valores válidos
    values_source: ValuesSource = "fixed"
    values: Optional[list] = None  # lista fija, solo cuando values_source == "fixed"

    # numeric_range: slider nativo sobre pasos discretos reales de catálogo
    # (ver plan: nunca un rango continuo arbitrario)
    unit: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    allowed_steps: Optional[list] = None

    def to_public_dict(self) -> dict:
        """Forma que viaja al frontend vía /api/categories/{slug}/schema.
        `valid_values` real (values_source=="dictionary") lo completa
        quien arma la respuesta del endpoint -- acá solo va lo estático."""
        d = {"label": self.label, "type": self.type, "core": self.core,
             "optional_detail": self.optional_detail}
        if self.values_source == "fixed" and self.values:
            d["values"] = self.values
        if self.unit:
            d["unit"] = self.unit
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.allowed_steps:
            d["allowed_steps"] = self.allowed_steps
        return d


@dataclass
class CategorySchema:
    slug: str  # "computadores" -- el que aparece en la URL/endpoints
    categoria_db: str  # "Computadores" -- la clave que ya usan attribute_dictionary.csv / PrecioCA
    label: str
    supports_convenio_marco: bool = False
    # Orden del wizard para los pasos "core". Una entrada puede ser un
    # nombre de atributo real, o una key lógica de paso que el agente de
    # extracción resuelve a uno de varios atributos posibles (ver
    # "linea_procesador_o_procesador" en computadores_schema.py).
    core_order: list = field(default_factory=list)
    attributes: dict = field(default_factory=dict)  # key -> AttributeSchema

    def core_attributes(self) -> list:
        """Solo los atributos reales (no keys lógicas de paso) marcados
        core=True -- lo que price_service necesita ver lleno antes de
        devolver un precio no-relajado."""
        return [a for a in self.attributes.values() if a.core]


_REGISTRY: dict = {}


def register(schema: CategorySchema) -> None:
    _REGISTRY[schema.slug] = schema


def get(slug: str) -> Optional[CategorySchema]:
    return _REGISTRY.get(slug)


def all_slugs() -> list:
    return list(_REGISTRY.keys())
