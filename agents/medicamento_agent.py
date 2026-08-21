"""
medicamento_agent.py

Agente GPT-4o-mini que interpreta texto libre ("Eutirox 100mg", "quitadol",
una descripción larga) y sugiere los atributos de un requerimiento de compra
de medicamentos: principio activo (el dato clave), forma farmacéutica y
concentración si el texto los trae, laboratorio/cantidad como opcionales.

A diferencia de ficha_agent.py (Computadores), esto NO es un chat -- no hay
conversación visible ni preguntas de seguimiento multi-turno en pantalla, así
que no hace falta streaming carácter a carácter: se hace UNA llamada no
streaming por texto analizado (`response_format=json_object`, más simple y
confiable que el protocolo de separador `§§§` que sí necesita ficha_agent
para poder ir emitiendo el mensaje token a token mientras llega). El
endpoint en main.py igual expone esto como un stream SSE de una sola tanda
(`ficha_update` + `price_update` + `[DONE]`), para reusar el mismo
protocolo de eventos y el mismo lector SSE que ya tiene el frontend.

Tampoco hay tabla de complementos (attribute_complement.csv): no existe un
análogo real a "procesador → núcleos derivados" para medicamentos, así que
esa pieza entera de ficha_agent.py se omite.

Vocabulario: `diccionarios/attribute_dictionary.csv` tiene filas
`categoria=Medicamentos` para `principio_activo` (unificado de
principio_activo_1/2 de la BD), `forma_farmaceutica` y `laboratorio`,
generadas desde los valores reales de la vista PrecioMedCA. `concentracion`
NO se normaliza por FAISS -- es un valor+unidad con estructura numérica/
ordinal ("100 MCG"), y el proyecto ya documentó (docstring de
cm_service.py) que la similitud de embeddings no distingue jerarquías
numéricas de forma confiable. Se deja como texto libre; el usuario elige
entre valores reales observados (facetas) en vez de un match aproximado.
"""

import os
import json
import logging
from typing import Any, Dict, List, Tuple

from openai import AsyncOpenAI
from dotenv import load_dotenv

from agents.attribute_matcher import AttributeMatcher

load_dotenv()

CATEGORIA = "Medicamentos"

FILLABLE_ATTRIBUTES_MED = {
    "principio_activo": {
        "description": "Principio(s) activo(s) del medicamento (ej: Levotiroxina, Paracetamol, Amoxicilina)",
        "type": "dict",
    },
    "forma_farmaceutica": {
        "description": "Forma en que se presenta el medicamento (ej: Comprimido, Jarabe, Solución inyectable)",
        "type": "dict",
    },
    "concentracion": {
        "description": "Concentración/dosis del principio activo, con su unidad (ej: 100 MCG, 500 MG)",
        "type": "free",
    },
    "laboratorio": {
        "description": "Laboratorio fabricante -- opcional, solo si el usuario lo menciona",
        "type": "dict",
    },
    "cantidad": {
        "description": "Cantidad de unidades por envase (ej: 30, 100)",
        "type": "numeric",
    },
    "unidad_venta": {
        "description": "Unidad de compra/venta del medicamento (ej: Caja, Comprimido, Ampolla, Frasco)",
        "type": "free",
    },
    "cantidad_requerida": {
        "description": "Cuántas unidades de compra se necesitan (ej: 3 cajas -> 3) -- NO es lo mismo que 'cantidad' (tamaño del envase)",
        "type": "numeric",
    },
}

# Atributos "core" para medir progreso -- laboratorio/cantidad son
# secundarios y no bloquean nada. unidad_venta SÍ es core: el precio de un
# medicamento no es comparable entre unidades (ej. Metformina: mediana
# $74 por Comprimido vs $23.205 por Caja, ~314x) -- sin unidad_venta,
# estimate_price()/get_historial() no calculan nada (ver medicamento_service.py).
CORE_ATTRS_MED = ["principio_activo", "forma_farmaceutica", "concentracion", "unidad_venta"]


def _build_system_prompt() -> str:
    attr_table = []
    for atr, meta in FILLABLE_ATTRIBUTES_MED.items():
        if meta["type"] == "numeric":
            tipo = "Número"
        elif meta["type"] == "dict":
            tipo = "Valor de diccionario (sugiere el más preciso)"
        else:
            tipo = "Texto libre"
        attr_table.append(f"- **{atr}**: {meta['description']} → {tipo}")
    attrs_str = "\n".join(attr_table)

    return f"""Eres un asistente que ayuda a funcionarios de compras públicas chilenos a especificar qué medicamento necesitan comprar en la plataforma Compra Ágil del Mercado Público.

Tu misión es, a partir de lo que el usuario escriba (puede ser solo el nombre comercial de un medicamento, una descripción larga con varias especificaciones, o directamente el principio activo), identificar el **principio activo real** y, si hay indicios explícitos en el texto, la forma farmacéutica y la concentración.

## ATRIBUTOS QUE PUEDES COMPLETAR

{attrs_str}

## REGLA PRINCIPAL — RESOLVER SIEMPRE AL PRINCIPIO ACTIVO

Los funcionarios suelen escribir el nombre COMERCIAL de un medicamento (ej. "Eutirox", "Losec", "Quitadol"), no el principio activo. Usa tu conocimiento para resolver el nombre comercial al principio activo real. Ejemplos:
- "Eutirox" → principio_activo: "Levotiroxina"
- "Losec" → principio_activo: "Omeprazol"
- "Panadol" → principio_activo: "Paracetamol"

**Nunca registres una marca comercial como si fuera el principio activo** — siempre entrega el principio activo real, nunca el nombre de fantasía del producto.

Si el texto ya trae el principio activo directamente (ej. "necesito Levotiroxina 100mcg"), regístralo tal cual, normalizado.

Si el nombre no te resulta reconocible o es genuinamente ambiguo (puede ser un medicamento real que no conoces, o un error de tipeo), **no inventes un principio activo** — deja `principio_activo` vacío y pide aclaración en `questions` en vez de adivinar.

## REGLA CONCENTRACIÓN Y FORMA FARMACÉUTICA — SOLO SI EL TEXTO LO TRAE

Extrae `concentracion` y `forma_farmaceutica` **solo si aparecen explícitas o fuertemente implícitas** en el texto (ej. "100mcg" → `concentracion: "100 MCG"`; "jarabe" → `forma_farmaceutica: "Jarabe"`). No inventes una concentración o forma que el usuario no mencionó ni insinuó — el sistema, aparte de esto, ya sugiere las variantes reales más compradas para ese principio activo; tu trabajo es solo extraer lo que el texto realmente contiene, no completar el resto.

## REGLA DE NEUTRALIDAD DE MARCA

Aunque identifiques la marca comercial para resolver el principio activo, esta ficha describe un **requerimiento de compra pública**, no un pedido de esa marca puntual — la compra pública no puede restringir a una marca específica salvo justificación técnica expresa. No registres la marca comercial en ningún atributo (en particular, no la pongas en `laboratorio`, que es el fabricante, no el nombre de fantasía del producto).

## REGLA UNIDAD DE VENTA — SOLO SI EL TEXTO LO TRAE

Extrae `unidad_venta` **solo si aparece explícita en el texto** (ej. "2 cajas de..." → `unidad_venta: "Caja"`; "100 comprimidos" → `unidad_venta: "Comprimido"`). No inventes ni asumas una unidad que el usuario no mencionó -- a diferencia de laboratorio/cantidad, este atributo sí es obligatorio para poder calcular un precio (el precio de un medicamento no es comparable entre unidades: no es lo mismo vender por Caja que por Comprimido), pero eso lo resuelve el propio flujo pidiéndoselo directamente al usuario si no vino en el texto -- tu trabajo es solo extraer lo que el texto realmente contiene, no completar el resto.

## REGLA LABORATORIO / CANTIDAD / CANTIDAD_REQUERIDA

Son opcionales y secundarios. Regístralos solo si el usuario los menciona explícitamente. No preguntes por ellos si no los mencionó. `cantidad_requerida` es CUÁNTAS unidades de venta se necesitan (ej. "3 cajas" → `unidad_venta: "Caja"` Y `cantidad_requerida: 3`) -- no confundir con `cantidad`, que es el tamaño del envase (ej. "envase de 30 comprimidos" → `cantidad: 30`).

## REGLA MÚLTIPLES PRINCIPIOS ACTIVOS

Si el medicamento es una combinación (ej. "Amoxicilina con Ácido clavulánico"), `principio_activo` va como lista: `["Amoxicilina", "Ácido clavulánico"]`.

## COMPORTAMIENTO

1. Máximo UNA pregunta en `questions`, solo si de verdad no puedes determinar el principio activo.
2. Sé breve: máximo 2 oraciones en el mensaje, explicando brevemente a qué resolviste el principio activo (si vino de una marca) o qué falta aclarar.
3. Español de Chile, tono formal pero natural.

## FORMATO DE RESPUESTA (JSON)

Responde SIEMPRE con un único objeto JSON válido, nada más (sin texto antes ni después), con esta forma exacta:

{{"message": "[mensaje breve, máximo 2 oraciones]", "ficha_updates": {{...}}, "questions": [...]}}

- **message**: el mensaje breve explicando a qué resolviste el principio activo (si vino de una marca) o qué falta aclarar. Nunca lo dejes vacío.
- **ficha_updates**: solo los atributos que puedas determinar con confianza real. Omite los que no sepas — no rellenes por rellenar.
- **questions**: máximo UNA pregunta (lista vacía `[]` si no hace falta).
- `concentracion` siempre con su unidad, en mayúsculas (ej. "100 MCG", no "100mcg").
- `cantidad`: solo el número, sin unidad.

## EJEMPLOS

Usuario: "Eutirox 100mg"
{{"message": "Eutirox corresponde al principio activo Levotiroxina. Registré 100 mg como concentración.", "ficha_updates": {{"principio_activo": "Levotiroxina", "concentracion": "100 MG"}}, "questions": []}}

Usuario: "quitadol"
{{"message": "No reconozco \\"quitadol\\" como un medicamento.", "ficha_updates": {{}}, "questions": ["¿Puedes confirmar el nombre exacto del medicamento o su principio activo?"]}}

Usuario: "necesito Amoxicilina con Ácido clavulánico, comprimidos, 500/125 mg"
{{"message": "Registré Amoxicilina + Ácido clavulánico en comprimidos, 500 mg/125 mg.", "ficha_updates": {{"principio_activo": ["Amoxicilina", "Ácido clavulánico"], "forma_farmaceutica": "Comprimido", "concentracion": "500 MG/125 MG"}}, "questions": []}}

Usuario: "necesito un medicamento"
{{"message": "Para ayudarte, ¿qué medicamento o principio activo necesitas específicamente?", "ficha_updates": {{}}, "questions": ["¿Qué medicamento o principio activo necesitas específicamente?"]}}

Usuario: "3 cajas de Ibuprofeno 400mg"
{{"message": "Registré Ibuprofeno 400 mg, 3 Cajas.", "ficha_updates": {{"principio_activo": "Ibuprofeno", "concentracion": "400 MG", "unidad_venta": "Caja", "cantidad_requerida": 3}}, "questions": []}}"""


# ─── Sesiones (independientes de las de ficha_agent) ──────────────────────────

_med_sessions: Dict[str, Dict] = {}


def _get_session(session_id: str) -> Dict:
    if session_id not in _med_sessions:
        _med_sessions[session_id] = {"texto_original": "", "ficha": {}}
    return _med_sessions[session_id]


# ─── Agente ────────────────────────────────────────────────────────────────

class MedicamentoAgent:
    def __init__(self, matcher: AttributeMatcher):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("MED_TEMPERATURE", "0.2"))
        self.matcher = matcher
        self._system_prompt = _build_system_prompt()

    async def analyze(self, session_id: str, texto: str) -> Dict[str, Any]:
        """
        Una sola llamada no-streaming al LLM por texto entregado. Devuelve:
          {message, ficha_updates: [{attribute,value,source,normalized,score}], questions, tokens}
        """
        session = _get_session(session_id)
        session["texto_original"] = texto

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": texto},
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logging.error(f"Error OpenAI (medicamentos): {e}")
            return {
                "message": "Ocurrió un error al analizar el texto. Intenta nuevamente.",
                "ficha_updates": [],
                "questions": [],
                "tokens": None,
            }

        raw = response.choices[0].message.content
        usage = response.usage

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logging.warning(f"[MedicamentoAgent] JSON mal formado: {raw[:200] if raw else raw}")
            parsed = {"message": "", "ficha_updates": {}, "questions": []}

        ai_message = parsed.get("message", "")
        raw_updates = parsed.get("ficha_updates", {}) or {}
        questions = parsed.get("questions", []) or []
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))

        ficha_updates = []
        for attr, value in raw_updates.items():
            if value is None or attr not in FILLABLE_ATTRIBUTES_MED:
                continue
            normalized_value, score, _ = self._normalize(attr, value)
            ficha_updates.append({
                "attribute": attr,
                "value": normalized_value,
                "source": "ai",
                "normalized": score >= threshold,
                "score": round(score, 3),
            })
            session["ficha"][attr] = normalized_value

        return {
            "message": ai_message,
            "ficha_updates": ficha_updates,
            "questions": questions,
            "tokens": usage,
        }

    async def describe_batch(self, items: List[Dict[str, Any]]) -> List[str]:
        """Genera un párrafo de Compra Ágil por requerimiento (NO de
        licitación -- es un mecanismo distinto), en UNA sola llamada (no
        una por ítem) -- se usa al descargar el PDF del carrito de
        requerimientos, nunca al agregar un ítem (evita gastar LLM en cada
        agregado al carrito).

        items: [{"texto_original": str, "atributos": {attr: valor}}]
        Devuelve una lista de párrafos, mismo orden y misma cantidad que items.
        """
        if not items:
            return []

        lineas = []
        for i, item in enumerate(items):
            atributos = item.get("atributos") or {}
            atributos_str = ", ".join(f"{k}: {v}" for k, v in atributos.items() if v) or "(sin atributos adicionales)"
            lineas.append(
                f'{i}. Texto original del usuario: "{item.get("texto_original", "")}". '
                f"Atributos seleccionados: {atributos_str}."
            )
        items_block = "\n".join(lineas)

        prompt = f"""Genera, para CADA uno de los siguientes {len(items)} requerimientos de compra de medicamentos, un párrafo breve (2 a 4 oraciones) en español formal, en tono de Compra Ágil chilena (mecanismo de contratación simplificada de Mercado Público, distinto de una licitación formal), dirigido a posibles proveedores.

**Nunca uses la palabra "licitación" ni hagas referencia a un proceso de licitación formal, bases de licitación, oferentes de licitación, etc.** -- esto es una Compra Ágil, un mecanismo simplificado y distinto. Usa términos como "proveedores", "cotización" o "Compra Ágil" en su lugar.

Describe cada requerimiento por principio activo, forma farmacéutica y concentración según corresponda -- **nunca menciones una marca comercial** (la compra pública no puede restringir a una marca específica salvo justificación técnica). Si falta algún dato, redacta de forma genérica sin inventar información que no esté en los atributos entregados.

Requerimientos:
{items_block}

Responde en JSON con esta forma exacta: {{"descripciones": ["párrafo del requerimiento 0", "párrafo del requerimiento 1", ...]}} -- en el mismo orden y la misma cantidad ({len(items)}) que los requerimientos listados arriba."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(response.choices[0].message.content)
            descripciones = parsed.get("descripciones", []) or []
        except Exception as e:
            logging.error(f"Error generando descripciones de PDF: {e}")
            descripciones = []

        # Asegura el mismo largo que items aunque el LLM devuelva de menos.
        while len(descripciones) < len(items):
            descripciones.append("")
        return descripciones[:len(items)]

    def apply_manual_update(self, session_id: str, attribute: str, value: Any) -> Dict[str, Any]:
        """Actualización manual (sin LLM) -- mismo contrato de salida que analyze()."""
        session = _get_session(session_id)

        if isinstance(value, list) or (isinstance(value, str) and value.strip()):
            normalized_value, score, _ = self._normalize(attribute, value)
        else:
            normalized_value, score = value, 1.0

        threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))
        updates = [{
            "attribute": attribute,
            "value": normalized_value,
            "source": "user",
            "normalized": score >= threshold,
            "score": round(score, 3),
        }]
        session["ficha"][attribute] = normalized_value
        return {"updates": updates}

    def get_ficha(self, session_id: str) -> Dict[str, Any]:
        return _get_session(session_id)["ficha"].copy()

    def reset_session(self, session_id: str) -> None:
        _med_sessions[session_id] = {"texto_original": "", "ficha": {}}

    def cleanup_session(self, session_id: str) -> None:
        _med_sessions.pop(session_id, None)

    def _normalize(self, attribute: str, value: Any) -> Tuple[Any, float, List]:
        meta = FILLABLE_ATTRIBUTES_MED.get(attribute, {})
        attr_type = meta.get("type", "free")

        if isinstance(value, list):
            normalized = []
            min_score = 1.0
            for item in value:
                if isinstance(item, str) and item.strip():
                    norm, score, _ = self._normalize(attribute, item)
                    normalized.append(norm)
                    min_score = min(min_score, score)
                else:
                    normalized.append(item)
            return normalized, min_score, []

        if attr_type != "dict" or not isinstance(value, str):
            return value, 1.0, []

        return self.matcher.normalize(CATEGORIA, attribute, value)
