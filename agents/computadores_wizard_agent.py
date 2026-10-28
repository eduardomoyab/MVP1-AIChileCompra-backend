"""
agents/computadores_wizard_agent.py

Agente de extracción del wizard de Computadores (Fase 1, ver plan de
implementación). Calcado en la forma de `agents/medicamento_agent.py`: UNA
sola llamada no-streaming por texto analizado (`response_format=json_object`),
sin protocolo de separador `§§§` ni preguntas de seguimiento multi-turno —
el wizard resuelve lo que falte con sus propios controles (chips, slider,
multi-select), no re-preguntando vía LLM.

Toda la lógica de negocio de la categoría (atributos, distinción línea vs
modelo exacto de procesador, neutralidad de marca, reglas de listas/rangos,
reconocimiento de gama del Convenio Marco, especificaciones mínimas por
uso) se importa de `agents/computadores_rules.py` -- el mismo módulo que
usa el chat legacy (`agents/ficha_agent.py`). Nada de esto se reescribe
acá; solo se omiten las secciones que son inherentemente conversacionales
(explorar el propósito con preguntas de seguimiento, anti-repetición,
anti-promesa, borrado por historial) porque no aplican a una extracción de
una sola pasada sin turnos previos.

Sí se preserva, igual que en ficha_agent.py: normalización FAISS vía
AttributeMatcher, derivación de complementos (attribute_complement.csv) y
derivación de linea_procesador a partir de un modelo exacto
(processor_family.extract_linea_procesador).
"""

import os
import json
import logging
from typing import Any, Dict, List, Tuple

from openai import AsyncOpenAI
from dotenv import load_dotenv

from agents.attribute_matcher import AttributeMatcher
from agents.processor_family import extract_linea_procesador
from agents.computadores_rules import (
    CATEGORIA,
    FILLABLE_ATTRIBUTES,
    MULTI_VALUE_CLEARS,
    COMPLEMENT_ONLY_ATTRIBUTES,
    MIN_NORMALIZE_LEN,
    attrs_table_text,
    PROCESADOR_DISTINCTION_TEXT,
    BRAND_NEUTRALITY_TEXT,
    MULTI_VALUE_RULE_TEXT,
    EXPLICIT_DECISION_RULE_TEXT,
    TECHNICAL_DESCRIPTION_RULE_TEXT,
)

load_dotenv()

# A diferencia del chat legacy, el wizard NO infiere un set completo de
# atributos a partir de un uso declarado o una gama mencionada (eso vivía
# en computadores_rules.gama_section_text()/min_specs_section_text(), que
# este agente deliberadamente NO importa). Acá el uso/gama es, a lo más,
# una sugerencia mencionada en el mensaje -- la ficha solo se completa con
# lo explícito del texto o con los controles del wizard (chips, slider).
USE_SUGGESTION_RULE_TEXT = """## USO O GAMA MENCIONADOS -- SOLO SUGERENCIA, NUNCA AUTOCOMPLETAR

Si el usuario menciona un uso (ej. "para oficina", "para diseño gráfico") o una gama del Convenio Marco (ej. "gama 2", "gama alta"), NO llenes el resto de los atributos a partir de eso -- esto no es una ficha de piso mínimo, es una extracción literal de lo que el usuario escribió. Puedes mencionarlo como una sugerencia breve en el campo `message` (ej. "para oficina suele bastar con 8 GB de RAM y 256 GB de disco; puedes ajustarlo en los pasos siguientes"), pero esos valores NO van en `ficha_updates` salvo que el usuario además haya dado un número o especificación concreta."""


def _build_system_prompt() -> str:
    attrs_str = attrs_table_text()

    return f"""Eres un asistente que ayuda a funcionarios de compras públicas chilenos a especificar qué computador necesitan comprar en la plataforma Compra Ágil del Mercado Público.

Tu misión es, a partir de la descripción que el usuario escriba, extraer todos los atributos técnicos que puedas determinar con confianza real -- sin inventar nada que no esté dicho ni razonablemente implícito por lo que sí se dijo.

## ATRIBUTOS QUE PUEDES COMPLETAR

{attrs_str}

**NUNCA incluyas en ficha_updates**: nucleos_procesador, hilos_procesador, frecuencia_turbo_procesador_mhz, frecuencia_ram_mhz, total_vram_gpu_gb, generacion_procesador, tecnologia_gpu_principal. Esos se completan solos.

{PROCESADOR_DISTINCTION_TEXT}

{BRAND_NEUTRALITY_TEXT}

{MULTI_VALUE_RULE_TEXT}

{EXPLICIT_DECISION_RULE_TEXT}

{TECHNICAL_DESCRIPTION_RULE_TEXT}

{USE_SUGGESTION_RULE_TEXT}

## COMPORTAMIENTO

1. Esto NO es una conversación -- es una sola extracción sobre el texto que te llega, no un chat con turnos. Nunca hagas preguntas de seguimiento abiertas tipo "¿para qué lo usarán?": extrae SOLO lo que el usuario dijo explícitamente -- si un dato no viene literalmente en el texto, déjalo vacío aunque se pueda adivinar por un uso o gama mencionados (ver regla de arriba). El usuario completa lo que falte después con los controles del asistente (chips, deslizadores), no respondiéndote a ti.
2. Si el texto no da ninguna pista de atributos técnicos explícitos (ej. "necesito un computador" a secas, o solo describe un uso sin números ni specs), no inventes especificaciones -- deja `ficha_updates` vacío o con solo lo obvio (ej. tipo_equipo si se menciona), y dilo brevemente en el mensaje.
3. `questions` casi siempre vacío -- solo úsalo si el texto es tan ambiguo que ni siquiera puedes determinar el tipo de equipo.
4. Sé breve: máximo 2 oraciones en el mensaje.
5. Español de Chile, tono formal pero natural.

## FORMATO DE RESPUESTA (JSON)

Responde SIEMPRE con un único objeto JSON válido, nada más (sin texto antes ni después), con esta forma exacta:

{{"titulo": "[título corto, 3-6 palabras]", "message": "[mensaje breve, máximo 2 oraciones]", "ficha_updates": {{...}}, "questions": [...]}}

- **titulo**: como el título de una conversación de ChatGPT -- 3 a 6 palabras que resuman el requerimiento (ej. "AIO oficina Intel i5", "Laptop gama 2 diseño"), nunca una oración completa ni repite literalmente el texto del usuario.
- **ficha_updates**: solo los atributos que puedas determinar con confianza real. Omite los que no sepas -- no rellenes por rellenar.
- **questions**: máximo UNA pregunta (lista vacía `[]` casi siempre).
- Para numéricos: solo el número, sin unidad. Ej: 16, no "16 GB".
- Para booleanos: true o false (sin comillas)."""


# ─── Sesiones (independientes de las del chat legacy) ─────────────────────────

_sessions: Dict[str, Dict] = {}


def _get_session(session_id: str) -> Dict:
    if session_id not in _sessions:
        _sessions[session_id] = {"texto_original": "", "ficha": {}}
    return _sessions[session_id]


# ─── Agente ────────────────────────────────────────────────────────────────

class ComputadoresWizardAgent:
    def __init__(self, matcher: AttributeMatcher):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("TEMPERATURE", "0.2"))
        self.matcher = matcher
        self._system_prompt = _build_system_prompt()

    async def analyze(self, session_id: str, texto: str) -> Dict[str, Any]:
        """
        Una sola llamada no-streaming al LLM por texto entregado. Devuelve:
          {message, ficha_updates: [...], complement_updates: [...], questions, tokens}
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
            logging.error(f"Error OpenAI (wizard computadores): {e}")
            return {
                "message": "Ocurrió un error al analizar el texto. Intenta nuevamente.",
                "titulo": "",
                "ficha_updates": [],
                "complement_updates": [],
                "questions": [],
                "tokens": None,
            }

        raw = response.choices[0].message.content
        usage = response.usage

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logging.warning(f"[ComputadoresWizardAgent] JSON mal formado: {raw[:200] if raw else raw}")
            parsed = {"message": "", "ficha_updates": {}, "questions": []}

        ai_message = parsed.get("message", "")
        titulo = (parsed.get("titulo") or "").strip()
        raw_updates = parsed.get("ficha_updates", {}) or {}
        questions = parsed.get("questions", []) or []
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))

        ficha_updates = []
        complement_updates = []

        for attr, value in raw_updates.items():
            # Nota: no se filtra por `attr not in FILLABLE_ATTRIBUTES` -- esa
            # tabla es solo la de atributos "sugeribles" documentados en el
            # prompt, pero linea_procesador (derivado, no está en la tabla)
            # SÍ es una extracción legítima del LLM. El chat legacy
            # (ficha_agent.py) tampoco filtra por esto, solo por
            # COMPLEMENT_ONLY_ATTRIBUTES -- mismo criterio acá.
            if value is None or attr in COMPLEMENT_ONLY_ATTRIBUTES:
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
            complement_updates.extend(self._apply_complements(session, attr, normalized_value))

        return {
            "message": ai_message,
            "titulo": titulo,
            "ficha_updates": ficha_updates,
            "complement_updates": complement_updates,
            "questions": questions,
            "tokens": usage,
        }

    async def describe_batch(self, items: List[Dict[str, Any]]) -> List[str]:
        """Genera un párrafo de Compra Ágil por requerimiento del carrito, en
        UNA sola llamada (no una por ítem) -- se usa al descargar el PDF,
        nunca al agregar un ítem al carrito. Mismo patrón que
        MedicamentoAgent.describe_batch.

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

        # Antes esto pedía un párrafo "de venta" (tono persuasivo) -- el
        # resultado real incluía frases calificativas sin respaldo técnico
        # ("asegurando un rendimiento óptimo", "garantizando una experiencia
        # fluida") y un cierre tipo invitación ("invitamos a los proveedores
        # a presentar sus cotizaciones"). Reporte real: debe ser una
        # solicitud de requerimiento de compra, no un texto de marketing --
        # solo los atributos obtenidos, sin calificativos ni cierre.
        prompt = f"""Genera, para CADA uno de los siguientes {len(items)} requerimientos de compra de computadores, una descripción breve y factual en español formal, en tono de Compra Ágil chilena (mecanismo de contratación simplificada de Mercado Público, distinto de una licitación formal). Es una especificación técnica, NO un texto de venta ni una invitación.

**Reglas estrictas**:
- Describe SOLO los atributos técnicos entregados (tipo de equipo, procesador, RAM, almacenamiento, GPU, etc. según corresponda) -- una enumeración factual, como una ficha técnica en prosa.
- **Prohibido el lenguaje calificativo o de venta**: nunca "asegurando un rendimiento óptimo", "garantizando una experiencia fluida", "de alta calidad", "eficiente", "óptimo" ni nada similar que no sea un dato técnico concreto. Si un adjetivo no está respaldado por un atributo entregado, no lo uses.
- **Prohibido cualquier cierre o llamado a la acción** -- nunca termines con frases como "invitamos a los proveedores a presentar sus cotizaciones" ni nada parecido. El texto termina en la última especificación, sin despedida.
- Nunca uses la palabra "licitación" ni hagas referencia a un proceso de licitación formal, bases de licitación, oferentes de licitación, etc. -- esto es una Compra Ágil.
- **Nunca menciones una marca de equipo como requisito excluyente** salvo que el usuario la haya pedido explícitamente (la compra pública no puede restringir a una marca específica sin justificación técnica).
- Si falta algún dato, redacta de forma genérica sin inventar información que no esté en los atributos entregados.

Ejemplo del tono esperado (no copiar el contenido, solo el estilo): "Se requiere un computador tipo AIO, con procesador de línea Intel Core i5 o AMD Ryzen 5, 16 GB de RAM y 1000 GB de almacenamiento."

Requerimientos:
{items_block}

Responde en JSON con esta forma exacta: {{"descripciones": ["descripción del requerimiento 0", "descripción del requerimiento 1", ...]}} -- en el mismo orden y la misma cantidad ({len(items)}) que los requerimientos listados arriba."""

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
            logging.error(f"Error generando descripciones de PDF (wizard computadores): {e}")
            descripciones = []

        while len(descripciones) < len(items):
            descripciones.append("")
        return descripciones[:len(items)]

    def apply_manual_update(self, session_id: str, attribute: str, value: Any) -> Dict[str, Any]:
        """Actualización manual (sin LLM) -- un control del wizard (chip,
        slider, toggle) llama esto directo. Mismo contrato de salida que
        analyze(), más los complementos/limpiezas que ya hacía
        ficha_agent.FichaAgent.apply_manual_update para el chat legacy."""
        session = _get_session(session_id)

        if attribute not in COMPLEMENT_ONLY_ATTRIBUTES:
            if isinstance(value, list) or (isinstance(value, dict) and ("min" in value or "max" in value)):
                normalized_value, score, _ = self._normalize(attribute, value)
            elif isinstance(value, str) and value.strip():
                normalized_value, score, _ = self._normalize(attribute, value)
            else:
                normalized_value = value
                score = 1.0
        else:
            normalized_value = value
            score = 1.0

        threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))
        updates = [{
            "attribute": attribute,
            "value": normalized_value,
            "source": "user",
            "normalized": score >= threshold,
            "score": round(score, 3),
        }]
        session["ficha"][attribute] = normalized_value

        complement_updates = self._apply_complements(session, attribute, normalized_value)

        # Limpiar complementos cuando se borra o se asignan múltiples valores
        # (mismo criterio que ficha_agent.py: un multi-select en
        # procesador_principal/gpu_dedicada_nombre invalida los derivados
        # de un modelo exacto anterior).
        should_clear = (
            normalized_value is None or isinstance(normalized_value, list)
        ) and attribute in MULTI_VALUE_CLEARS
        if should_clear:
            for clr_attr in MULTI_VALUE_CLEARS[attribute]:
                if session["ficha"].get(clr_attr) is not None:
                    session["ficha"][clr_attr] = None
                    complement_updates.append({
                        "attribute": clr_attr,
                        "value": None,
                        "source": "complement",
                        "triggered_by": attribute,
                        "triggered_value": None,
                    })

        return {"updates": updates, "complement_updates": complement_updates}

    def get_ficha(self, session_id: str) -> Dict[str, Any]:
        return _get_session(session_id)["ficha"].copy()

    def reset_session(self, session_id: str) -> None:
        _sessions[session_id] = {"texto_original": "", "ficha": {}}

    def cleanup_session(self, session_id: str) -> None:
        _sessions.pop(session_id, None)

    def seed_session(self, session_id: str, ficha: Dict[str, Any], texto_original: str = "") -> None:
        """Re-siembra la ficha en memoria al retomar un formulario guardado
        (ver category_session_service.py) -- este proceso puede no tener
        nunca visto este session_id antes (otra pestaña, otro día, o el
        backend se reinició entre medio)."""
        _sessions[session_id] = {"texto_original": texto_original, "ficha": dict(ficha)}

    def _apply_complements(self, session: Dict, attr: str, normalized_value: Any) -> List[Dict[str, Any]]:
        """Deriva complement_updates para un atributo recién fijado --
        attribute_complement.csv (núcleos/hilos/frecuencia a partir de
        procesador o RAM, VRAM/tecnología a partir de GPU) más la
        derivación especial procesador_principal -> linea_procesador
        (processor_family.extract_linea_procesador), igual que
        ficha_agent.FichaAgent."""
        complement_updates = []

        comps = self.matcher.get_complements(CATEGORIA, attr, str(normalized_value))
        for comp_attr, comp_val in comps.items():
            complement_updates.append({
                "attribute": comp_attr,
                "value": comp_val,
                "source": "complement",
                "triggered_by": attr,
                "triggered_value": str(normalized_value),
            })
            session["ficha"][comp_attr] = comp_val

        if attr == "procesador_principal" and isinstance(normalized_value, str):
            linea = extract_linea_procesador(normalized_value)
            if linea and not session["ficha"].get("linea_procesador"):
                complement_updates.append({
                    "attribute": "linea_procesador",
                    "value": linea,
                    "source": "complement",
                    "triggered_by": "procesador_principal",
                    "triggered_value": normalized_value,
                })
                session["ficha"]["linea_procesador"] = linea

        return complement_updates

    def _normalize(self, attribute: str, value: Any) -> Tuple[Any, float, list]:
        meta = FILLABLE_ATTRIBUTES.get(attribute, {})
        attr_type = meta.get("type", "free")

        # Lista de valores → normalizar cada elemento individualmente
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

        # Rango (dict con min/max) → pasar tal cual sin normalizar
        if isinstance(value, dict) and ("min" in value or "max" in value):
            return value, 1.0, []

        # Para enums y booleanos no usamos FAISS
        if attr_type in ("enum", "boolean") or not isinstance(value, str):
            return value, 1.0, []

        # Para atributos de diccionario: FAISS (ver docstring de
        # ficha_agent.FichaAgent._normalize -- strings cortas/ambiguas no
        # son seguras de normalizar por similitud semántica).
        if attr_type == "dict":
            if len(value.strip()) < MIN_NORMALIZE_LEN:
                return value, 0.0, []
            return self.matcher.normalize(CATEGORIA, attribute, value)

        return value, 1.0, []
