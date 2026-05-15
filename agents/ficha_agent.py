"""
ficha_agent.py

Agente GPT-4o-mini que mantiene conversación y actualiza la ficha técnica
de computadores en tiempo real.

Flujo por mensaje:
  1. Usuario envía descripción/respuesta
  2. GPT extrae atributos y devuelve JSON estructurado
  3. Backend normaliza valores de diccionario via FAISS
  4. Backend aplica reglas de complemento (attribute_complement.csv)
  5. Retorna {message, ficha_updates, complement_updates, questions}
"""

import os
import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from dotenv import load_dotenv

from agents.attribute_matcher import AttributeMatcher

load_dotenv()

CATEGORIA = "Computadores"

# Atributos que el LLM puede sugerir (excluye los que se auto-completan por complemento)
FILLABLE_ATTRIBUTES = {
    "tipo_equipo": {
        "description": "Factor de forma del equipo",
        "type": "enum",
        "values": ["Laptop", "AIO", "Desktop", "Otro"],
    },
    "marca": {"description": "Fabricante del equipo (ej: HP, Dell, Lenovo, Apple)", "type": "dict"},
    "nombre_modelo": {"description": "Nombre comercial completo del modelo (opcional)", "type": "free"},
    "procesador_principal": {
        "description": "Modelo completo del procesador (ej: Intel Core i5-1335U, AMD Ryzen 5 7530U)",
        "type": "dict",
    },
    "total_ram_gb": {"description": "Memoria RAM total en GB", "type": "numeric"},
    "tecnologia_ram": {
        "description": "Estándar de memoria RAM",
        "type": "enum",
        "values": ["DDR5", "DDR4", "LPDDR5X", "LPDDR5", "LPDDR4X", "LPDDR4", "DDR3"],
    },
    "total_almacenamiento_gb": {"description": "Almacenamiento total en GB", "type": "numeric"},
    "tecnologia_disco_principal": {
        "description": "Tecnología del disco de mayor jerarquía",
        "type": "enum",
        "values": ["NVMe SSD", "SATA SSD", "SSD", "HDD", "eMMC", "mSATA"],
    },
    "tipo_configuracion_discos": {
        "description": "Combinación de discos instalados",
        "type": "enum",
        "values": ["solo SSD", "SSD+HDD", "solo HDD", "otro"],
    },
    "tiene_gpu_dedicada": {
        "description": "¿El equipo tiene GPU discreta?",
        "type": "boolean",
    },
    "gpu_dedicada_nombre": {
        "description": "Nombre completo de la GPU dedicada (solo si tiene_gpu_dedicada=true)",
        "type": "dict",
    },
    "pantalla_pulgadas": {
        "description": "Tamaño diagonal de pantalla en pulgadas (ej: 14.0, 15.6, 27.0)",
        "type": "numeric",
    },
    "sistema_operativo": {
        "description": "Sistema operativo preinstalado (ej: Microsoft Windows 11 Home, Sin sistema operativo)",
        "type": "dict",
    },
    "wifi_generacion": {
        "description": "Estándar Wi-Fi más alto disponible",
        "type": "enum",
        "values": ["Wi-Fi 7", "Wi-Fi 6E", "Wi-Fi 6", "Wi-Fi 5", "Wi-Fi 4"],
    },
}

# Atributos que se completan automáticamente por complemento (el LLM no los toca)
COMPLEMENT_ONLY_ATTRIBUTES = {
    "nucleos_procesador",
    "hilos_procesador",
    "frecuencia_turbo_procesador_mhz",
    "frecuencia_ram_mhz",
    "total_vram_gpu_gb",
    "tecnologia_gpu_principal",
    "linea_procesador",
    "generacion_procesador",
}


# Extrae la línea canónica del procesador (valores alineados al vocabulario del modelo LGBM)
_LINEA_RULES = [
    (r"Core\s+Ultra\s+9",   "Intel Core Ultra 9"),
    (r"Core\s+Ultra\s+7",   "Intel Core Ultra 7"),
    (r"Core\s+Ultra\s+5",   "Intel Core Ultra 5"),
    (r"Core\s+i9",          "Intel Core i9"),
    (r"Core\s+i7",          "Intel Core i7"),
    (r"Core\s+i5",          "Intel Core i5"),
    (r"Core\s+i3",          "Intel Core i3"),
    (r"\bCeleron\b",        "Intel Celeron"),
    (r"\bPentium\b",        "Intel Pentium"),
    (r"\bXeon\b",           "Intel Xeon"),
    (r"Ryzen\s+AI\s+9",     "AMD Ryzen AI 9"),
    (r"Ryzen\s+AI\s+7",     "AMD Ryzen AI 7"),
    (r"Ryzen\s+AI\s+5",     "AMD Ryzen AI 5"),
    (r"Ryzen\s+9",          "AMD Ryzen 9"),
    (r"Ryzen\s+7",          "AMD Ryzen 7"),
    (r"Ryzen\s+5",          "AMD Ryzen 5"),
    (r"Ryzen\s+3",          "AMD Ryzen 3"),
    (r"\bAthlon\b",         "AMD Athlon"),
    (r"Apple\s+M4\s+Max",   "Apple M4 Max"),
    (r"Apple\s+M4\s+Pro",   "Apple M4 Pro"),
    (r"Apple\s+M4",         "Apple M4"),
    (r"Apple\s+M3\s+Pro",   "Apple M3 Pro"),
    (r"Apple\s+M3",         "Apple M3"),
    (r"Apple\s+M[12]",      "Apple M-Series"),
    (r"Apple\s+M\d",        "Apple M-Series"),
]


def _extract_linea_procesador(procesador: str) -> Optional[str]:
    for pattern, canonical in _LINEA_RULES:
        if re.search(pattern, procesador, re.IGNORECASE):
            return canonical
    return None


_SEPARATOR = "§§§"


def _build_system_prompt() -> str:
    attr_table = []
    for atr, meta in FILLABLE_ATTRIBUTES.items():
        if meta["type"] == "enum":
            vals = " / ".join(meta["values"])
            tipo = f"SOLO uno de: {vals}"
        elif meta["type"] == "boolean":
            tipo = "SOLO: true / false"
        elif meta["type"] == "numeric":
            tipo = "Número"
        elif meta["type"] == "dict":
            tipo = "Valor de diccionario (sugiere el más preciso)"
        else:
            tipo = "Texto libre"
        attr_table.append(f"- **{atr}**: {meta['description']} → {tipo}")

    attrs_str = "\n".join(attr_table)

    return f"""Eres un asistente que ayuda a funcionarios de compras públicas chilenos a preparar fichas técnicas para la plataforma Compra Ágil del Mercado Público.

Tu misión es entender qué necesita el usuario y completar la ficha con las especificaciones **mínimas necesarias** para ese uso, sin pedir más de lo que realmente se necesita.

## ATRIBUTOS QUE PUEDES COMPLETAR

{attrs_str}

**NUNCA incluyas en ficha_updates**: nucleos_procesador, hilos_procesador, frecuencia_turbo_procesador_mhz, frecuencia_ram_mhz, total_vram_gpu_gb, linea_procesador, generacion_procesador, tecnologia_gpu_principal. Esos se completan solos.

## REGLA PRINCIPAL: PARA QUÉ SE USA EL EQUIPO

**Antes de sugerir especificaciones, debes saber para qué se va a usar el equipo.**
- Si el usuario no lo mencionó, tu única pregunta en ese turno debe ser para qué lo van a usar.
- Sin conocer el uso, NO completes RAM, almacenamiento, procesador ni GPU. Solo puedes inferir tipo_equipo si es obvio.
- Una vez que sepas el uso, llena todos los atributos que puedas.

## ESPECIFICACIONES MÍNIMAS SEGÚN USO

Usa siempre el mínimo adecuado. No pongas más de lo necesario.

- **Trabajo de oficina básico** (Word, Excel, correo, navegación): 8 GB RAM, 256 GB disco, sin GPU dedicada
- **Trabajo de oficina con programas exigentes** (Excel con macros complejas, muchas aplicaciones abiertas a la vez, bases de datos): 16 GB RAM, 256-512 GB disco, sin GPU dedicada
- **Programación y desarrollo de software** (IDEs, compiladores, servidores locales): 16 GB RAM, 512 GB disco, sin GPU dedicada
- **Diseño gráfico y edición de fotos** (Photoshop, Illustrator, Canva Pro): 16 GB RAM, 512 GB disco, sin GPU dedicada
- **Edición de video, animación o trabajo en 3D**: 32 GB RAM, 1000 GB disco, con GPU dedicada
- **Análisis de datos o inteligencia artificial**: 32 GB RAM, 512 GB disco, GPU dedicada opcional
- **Uso mixto o sin especificar**: 8 GB RAM, 256 GB disco (pide aclaración)

## REGLAS DE COMPORTAMIENTO

1. Conocido el uso, llena TODOS los atributos que puedas determinar con confianza.
2. Para los atributos con valores fijos (SOLO), usa exactamente uno de los valores indicados.
3. Para atributos de diccionario, sugiere el valor más específico posible.
4. Máximo UNA pregunta por turno. Prioridad: para qué se usa → tipo de equipo → otros.
5. NO pidas ni completes: nombre_modelo, wifi_generacion, pantalla_pulgadas.
6. NO preguntes por marca, sistema operativo ni pantalla a menos que el usuario los mencione.
7. Si el usuario menciona una marca, modelo o especificaciones concretas, úsalas directamente.
8. Sé breve y directo. No repitas la ficha en el mensaje. Una o dos oraciones bastan.
9. Escribe en español de Chile, con un tono formal pero natural. Sin tecnicismos innecesarios.

## FORMATO DE RESPUESTA

Responde SIEMPRE con este formato exacto, nada más:

[Mensaje conversacional aquí, máximo 3 oraciones]
{_SEPARATOR}
{{"ficha_updates": {{...}}, "questions": [...]}}

- **ficha_updates**: solo los atributos que puedes determinar con confianza. Omite los que no conoces.
- **questions**: máximo UNA pregunta (lista vacía [] si no necesitas más info).
- Para numéricos: solo el número, sin unidad. Ej: 8, no "8 GB".
- Para booleanos: true o false (sin comillas).

## EJEMPLOS

Usuario: "quiero un equipo para la oficina, que use excel, ppt, word"
Para trabajo de oficina con Office, 8 GB de RAM y 256 GB de disco son suficientes sin gastar de más.
{_SEPARATOR}
{{"ficha_updates": {{"tipo_equipo": "Laptop", "procesador_principal": "Intel Core i5-1335U", "total_ram_gb": 8, "tecnologia_ram": "DDR4", "total_almacenamiento_gb": 256, "tecnologia_disco_principal": "NVMe SSD", "tipo_configuracion_discos": "solo SSD", "tiene_gpu_dedicada": false, "sistema_operativo": "Microsoft Windows 11 Home"}}, "questions": []}}

Usuario: "necesito un equipo"
Para recomendarte las especificaciones correctas, necesito saber para qué lo van a usar.
{_SEPARATOR}
{{"ficha_updates": {{}}, "questions": ["¿Para qué se usará el equipo? (por ejemplo: trabajo de oficina, programación, diseño, edición de video)"]}}"""


# ─── Sesiones ────────────────────────────────────────────────────────────────

_sessions: Dict[str, Dict] = {}


def _get_session(session_id: str) -> Dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "history": [],
            "ficha": {},
        }
    return _sessions[session_id]


# ─── Agente ──────────────────────────────────────────────────────────────────

class FichaAgent:
    def __init__(self, matcher: AttributeMatcher):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("TEMPERATURE", "0.2"))
        self.matcher = matcher
        self._system_prompt = _build_system_prompt()

    async def stream_process_message(self, session_id: str, user_content: str):
        """
        Async generator.
        Yields ("chunk", str)  — fragmento de texto del mensaje del asistente.
        Yields ("done",  dict) — resultado final con ficha_updates, complement_updates, questions.

        Formato esperado del modelo:
            [Mensaje aquí]
            §§§
            {"ficha_updates": {...}, "questions": [...]}
        """
        session = _get_session(session_id)
        session["history"].append({"role": "user", "content": user_content})
        messages = [{"role": "system", "content": self._system_prompt}] + session["history"]

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )
        except Exception as e:
            logging.error(f"Error OpenAI streaming: {e}")
            err = "Lo siento, ocurrió un error al procesar tu mensaje. Intenta nuevamente."
            yield ("chunk", err)
            yield ("done", {"message": err, "ficha_updates": [], "complement_updates": [], "questions": []})
            return

        buf = ""          # caracteres recibidos aún no enviados al cliente
        msg_text = ""     # todo lo enviado como chunks de mensaje
        json_text = ""    # texto acumulado después del separador
        in_json = False

        async for raw_chunk in stream:
            delta = raw_chunk.choices[0].delta.content or ""
            if not delta:
                continue

            if in_json:
                json_text += delta
                continue

            buf += delta
            sep_pos = buf.find(_SEPARATOR)

            if sep_pos >= 0:
                # Emitir el tramo de mensaje antes del separador
                pre = buf[:sep_pos]
                if pre:
                    yield ("chunk", pre)
                    msg_text += pre
                json_text = buf[sep_pos + len(_SEPARATOR):]
                buf = ""
                in_json = True
            else:
                # Emitir lo que es seguro (mantener en buffer los últimos len(sep)-1 chars
                # por si el separador llegó partido entre chunks)
                safe = max(0, len(buf) - len(_SEPARATOR) + 1)
                if safe > 0:
                    to_yield = buf[:safe]
                    yield ("chunk", to_yield)
                    msg_text += to_yield
                    buf = buf[safe:]

        # Si el stream terminó sin separador (respuesta inesperada), tratar todo como mensaje
        if not in_json:
            if buf:
                yield ("chunk", buf)
                msg_text += buf
            session["history"].append({"role": "assistant", "content": msg_text})
            yield ("done", {"message": msg_text, "ficha_updates": [], "complement_updates": [], "questions": []})
            return

        # Guardar en historial
        session["history"].append({
            "role": "assistant",
            "content": f"{msg_text.strip()}\n{_SEPARATOR}\n{json_text.strip()}",
        })

        # Parsear JSON de la ficha
        try:
            parsed = json.loads(json_text.strip())
        except json.JSONDecodeError:
            logging.warning(f"[FichaAgent] JSON mal formado: {json_text[:200]}")
            parsed = {"ficha_updates": {}, "questions": []}

        raw_updates = parsed.get("ficha_updates", {}) or {}
        questions = parsed.get("questions", []) or []
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))

        ficha_updates = []
        complement_updates = []

        for attr, value in raw_updates.items():
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
                linea = _extract_linea_procesador(normalized_value)
                if linea and not session["ficha"].get("linea_procesador"):
                    complement_updates.append({
                        "attribute": "linea_procesador",
                        "value": linea,
                        "source": "complement",
                        "triggered_by": "procesador_principal",
                        "triggered_value": normalized_value,
                    })
                    session["ficha"]["linea_procesador"] = linea

        yield ("done", {
            "message": msg_text,
            "ficha_updates": ficha_updates,
            "complement_updates": complement_updates,
            "questions": questions,
        })

    async def process_message(
        self, session_id: str, user_content: str
    ) -> Dict[str, Any]:
        """
        Procesa un mensaje del usuario y devuelve:
          {
            message: str,
            ficha_updates: [ {attribute, value, source, normalized, score} ],
            complement_updates: [ {attribute, value, source, triggered_by, triggered_value} ],
            questions: [str],
          }
        """
        session = _get_session(session_id)
        session["history"].append({"role": "user", "content": user_content})

        messages = [{"role": "system", "content": self._system_prompt}] + session["history"]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logging.error(f"Error OpenAI: {e}")
            return {
                "message": "Lo siento, ocurrió un error al procesar tu mensaje. Intenta nuevamente.",
                "ficha_updates": [],
                "complement_updates": [],
                "questions": [],
            }

        raw = response.choices[0].message.content
        session["history"].append({"role": "assistant", "content": raw})

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"message": raw, "ficha_updates": {}, "questions": []}

        ai_message = parsed.get("message", "")
        raw_updates = parsed.get("ficha_updates", {}) or {}
        questions = parsed.get("questions", []) or []

        ficha_updates = []
        complement_updates = []

        for attr, value in raw_updates.items():
            if value is None:
                continue
            if attr in COMPLEMENT_ONLY_ATTRIBUTES:
                continue

            normalized_value, score, candidates = self._normalize(attr, value)
            entry = {
                "attribute": attr,
                "value": normalized_value,
                "source": "ai",
                "normalized": score >= float(os.getenv("SIMILARITY_THRESHOLD", "0.82")),
                "score": round(score, 3),
            }
            ficha_updates.append(entry)
            session["ficha"][attr] = normalized_value

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
                linea = _extract_linea_procesador(normalized_value)
                if linea and not session["ficha"].get("linea_procesador"):
                    complement_updates.append({
                        "attribute": "linea_procesador",
                        "value": linea,
                        "source": "complement",
                        "triggered_by": "procesador_principal",
                        "triggered_value": normalized_value,
                    })
                    session["ficha"]["linea_procesador"] = linea

        return {
            "message": ai_message,
            "ficha_updates": ficha_updates,
            "complement_updates": complement_updates,
            "questions": questions,
        }

    def apply_manual_update(
        self, session_id: str, attribute: str, value: Any
    ) -> Dict[str, Any]:
        """
        Procesa una actualización manual del usuario en la ficha.
        Normaliza via FAISS si el atributo está en el diccionario.
        Aplica complementos sobre el valor normalizado.
        """
        session = _get_session(session_id)

        # Intentar normalizar si el atributo está en el diccionario
        if attribute not in COMPLEMENT_ONLY_ATTRIBUTES and isinstance(value, str) and value.strip():
            normalized_value, score, _ = self._normalize(attribute, value)
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

        complement_updates = []
        comps = self.matcher.get_complements(CATEGORIA, attribute, str(normalized_value))
        for comp_attr, comp_val in comps.items():
            complement_updates.append({
                "attribute": comp_attr,
                "value": comp_val,
                "source": "complement",
                "triggered_by": attribute,
                "triggered_value": str(normalized_value),
            })
            session["ficha"][comp_attr] = comp_val

        if attribute == "procesador_principal" and isinstance(normalized_value, str):
            linea = _extract_linea_procesador(normalized_value)
            if linea:
                complement_updates.append({
                    "attribute": "linea_procesador",
                    "value": linea,
                    "source": "complement",
                    "triggered_by": "procesador_principal",
                    "triggered_value": str(normalized_value),
                })
                session["ficha"]["linea_procesador"] = linea

        return {"updates": updates, "complement_updates": complement_updates}

    def get_ficha(self, session_id: str) -> Dict[str, Any]:
        return _get_session(session_id)["ficha"].copy()

    def reset_session(self, session_id: str) -> None:
        _sessions[session_id] = {"history": [], "ficha": {}}

    def cleanup_session(self, session_id: str) -> None:
        _sessions.pop(session_id, None)

    def _normalize(self, attribute: str, value: Any) -> Tuple[Any, float, list]:
        meta = FILLABLE_ATTRIBUTES.get(attribute, {})
        attr_type = meta.get("type", "free")

        # Para enums y booleanos no usamos FAISS
        if attr_type in ("enum", "boolean") or not isinstance(value, str):
            return value, 1.0, []

        # Para atributos de diccionario: FAISS
        if attr_type == "dict":
            return self.matcher.normalize(CATEGORIA, attribute, value)

        return value, 1.0, []
