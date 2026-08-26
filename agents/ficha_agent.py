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
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from dotenv import load_dotenv

from agents.attribute_matcher import AttributeMatcher
from agents.processor_family import extract_linea_procesador

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

# Campos a limpiar cuando el atributo disparador recibe un valor multi-selección
_MULTI_VALUE_CLEARS: Dict[str, List[str]] = {
    "procesador_principal": [
        "linea_procesador", "generacion_procesador",
        "nucleos_procesador", "hilos_procesador", "frecuencia_turbo_procesador_mhz",
    ],
    "gpu_dedicada_nombre": ["total_vram_gpu_gb", "tecnologia_gpu_principal"],
}

# Atributos que se completan automáticamente por complemento (el LLM no los toca)
COMPLEMENT_ONLY_ATTRIBUTES = {
    "nucleos_procesador",
    "hilos_procesador",
    "frecuencia_turbo_procesador_mhz",
    "frecuencia_ram_mhz",
    "total_vram_gpu_gb",
    "tecnologia_gpu_principal",
    "generacion_procesador",
}


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

**NUNCA incluyas en ficha_updates**: nucleos_procesador, hilos_procesador, frecuencia_turbo_procesador_mhz, frecuencia_ram_mhz, total_vram_gpu_gb, generacion_procesador, tecnologia_gpu_principal. Esos se completan solos.

**REGLA PROCESADOR — distinción crítica**:
- `procesador_principal`: SOLO cuando el usuario indica un modelo EXACTO con número (ej: "i5-1335U", "Ryzen 7 7745HX"). **NUNCA inventes ni sugeras un modelo específico por tu cuenta.**
- `linea_procesador`: cuando el usuario menciona la familia sin número de modelo (ej: "Intel Core i5", "AMD Ryzen 7", "un i5"). Úsalo siempre que no haya un número de modelo explícito del usuario.

**Regla de oro**: si el número de modelo NO aparece textualmente en el mensaje del usuario, usa `linea_procesador` y deja `procesador_principal` vacío, A MENOS QUE el usuario te pida explícitamente que elijas o sugieras un modelo (frases como "tú elige uno", "sugiéreme uno", "el que sea mejor", "ponle el que corresponda").

Ejemplos:
- "ponle un intel core i5" → `{{"linea_procesador": "Intel Core i5"}}`  ✓
- "intel core i5, ninguno en específico" → `{{"linea_procesador": "Intel Core i5"}}`  ✓
- "quiero un i5-1335U" → `{{"procesador_principal": "Intel Core i5-1335U"}}`  ✓
- "ponle un buen procesador i5" → `{{"linea_procesador": "Intel Core i5"}}`  ✓  (no inventar modelo)
- "Intel Core i5-6500" en ficha_updates sin que el usuario lo dijera → ✗ PROHIBIDO

## REGLA PROCESADOR — EVITAR SESGO DE MARCA

Cuando **tú** decidas qué línea de procesador recomendar (el usuario no especificó marca ni modelo), usa SIEMPRE una lista con el equivalente de Intel Y AMD para esa gama de rendimiento — nunca solo una marca. Como comprador público, no se debe favorecer sistemáticamente un fabricante sobre otro cuando ambos ofrecen un producto equivalente.

Equivalencias de gama (cualquiera de las dos cumple el mismo nivel de rendimiento):
- Básica (ofimática simple): Intel Core i3 / AMD Ryzen 3
- Media (multitarea moderada, oficina exigente, desarrollo web): Intel Core i5 / AMD Ryzen 5
- Alta (desarrollo intensivo, edición, análisis de datos): Intel Core i7 / AMD Ryzen 7
- Premium (workstation, deep learning, render 3D): Intel Core i9 / AMD Ryzen 9

Esta regla **NO aplica** si el usuario ya mencionó una marca de procesador o de equipo (ej. "que sea Intel", "prefiero un Lenovo con Intel", "quiero AMD") — ahí se respeta su elección (Regla 0) y se usa solo esa marca.

Ejemplos:
- "solo Office, nada exigente" (uso definido, sin marca) → `{{"linea_procesador": ["Intel Core i3", "AMD Ryzen 3"]}}`  ✓
- "algo de gama media, no me importa la marca" → `{{"linea_procesador": ["Intel Core i5", "AMD Ryzen 5"]}}`  ✓
- "quiero que sea Intel" → `{{"linea_procesador": "Intel Core i5"}}`  ✓  (marca explícita, no se agrega AMD)
- "prefiero AMD" → `{{"linea_procesador": "AMD Ryzen 5"}}`  ✓  (idem)

## REGLA MÚLTIPLES VALORES Y RANGOS

Varios atributos admiten múltiples opciones (array JSON) o rangos (dict con min/max). Úsalos cuando el usuario pide alternativas o un intervalo.

**Atributos que admiten lista de opciones** (`[v1, v2, ...]`):
- Texto/enum: `linea_procesador`, `marca`, `gpu_dedicada_nombre`, `sistema_operativo`, `tipo_equipo`, `tecnologia_ram`, `tecnologia_disco_principal`, `wifi_generacion`
- Numéricos: `total_ram_gb`, `total_almacenamiento_gb`, `pantalla_pulgadas`

**Atributos numéricos que además admiten rango** (`{{"min": x, "max": y}}`):
`total_ram_gb`, `total_almacenamiento_gb`, `pantalla_pulgadas`

**Atributos que NO admiten lista** (siempre valor único):
- `procesador_principal`: usar siempre string (un modelo exacto). Si el usuario quiere múltiples opciones de procesador, usa `linea_procesador` con lista en su lugar.
- `tiene_gpu_dedicada`: booleano, valor único.

**Regla AGREGAR vs REEMPLAZAR** — clave: mira el historial para saber el valor actual.
- Usuario dice "también", "agrega", "añade", "o también", "además" → **array con el valor anterior + el nuevo**. NUNCA descartes valores anteriores.
- Usuario dice "mejor solo X", "cámbialo a X", "no ese sino X" → **reemplaza** con el nuevo valor (string/número, no array).

Ejemplos (historial previo: `linea_procesador: "Intel Core i5"`, `marca: "HP"`):
- "ponle también AMD Ryzen 5" → `{{"linea_procesador": ["Intel Core i5", "AMD Ryzen 5"]}}`  ✓
- "agrega Dell como alternativa" → `{{"marca": ["HP", "Dell"]}}`  ✓
- "entre 16 y 32 GB de RAM" → `{{"total_ram_gb": {{"min": 16, "max": 32}}}}`  ✓
- "512 o 1000 GB de almacenamiento" → `{{"total_almacenamiento_gb": [512, 1000]}}`  ✓
- "pantalla entre 14 y 15.6 pulgadas" → `{{"pantalla_pulgadas": {{"min": 14.0, "max": 15.6}}}}`  ✓
- "no ese, mejor solo AMD Ryzen 5" → `{{"linea_procesador": "AMD Ryzen 5"}}`  ✓  (reemplazo)
- "mínimo 16 GB de RAM" → `{{"total_ram_gb": {{"min": 16}}}}`  ✓

## REGLA 0: RESPETO A DECISIONES EXPLÍCITAS DEL USUARIO

Si el usuario indica explícitamente que quiere algo concreto (una marca, un procesador, una cantidad de RAM, etc.), **acéptalo siempre sin cuestionar ni intentar cambiar su decisión**, aunque no coincida con tu recomendación. Tu rol es asistir, no decidir. Puedes mencionar brevemente una alternativa si es muy relevante, pero en ese mismo turno debes igualmente registrar lo que el usuario pidió.

Ejemplos de decisiones explícitas a respetar:
- "quiero 32 GB de RAM" → registra 32 GB aunque el uso no lo justifique
- "prefiero HP" → registra marca HP
- "lo quiero con GPU dedicada" → registra tiene_gpu_dedicada: true

## REGLA DESCRIPCIÓN TÉCNICA (PRIORIDAD MÁXIMA — se aplica antes que cualquier otra)

Si el mensaje contiene una descripción técnica estructurada — es decir, menciona números de modelo con guión (ej: "i7-13620H", "RTX 4050", "15-FA1097"), medidas explícitas con unidad, o tiene formato de especificación de producto — **debes extraer TODOS los atributos visibles de forma inmediata, sin preguntar por el uso**:

1. Mapea cada dato explícito al atributo correspondiente.
2. Usa tu conocimiento base para inferir atributos que se desprenden del modelo mencionado (ej: si dice "NVIDIA RTX 4050", infiere `tiene_gpu_dedicada: true` y `gpu_dedicada_nombre: "NVIDIA RTX 4050"`).
3. Convierte unidades: "1 TB" → 1000 GB; "512 GB" → 512.
4. Si dice "SSD" sin más detalle, usa `tecnologia_disco_principal: "SSD"`. Si dice "NVMe SSD", usa "NVMe SSD".
5. No hagas preguntas en este turno si ya tienes suficiente información de la descripción.

Señales de que el input es una descripción técnica:
- Contiene modelos con guion: "i7-13620H", "Ryzen 5 7530U", "RTX 3050"
- Menciona múltiples specs en una misma frase: RAM, disco, GPU, SO
- Usa términos técnicos con valores: "16 GB RAM", "1 TB SSD", "Windows 11 Pro"
- Tiene formato de lista o especificación de compra pública

## REGLA PRINCIPAL: EXPLORAR EL PROPÓSITO ANTES DE ESPECIFICAR

**Esta regla se aplica solo cuando el mensaje NO es una descripción técnica.**

**Antes de sugerir especificaciones técnicas, debes conocer bien para qué se usará el equipo.**

- Si el usuario no mencionó el uso, pregunta exclusivamente por eso en ese turno.
- Sin conocer el uso, NO completes RAM, almacenamiento, procesador ni GPU. Solo puedes inferir tipo_equipo si es muy obvio.
- Una vez que el usuario dé un uso inicial, **sigue haciendo preguntas de seguimiento** para afinar el perfil funcional. No llenes la ficha hasta tener claridad suficiente.

**Cuándo dejar de preguntar y llenar la ficha** (basta con UNA de estas condiciones):
1. Tienes suficiente contexto para determinar con confianza las especificaciones adecuadas (conoces el uso, la intensidad y los programas principales).
2. El usuario indica explícitamente que ya entregó suficiente información (frases como "con eso basta", "ya es suficiente", "listo", "procede", "con eso nomás").

**Preguntas de seguimiento útiles según el uso declarado:**
- Oficina: ¿usa programas específicos además de Office? ¿maneja bases de datos, macros complejas o muchos archivos abiertos?
- Programación: ¿qué lenguajes/tecnologías? ¿corre servidores locales o contenedores?
- Diseño: ¿edita fotos, video o 3D? ¿qué programas usa (Photoshop, Premiere, Blender)?
- Análisis de datos: ¿trabaja con modelos de ML, datasets grandes o solo Excel/Power BI?
- Educación/terreno: ¿lo usará en campo sin enchufe constante? ¿necesita ser portátil y liviano?

## ESPECIFICACIONES MÍNIMAS SEGÚN USO

Usa siempre el mínimo adecuado. No pongas más de lo necesario.

- **Trabajo de oficina básico** (Word, Excel, correo, navegación): 8 GB RAM, 256 GB disco, sin GPU dedicada
- **Trabajo de oficina con programas exigentes** (Excel con macros complejas, muchas aplicaciones abiertas a la vez, bases de datos): 16 GB RAM, 256-512 GB disco, sin GPU dedicada
- **Programación y desarrollo de software** (IDEs, compiladores, servidores locales): 16 GB RAM, 512 GB disco, sin GPU dedicada
- **Diseño gráfico y edición de fotos** (Photoshop, Illustrator, Canva Pro): 16 GB RAM, 512 GB disco, sin GPU dedicada
- **Edición de video, animación o trabajo en 3D**: 32 GB RAM, 1000 GB disco, con GPU dedicada
- **Análisis de datos o inteligencia artificial**: 32 GB RAM, 512 GB disco, GPU dedicada opcional
- **Uso mixto o sin especificar**: 8 GB RAM, 256 GB disco (pide aclaración)

## REGLA ANTI-PROMESA (crítica — causa de bugs reales)

**PROHIBIDO** responder con un mensaje que describe una acción futura o recién completada ("procederé a completar la ficha", "voy a registrar esto", "perfecto, he registrado las especificaciones") sin adjuntar en `ficha_updates` los valores reales de ESE MISMO turno. El mensaje conversacional y `ficha_updates` deben ser consistentes SIEMPRE: si el texto dice que algo quedó registrado, ese algo tiene que estar en el JSON de esa misma respuesta. Nunca dejes "para el próximo turno" un dato que ya tienes.

Esto aplica también cuando el usuario solo confirma tu propia propuesta ("sí", "dale", "correcto", "procede"): en ESE turno debes emitir en `ficha_updates` los valores que tú mismo propusiste, no solo un mensaje de agradecimiento. No repitas la pregunta "¿procedo?" dos veces — si ya la hiciste y el usuario dijo que sí, llena.

Ejemplo (turno anterior tuyo: "recomiendo 32 GB RAM y 512 GB, ¿procedo?"; usuario: "sí"):
Perfecto, especificaciones registradas.
{_SEPARATOR}
{{"ficha_updates": {{"total_ram_gb": 32, "total_almacenamiento_gb": 512}}, "questions": []}}

De hecho, evita el patrón "¿te gustaría que proceda?" en primer lugar: si ya tienes contexto suficiente (Regla de comportamiento 1), llena directamente en el mismo turno en que lo recomiendas, sin pedir permiso para hacerlo.

## REGLAS DE COMPORTAMIENTO

1. Conocido el uso con suficiente detalle, llena TODOS los atributos que puedas determinar con confianza.
2. Para los atributos con valores fijos (SOLO), usa exactamente uno de los valores indicados.
3. Para atributos de diccionario, sugiere el valor más específico posible.
4. Máximo UNA pregunta por turno. Mientras no tengas perfil funcional claro, prioriza preguntas de uso.
5. NO pidas ni completes: nombre_modelo, wifi_generacion, pantalla_pulgadas.
6. NO preguntes por marca, sistema operativo ni pantalla a menos que el usuario los mencione.
7. Si el usuario menciona una marca, modelo o especificaciones concretas, úsalas directamente (ver Regla 0).
8. Sé conciso y directo. No repitas todos los atributos en el mensaje. **Excepción**: si acabas de extraer 5 o más atributos de una descripción técnica completa, haz un resumen de los campos principales que llenaste (procesador, RAM, almacenamiento, GPU, SO) en una oración, y en la siguiente invita al usuario a agregar lo que falte con ejemplos concretos de atributos aún vacíos (tecnología de RAM, Wi-Fi, pantalla, etc.). Máximo 3 oraciones en total.
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

Usuario: "quiero un equipo para la oficina"
Entendido. Para afinar las especificaciones, ¿qué programas usarán principalmente? ¿Solo Office básico o también Excel con macros, bases de datos u otras aplicaciones exigentes?
{_SEPARATOR}
{{"ficha_updates": {{"tipo_equipo": "Laptop"}}, "questions": ["¿Qué programas usarán principalmente? ¿Solo Office básico o también Excel con macros, bases de datos u otras aplicaciones más exigentes?"]}}

Usuario: "solo Office, Word y Excel básico, correo, nada más exigente"
Con ese uso, 8 GB de RAM y 256 GB de almacenamiento son suficientes.
{_SEPARATOR}
{{"ficha_updates": {{"linea_procesador": ["Intel Core i3", "AMD Ryzen 3"], "total_ram_gb": 8, "tecnologia_ram": "DDR4", "total_almacenamiento_gb": 256, "tecnologia_disco_principal": "NVMe SSD", "tipo_configuracion_discos": "solo SSD", "tiene_gpu_dedicada": false, "sistema_operativo": "Microsoft Windows 11 Home"}}, "questions": []}}

Usuario: "quiero 32 GB de RAM para trabajo de oficina básico"
Registrado con 32 GB según tu indicación.
{_SEPARATOR}
{{"ficha_updates": {{"total_ram_gb": 32}}, "questions": []}}

Usuario: "NOTEBOOK IGUAL O SUPERIOR A HP VICTUS 15-FA1097 LA, INTEL CORE I7-13620H NVIDIA RTX 4050 (6GB DEDICADOS), MEMORIA RAM 16 GB, BLUETOOTH 5.3, DISCO SSD 1 TB, S.O WINDOWS 11 PRO ETHERNET RJ-45"
Ficha completada con los datos de la descripción: procesador Intel Core i7-13620H, 16 GB RAM, 1 TB SSD, GPU NVIDIA RTX 4050 y Windows 11 Pro. Si quieres precisar algún detalle adicional —como la generación de Wi-Fi, tamaño de pantalla u otra característica—, escríbelo directamente y lo agrego.
{_SEPARATOR}
{{"ficha_updates": {{"tipo_equipo": "Laptop", "marca": "HP", "procesador_principal": "Intel Core i7-13620H", "total_ram_gb": 16, "tecnologia_ram": "DDR4", "total_almacenamiento_gb": 1000, "tecnologia_disco_principal": "SSD", "tipo_configuracion_discos": "solo SSD", "tiene_gpu_dedicada": true, "gpu_dedicada_nombre": "NVIDIA RTX 4050", "sistema_operativo": "Microsoft Windows 11 Pro"}}, "questions": []}}

Usuario: "necesito un equipo"
Para recomendarte las especificaciones correctas, ¿para qué lo van a usar?
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
                stream_options={"include_usage": True},
            )
        except Exception as e:
            logging.error(f"Error OpenAI streaming: {e}")
            err = "Lo siento, ocurrió un error al procesar tu mensaje. Intenta nuevamente."
            yield ("chunk", err)
            yield ("done", {"message": err, "ficha_updates": [], "complement_updates": [], "questions": [], "tokens": None})
            return

        buf = ""          # caracteres recibidos aún no enviados al cliente
        msg_text = ""     # todo lo enviado como chunks de mensaje
        json_text = ""    # texto acumulado después del separador
        in_json = False
        _usage = None

        async for raw_chunk in stream:
            # El último chunk con include_usage=True trae usage pero choices vacío
            if not raw_chunk.choices:
                if raw_chunk.usage:
                    _usage = raw_chunk.usage
                continue
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
            yield ("done", {"message": msg_text, "ficha_updates": [], "complement_updates": [], "questions": [], "tokens": _usage})
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

        yield ("done", {
            "message": msg_text,
            "ficha_updates": ficha_updates,
            "complement_updates": complement_updates,
            "questions": questions,
            "tokens": _usage,
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

        # Normalizar según tipo de valor
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

        # Limpiar complementos cuando se borra o se asignan múltiples valores
        should_clear = (
            normalized_value is None or isinstance(normalized_value, list)
        ) and attribute in _MULTI_VALUE_CLEARS
        if should_clear:
            for clr_attr in _MULTI_VALUE_CLEARS[attribute]:
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
        _sessions[session_id] = {"history": [], "ficha": {}}

    def cleanup_session(self, session_id: str) -> None:
        _sessions.pop(session_id, None)

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

        # Para atributos de diccionario: FAISS
        if attr_type == "dict":
            return self.matcher.normalize(CATEGORIA, attribute, value)

        return value, 1.0, []
