"""
agents/computadores_rules.py

Reglas de negocio y datos de la categoría Computadores, compartidos entre
el chat legacy (`agents/ficha_agent.py`) y el agente de extracción del
wizard nuevo (`agents/computadores_wizard_agent.py`).

Este módulo existe porque, antes de él, TODO el conocimiento de la
categoría (qué atributos existen, cómo distinguir línea de procesador vs
modelo exacto, la regla de neutralidad de marca, los pisos mínimos del
Convenio Marco por gama) vivía exclusivamente dentro del prompt gigante
de `ficha_agent.py` -- si el wizard se escribía aparte, esas reglas se
iban a tener que reinventar o duplicar.

Extraído literal desde `ficha_agent.py` (ver historial de git para el
original) -- es un refactor puro: el texto de cada bloque es
carácter-por-carácter el mismo que estaba inline, solo que ahora vive acá
para que ambos agentes lo importen. `ficha_agent.py` arma su prompt
completo intercalando estos bloques con sus propias secciones
conversacionales (anti-repetición, anti-promesa, borrado, el flujo de
preguntas de seguimiento) -- esas SÍ son específicas del chat multi-turno
y no se movieron acá porque no aplican a una extracción de una sola
pasada como la del wizard.
"""

from agents import convenio_marco_gamas

CATEGORIA = "Computadores"

# Atributos que el LLM puede sugerir (excluye los que se auto-completan por complemento)
FILLABLE_ATTRIBUTES = {
    "tipo_equipo": {
        "description": "Factor de forma del equipo",
        "type": "enum",
        "values": ["Laptop", "AIO", "Desktop", "Otro"],
    },
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
    "marca": {"description": "Fabricante del EQUIPO/laptop-desktop (ej: HP, Dell, Lenovo, Apple, Acer) — NUNCA el fabricante del procesador (Intel/AMD no son valores válidos acá, esos van en procesador_principal/linea_procesador). Opcional, no preguntar salvo que el usuario lo mencione (ver regla de neutralidad de marca)", "type": "dict"},
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
MULTI_VALUE_CLEARS = {
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

# Bajo esta longitud, no se confía en el matching por embeddings (ver
# memoria "embeddings no discriminan CPU" -- strings cortas y ambiguas
# pueden dar score alto contra un vecino sin relación real).
MIN_NORMALIZE_LEN = 4


def attrs_table_text(fillable: dict = FILLABLE_ATTRIBUTES) -> str:
    """La tabla "- **atributo**: descripción → tipo" que arma el prompt a
    partir de FILLABLE_ATTRIBUTES. Idéntica a lo que antes estaba inline
    al principio de _build_system_prompt en ficha_agent.py."""
    attr_table = []
    for atr, meta in fillable.items():
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
    return "\n".join(attr_table)


def gama_section_text() -> str:
    """Bloque completo de "RECONOCIMIENTO DE GAMA" -- se arma aparte (no
    directo en el f-string del prompt) porque es opcional: en el chat solo
    se agrega cuando el mensaje del turno actual realmente menciona una
    gama (ver convenio_marco_gamas.mentions_gama()). El wizard, al ser
    extracción de una sola pasada sobre una descripción completa, la
    incluye siempre -- no hay "turnos" sobre los que decidir si vale la
    pena el costo en tokens."""
    gamas_convenio_marco = convenio_marco_gamas.format_for_prompt()
    return f"""## RECONOCIMIENTO DE GAMA (Convenio Marco vigente)

Si el usuario menciona una "gama" (ej. "laptop gama 1", "necesito algo de gama alta", "equipo gama 3"), reconócelo como un criterio de primer nivel — no lo ignores ni lo trates como si solo hubiera dicho el tipo de equipo. Traduce el término a la gama oficial (2, 3 o 4) y llena la ficha con el PISO MÍNIMO exacto de esa gama, indicando en tu mensaje que son las especificaciones mínimas del Convenio Marco vigente (no una estimación tuya):

{gamas_convenio_marco}

Sinónimos → gama oficial: "básica"/"entrada"/"baja" ≈ por debajo de Gama 2 (este convenio NO define una Gama 1 oficial para Laptop/Desktop/AIO — no inventes specs para "gama 1"; en ese caso trátalo como el nivel de uso más básico posible, igual que "oficina básico" en la tabla de uso más abajo, y acláraselo al usuario). "media"/"intermedia" = Gama 2. "alta"/"avanzada" = Gama 3. "premium"/"workstation" = Gama 4 (solo existe para Desktop en este convenio).

Si la combinación tipo_equipo+gama que pide el usuario no está en la tabla (ej. "AIO gama 4", que no existe en este convenio), dilo explícitamente en vez de inventar una fila que no existe, y ofrece la gama más cercana disponible.

Ejemplo:
Usuario: "necesito laptops gama 2 para la oficina"
Con la Gama 2 del Convenio Marco vigente, estas son las especificaciones mínimas: procesador Intel Core i5 o AMD Ryzen 5, 16 GB de RAM, 960 GB SSD NVMe, Windows 11 Pro.
"""


PROCESADOR_DISTINCTION_TEXT = """**REGLA PROCESADOR — distinción crítica**:
- `procesador_principal`: SOLO cuando el usuario indica un modelo EXACTO con número (ej: "i5-1335U", "Ryzen 7 7745HX"). **NUNCA inventes ni sugeras un modelo específico por tu cuenta.**
- `linea_procesador`: cuando el usuario menciona la familia sin número de modelo (ej: "Intel Core i5", "AMD Ryzen 7", "un i5"). Úsalo siempre que no haya un número de modelo explícito del usuario.

**Regla de oro**: si el número de modelo NO aparece textualmente en el mensaje del usuario, usa `linea_procesador` y deja `procesador_principal` vacío, A MENOS QUE el usuario te pida explícitamente que elijas o sugieras un modelo (frases como "tú elige uno", "sugiéreme uno", "el que sea mejor", "ponle el que corresponda").

Ejemplos:
- "ponle un intel core i5" → `{"linea_procesador": "Intel Core i5"}`  ✓
- "intel core i5, ninguno en específico" → `{"linea_procesador": "Intel Core i5"}`  ✓
- "quiero un i5-1335U" → `{"procesador_principal": "Intel Core i5-1335U"}`  ✓
- "ponle un buen procesador i5" → `{"linea_procesador": "Intel Core i5"}`  ✓  (no inventar modelo)
- "Intel Core i5-6500" en ficha_updates sin que el usuario lo dijera → ✗ PROHIBIDO"""


BRAND_NEUTRALITY_TEXT = """## REGLA PROCESADOR — EVITAR SESGO DE MARCA

Cuando **tú** decidas qué línea de procesador recomendar (el usuario no especificó marca ni modelo), usa SIEMPRE una lista con el equivalente de Intel Y AMD para esa gama de rendimiento — nunca solo una marca. Como comprador público, no se debe favorecer sistemáticamente un fabricante sobre otro cuando ambos ofrecen un producto equivalente.

Equivalencias de gama (cualquiera de las dos cumple el mismo nivel de rendimiento):
- Básica (ofimática simple): Intel Core i3 / AMD Ryzen 3
- Media (multitarea moderada, oficina exigente, desarrollo web): Intel Core i5 / AMD Ryzen 5
- Alta (desarrollo intensivo, edición, análisis de datos): Intel Core i7 / AMD Ryzen 7
- Premium (workstation, deep learning, render 3D): Intel Core i9 / AMD Ryzen 9

Esta regla **NO aplica** si el usuario ya mencionó un fabricante de procesador o de equipo (ej. "que sea Intel", "prefiero un Lenovo con Intel", "quiero AMD") — ahí se respeta su elección (Regla 0) y se usa solo esa opción.

**Tampoco aplica si `marca` ya es Apple**: los Mac actuales no traen procesador Intel ni AMD (usan Apple Silicon, familia M — M1/M2/M3/M4...). Si el equipo es Apple, sugiere directamente una línea Apple M de la gama adecuada, nunca la tabla de equivalencias Intel/AMD de abajo.

**Cuidado — error real observado**: "Intel"/"AMD" son fabricantes de PROCESADOR, nunca van en el atributo `marca` (que es el fabricante del EQUIPO: HP, Lenovo, Acer, Dell, Apple...). Si el usuario dice "quiero AMD" o "prefiero Intel" sin mencionar una marca de equipo, eso llena `linea_procesador`, y el atributo `marca` se deja como está (vacío si no se ha dicho nada de marca de equipo). Nunca escribas "AMD" o "Intel" como valor de `marca`.

Ejemplos:
- "solo Office, nada exigente" (uso definido, sin marca) → `{"linea_procesador": ["Intel Core i3", "AMD Ryzen 3"]}`  ✓
- "algo de gama media, no me importa la marca" → `{"linea_procesador": ["Intel Core i5", "AMD Ryzen 5"]}`  ✓
- "quiero que sea Intel" → `{"linea_procesador": "Intel Core i5"}`  ✓  (marca explícita, no se agrega AMD)
- "prefiero AMD" → `{"linea_procesador": "AMD Ryzen 5"}`  ✓  (idem — NUNCA `{"marca": "AMD"}`)
- "quiero AMD" → `{"marca": "AMD"}`  ✗ PROHIBIDO — AMD no es una marca de equipo"""


MULTI_VALUE_RULE_TEXT = """## REGLA MÚLTIPLES VALORES Y RANGOS

Varios atributos admiten múltiples opciones (array JSON) o rangos (dict con min/max). Úsalos cuando el usuario pide alternativas o un intervalo.

**Atributos que admiten lista de opciones** (`[v1, v2, ...]`):
- Texto/enum: `linea_procesador`, `marca`, `gpu_dedicada_nombre`, `sistema_operativo`, `tipo_equipo`, `tecnologia_ram`, `tecnologia_disco_principal`, `wifi_generacion`
- Numéricos: `total_ram_gb`, `total_almacenamiento_gb`, `pantalla_pulgadas`

**Atributos numéricos que además admiten rango** (`{"min": x, "max": y}`):
`total_ram_gb`, `total_almacenamiento_gb`, `pantalla_pulgadas`

**Atributos que NO admiten lista** (siempre valor único):
- `procesador_principal`: usar siempre string (un modelo exacto). Si el usuario quiere múltiples opciones de procesador, usa `linea_procesador` con lista en su lugar.
- `tiene_gpu_dedicada`: booleano, valor único.

**Regla AGREGAR vs REEMPLAZAR** — clave: mira el historial para saber el valor actual.
- Usuario dice "también", "agrega", "añade", "o también", "además" → **array con el valor anterior + el nuevo**. NUNCA descartes valores anteriores.
- Usuario dice "mejor solo X", "cámbialo a X", "no ese sino X" → **reemplaza** con el nuevo valor (string/número, no array).

Ejemplos (historial previo: `linea_procesador: "Intel Core i5"`, `marca: "HP"`):
- "ponle también AMD Ryzen 5" → `{"linea_procesador": ["Intel Core i5", "AMD Ryzen 5"]}`  ✓
- "agrega Dell como alternativa" → `{"marca": ["HP", "Dell"]}`  ✓
- "entre 16 y 32 GB de RAM" → `{"total_ram_gb": {"min": 16, "max": 32}}`  ✓
- "512 o 1000 GB de almacenamiento" → `{"total_almacenamiento_gb": [512, 1000]}`  ✓
- "pantalla entre 14 y 15.6 pulgadas" → `{"pantalla_pulgadas": {"min": 14.0, "max": 15.6}}`  ✓
- "no ese, mejor solo AMD Ryzen 5" → `{"linea_procesador": "AMD Ryzen 5"}`  ✓  (reemplazo)
- "mínimo 16 GB de RAM" → `{"total_ram_gb": {"min": 16}}`  ✓"""


EXPLICIT_DECISION_RULE_TEXT = """## REGLA 0: RESPETO A DECISIONES EXPLÍCITAS DEL USUARIO

Si el usuario indica explícitamente que quiere algo concreto (una marca, un procesador, una cantidad de RAM, etc.), **acéptalo siempre sin cuestionar ni intentar cambiar su decisión**, aunque no coincida con tu recomendación. Tu rol es asistir, no decidir. Puedes mencionar brevemente una alternativa si es muy relevante, pero en ese mismo turno debes igualmente registrar lo que el usuario pidió.

Ejemplos de decisiones explícitas a respetar:
- "quiero 32 GB de RAM" → registra 32 GB aunque el uso no lo justifique
- "prefiero HP" → registra marca HP
- "lo quiero con GPU dedicada" → registra tiene_gpu_dedicada: true"""


TECHNICAL_DESCRIPTION_RULE_TEXT = """## REGLA DESCRIPCIÓN TÉCNICA (PRIORIDAD MÁXIMA — se aplica antes que cualquier otra)

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
- Tiene formato de lista o especificación de compra pública"""


def min_specs_section_text() -> str:
    """Bloque "ESPECIFICACIONES MÍNIMAS SEGÚN USO" -- los pisos mínimos por
    tipo de uso (oficina básico, oficina exigente, diseño/video/IA), que
    para los usos de Gama 2/3/4 citan los mismos resúmenes de gama que ya
    usa gama_section_text(). Se separó en función (no constante estática)
    porque depende de convenio_marco_gamas.format_gama_summary(), igual
    que en el original."""
    gama2_summary = convenio_marco_gamas.format_gama_summary(2)
    gama3_summary = convenio_marco_gamas.format_gama_summary(3)
    gama4_summary = convenio_marco_gamas.format_gama_summary(4)
    return f"""## ESPECIFICACIONES MÍNIMAS SEGÚN USO

Usa siempre el mínimo adecuado. No pongas más de lo necesario. Cuando el uso declarado corresponda razonablemente a una Gama oficial del Convenio Marco vigente, usa esa Gama como base (mismos valores que en RECONOCIMIENTO DE GAMA, no inventes otros números) y dilo explícitamente en tu mensaje — igual que si el usuario hubiera pedido la gama directamente. Si después el usuario pide algo puntual distinto a lo que sugiere la gama, respeta lo que pide (Regla 0) por sobre la gama — la gama es solo el punto de partida.

**No te olvides de `tipo_equipo` (causa de bugs reales — la estimación de precio depende de este atributo y nunca aparece si falta):** si el mismo mensaje donde describe el uso también dice o deja claro el tipo de equipo ("laptop", "notebook", "desktop", "PC de escritorio", "all-in-one", "AIO"), inclúyelo en `ficha_updates` en ESE MISMO turno, junto con las specs de la gama — no dejes que la atención en mapear RAM/procesador/disco de la gama te haga olvidar este atributo, que es independiente y tan importante como los demás.

- **Trabajo de oficina básico** (Word, Excel, correo, navegación): por debajo de la Gama 2 oficial — 8 GB RAM, 256 GB disco, sin GPU dedicada. Es una estimación propia, no una gama oficial (el Convenio Marco no define una Gama 1 para este uso tan liviano); acláraselo al usuario igual que en RECONOCIMIENTO DE GAMA.
- **Trabajo de oficina con programas exigentes / Programación y desarrollo de software / Diseño gráfico y edición de fotos**: equivale a **Gama 2** del Convenio Marco vigente — {gama2_summary}.
- **Edición de video/animación/3D / Análisis de datos o inteligencia artificial**: equivale a **Gama 3** del Convenio Marco vigente — {gama3_summary}. Si además necesita GPU dedicada fuerte y el equipo es un Desktop, ofrece **Gama 4** en su lugar — {gama4_summary} (Gama 4 solo existe para Desktop en este convenio; para Laptop/AIO no hay equivalente oficial a ese nivel, quedate en Gama 3 y aclara que la GPU dedicada sería una adición propia, no parte de una gama oficial).
- **Uso mixto o sin especificar**: 8 GB RAM, 256 GB disco (pide aclaración)

Al sugerir specs por gama en esta sección (no porque el usuario haya dicho "gama X", sino porque el uso descrito calza con una), responde en el mismo estilo que RECONOCIMIENTO DE GAMA: cita que son las especificaciones de esa Gama del Convenio Marco vigente, no una estimación tuya."""
