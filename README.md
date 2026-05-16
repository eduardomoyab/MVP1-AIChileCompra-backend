# AIChileCompra — Backend

API REST construida con **FastAPI (Python)** que actúa como núcleo inteligente del asistente de especificación técnica para el catálogo Compra Ágil. Procesa mensajes en lenguaje natural, extrae y estructura atributos de ficha técnica, y estima precios de referencia consultando datos históricos reales de compras públicas.

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Framework API | FastAPI + Uvicorn |
| LLM | OpenAI GPT-4o-mini |
| Búsqueda semántica | FAISS + `paraphrase-multilingual-MiniLM-L12-v2` |
| Estimación de precios (CA) | PostgreSQL + LLM-driven query refinement |
| Estimación de precios (mercado) | LightGBM (modelos `.pkl` preentrenados) |
| Base de datos | PostgreSQL (Railway) |
| ORM / queries | SQLAlchemy (Core) |
| Streaming | Server-Sent Events (SSE) |
| Contenerización | Docker |
| Despliegue | Railway |

---

## Arquitectura general

```mermaid
graph TB
    subgraph Railway["☁️ Railway — Red Privada Interna"]
        FE["Frontend Flask\n:5000"]
        BE["Backend FastAPI\n:8000"]
        PG[("PostgreSQL\nTabla PrecioCA")]
    end

    Browser["🌐 Browser"] -->|HTTPS| FE
    FE -->|HTTP interno · x-api-key| BE
    BE -->|SQLAlchemy| PG
    BE -->|REST API| OAI["☁️ OpenAI\nGPT-4o-mini"]
    BE -.->|"Vector search\n(índice en memoria)"| FAISS["FAISS\nMiniLM embeddings"]

```

El backend **no tiene dominio público expuesto**: solo el frontend puede alcanzarlo mediante la red interna de Railway, eliminando cualquier acceso externo directo.

---

## Endpoints

Todos los endpoints requieren el header `x-api-key` y retornan un stream `text/event-stream` (SSE).

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/chat/{session_id}` | Procesa un mensaje del usuario y actualiza la ficha |
| `POST` | `/api/manual_update/{session_id}` | Aplica una edición manual de atributo |
| `POST` | `/api/reset/{session_id}` | Reinicia el estado de la sesión |

---

## Flujo de una petición (SSE)

```mermaid
sequenceDiagram
    participant B as Browser
    participant F as Flask (Proxy)
    participant A as FastAPI
    participant O as OpenAI
    participant P as PostgreSQL

    B->>F: POST /api/chat/{session_id}
    F->>A: POST /api/chat/{session_id} + x-api-key
    A-->>B: SSE · type: thinking
    A->>O: chat.completions (streaming)
    loop Tokens del LLM
        O-->>A: chunk de texto
        A-->>B: SSE · type: assistant_chunk
    end
    A-->>B: SSE · type: assistant_done
    A-->>B: SSE · type: ficha_update (atributos extraídos)
    A-->>B: SSE · type: ficha_update (atributos complementados)
    A->>P: SELECT estadísticas FROM PrecioCA WHERE ...
    P-->>A: distribución de precios
    A-->>B: SSE · type: price_update / price_not_found
    A->>A: LightGBM predict(ficha)
    A-->>B: SSE · type: lgbm_price_update / lgbm_price_not_found
    A-->>B: data: [DONE]
```

---

## Módulos internos

```mermaid
graph LR
    subgraph main["main.py — FastAPI"]
        EP1["/api/chat"]
        EP2["/api/manual_update"]
        EP3["/api/reset"]
    end

    subgraph agents["agents/"]
        AM["attribute_matcher.py\nNormalización FAISS\nComplemento automático"]
        GE["get_embeddings.py\nSentenceTransformer loader"]
    end

    subgraph services["services/"]
        PS["price_service.py\nEstimación precios CA\nLLM-driven query refinement"]
        LS["lgbm_price_service.py\nEstimación mercado externo\nModelos LightGBM (.pkl)"]
    end

    subgraph data["diccionarios/"]
        D1["attribute_dictionary.csv\nValores canónicos por atributo"]
        D2["attribute_complement.csv\nAtributos derivados automáticos"]
    end

    subgraph models["models/lgbm/"]
        M1["p25/p75/mean_agg_notebooks.pkl"]
        M2["p25/p75/mean_agg_all_in_one.pkl"]
    end

    EP1 --> AM
    EP2 --> AM
    EP1 --> PS
    EP2 --> PS
    EP1 --> LS
    EP2 --> LS
    AM --> GE
    AM --> D1
    AM --> D2
    LS --> M1
    LS --> M2
```

### `agents/attribute_matcher.py`
Carga dos CSVs con el diccionario de atributos y las reglas de complemento. Para cada atributo editable construye un índice FAISS con embeddings de los valores canónicos, permitiendo normalizar valores escritos en lenguaje natural por similitud semántica. El complemento infiere automáticamente atributos derivados: dado un `procesador_principal`, extrae `linea_procesador`, `generacion_procesador`, `nucleos_procesador`, `hilos_procesador` y `frecuencia_turbo_procesador_mhz`.

### `services/price_service.py`
Consulta la tabla `PrecioCA` en PostgreSQL con filtros dinámicos sobre los atributos de la ficha. Si la consulta inicial no retorna suficientes registros, invoca al LLM para reformular la query relajando atributos según una jerarquía de especificidad (hasta 6 iteraciones). Los registros se filtran por `precio_unitario` entre 200 000 y 5 000 000 CLP (sin IVA).

### `services/lgbm_price_service.py`
Estima el precio de mercado externo usando seis modelos LightGBM preentrenados (P25, P75 y media para `notebooks` y `all_in_one`). Construye un DataFrame de una fila con los atributos de la ficha (incluyendo features temporales: `year`, `month`, `week_of_year`, `days_since_start`, `semester_idx`), convierte columnas numéricas con `pd.to_numeric(..., errors="coerce")` para manejar valores `None` sin errores de dtype, y devuelve P25, P75 y media en CLP. Los modelos se cargan una sola vez al iniciar la aplicación.

### `agents/ficha_agent.py`
Gestiona el estado de la sesión y la conversación con el LLM. Extrae y estructura los atributos de la ficha técnica a partir del lenguaje natural. Los atributos marcados como `COMPLEMENT_ONLY` (e.g., `linea_procesador`, `generacion_procesador`) son derivados automáticamente por regex desde `procesador_principal` y nunca son solicitados directamente al usuario.

---

## Modelo de datos — tabla `PrecioCA`

Cada fila representa una orden de compra real del catálogo Compra Ágil.

```mermaid
erDiagram
    PrecioCA {
        text tipo_equipo "Laptop / AIO / Desktop / Otro"
        text procesador_principal "Nombre completo del procesador"
        text linea_procesador "Core i5 / Ryzen 7 / Apple M2..."
        text generacion_procesador "13th Gen / Zen 4 / M2..."
        integer nucleos_procesador
        integer hilos_procesador
        numeric total_ram_gb "8 / 16 / 32 GB"
        text tecnologia_ram "DDR5 / DDR4 / LPDDR5X / LPDDR4..."
        numeric total_almacenamiento_gb "256 / 512 / 1024 GB"
        text tecnologia_disco_principal "NVMe SSD / SATA SSD / HDD / eMMC"
        text tipo_configuracion_discos "solo SSD / SSD+HDD / solo HDD"
        boolean tiene_gpu_dedicada
        text marca "HP / Dell / Lenovo / Apple..."
        text sistema_operativo "Windows 11 / macOS / Linux..."
        numeric precio_unitario "Precio neto en CLP"
        numeric precio_unitario_iva "Precio con IVA en CLP"
        boolean es_accesorio "Se excluyen accesorios de las consultas"
    }
```

---

## Lógica de estimación de precios

```mermaid
flowchart TD
    A["Ficha técnica del usuario"] --> B["Query inicial\ncon todos los atributos disponibles"]
    B --> C{"¿≥ 5 registros\ncoincidentes?"}
    C -->|Sí| D["✅ Retorna distribución\nde precios históricos"]
    C -->|No| E["LLM elige qué atributos\nrelajar según jerarquía"]
    E --> F{"¿Combinación válida\ny no repetida?"}
    F -->|No| G["❌ price_not_found"]
    F -->|Sí| H{"¿Iteración < 6?"}
    H -->|No| G
    H -->|Sí| B

```

**Jerarquía de especificidad del procesador:**

`procesador_principal` → `linea_procesador + generacion_procesador` → `nucleos_procesador` → _(sin procesador)_

---

## Variables de entorno

```env
OPENAI_API_KEY=           # API key de OpenAI
OPENAI_MODEL=gpt-4o-mini  # Modelo LLM

# Proveedor de embeddings: "openai" (~0 MB RAM, llama a la API) o "local" (~500 MB RAM, SentenceTransformer)
EMBEDDING_PROVIDER=openai
# Modelo según el proveedor elegido:
#   openai → text-embedding-3-small
#   local  → paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_MODEL=text-embedding-3-small

DATABASE_URL=             # Connection string PostgreSQL (interno Railway)
FRONTEND_API_KEY=         # Clave compartida con el frontend para autenticación
ALLOWED_ORIGINS=          # URL pública del frontend (CORS)
FAISS_CACHE_DIR=./cache/faiss_dict
TEMPERATURE=0.2
SIMILARITY_THRESHOLD=0.82
PORT=8000
```

> **Cambio de proveedor:** al modificar `EMBEDDING_PROVIDER` o `EMBEDDING_MODEL`, el caché FAISS se elimina automáticamente en el próximo arranque y se reconstruye con el nuevo proveedor. No es necesario limpiar la carpeta manualmente.

---

## Ejecución local

```bash
cd MVP1-AIChileCompra-backend
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# Crear .env con las variables requeridas
uvicorn main:app --reload --port 8000
```

## Despliegue (Railway)

Railway detecta el `Dockerfile` automáticamente. Las variables de entorno se configuran en **Railway → Backend service → Variables**. El servicio no necesita dominio público habilitado: opera exclusivamente dentro de la red privada interna de Railway.
