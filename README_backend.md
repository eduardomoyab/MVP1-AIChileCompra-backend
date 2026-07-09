# Backend — Asistente IA Compra Ágil

API REST con streaming SSE construida en **FastAPI**. Contiene toda la lógica de negocio: agente conversacional, normalización semántica vectorial (FAISS), estimación de precio histórico (PostgreSQL) y guardrail de seguridad.

> Repositorio: `github.com/eduardomoyab/MVP1-AIChileCompra-backend` · versión `v1.0.0`

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Framework API | FastAPI 0.115+ + Uvicorn |
| LLM principal | OpenAI GPT-4o-mini |
| LLM guardrail | OpenAI GPT-4o-mini (temperatura 0, fijo) |
| Búsqueda semántica | FAISS (`IndexFlatIP`) + OpenAI `text-embedding-3-small` (nube) / sentence-transformers (local) |
| Estimación de precios | PostgreSQL · `PERCENTILE_CONT` sobre tabla `PrecioCA` |
| Base de datos | PostgreSQL (Railway Managed) |
| ORM / queries | SQLAlchemy 2.0 Core |
| Streaming | Server-Sent Events (SSE) |
| Despliegue | Railway |

---

## Arquitectura de módulos

```mermaid
flowchart TD
    EP["main.py\nEndpoints FastAPI\nAuth · CORS · SSE"]

    subgraph agents["agents/"]
        FA["ficha_agent.py\nFichaAgent\nGPT-4o-mini · sesiones · streaming"]
        AM["attribute_matcher.py\nAttributeMatcher\nFAISS + embeddings + complementos"]
        GE["get_embeddings.py\nSingleton encode_fn\nlocal · openai"]
    end

    subgraph services["services/"]
        GR["guardrail_service.py\nGuardrailService\nClasificador LLM · temp=0"]
        PS["price_service.py\nPriceService\nPrecioCA · PERCENTILE_CONT SQL"]
        AN["analytics_service.py\nlog fire-and-forget"]
    end

    subgraph data["Datos"]
        CSV1["diccionarios/attribute_dictionary.csv"]
        CSV2["diccionarios/attribute_complement.csv"]
        DB[("PostgreSQL\nPrecioCA · metricas")]
    end

    EP --> GR
    EP --> FA
    EP --> PS
    EP --> AN
    FA --> AM
    AM --> GE
    AM --> CSV1
    AM --> CSV2
    PS --> DB
    AN --> DB
```

---

## Endpoints

| Método | Ruta | Auth | Streaming | Descripción |
|---|---|---|---|---|
| `GET` | `/health` | No | No | Estado del servicio |
| `GET` | `/api/schema` | No | No | Atributos válidos y valores permitidos |
| `GET` | `/api/dropdowns` | No | No | Valores únicos de BD para editor manual |
| `POST` | `/api/chat/{session_id}` | **Sí** | **SSE** | Chat con agente IA |
| `POST` | `/api/manual_update/{session_id}` | **Sí** | **SSE** | Edición manual de un atributo |
| `GET` | `/api/offers/{session_id}` | **Sí** | No | Historial de OC reales coincidentes |
| `POST` | `/api/track/{session_id}` | **Sí** | No | Evento de analytics del frontend |
| `POST` | `/api/reset/{session_id}` | **Sí** | No | Reiniciar sesión |

Autenticación: header `x-api-key: <FRONTEND_API_KEY>`.

---

## Eventos SSE

Los endpoints `/api/chat` y `/api/manual_update` retornan un stream `text/event-stream` con eventos JSON.

| `type` | Cuándo | Payload |
|---|---|---|
| `thinking` | Inicio del procesamiento | — |
| `blocked` | Guardrail rechazó el mensaje | `message` (razón) |
| `assistant_chunk` | Fragmento de texto (streaming) | `delta: string` |
| `assistant_done` | Texto completo recibido | — |
| `ficha_update` | Atributos actualizados (IA, usuario o complemento) | `updates: array` |
| `questions` | Preguntas de seguimiento | `questions: array` |
| `price_update` | Estimación de precio disponible | `data` (percentiles CLP) |
| `price_not_found` | Sin registros suficientes | — |
| `error` | Error interno | `message` |

---

## Temperatura y modelos LLM

El MVP realiza **dos llamadas LLM independientes** con configuraciones distintas:

| Componente | Modelo | Temperatura | Configurable | Rol |
|---|---|---|---|---|
| `FichaAgent` | `OPENAI_MODEL` (default `gpt-4o-mini`) | **0.2** — env `TEMPERATURE` | **Sí** | Conversación y extracción de atributos. Balance entre respuestas consistentes y lenguaje natural variado. |
| `GuardrailService` | `GUARDRAIL_MODEL` (default `gpt-4o-mini`) | **0** — hardcoded | No | Clasificación de seguridad. Determinístico: la misma entrada siempre produce la misma decisión. |

> Para ajustar el comportamiento del agente (más conservador o más flexible) modificar `TEMPERATURE` en `.env`. No afecta al guardrail.

---

## Lógica de estimación de precio (PriceService)

El precio estimado al comprador se calcula sobre **Órdenes de Compra históricas reales de Compra Ágil** almacenadas en la tabla `PrecioCA` de PostgreSQL.

### Consulta SQL (generada dinámicamente)

```sql
SELECT
    COUNT(*)                                                                        AS n,
    MIN(precio_unitario::numeric)                                                   AS precio_min,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY precio_unitario::numeric)         AS p25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY precio_unitario::numeric)         AS mediana,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY precio_unitario::numeric)         AS p75,
    MAX(precio_unitario::numeric)                                                   AS precio_max,
    ROUND(AVG(precio_unitario::numeric), 0)                                        AS promedio,
    -- ídem para precio_unitario_iva (P25, mediana, P75, min, max)
FROM "PrecioCA"
WHERE
    LOWER(COALESCE(es_accesorio::text, 'false')) != 'true'   -- excluye accesorios
    AND precio_unitario::numeric > 200000                     -- mínimo $200.000 CLP
    AND precio_unitario::numeric < 5000000                    -- máximo $5.000.000 CLP
    AND <filtros dinámicos según atributos de la ficha>
```

### Tipo de filtro por atributo

| Tipo de valor | SQL generado |
|---|---|
| Texto único | `columna ILIKE '%valor%'` |
| Lista de alternativas `[v1, v2]` | `(columna ILIKE '%v1%' OR columna ILIKE '%v2%')` |
| Rango numérico `{min, max}` | `SPLIT_PART(col, ' ', 1)::numeric BETWEEN min AND max` |
| Booleano | `tiene_gpu_dedicada = 'true'` / `'false'` |

### Atributos usados como filtros

`tipo_equipo` · `procesador_principal` · `linea_procesador` · `generacion_procesador` · `nucleos_procesador` · `total_ram_gb` · `tecnologia_ram` · `total_almacenamiento_gb` · `tecnologia_disco_principal` · `tipo_configuracion_discos` · `tiene_gpu_dedicada` · `marca` · `sistema_operativo`

### Resultado

```json
{
  "count": 94,
  "p25": 412000,        "median": 489000,     "p75": 567000,
  "p25_iva": 490280,    "median_iva": 581910, "p75_iva": 674730,
  "min": 298000,        "max": 890000,
  "currency": "CLP",
  "match_attrs": ["tipo_equipo", "linea_procesador", "total_ram_gb", "tiene_gpu_dedicada"],
  "match_description": "tipo, línea proc., RAM, GPU dedicada",
  "broad_warning": false
}
```

`broad_warning: true` cuando el resultado supera 1.000 registros (ficha demasiado genérica). Se requiere al menos 1 registro para reportar estimación.

---

## Ejemplo: input → output

**Endpoint:** `POST /api/chat/{session_id}`  
**Header:** `x-api-key: <FRONTEND_API_KEY>`  
**Body:**
```json
{"message": "Necesito laptops para la oficina, trabajan con Excel con macros y SAP"}
```

**Stream SSE de respuesta:**
```
data: {"type": "thinking"}

data: {"type": "assistant_chunk", "delta": "Para SAP y Excel con macros recomiendo al menos 16 GB RAM y procesador Core i5 o Ryzen 5."}

data: {"type": "assistant_done"}

data: {"type": "ficha_update", "updates": [
  {"attribute": "tipo_equipo",                "value": "Laptop",                    "source": "ai",        "normalized": true,  "score": 1.0},
  {"attribute": "linea_procesador",           "value": "Intel Core i5",            "source": "ai",        "normalized": false, "score": 1.0},
  {"attribute": "total_ram_gb",               "value": 16,                          "source": "ai",        "normalized": false, "score": 1.0},
  {"attribute": "tecnologia_ram",             "value": "DDR4",                      "source": "ai",        "normalized": false, "score": 1.0},
  {"attribute": "total_almacenamiento_gb",    "value": 512,                         "source": "ai",        "normalized": false, "score": 1.0},
  {"attribute": "tecnologia_disco_principal", "value": "NVMe SSD",                 "source": "ai",        "normalized": false, "score": 1.0},
  {"attribute": "tipo_configuracion_discos",  "value": "solo SSD",                 "source": "ai",        "normalized": false, "score": 1.0},
  {"attribute": "tiene_gpu_dedicada",         "value": false,                       "source": "ai",        "normalized": false, "score": 1.0},
  {"attribute": "sistema_operativo",          "value": "Microsoft Windows 11 Pro", "source": "ai",        "normalized": true,  "score": 0.96},
  {"attribute": "frecuencia_ram_mhz",         "value": "3200",                     "source": "complement","triggered_by": "tecnologia_ram"}
]}

data: {"type": "questions", "questions": ["¿Cuántas unidades necesita?", "¿Requiere pantalla táctil?"]}

data: {"type": "price_update", "data": {
  "count": 94,
  "p25": 412000,      "median": 489000,     "p75": 567000,
  "p25_iva": 490280,  "median_iva": 581910, "p75_iva": 674730,
  "min": 298000,      "max": 890000,
  "currency": "CLP",
  "match_attrs": ["tipo_equipo","linea_procesador","total_ram_gb","tecnologia_ram","total_almacenamiento_gb","tiene_gpu_dedicada","sistema_operativo"],
  "match_description": "tipo, línea proc., RAM, tecnología RAM, almacenamiento, GPU dedicada, SO",
  "broad_warning": false
}}

data: [DONE]
```

> El evento `price_update` se emite en el mismo stream SSE, después de los atributos. Los precios provienen de OC históricas de Compra Ágil filtradas por los atributos de la ficha (ver sección SQL más abajo).

### Edición manual de atributo — `POST /api/manual_update/{session_id}`

El usuario edita un campo directamente en la ficha (sin pasar por el chat). El backend aplica la normalización FAISS, recalcula complementos y re-estima el precio.

**Body:**
```json
{"attribute": "total_ram_gb", "value": 32}
```

**Stream SSE de respuesta:**
```
data: {"type": "ficha_update", "updates": [
  {"attribute": "total_ram_gb", "value": 32, "source": "user", "normalized": false, "score": 1.0}
]}

data: {"type": "price_update", "data": {
  "count": 18,
  "p25": 561000,      "median": 648000,     "p75": 742000,
  "p25_iva": 667590,  "median_iva": 771120, "p75_iva": 869980,
  "currency": "CLP",
  "match_attrs": ["tipo_equipo","linea_procesador","total_ram_gb","tecnologia_ram","total_almacenamiento_gb","tiene_gpu_dedicada","sistema_operativo"],
  "match_description": "tipo, línea proc., RAM, tecnología RAM, almacenamiento, GPU dedicada, SO",
  "broad_warning": false
}}

data: [DONE]
```

> Si el atributo editado tiene complementos definidos en `attribute_complement.csv`, el SSE incluye un segundo evento `ficha_update` con los atributos derivados actualizados (p. ej., editar `tecnologia_ram` recalcula `frecuencia_ram_mhz`).

---

### Transacciones históricas — `GET /api/offers/{session_id}`

Llamada separada (no SSE) que devuelve hasta 30 OC reales adjudicadas en Compra Ágil que coinciden con la ficha actual, filtradas dentro del rango P25–P75. Incluye link directo a Mercado Público por cada transacción.

**Respuesta (fragmento con 3 registros):**
```json
{
  "offers": [
    {
      "codigo_requerimiento": "XXXX-1-COT24",
      "precio_unitario":      489000,
      "precio_unitario_iva":  581910,
      "descripcion":          "Laptop Intel Core i5-1235U, 16GB DDR4, 512GB NVMe SSD, Windows 11 Pro",
      "fecha_modificacion":   "2024-03-12",
      "id_oferta_aquiles":    9000001,
      "codigo_oc":            "XXXX-1-AG24",
      "razon_social":         "Proveedor A SpA",
      "oc_codes":             ["XXXX-1-AG24"],
      "oc_urls":              ["https://www.mercadopublico.cl/PurchaseOrder/Modules/PO/DetailsPurchaseOrder.aspx?CodigoOC=XXXX-1-AG24"],
      "ca_url":               "https://buscador.mercadopublico.cl/ficha?code=XXXX-1-COT24",
      "ca_available":         true
    },
    {
      "codigo_requerimiento": "XXXX-2-COT23",
      "precio_unitario":      462000,
      "precio_unitario_iva":  549780,
      "descripcion":          "Notebook Core i5 11va Gen, RAM 16GB, Disco SSD 512GB, SO Win 11 Pro",
      "fecha_modificacion":   "2023-11-08",
      "id_oferta_aquiles":    9000002,
      "codigo_oc":            "XXXX-2-AG23",
      "razon_social":         "Proveedor B Ltda.",
      "oc_codes":             ["XXXX-2-AG23"],
      "oc_urls":              ["https://www.mercadopublico.cl/PurchaseOrder/Modules/PO/DetailsPurchaseOrder.aspx?CodigoOC=XXXX-2-AG23"],
      "ca_url":               "https://buscador.mercadopublico.cl/ficha?code=XXXX-2-COT23",
      "ca_available":         true
    },
    {
      "codigo_requerimiento": "XXXX-3-COT23",
      "precio_unitario":      531000,
      "precio_unitario_iva":  631890,
      "descripcion":          "Laptop i5 12th Gen 16GB RAM 512GB SSD NVMe W11Pro",
      "fecha_modificacion":   "2023-09-21",
      "id_oferta_aquiles":    9000003,
      "codigo_oc":            "XXXX-3-AG23",
      "razon_social":         "Proveedor C S.A.",
      "oc_codes":             ["XXXX-3-AG23"],
      "oc_urls":              ["https://www.mercadopublico.cl/PurchaseOrder/Modules/PO/DetailsPurchaseOrder.aspx?CodigoOC=XXXX-3-AG23"],
      "ca_url":               "https://buscador.mercadopublico.cl/ficha?code=XXXX-3-COT23",
      "ca_available":         true
    }
  ]
}
```

> Los campos `oc_urls` y `ca_url` son links directos a Mercado Público. El frontend los usa para que el comprador pueda revisar cada OC original. `razon_social` es el proveedor adjudicado en esa transacción.

---

## Normalización semántica (FAISS)

```mermaid
flowchart LR
    IN["Valor en texto libre\nej: 'hp victus'"]
    EMB["Embedding\nOpenAI text-embedding-3-small\n(en la nube) o MiniLM (local)"]
    IDX["Índice FAISS\nIndexFlatIP\npor categoria+atributo"]
    TOP["Top-3 candidatos\npor similitud coseno"]
    THR{"score ≥\nSIMILARITY_THRESHOLD\n(default 0.82)?"}
    OK["Valor canónico\nej: 'HP'"]
    ORIG["Valor original\nsin cambios"]

    IN --> EMB --> IDX --> TOP --> THR
    THR -- Sí --> OK
    THR -- No --> ORIG
```

Los índices se construyen al inicio (background thread) y se persisten en `FAISS_CACHE_DIR`. Si cambia `EMBEDDING_PROVIDER` o `EMBEDDING_MODEL`, el caché se invalida automáticamente.

---

## Configuración (`.env`)

| Variable | Default | Requerida | Descripción |
|---|---|---|---|
| `OPENAI_API_KEY` | — | **Sí** | Clave OpenAI para el agente y el guardrail |
| `OPENAI_MODEL` | `gpt-4o-mini` | No | Modelo del agente conversacional (`FichaAgent`) |
| `GUARDRAIL_MODEL` | `gpt-4o-mini` | No | Modelo del clasificador de seguridad |
| `TEMPERATURE` | `0.2` | No | Temperatura del agente conversacional (no afecta al guardrail) |
| `DATABASE_URL` | — | **Sí** | URL PostgreSQL (`postgresql://user:pass@host:port/db`) |
| `FRONTEND_API_KEY` | — | **Sí** | Clave compartida con el frontend para autenticar peticiones |
| `ALLOWED_ORIGINS` | `http://localhost:5000` | No | Orígenes CORS (URL pública del frontend) |
| `EMBEDDING_PROVIDER` | `local` | No | `openai` (en la nube) o `local` (sentence-transformers, ~500 MB RAM) |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | No | `text-embedding-3-small` con `openai`, o nombre de modelo HuggingFace con `local` |
| `SIMILARITY_THRESHOLD` | `0.82` | No | Umbral mínimo FAISS para aceptar normalización (0.0–1.0) |
| `BROAD_SEARCH_THRESHOLD` | `1000` | No | Registros a partir de los cuales se marca `broad_warning` |
| `FAISS_CACHE_DIR` | `./cache/faiss_dict` | No | Directorio para persistir índices FAISS entre reinicios |
| `API_HOST` | `0.0.0.0` | No | Host del servidor Uvicorn |
| `API_PORT` | `8000` | No | Puerto del servidor |

---

## Instalación local

```bash
cd MVP1-AIChileCompra-backend

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

cp .env.example .env           # completar OPENAI_API_KEY, DATABASE_URL, FRONTEND_API_KEY

python main.py
# → http://localhost:8000
# → http://localhost:8000/health
```

> **En la nube (Railway):** `EMBEDDING_PROVIDER=openai` + `EMBEDDING_MODEL=text-embedding-3-small` — sin dependencia local, usa la misma clave `OPENAI_API_KEY`.  
> **Local:** `EMBEDDING_PROVIDER=local` + `EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2` — descarga el modelo (~500 MB RAM) al primer arranque.
