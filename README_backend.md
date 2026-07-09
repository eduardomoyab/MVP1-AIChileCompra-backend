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
| Búsqueda semántica | FAISS (`IndexFlatIP`) + sentence-transformers |
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

## Normalización semántica (FAISS)

```mermaid
flowchart LR
    IN["Valor en texto libre\nej: 'hp victus'"]
    EMB["Embedding\nparaphrase-multilingual\nMiniLM-L12-v2"]
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
| `EMBEDDING_PROVIDER` | `local` | No | `local` (sentence-transformers, ~500 MB RAM) u `openai` (API) |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | No | Modelo de embeddings (solo si `EMBEDDING_PROVIDER=local`) |
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

> Para entornos con poca RAM: usar `EMBEDDING_PROVIDER=openai` y comentar `sentence-transformers` en `requirements.txt`.
