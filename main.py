"""
main.py — Backend Asistente Compra Ágil
FastAPI + HTTP SSE

Endpoints:
  POST /api/chat/{session_id}                     → Chat streaming (SSE) — Computadores
  POST /api/manual_update/{session_id}             → Actualización manual de atributo (SSE) — Computadores
  GET  /api/sessions                               → Lista de conversaciones del usuario (sidebar) — Computadores
  GET  /api/sessions/{session_id}                  → Ficha+mensajes+carrito+precio de una conversación — Computadores
  POST /api/sessions/{session_id}/rename           → Renombrar conversación — Computadores
  POST /api/sessions/{session_id}/delete           → Eliminar conversación — Computadores
  GET  /api/compare/{session_id}                   → Carrito de comparación de esa conversación — Computadores
  POST /api/compare/{session_id}                   → Guardar carrito de comparación de esa conversación — Computadores
  POST /api/medicamentos/analizar/{session_id}     → Análisis de texto libre por LLM (SSE) — Medicamentos
  POST /api/medicamentos/manual_update/{session_id}→ Actualización manual de atributo (SSE) — Medicamentos
  GET  /api/medicamentos/facets/{session_id}       → Valores reales+conteo de un atributo, acotados por lo ya elegido
  GET  /api/medicamentos/precio_por_unidad/{session_id} → Precio (p25/mediana/p75) por cada unidad_venta -- para comparar antes de elegir
  GET  /api/medicamentos/historial/{session_id}    → Compras anteriores acotadas (acordeón opcional)
  POST /api/medicamentos/describe                  → Descripciones IA por requerimiento (para el PDF del carrito)
  POST /api/medicamentos/reset/{session_id}        → Resetear sesión — Medicamentos
  GET  /health                                     → Estado del servicio
  GET  /api/schema                                 → Esquema de atributos y valores válidos — Computadores (legacy)
  GET  /api/medicamentos/schema                    → Esquema de atributos y valores válidos — Medicamentos

  Wizard genérico por categoría (ver plan de implementación -- Fase 1 solo
  Computadores registrado; "/api/categories/medicamentos/*" da 404 a propósito):
  GET  /api/categories/{slug}/schema                    → Esquema de atributos de la categoría
  POST /api/categories/{slug}/analizar/{session_id}     → Análisis de texto libre por LLM (SSE)
  POST /api/categories/{slug}/manual_update/{session_id}→ Actualización manual de atributo (SSE)
  GET  /api/categories/{slug}/facets/{session_id}       → Valores válidos de un atributo, acotados por lo ya elegido
  GET  /api/categories/{slug}/price/{session_id}        → Precio Compra Ágil + Convenio Marco (si la categoría lo soporta)
  POST /api/categories/{slug}/describe                  → Descripciones IA por ítem del carrito (para el PDF)
  POST /api/categories/{slug}/reset/{session_id}        → Resetear sesión
"""

import os
import json
import logging
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv

from agents.attribute_matcher import AttributeMatcher
from agents.ficha_agent import FichaAgent, FILLABLE_ATTRIBUTES
from agents.medicamento_agent import MedicamentoAgent, FILLABLE_ATTRIBUTES_MED, CORE_ATTRS_MED
from agents.computadores_wizard_agent import ComputadoresWizardAgent
from services.price_service import PriceService
from services.cm_service import CMService
from services import currency_service
from services.lgbm_price_service import LgbmPriceService
from services.guardrail_service import GuardrailService
from services import analytics_service
from services import access_service
from services import admin_service
from services import usage_service
from services import chat_session_service
from services import category_session_service
from services.medicamento_service import MedicamentoService
from services import category_service
import categories.schema as categories_schema
from categories import computadores_schema  # noqa: F401 -- side-effect: registra el esquema de Computadores

load_dotenv()
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

matcher = AttributeMatcher()
price_service = PriceService()
cm_service = CMService()
medicamento_service = MedicamentoService()
lgbm_service = LgbmPriceService()
guardrail = GuardrailService()
agent: FichaAgent = None
medicamento_agent: MedicamentoAgent = None
computadores_wizard_agent: ComputadoresWizardAgent = None

# Historial liviano por sesión para el guardrail (máx 20 mensajes por sesión)
_session_histories: dict[str, list[dict]] = {}
_HISTORY_MAX = 20

# Caché del último estimate por sesión para evitar re-query en /api/offers
_session_price_cache: dict[str, dict] = {}
_session_cm_price_cache: dict[str, dict] = {}

# Mismo propósito que _session_price_cache, pero para el wizard genérico de
# categorías -- ver /api/categories/{slug}/price y /api/categories/{slug}/offers.
_category_price_cache: dict[str, dict] = {}

# ─── API Key ──────────────────────────────────────────────────────────────────

_raw_keys = os.getenv("FRONTEND_API_KEY", "")
API_KEYS = {k.strip() for k in _raw_keys.split(",") if k.strip()}

_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def require_api_key(x_api_key: Optional[str] = Depends(_api_key_header)):
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API Key inválida")
    return x_api_key


# Correo de la persona logueada, mandado por el frontend (que ya lo sabe vía
# su propia sesión) en el header x-user-email. Confiable porque solo el
# frontend conoce FRONTEND_API_KEY, mismo modelo de confianza que el resto
# del backend -- no hay verificación de identidad adicional por request.
_user_email_header = APIKeyHeader(name="x-user-email", auto_error=False)

async def get_user_email(x_user_email: Optional[str] = Depends(_user_email_header)) -> str:
    return (x_user_email or "").strip().lower()


async def require_admin(
    user_email: str = Depends(get_user_email), _: str = Depends(require_api_key)
) -> str:
    """Gate de los endpoints /api/admin/* -- el correo debe pertenecer al
    grupo 'Admins' de mvp1-compra-agil (ver access_service.is_admin)."""
    if not await asyncio.to_thread(access_service.is_admin, user_email):
        raise HTTPException(status_code=403, detail="No tienes acceso al panel de administrador")
    return user_email


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, medicamento_agent, computadores_wizard_agent
    agent = FichaAgent(matcher)
    medicamento_agent = MedicamentoAgent(matcher)
    computadores_wizard_agent = ComputadoresWizardAgent(matcher)
    lgbm_service._load()
    usage_service.ensure_table()
    analytics_service.ensure_table()
    chat_session_service.ensure_table()
    category_session_service.ensure_table()
    logging.info("Servidor listo — construyendo índices FAISS y dropdowns en background...")
    asyncio.create_task(asyncio.to_thread(matcher.warm))
    asyncio.create_task(asyncio.to_thread(price_service.warmup_dropdowns))
    asyncio.create_task(asyncio.to_thread(cm_service.warmup))
    asyncio.create_task(asyncio.to_thread(currency_service.warmup))
    yield
    logging.info("Servidor apagado")


app = FastAPI(
    title="Asistente Compra Ágil",
    version="2.0.0",
    lifespan=lifespan,
)

_allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5000").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "x-api-key"],
)

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ─── Modelos ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    content: str

class ManualUpdateRequest(BaseModel):
    attribute: str
    value: Optional[Any] = None


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Asistente Compra Ágil"}


# ─── Acceso (whitelist consultada por el frontend en el login) ────────────────

@app.get("/api/auth/check_access")
async def check_access_endpoint(email: str, _: str = Depends(require_api_key)):
    allowed = await asyncio.to_thread(access_service.is_email_allowed, email)
    sections = await asyncio.to_thread(access_service.get_allowed_sections, email) if allowed else []
    is_admin = await asyncio.to_thread(access_service.is_admin, email) if allowed else False
    return {"allowed": allowed, "sections": sections, "is_admin": is_admin}


# ─── Uso diario de tokens (panel de cuenta) ────────────────────────────────────

@app.get("/api/usage")
async def usage_endpoint(user_email: str = Depends(get_user_email), _: str = Depends(require_api_key)):
    usage = await asyncio.to_thread(usage_service.get_usage, user_email)
    return usage_service.to_public(usage)


# ─── Comparador de candidatos (persistencia por conversación) ─────────────────
# El comparador de equipos (ver 1.3 en la bitácora) vivía en memoria del
# navegador y luego por usuario a secas -- ahora es por conversación, como el
# resto de lo que compone una sesión de chat (ver chat_session_service.py).
# El frontend manda la lista completa de candidatos ya normalizada (ver
# _candidateFromCA/_candidateFromCM en app.js), acá solo se guarda/devuelve
# tal cual, sin reconstruir nada.

class CompareSaveRequest(BaseModel):
    items: list

@app.get("/api/compare/{session_id}")
async def get_compare_endpoint(
    session_id: str, user_email: str = Depends(get_user_email), _: str = Depends(require_api_key)
):
    session = await asyncio.to_thread(chat_session_service.get_full_session, session_id, user_email)
    return {"items": session["compare_items"] if session else []}

@app.post("/api/compare/{session_id}")
async def save_compare_endpoint(
    session_id: str,
    body: CompareSaveRequest,
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    await asyncio.to_thread(chat_session_service.save_compare_items, session_id, user_email, body.items)
    return {"ok": True}


# ─── Conversaciones (sidebar, Computadores) ────────────────────────────────────
# Reemplaza el modelo de "una sola sesión que se pisa con el botón Nueva
# sesión" por varias conversaciones guardadas por usuario -- ficha, mensajes
# del chat y carrito de comparación, todo scoped al session_id de cada una
# (chat_session_service.py) y nunca visible entre usuarios distintos
# (ownership check por email en cada lectura/escritura). El precio (Compra
# Ágil y Convenio Marco) NUNCA se persiste -- se recalcula en caliente acá
# mismo a partir de la ficha, igual que tras cada turno de chat.

class RenameSessionRequest(BaseModel):
    title: str


def _recompute_prices(session_id: str, ficha: dict) -> tuple[Optional[dict], Optional[dict]]:
    price_data = None
    cm_price_data = None
    if not ficha.get("tipo_equipo"):
        return None, None
    try:
        price_data = price_service.estimate(ficha)
        if price_data:
            _session_price_cache[session_id] = price_data
        else:
            _session_price_cache.pop(session_id, None)
    except Exception as e:
        logging.warning(f"Error en estimación de precio (sessions): {e}")
    try:
        cm_price_data = cm_service.estimate(ficha)
        if cm_price_data:
            _session_cm_price_cache[session_id] = cm_price_data
        else:
            _session_cm_price_cache.pop(session_id, None)
    except Exception as e:
        logging.warning(f"Error en estimación de precio Convenio Marco (sessions): {e}")
    return price_data, cm_price_data


@app.get("/api/sessions")
async def list_sessions_endpoint(user_email: str = Depends(get_user_email), _: str = Depends(require_api_key)):
    sessions = await asyncio.to_thread(chat_session_service.list_sessions, user_email)
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def get_session_endpoint(
    session_id: str, user_email: str = Depends(get_user_email), _: str = Depends(require_api_key)
):
    session = await asyncio.to_thread(chat_session_service.get_full_session, session_id, user_email)
    if not session:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")

    # Re-sembrar el contexto del LLM (ficha + historial) en memoria, para que
    # el agente siga la conversación sin perder contexto tras el cambio.
    ficha = await asyncio.to_thread(agent.get_ficha, session_id)

    # Re-sembrar el historial liviano del guardrail (incluye turnos
    # bloqueados, que el agente principal nunca vio pero el guardrail sí
    # necesita para la REGLA DE CONTEXTO).
    history = [
        {"role": m["role"], "content": m["content"][:400] if m["role"] == "assistant" else m["content"]}
        for m in session["messages"]
        if m.get("role") in ("user", "assistant")
    ]
    _session_histories[session_id] = history[-_HISTORY_MAX:]

    price_data, cm_price_data = await asyncio.to_thread(_recompute_prices, session_id, ficha)

    return {
        "title": session["title"],
        "ficha": ficha,
        "messages": session["messages"],
        "compare_items": session["compare_items"],
        "price_data": price_data,
        "cm_price_data": cm_price_data,
    }


@app.post("/api/sessions/{session_id}/rename")
async def rename_session_endpoint(
    session_id: str,
    body: RenameSessionRequest,
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    ok = await asyncio.to_thread(chat_session_service.rename, session_id, user_email, body.title)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    return {"ok": True}


@app.post("/api/sessions/{session_id}/delete")
async def delete_session_endpoint(
    session_id: str, user_email: str = Depends(get_user_email), _: str = Depends(require_api_key)
):
    ok = await asyncio.to_thread(chat_session_service.delete, session_id, user_email)
    agent.cleanup_session(session_id)
    _session_histories.pop(session_id, None)
    _session_price_cache.pop(session_id, None)
    _session_cm_price_cache.pop(session_id, None)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    return {"ok": True}


# ─── Schema ───────────────────────────────────────────────────────────────────

@app.get("/api/schema")
async def get_schema():
    schema = {}
    for attr, meta in FILLABLE_ATTRIBUTES.items():
        entry = {"description": meta["description"], "type": meta["type"]}
        if meta.get("values"):
            entry["values"] = meta["values"]
            entry["valid_values"] = matcher.get_valid_values("Computadores", attr)
        schema[attr] = entry
    return {"categoria": "Computadores", "attributes": schema}


# ─── Dropdowns ────────────────────────────────────────────────────────────────

# Atributos que se resuelven desde el diccionario FAISS (no están en PrecioCA)
_DICT_DROPDOWN_ATTRS = [
    "gpu_dedicada_nombre",
    "total_vram_gpu_gb",
]

@app.get("/api/dropdowns")
async def get_dropdowns():
    db_values = await asyncio.to_thread(price_service.get_dropdown_values)
    for attr in _DICT_DROPDOWN_ATTRS:
        db_values[attr] = matcher.get_valid_values("Computadores", attr)
    return db_values


@app.get("/api/dropdowns/filtered")
async def get_filtered_dropdown(
    field: str,
    marca: Optional[str] = None,
    tipo_equipo: Optional[str] = None,
):
    """Opciones de un solo campo acotadas por marca/tipo_equipo -- para que
    el editor de Procesador (u otro atributo correlacionado con la marca) no
    ofrezca combinaciones que no existen en la realidad (ej. procesadores
    AMD/Intel para un equipo Apple)."""
    values = await asyncio.to_thread(
        price_service.get_filtered_dropdown_values, field, marca, tipo_equipo
    )
    return {"values": values}


# ─── Chat (SSE) ───────────────────────────────────────────────────────────────

@app.post("/api/chat/{session_id}")
async def chat_endpoint(
    session_id: str,
    body: ChatRequest,
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    content = body.content.strip()
    if not content:
        async def empty():
            yield sse({"type": "error", "message": "Mensaje vacío"})
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream", headers=_SSE_HEADERS)

    async def event_gen():
        t0 = time.perf_counter()
        try:
            # ── Límite diario de tokens: se chequea ANTES de gastar nada,
            # ni siquiera el guardrail corre si ya está bloqueado — es lo
            # que de verdad ahorra el gasto, no solo lo reporta después.
            if await asyncio.to_thread(usage_service.is_blocked, user_email):
                usage_info = await asyncio.to_thread(usage_service.get_usage, user_email)
                yield sse({"type": "usage_limit_reached", "data": usage_service.to_public(usage_info)})
                yield "data: [DONE]\n\n"
                analytics_service.log(
                    session_id=session_id,
                    user_email=user_email,
                    tipo="usage_blocked",
                    user_msg=content,
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                    blocked=True,
                    block_reason="límite diario de tokens alcanzado",
                )
                return

            # ── Guardrail: validar mensaje antes de procesarlo ──────────────
            history = _session_histories.get(session_id, [])
            allowed, block_reason, clean_message, friendly_reply, guardrail_usage = await guardrail.check(content, history)
            guardrail_tokens = guardrail_usage.total_tokens if guardrail_usage else 0

            if not allowed:
                logging.info(f"[guardrail] Bloqueado session={session_id}: {block_reason}")
                # El mensaje "message" acá es friendly_reply -- una respuesta
                # conversacional (la escribe el mismo clasificador), no el
                # motivo interno de bloqueo. El frontend la muestra como un
                # turno normal del chat, no como el antiguo cartel fijo de
                # "Consulta fuera del ámbito" que cortaba la conversación.
                # El texto original del usuario NUNCA llega al agente
                # principal en este caso -- esa sigue siendo la capa de
                # seguridad real; lo único que cambió es cómo se comunica.
                yield sse({"type": "blocked", "message": friendly_reply})
                yield "data: [DONE]\n\n"
                analytics_service.log(
                    session_id=session_id,
                    user_email=user_email,
                    tipo="blocked",
                    user_msg=content,
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                    blocked=True,
                    block_reason=block_reason,
                )
                # Se guarda en el historial liviano del guardrail (no en el
                # del agente principal) para que la REGLA DE CONTEXTO pueda
                # seguir reconociendo referencias a este intercambio en el
                # turno siguiente (ej. "por qué no puedes ayudarme con eso").
                history.append({"role": "user", "content": content})
                history.append({"role": "assistant", "content": friendly_reply})
                if len(history) > _HISTORY_MAX:
                    history = history[-_HISTORY_MAX:]
                _session_histories[session_id] = history
                # El guardrail igual gastó tokens aunque el mensaje se bloqueara.
                await asyncio.to_thread(usage_service.add_usage, user_email, guardrail_tokens)
                # Transcript limpio para el sidebar -- el turno bloqueado
                # también se muestra (con blocked=true) aunque nunca haya
                # llegado al agente principal.
                await asyncio.to_thread(
                    chat_session_service.append_messages, session_id, user_email,
                    [
                        {"role": "user", "content": content, "blocked": True},
                        {"role": "assistant", "content": friendly_reply, "blocked": True},
                    ],
                )
                return

            # Usar la versión limpia si el guardrail extrajo solo la parte válida
            effective_content = clean_message if clean_message else content
            if clean_message:
                logging.info(f"[guardrail] Mensaje parcial limpiado session={session_id}: {block_reason}")

            # Registrar mensaje original del usuario en el historial
            history.append({"role": "user", "content": content})
            if len(history) > _HISTORY_MAX:
                history = history[-_HISTORY_MAX:]
            _session_histories[session_id] = history

            yield sse({"type": "thinking"})

            result = {}
            async for event_type, event_data in agent.stream_process_message(session_id, effective_content, user_email):
                if event_type == "chunk":
                    yield sse({"type": "assistant_chunk", "delta": event_data})
                elif event_type == "done":
                    result = event_data

            yield sse({"type": "assistant_done"})

            # Transcript limpio para el sidebar (se guarda el texto original
            # del usuario, no la versión limpiada por el guardrail).
            await asyncio.to_thread(
                chat_session_service.append_messages, session_id, user_email,
                [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": result.get("message", "")},
                ],
            )

            # Registrar respuesta del asistente para contexto futuro del guardrail
            if result.get("message"):
                history.append({"role": "assistant", "content": result["message"][:400]})
                _session_histories[session_id] = history[-_HISTORY_MAX:]

            if result.get("ficha_updates"):
                yield sse({"type": "ficha_update", "updates": result["ficha_updates"]})

            if result.get("complement_updates"):
                yield sse({"type": "ficha_update", "updates": result["complement_updates"]})

            if result.get("questions"):
                yield sse({"type": "questions", "questions": result["questions"]})

            price_found = False
            ficha = agent.get_ficha(session_id)
            if ficha.get("tipo_equipo"):
                try:
                    price = await asyncio.to_thread(price_service.estimate, ficha)
                    if price:
                        _session_price_cache[session_id] = price
                        price_found = True
                        yield sse({"type": "price_update", "data": price})
                    else:
                        _session_price_cache.pop(session_id, None)
                        yield sse({"type": "price_not_found"})
                except Exception as e:
                    logging.warning(f"Error en estimación de precio: {e}")
                    yield sse({"type": "price_not_found"})

                try:
                    cm_price = await asyncio.to_thread(cm_service.estimate, ficha)
                    if cm_price:
                        _session_cm_price_cache[session_id] = cm_price
                        yield sse({"type": "cm_price_update", "data": cm_price})
                    else:
                        _session_cm_price_cache.pop(session_id, None)
                        yield sse({"type": "cm_price_not_found"})
                except Exception as e:
                    logging.warning(f"Error en estimación de precio Convenio Marco: {e}")
                    yield sse({"type": "cm_price_not_found"})

            yield "data: [DONE]\n\n"

            usage = result.get("tokens")
            ficha_updates = result.get("ficha_updates", [])
            analytics_service.log(
                session_id=session_id,
                user_email=user_email,
                tipo="chat",
                user_msg=content,
                ai_msg=result.get("message", ""),
                tokens_in=usage.prompt_tokens if usage else None,
                tokens_out=usage.completion_tokens if usage else None,
                duration_ms=int((time.perf_counter() - t0) * 1000),
                n_updates=len(ficha_updates),
                price_found=price_found,
                model=agent.model,
                attrs_updated=",".join(u["attribute"] for u in ficha_updates) or None,
            )
            ficha_tokens = usage.total_tokens if usage else 0
            await asyncio.to_thread(usage_service.add_usage, user_email, guardrail_tokens + ficha_tokens)

        except Exception as e:
            logging.error(f"Error en chat SSE {session_id}: {e}", exc_info=True)
            yield sse({"type": "error", "message": "Error interno del servidor"})
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


# ─── Manual update (SSE) ─────────────────────────────────────────────────────

@app.post("/api/manual_update/{session_id}")
async def manual_update_endpoint(
    session_id: str,
    body: ManualUpdateRequest,
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    if not body.attribute:
        async def err():
            yield sse({"type": "error", "message": "Atributo requerido"})
            yield "data: [DONE]\n\n"
        return StreamingResponse(err(), media_type="text/event-stream", headers=_SSE_HEADERS)

    async def event_gen():
        t0 = time.perf_counter()
        try:
            result = agent.apply_manual_update(session_id, body.attribute, body.value, user_email)

            if result["updates"]:
                yield sse({"type": "ficha_update", "updates": result["updates"]})

            if result["complement_updates"]:
                yield sse({"type": "ficha_update", "updates": result["complement_updates"]})

            price_found = False
            ficha = agent.get_ficha(session_id)
            if ficha.get("tipo_equipo"):
                try:
                    price = await asyncio.to_thread(price_service.estimate, ficha)
                    if price:
                        _session_price_cache[session_id] = price
                        price_found = True
                        yield sse({"type": "price_update", "data": price})
                    else:
                        _session_price_cache.pop(session_id, None)
                        yield sse({"type": "price_not_found"})
                except Exception as e:
                    logging.warning(f"Error en estimación de precio: {e}")
                    yield sse({"type": "price_not_found"})

                try:
                    cm_price = await asyncio.to_thread(cm_service.estimate, ficha)
                    if cm_price:
                        _session_cm_price_cache[session_id] = cm_price
                        yield sse({"type": "cm_price_update", "data": cm_price})
                    else:
                        _session_cm_price_cache.pop(session_id, None)
                        yield sse({"type": "cm_price_not_found"})
                except Exception as e:
                    logging.warning(f"Error en estimación de precio Convenio Marco: {e}")
                    yield sse({"type": "cm_price_not_found"})

            yield "data: [DONE]\n\n"

            analytics_service.log(
                session_id=session_id,
                user_email=user_email,
                tipo="manual_update",
                user_msg=f"{body.attribute}={body.value}",
                duration_ms=int((time.perf_counter() - t0) * 1000),
                n_updates=len(result.get("updates", [])),
                price_found=price_found,
                attrs_updated=body.attribute,
            )

        except Exception as e:
            logging.error(f"Error en manual_update SSE {session_id}: {e}", exc_info=True)
            yield sse({"type": "error", "message": "Error interno"})
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


# ─── Offers ───────────────────────────────────────────────────────────────────

_CA_PO_BASE       = "https://www.mercadopublico.cl/PurchaseOrder/Modules/PO/DetailsPurchaseOrder.aspx"
_CA_BUSCADOR_BASE = "https://buscador.mercadopublico.cl/ficha"


def _enrich_offers(rows: list) -> list:
    result = []
    for row in rows:
        req = row["codigo_requerimiento"]
        oc_code = row.get("codigo_oc")
        oc_codes = [oc_code] if oc_code else []
        result.append({
            **row,
            "oc_codes":     oc_codes,
            "oc_urls":      [f"{_CA_PO_BASE}?CodigoOC={c}" for c in oc_codes],
            "ca_url":       f"{_CA_BUSCADOR_BASE}?code={req}",
            "ca_available": bool(req),
        })
    return result


@app.get("/api/offers/{session_id}")
async def get_offers_endpoint(
    session_id: str,
    price_min: Optional[int] = None,
    price_max: Optional[int] = None,
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    ficha = agent.get_ficha(session_id)
    if not ficha.get("tipo_equipo"):
        return {"offers": []}

    # Mismos atributos que se relajaron para llegar al precio ya mostrado
    # (cacheado en /api/chat al llamar price_service.estimate()) -- si no se
    # aplica el mismo relajo acá, esta lista filtra con la ficha completa sin
    # relajar y puede devolver 0 filas mientras el panel de precio, arriba,
    # sigue mostrando el conteo relajado (bug real: "77 ofertas" arriba,
    # "sin transacciones" en el historial, al mismo tiempo).
    relaxed_attrs = _session_price_cache.get(session_id, {}).get("relaxed_attrs")

    # Sin filtro por default -- se muestran TODAS las transacciones que
    # calzan con la ficha (antes se acotaba en silencio al rango p25-p75,
    # lo que hacía parecer que había menos evidencia de la que en realidad
    # hay). Solo se acota si el usuario aplicó su propio filtro de precio
    # (price_min/price_max en la query, ver sección 3.2 del feedback).
    rows = await asyncio.to_thread(price_service.get_offer_rows, ficha, 30, price_min, price_max, relaxed_attrs)
    analytics_service.log(session_id=session_id, user_email=user_email, tipo="ver_historial")
    return {"offers": _enrich_offers(rows)}


@app.get("/api/price_methodology/{session_id}")
async def get_price_methodology_endpoint(
    session_id: str,
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    """Trazabilidad de la estimación de Compra Ágil: cantidad de
    transacciones, período y desagregación por tipo de organismo -- sobre
    el conjunto COMPLETO que calza con la ficha, no solo la página de hasta
    30 filas que se lista en /api/offers."""
    ficha = agent.get_ficha(session_id)
    if not ficha.get("tipo_equipo"):
        return {"methodology": None}
    relaxed_attrs = _session_price_cache.get(session_id, {}).get("relaxed_attrs")
    data = await asyncio.to_thread(price_service.get_methodology, ficha, relaxed_attrs)
    return {"methodology": data}


@app.get("/api/cm_offers/{session_id}")
async def get_cm_offers_endpoint(
    session_id: str,
    price_min: Optional[int] = None,
    price_max: Optional[int] = None,
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    ficha = agent.get_ficha(session_id)
    if not ficha.get("tipo_equipo"):
        return {"offers": []}

    rows = await asyncio.to_thread(cm_service.get_offer_rows, ficha, 30, price_min, price_max)
    analytics_service.log(session_id=session_id, user_email=user_email, tipo="ver_catalogo_cm")
    return {"offers": rows}


# ─── Track evento frontend ────────────────────────────────────────────────────

class TrackRequest(BaseModel):
    tipo: str

@app.post("/api/track/{session_id}")
async def track_endpoint(
    session_id: str,
    body: TrackRequest,
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    analytics_service.log(session_id=session_id, user_email=user_email, tipo=body.tipo)
    return {"ok": True}


# ─── Medicamentos: selector guiado de atributos (sin buscador de texto) ───────

def _enrich_medicamentos(rows: list) -> list:
    # Mismo patrón que _enrich_offers para Computadores: el enlace real a
    # Mercado Público ("Detalle Compra Ágil") solo necesita codigo_requerimiento,
    # que acá SÍ existe (100% de las filas, a diferencia de codigo_oc/OC_productos
    # que no tiene calce para medicamentos -- por eso no hay "Ver OC" acá).
    result = []
    for row in rows:
        codigos = row.get("codigos_requerimiento") or []
        result.append({
            **row,
            "ca_urls": [f"{_CA_BUSCADOR_BASE}?code={c}" for c in codigos],
        })
    return result


class MedicamentoAnalizarRequest(BaseModel):
    texto: str

class MedicamentoManualUpdateRequest(BaseModel):
    attribute: str
    value: Optional[Any] = None

class DescribeItem(BaseModel):
    texto_original: str
    atributos: dict

class DescribeRequest(BaseModel):
    items: list[DescribeItem]


class CategoryAnalizarRequest(BaseModel):
    texto: str

class CategoryManualUpdateRequest(BaseModel):
    attribute: str
    value: Optional[Any] = None

class CategoryDescribeItem(BaseModel):
    texto_original: str
    atributos: dict

class CategoryDescribeRequest(BaseModel):
    items: list[CategoryDescribeItem]


@app.get("/api/medicamentos/schema")
async def get_medicamentos_schema():
    schema = {}
    for attr, meta in FILLABLE_ATTRIBUTES_MED.items():
        entry = {"description": meta["description"], "type": meta["type"]}
        if meta["type"] == "dict":
            entry["valid_values"] = matcher.get_valid_values("Medicamentos", attr)
        schema[attr] = entry
    return {"categoria": "Medicamentos", "attributes": schema, "core_attrs": CORE_ATTRS_MED}


@app.post("/api/medicamentos/analizar/{session_id}")
async def medicamentos_analizar_endpoint(
    session_id: str,
    body: MedicamentoAnalizarRequest,
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    texto = body.texto.strip()
    if not texto:
        async def empty():
            yield sse({"type": "error", "message": "Texto vacío"})
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream", headers=_SSE_HEADERS)

    async def event_gen():
        t0 = time.perf_counter()
        try:
            if await asyncio.to_thread(usage_service.is_blocked, user_email):
                usage_info = await asyncio.to_thread(usage_service.get_usage, user_email)
                yield sse({"type": "usage_limit_reached", "data": usage_service.to_public(usage_info)})
                yield "data: [DONE]\n\n"
                analytics_service.log(
                    session_id=session_id, user_email=user_email, tipo="medicamento_usage_blocked",
                    user_msg=texto, duration_ms=int((time.perf_counter() - t0) * 1000),
                    blocked=True, block_reason="límite diario de tokens alcanzado",
                )
                return

            result = await medicamento_agent.analyze(session_id, texto)

            if result.get("ficha_updates"):
                yield sse({"type": "ficha_update", "updates": result["ficha_updates"]})
            if result.get("message"):
                yield sse({"type": "message", "text": result["message"]})
            if result.get("questions"):
                yield sse({"type": "questions", "questions": result["questions"]})

            price_found = False
            ficha = medicamento_agent.get_ficha(session_id)
            if ficha.get("principio_activo"):
                try:
                    price = await asyncio.to_thread(medicamento_service.estimate_price, ficha)
                    if price:
                        price_found = True
                        yield sse({"type": "price_update", "data": price})
                    else:
                        yield sse({"type": "price_not_found"})
                except Exception as e:
                    logging.warning(f"Error en estimación de precio medicamentos: {e}")
                    yield sse({"type": "price_not_found"})

            yield "data: [DONE]\n\n"

            usage = result.get("tokens")
            ficha_updates = result.get("ficha_updates", [])
            analytics_service.log(
                session_id=session_id,
                user_email=user_email,
                tipo="medicamento_analizar",
                user_msg=texto,
                ai_msg=result.get("message", ""),
                tokens_in=usage.prompt_tokens if usage else None,
                tokens_out=usage.completion_tokens if usage else None,
                duration_ms=int((time.perf_counter() - t0) * 1000),
                n_updates=len(ficha_updates),
                price_found=price_found,
                model=medicamento_agent.model,
                attrs_updated=",".join(u["attribute"] for u in ficha_updates) or None,
            )
            tokens_total = usage.total_tokens if usage else 0
            await asyncio.to_thread(usage_service.add_usage, user_email, tokens_total)

        except Exception as e:
            logging.error(f"Error en medicamentos analizar SSE {session_id}: {e}", exc_info=True)
            yield sse({"type": "error", "message": "Error interno del servidor"})
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.post("/api/medicamentos/manual_update/{session_id}")
async def medicamentos_manual_update_endpoint(
    session_id: str,
    body: MedicamentoManualUpdateRequest,
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    if not body.attribute:
        async def err():
            yield sse({"type": "error", "message": "Atributo requerido"})
            yield "data: [DONE]\n\n"
        return StreamingResponse(err(), media_type="text/event-stream", headers=_SSE_HEADERS)

    async def event_gen():
        t0 = time.perf_counter()
        try:
            result = medicamento_agent.apply_manual_update(session_id, body.attribute, body.value)

            if result["updates"]:
                yield sse({"type": "ficha_update", "updates": result["updates"]})

            price_found = False
            ficha = medicamento_agent.get_ficha(session_id)
            if ficha.get("principio_activo"):
                try:
                    price = await asyncio.to_thread(medicamento_service.estimate_price, ficha)
                    if price:
                        price_found = True
                        yield sse({"type": "price_update", "data": price})
                    else:
                        yield sse({"type": "price_not_found"})
                except Exception as e:
                    logging.warning(f"Error en estimación de precio medicamentos: {e}")
                    yield sse({"type": "price_not_found"})

            yield "data: [DONE]\n\n"

            analytics_service.log(
                session_id=session_id,
                user_email=user_email,
                tipo="medicamento_manual_update",
                user_msg=f"{body.attribute}={body.value}",
                duration_ms=int((time.perf_counter() - t0) * 1000),
                n_updates=len(result.get("updates", [])),
                price_found=price_found,
                attrs_updated=body.attribute,
            )

        except Exception as e:
            logging.error(f"Error en medicamentos manual_update SSE {session_id}: {e}", exc_info=True)
            yield sse({"type": "error", "message": "Error interno"})
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.get("/api/medicamentos/facets/{session_id}")
async def get_medicamentos_facets(
    session_id: str,
    attr: str,
    _: str = Depends(require_api_key),
):
    ficha = medicamento_agent.get_ficha(session_id)
    values = await asyncio.to_thread(medicamento_service.get_facet_values, attr, ficha)
    return {"attribute": attr, "values": values}


@app.get("/api/medicamentos/precio_por_unidad/{session_id}")
async def get_medicamentos_precio_por_unidad(
    session_id: str,
    _: str = Depends(require_api_key),
):
    ficha = medicamento_agent.get_ficha(session_id)
    values = await asyncio.to_thread(medicamento_service.get_price_by_unidad, ficha)
    return {"attribute": "unidad_venta", "values": values}


@app.get("/api/medicamentos/companions/{session_id}")
async def get_medicamentos_companions(session_id: str, _: str = Depends(require_api_key)):
    ficha = medicamento_agent.get_ficha(session_id)
    values = await asyncio.to_thread(medicamento_service.get_companion_values, ficha)
    return {"attribute": "principio_activo", "values": values}


@app.get("/api/medicamentos/historial/{session_id}")
async def get_medicamentos_historial(
    session_id: str,
    sort: str = "fecha_desc",
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    ficha = medicamento_agent.get_ficha(session_id)
    result = await asyncio.to_thread(medicamento_service.get_historial, ficha, sort, 30)
    result["results"] = _enrich_medicamentos(result.get("results", []))
    analytics_service.log(session_id=session_id, user_email=user_email, tipo="medicamento_ver_historial")
    return result


@app.post("/api/medicamentos/describe")
async def medicamentos_describe_endpoint(
    body: DescribeRequest,
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    items = [item.dict() for item in body.items]
    if not items:
        return {"descripciones": []}
    descripciones = await medicamento_agent.describe_batch(items)
    analytics_service.log(
        session_id="", user_email=user_email, tipo="medicamento_pdf_describe", n_updates=len(items),
    )
    return {"descripciones": descripciones}


@app.post("/api/medicamentos/reset/{session_id}")
async def medicamentos_reset_endpoint(session_id: str, _: str = Depends(require_api_key)):
    medicamento_agent.reset_session(session_id)
    return {"type": "reset_ok"}


# ─── Wizard genérico por categoría (Fase 1: solo Computadores, ver plan) ──────
#
# El slug se valida contra categories.schema -- cualquier categoría no
# registrada (incluida "medicamentos", que sigue 100% en sus propios
# endpoints de arriba) da 404 automático, nunca se confunde con el chat
# legacy de Computadores (/api/chat/*, /api/manual_update/*, /api/cm_offers/*
# más abajo) ni con /api/medicamentos/*.
#
# _get_wizard_agent(slug) hoy solo resuelve a computadores_wizard_agent --
# es el punto de extensión obvio para cuando se registre un segundo agente
# de categoría en una fase futura (ver plan, Fase 2: Medicamentos).

def _get_category_schema_or_404(slug: str):
    schema = categories_schema.get(slug)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Categoría '{slug}' no soportada por el wizard")
    return schema


def _get_wizard_agent(slug: str) -> ComputadoresWizardAgent:
    if slug == "computadores":
        return computadores_wizard_agent
    raise HTTPException(status_code=404, detail=f"Categoría '{slug}' no soportada por el wizard")


@app.get("/api/categories/{slug}/schema")
async def get_category_schema(slug: str):
    schema = _get_category_schema_or_404(slug)
    attributes = {}
    for key, meta in schema.attributes.items():
        entry = meta.to_public_dict()
        if meta.values_source == "dictionary":
            entry["valid_values"] = matcher.get_valid_values(schema.categoria_db, key)
        attributes[key] = entry
    return {
        "slug": schema.slug,
        "label": schema.label,
        "supports_convenio_marco": schema.supports_convenio_marco,
        "core_order": schema.core_order,
        "attributes": attributes,
    }


@app.post("/api/categories/{slug}/analizar/{session_id}")
async def category_analizar_endpoint(
    slug: str,
    session_id: str,
    body: CategoryAnalizarRequest,
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    schema = _get_category_schema_or_404(slug)
    wiz_agent = _get_wizard_agent(slug)
    texto = body.texto.strip()
    if not texto:
        async def empty():
            yield sse({"type": "error", "message": "Texto vacío"})
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream", headers=_SSE_HEADERS)

    async def event_gen():
        t0 = time.perf_counter()
        try:
            if await asyncio.to_thread(usage_service.is_blocked, user_email):
                usage_info = await asyncio.to_thread(usage_service.get_usage, user_email)
                yield sse({"type": "usage_limit_reached", "data": usage_service.to_public(usage_info)})
                yield "data: [DONE]\n\n"
                analytics_service.log(
                    session_id=session_id, user_email=user_email, tipo=f"category_{slug}_usage_blocked",
                    user_msg=texto, duration_ms=int((time.perf_counter() - t0) * 1000),
                    blocked=True, block_reason="límite diario de tokens alcanzado",
                )
                return

            result = await wiz_agent.analyze(session_id, texto)

            if result.get("ficha_updates"):
                yield sse({"type": "ficha_update", "updates": result["ficha_updates"]})
            if result.get("complement_updates"):
                yield sse({"type": "ficha_update", "updates": result["complement_updates"]})
            if result.get("message"):
                yield sse({"type": "message", "text": result["message"]})
            if result.get("questions"):
                yield sse({"type": "questions", "questions": result["questions"]})

            price_found = False
            ficha = wiz_agent.get_ficha(session_id)
            # Pendiente real reportado: sin esto, el formulario se perdía al
            # refrescar/cerrar la pestaña -- ver category_session_service.py.
            # texto_original solo se guarda acá (único punto donde existe).
            # title_hint: título corto que la IA ya armó en la misma pasada
            # de extracción (ver "titulo" en ComputadoresWizardAgent.analyze)
            # -- como ChatGPT nombrando la conversación, no el texto crudo.
            await asyncio.to_thread(
                category_session_service.save_ficha,
                session_id, slug, user_email, ficha, texto, result.get("titulo"),
            )
            for core_attr in schema.core_attributes():
                if ficha.get(core_attr.key) is None:
                    break
            else:
                try:
                    price_data, cm_price_data = await asyncio.to_thread(
                        category_service.estimate_price, schema, ficha, price_service, cm_service
                    )
                    if price_data:
                        price_found = True
                        yield sse({"type": "price_update", "data": price_data})
                    else:
                        yield sse({"type": "price_not_found"})
                    if schema.supports_convenio_marco:
                        if cm_price_data:
                            yield sse({"type": "cm_price_update", "data": cm_price_data})
                        else:
                            yield sse({"type": "cm_price_not_found"})
                except Exception as e:
                    logging.warning(f"Error en estimación de precio (categories/{slug}): {e}")
                    yield sse({"type": "price_not_found"})

            yield "data: [DONE]\n\n"

            usage = result.get("tokens")
            ficha_updates = result.get("ficha_updates", [])
            analytics_service.log(
                session_id=session_id,
                user_email=user_email,
                tipo=f"category_{slug}_analizar",
                user_msg=texto,
                ai_msg=result.get("message", ""),
                tokens_in=usage.prompt_tokens if usage else None,
                tokens_out=usage.completion_tokens if usage else None,
                duration_ms=int((time.perf_counter() - t0) * 1000),
                n_updates=len(ficha_updates),
                price_found=price_found,
                model=wiz_agent.model,
                attrs_updated=",".join(u["attribute"] for u in ficha_updates) or None,
            )
            tokens_total = usage.total_tokens if usage else 0
            await asyncio.to_thread(usage_service.add_usage, user_email, tokens_total)

        except Exception as e:
            logging.error(f"Error en categories/{slug} analizar SSE {session_id}: {e}", exc_info=True)
            yield sse({"type": "error", "message": "Error interno del servidor"})
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.post("/api/categories/{slug}/manual_update/{session_id}")
async def category_manual_update_endpoint(
    slug: str,
    session_id: str,
    body: CategoryManualUpdateRequest,
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    schema = _get_category_schema_or_404(slug)
    wiz_agent = _get_wizard_agent(slug)
    if not body.attribute:
        async def err():
            yield sse({"type": "error", "message": "Atributo requerido"})
            yield "data: [DONE]\n\n"
        return StreamingResponse(err(), media_type="text/event-stream", headers=_SSE_HEADERS)

    async def event_gen():
        t0 = time.perf_counter()
        try:
            result = wiz_agent.apply_manual_update(session_id, body.attribute, body.value)

            if result["updates"]:
                yield sse({"type": "ficha_update", "updates": result["updates"]})
            if result.get("complement_updates"):
                yield sse({"type": "ficha_update", "updates": result["complement_updates"]})

            price_found = False
            ficha = wiz_agent.get_ficha(session_id)
            # texto_original=None -- no lo pisa, solo /analizar lo setea (ver
            # category_session_service.save_ficha).
            await asyncio.to_thread(category_session_service.save_ficha, session_id, slug, user_email, ficha)
            for core_attr in schema.core_attributes():
                if ficha.get(core_attr.key) is None:
                    break
            else:
                try:
                    price_data, cm_price_data = await asyncio.to_thread(
                        category_service.estimate_price, schema, ficha, price_service, cm_service
                    )
                    if price_data:
                        price_found = True
                        yield sse({"type": "price_update", "data": price_data})
                    else:
                        yield sse({"type": "price_not_found"})
                    if schema.supports_convenio_marco:
                        if cm_price_data:
                            yield sse({"type": "cm_price_update", "data": cm_price_data})
                        else:
                            yield sse({"type": "cm_price_not_found"})
                except Exception as e:
                    logging.warning(f"Error en estimación de precio (categories/{slug}): {e}")
                    yield sse({"type": "price_not_found"})

            yield "data: [DONE]\n\n"

            analytics_service.log(
                session_id=session_id,
                user_email=user_email,
                tipo=f"category_{slug}_manual_update",
                user_msg=f"{body.attribute}={body.value}",
                duration_ms=int((time.perf_counter() - t0) * 1000),
                n_updates=len(result.get("updates", [])),
                price_found=price_found,
                attrs_updated=body.attribute,
            )

        except Exception as e:
            logging.error(f"Error en categories/{slug} manual_update SSE {session_id}: {e}", exc_info=True)
            yield sse({"type": "error", "message": "Error interno"})
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.get("/api/categories/{slug}/facets/{session_id}")
async def get_category_facets(
    slug: str,
    session_id: str,
    attr: str,
    _: str = Depends(require_api_key),
):
    schema = _get_category_schema_or_404(slug)
    wiz_agent = _get_wizard_agent(slug)
    ficha = wiz_agent.get_ficha(session_id)
    values = await asyncio.to_thread(category_service.get_facet_values, schema, attr, ficha, matcher, price_service)
    return {"attribute": attr, "values": values}


@app.get("/api/categories/{slug}/price/{session_id}")
async def get_category_price(
    slug: str,
    session_id: str,
    _: str = Depends(require_api_key),
):
    schema = _get_category_schema_or_404(slug)
    wiz_agent = _get_wizard_agent(slug)
    ficha = wiz_agent.get_ficha(session_id)
    price_data, cm_price_data = await asyncio.to_thread(
        category_service.estimate_price, schema, ficha, price_service, cm_service
    )
    # Mismo criterio que _session_price_cache del chat legacy -- /offers
    # necesita el mismo relajo que usó estimate() para no filtrar con la
    # ficha completa sin relajar y devolver 0 filas (ver docstring de
    # category_service.get_offer_rows).
    if price_data:
        _category_price_cache[session_id] = {"relaxed_attrs": price_data.get("relaxed_attrs")}
    return {"price_data": price_data, "cm_price_data": cm_price_data}


@app.get("/api/categories/{slug}/offers/{session_id}")
async def get_category_offers(
    slug: str,
    session_id: str,
    price_min: Optional[int] = None,
    price_max: Optional[int] = None,
    page: int = 1,
    page_size: int = 10,
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    """Transacciones individuales de Compra Ágil detrás de la estimación
    (panel de precio del wizard solo muestra el agregado). Paginado real
    (page/page_size), no un tope fijo -- antes esto traía siempre un máximo
    de 30 filas sin forma de ver el resto; con categorías de specs comunes
    el total real puede ser miles de transacciones. Convenio Marco no
    necesita un endpoint equivalente -- CMService.estimate() ya trae las
    filas de producto en su propio resultado (cm_price_data.products)."""
    _get_category_schema_or_404(slug)
    wiz_agent = _get_wizard_agent(slug)
    ficha = wiz_agent.get_ficha(session_id)
    if not ficha.get("tipo_equipo"):
        return {"offers": [], "total": 0, "page": 1, "page_size": page_size}

    page = max(1, page)
    page_size = max(1, min(page_size, 50))
    relaxed_attrs = _category_price_cache.get(session_id, {}).get("relaxed_attrs")
    rows, total = await asyncio.gather(
        asyncio.to_thread(
            category_service.get_offer_rows, ficha, price_service, page_size, price_min, price_max,
            relaxed_attrs, (page - 1) * page_size,
        ),
        asyncio.to_thread(
            category_service.count_offer_rows, ficha, price_service, price_min, price_max, relaxed_attrs
        ),
    )
    analytics_service.log(session_id=session_id, user_email=user_email, tipo=f"category_{slug}_ver_historial")
    return {"offers": _enrich_offers(rows), "total": total, "page": page, "page_size": page_size}


@app.post("/api/categories/{slug}/describe")
async def category_describe_endpoint(
    slug: str,
    body: CategoryDescribeRequest,
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    _get_category_schema_or_404(slug)
    wiz_agent = _get_wizard_agent(slug)
    items = [item.dict() for item in body.items]
    if not items:
        return {"descripciones": []}
    descripciones = await wiz_agent.describe_batch(items)
    analytics_service.log(
        session_id="", user_email=user_email, tipo=f"category_{slug}_pdf_describe", n_updates=len(items),
    )
    return {"descripciones": descripciones}


@app.post("/api/categories/{slug}/reset/{session_id}")
async def category_reset_endpoint(slug: str, session_id: str, _: str = Depends(require_api_key)):
    _get_category_schema_or_404(slug)
    wiz_agent = _get_wizard_agent(slug)
    wiz_agent.reset_session(session_id)
    return {"type": "reset_ok"}


# ─── Historial de formularios (ver category_session_service.py) ───────────────
# Pendiente real reportado: el wizard no tenía ningún mecanismo para volver
# a ver/retomar un formulario ya empezado, a diferencia del chat legacy
# (/api/sessions más arriba). El guardado incremental vive en los propios
# endpoints de analizar/manual_update (ver save_ficha ahí); acá solo se
# lista, retoma, renombra y borra.

@app.get("/api/categories/{slug}/sessions")
async def list_category_sessions_endpoint(
    slug: str, user_email: str = Depends(get_user_email), _: str = Depends(require_api_key)
):
    _get_category_schema_or_404(slug)
    sessions = await asyncio.to_thread(category_session_service.list_sessions, slug, user_email)
    return {"sessions": sessions}


@app.post("/api/categories/{slug}/sessions/{session_id}/resume")
async def resume_category_session_endpoint(
    slug: str, session_id: str, user_email: str = Depends(get_user_email), _: str = Depends(require_api_key)
):
    _get_category_schema_or_404(slug)
    wiz_agent = _get_wizard_agent(slug)
    saved = await asyncio.to_thread(category_session_service.get_full_session, session_id, slug, user_email)
    if not saved:
        raise HTTPException(status_code=404, detail="Formulario no encontrado")
    # Re-sembrar en memoria -- este proceso puede no tener nunca visto este
    # session_id (otra pestaña, otro día, o el backend se reinició entre
    # medio), y get_ficha()/estimate_price() leen del estado en memoria.
    wiz_agent.seed_session(session_id, saved["ficha"], saved["texto_original"])
    return {"title": saved["title"], "ficha": saved["ficha"], "texto_original": saved["texto_original"]}


@app.post("/api/categories/{slug}/sessions/{session_id}/rename")
async def rename_category_session_endpoint(
    slug: str,
    session_id: str,
    body: RenameSessionRequest,
    user_email: str = Depends(get_user_email),
    _: str = Depends(require_api_key),
):
    ok = await asyncio.to_thread(category_session_service.rename, session_id, slug, user_email, body.title)
    if not ok:
        raise HTTPException(status_code=404, detail="Formulario no encontrado")
    return {"ok": True}


@app.post("/api/categories/{slug}/sessions/{session_id}/delete")
async def delete_category_session_endpoint(
    slug: str, session_id: str, user_email: str = Depends(get_user_email), _: str = Depends(require_api_key)
):
    ok = await asyncio.to_thread(category_session_service.delete, session_id, slug, user_email)
    if not ok:
        raise HTTPException(status_code=404, detail="Formulario no encontrado")
    return {"ok": True}


# ─── Administración (grupos, usuarios, tokens, módulos) ───────────────────────
# Administra las MISMAS tablas de panel_admin que ya usa db-admin-panel (ver
# access_service.py / admin_service.py) -- un cambio hecho acá se ve
# reflejado allá y viceversa, misma fuente de verdad. Todo gateado por
# require_admin (correo en el grupo 'Admins' de mvp1-compra-agil).

class AdminSectionRequest(BaseModel):
    slug: str
    nombre: str

class AdminLimitRequest(BaseModel):
    daily_token_limit: Optional[str] = None
    unlimited: bool = False

class AdminUserRequest(AdminLimitRequest):
    email: str

class AdminToggleRequest(BaseModel):
    enabled: bool

class AdminGroupRequest(AdminLimitRequest):
    nombre: str

class AdminGroupMembersRequest(BaseModel):
    emails: str


def _admin_error_response(e: admin_service.AdminServiceError):
    raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/admin/overview")
async def admin_overview_endpoint(_: str = Depends(require_admin)):
    sections = await asyncio.to_thread(admin_service.list_sections)
    users = await asyncio.to_thread(admin_service.list_users_with_sections)
    groups = await asyncio.to_thread(admin_service.list_groups)

    # Consumo real de hoy por usuario -- no vive en panel_admin (esquema
    # administrado desde db-admin-panel), sino en la tabla daily_usage
    # propia de MVP1 (ver usage_service.py). Valor agregado respecto al
    # otro panel, que solo puede mostrar el tope configurado, no el gasto.
    for u in users:
        usage = await asyncio.to_thread(usage_service.get_usage, u["email"])
        u["tokens_used_today"] = usage["tokens_used"]

    group_members = {}
    for g in groups:
        group_members[g["id"]] = await asyncio.to_thread(admin_service.list_group_members, g["id"])

    return {"sections": sections, "users": users, "groups": groups, "group_members": group_members}


@app.post("/api/admin/sections")
async def admin_create_section(body: AdminSectionRequest, _: str = Depends(require_admin)):
    try:
        await asyncio.to_thread(admin_service.create_section, body.slug, body.nombre)
    except admin_service.AdminServiceError as e:
        _admin_error_response(e)
    return {"ok": True}


@app.post("/api/admin/sections/{section_id}/delete")
async def admin_delete_section(section_id: int, _: str = Depends(require_admin)):
    await asyncio.to_thread(admin_service.delete_section, section_id)
    return {"ok": True}


@app.post("/api/admin/users")
async def admin_add_user(body: AdminUserRequest, user_email: str = Depends(require_admin)):
    try:
        await asyncio.to_thread(
            admin_service.add_user, body.email, body.daily_token_limit, body.unlimited, user_email
        )
    except admin_service.AdminServiceError as e:
        _admin_error_response(e)
    return {"ok": True}


@app.post("/api/admin/users/{user_id}/limit")
async def admin_update_user_limit(user_id: int, body: AdminLimitRequest, _: str = Depends(require_admin)):
    try:
        await asyncio.to_thread(admin_service.update_user_limit, user_id, body.daily_token_limit, body.unlimited)
    except admin_service.AdminServiceError as e:
        _admin_error_response(e)
    return {"ok": True}


@app.post("/api/admin/users/{user_id}/delete")
async def admin_remove_user(user_id: int, _: str = Depends(require_admin)):
    await asyncio.to_thread(admin_service.remove_user, user_id)
    return {"ok": True}


@app.post("/api/admin/users/{user_id}/sections/{section_id}/toggle")
async def admin_toggle_user_section(
    user_id: int, section_id: int, body: AdminToggleRequest, _: str = Depends(require_admin)
):
    await asyncio.to_thread(admin_service.set_user_section, user_id, section_id, body.enabled)
    return {"ok": True}


@app.post("/api/admin/groups")
async def admin_create_group(body: AdminGroupRequest, user_email: str = Depends(require_admin)):
    try:
        await asyncio.to_thread(
            admin_service.create_group, body.nombre, body.daily_token_limit, body.unlimited, user_email
        )
    except admin_service.AdminServiceError as e:
        _admin_error_response(e)
    return {"ok": True}


@app.post("/api/admin/groups/{group_id}/delete")
async def admin_delete_group(group_id: int, _: str = Depends(require_admin)):
    await asyncio.to_thread(admin_service.delete_group, group_id)
    return {"ok": True}


@app.post("/api/admin/groups/{group_id}/limit")
async def admin_update_group_limit(group_id: int, body: AdminLimitRequest, _: str = Depends(require_admin)):
    try:
        await asyncio.to_thread(admin_service.update_group_limit, group_id, body.daily_token_limit, body.unlimited)
    except admin_service.AdminServiceError as e:
        _admin_error_response(e)
    return {"ok": True}


@app.post("/api/admin/groups/{group_id}/sections/{section_id}/toggle")
async def admin_toggle_group_section(
    group_id: int, section_id: int, body: AdminToggleRequest, _: str = Depends(require_admin)
):
    await asyncio.to_thread(admin_service.set_group_section, group_id, section_id, body.enabled)
    return {"ok": True}


@app.post("/api/admin/groups/{group_id}/members")
async def admin_add_group_members(group_id: int, body: AdminGroupMembersRequest, user_email: str = Depends(require_admin)):
    try:
        result = await asyncio.to_thread(admin_service.bulk_add_emails_to_group, group_id, body.emails, user_email)
    except admin_service.AdminServiceError as e:
        _admin_error_response(e)
    return result


@app.post("/api/admin/groups/{group_id}/members/{user_id}/remove")
async def admin_remove_group_member(group_id: int, user_id: int, _: str = Depends(require_admin)):
    await asyncio.to_thread(admin_service.remove_member_from_group, user_id)
    return {"ok": True}


# ─── Punto de entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
