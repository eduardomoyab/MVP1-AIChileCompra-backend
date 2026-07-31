"""
main.py — Backend Asistente Compra Ágil
FastAPI + HTTP SSE

Endpoints:
  POST /api/chat/{session_id}          → Chat streaming (SSE)
  POST /api/manual_update/{session_id} → Actualización manual de atributo (SSE)
  POST /api/reset/{session_id}         → Resetear sesión
  GET  /health                         → Estado del servicio
  GET  /api/schema                     → Esquema de atributos y valores válidos
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
from services.price_service import PriceService
from services.cm_service import CMService
from services import currency_service
from services.lgbm_price_service import LgbmPriceService
from services.guardrail_service import GuardrailService
from services import analytics_service
from services import access_service
from services import usage_service

load_dotenv()
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

matcher = AttributeMatcher()
price_service = PriceService()
cm_service = CMService()
lgbm_service = LgbmPriceService()
guardrail = GuardrailService()
agent: FichaAgent = None

# Historial liviano por sesión para el guardrail (máx 20 mensajes por sesión)
_session_histories: dict[str, list[dict]] = {}
_HISTORY_MAX = 20

# Caché del último estimate por sesión para evitar re-query en /api/offers
_session_price_cache: dict[str, dict] = {}
_session_cm_price_cache: dict[str, dict] = {}

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = FichaAgent(matcher)
    lgbm_service._load()
    usage_service.ensure_table()
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
    return {"allowed": allowed}


# ─── Uso diario de tokens (panel de cuenta) ────────────────────────────────────

@app.get("/api/usage")
async def usage_endpoint(user_email: str = Depends(get_user_email), _: str = Depends(require_api_key)):
    usage = await asyncio.to_thread(usage_service.get_usage, user_email)
    return usage_service.to_public(usage)


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
                    tipo="usage_blocked",
                    user_msg=content,
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                    blocked=True,
                    block_reason="límite diario de tokens alcanzado",
                )
                return

            # ── Guardrail: validar mensaje antes de procesarlo ──────────────
            history = _session_histories.get(session_id, [])
            allowed, block_reason, clean_message, guardrail_usage = await guardrail.check(content, history)
            guardrail_tokens = guardrail_usage.total_tokens if guardrail_usage else 0

            if not allowed:
                logging.info(f"[guardrail] Bloqueado session={session_id}: {block_reason}")
                yield sse({"type": "blocked", "message": block_reason})
                yield "data: [DONE]\n\n"
                analytics_service.log(
                    session_id=session_id,
                    tipo="blocked",
                    user_msg=content,
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                    blocked=True,
                    block_reason=block_reason,
                )
                # El guardrail igual gastó tokens aunque el mensaje se bloqueara.
                await asyncio.to_thread(usage_service.add_usage, user_email, guardrail_tokens)
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
            async for event_type, event_data in agent.stream_process_message(session_id, effective_content):
                if event_type == "chunk":
                    yield sse({"type": "assistant_chunk", "delta": event_data})
                elif event_type == "done":
                    result = event_data

            yield sse({"type": "assistant_done"})

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
async def manual_update_endpoint(session_id: str, body: ManualUpdateRequest, _: str = Depends(require_api_key)):
    if not body.attribute:
        async def err():
            yield sse({"type": "error", "message": "Atributo requerido"})
            yield "data: [DONE]\n\n"
        return StreamingResponse(err(), media_type="text/event-stream", headers=_SSE_HEADERS)

    async def event_gen():
        t0 = time.perf_counter()
        try:
            result = agent.apply_manual_update(session_id, body.attribute, body.value)

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
async def get_offers_endpoint(session_id: str, _: str = Depends(require_api_key)):
    ficha = agent.get_ficha(session_id)
    if not ficha.get("tipo_equipo"):
        return {"offers": []}

    cached = _session_price_cache.get(session_id, {})
    p25 = cached.get("p25")
    p75 = cached.get("p75")

    rows = await asyncio.to_thread(price_service.get_offer_rows, ficha, 30, p25, p75)
    analytics_service.log(session_id=session_id, tipo="ver_historial")
    return {"offers": _enrich_offers(rows)}


@app.get("/api/cm_offers/{session_id}")
async def get_cm_offers_endpoint(session_id: str, _: str = Depends(require_api_key)):
    ficha = agent.get_ficha(session_id)
    if not ficha.get("tipo_equipo"):
        return {"offers": []}

    rows = await asyncio.to_thread(cm_service.get_offer_rows, ficha, 30)
    analytics_service.log(session_id=session_id, tipo="ver_catalogo_cm")
    return {"offers": rows}


# ─── Track evento frontend ────────────────────────────────────────────────────

class TrackRequest(BaseModel):
    tipo: str

@app.post("/api/track/{session_id}")
async def track_endpoint(session_id: str, body: TrackRequest, _: str = Depends(require_api_key)):
    analytics_service.log(session_id=session_id, tipo=body.tipo)
    return {"ok": True}


# ─── Reset ────────────────────────────────────────────────────────────────────

@app.post("/api/reset/{session_id}")
async def reset_endpoint(session_id: str, _: str = Depends(require_api_key)):
    agent.reset_session(session_id)
    _session_histories.pop(session_id, None)
    _session_price_cache.pop(session_id, None)
    _session_cm_price_cache.pop(session_id, None)
    return {"type": "reset_ok"}


# ─── Punto de entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
