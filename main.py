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
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv

from agents.attribute_matcher import AttributeMatcher
from agents.ficha_agent import FichaAgent, FILLABLE_ATTRIBUTES
from services.price_service import PriceService
from services.lgbm_price_service import LgbmPriceService
from services.guardrail_service import GuardrailService

load_dotenv()
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

matcher = AttributeMatcher()
price_service = PriceService()
lgbm_service = LgbmPriceService()
guardrail = GuardrailService()
agent: FichaAgent = None

# Historial liviano por sesión para el guardrail (máx 20 mensajes por sesión)
_session_histories: dict[str, list[dict]] = {}
_HISTORY_MAX = 20

# ─── API Key ──────────────────────────────────────────────────────────────────

_raw_keys = os.getenv("FRONTEND_API_KEY", "")
API_KEYS = {k.strip() for k in _raw_keys.split(",") if k.strip()}

_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def require_api_key(x_api_key: Optional[str] = Depends(_api_key_header)):
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API Key inválida")
    return x_api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = FichaAgent(matcher)
    lgbm_service._load()
    logging.info("Servidor listo — construyendo índices FAISS y dropdowns en background...")
    asyncio.create_task(asyncio.to_thread(matcher.warm))
    asyncio.create_task(asyncio.to_thread(price_service.warmup_dropdowns))
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
async def chat_endpoint(session_id: str, body: ChatRequest, _: str = Depends(require_api_key)):
    content = body.content.strip()
    if not content:
        async def empty():
            yield sse({"type": "error", "message": "Mensaje vacío"})
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream", headers=_SSE_HEADERS)

    async def event_gen():
        try:
            # ── Guardrail: validar mensaje antes de procesarlo ──────────────
            history = _session_histories.get(session_id, [])
            allowed, block_reason, clean_message = await guardrail.check(content, history)

            if not allowed:
                logging.info(f"[guardrail] Bloqueado session={session_id}: {block_reason}")
                yield sse({"type": "blocked", "message": block_reason})
                yield "data: [DONE]\n\n"
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

            ficha = agent.get_ficha(session_id)
            if ficha.get("tipo_equipo"):
                try:
                    price = await asyncio.to_thread(price_service.estimate, ficha)
                    if price:
                        yield sse({"type": "price_update", "data": price})
                    else:
                        yield sse({"type": "price_not_found"})
                except Exception as e:
                    logging.warning(f"Error en estimación de precio: {e}")
                    yield sse({"type": "price_not_found"})

            yield "data: [DONE]\n\n"

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
        try:
            result = agent.apply_manual_update(session_id, body.attribute, body.value)

            if result["updates"]:
                yield sse({"type": "ficha_update", "updates": result["updates"]})

            if result["complement_updates"]:
                yield sse({"type": "ficha_update", "updates": result["complement_updates"]})

            ficha = agent.get_ficha(session_id)
            if ficha.get("tipo_equipo"):
                try:
                    price = await asyncio.to_thread(price_service.estimate, ficha)
                    if price:
                        yield sse({"type": "price_update", "data": price})
                    else:
                        yield sse({"type": "price_not_found"})
                except Exception as e:
                    logging.warning(f"Error en estimación de precio: {e}")
                    yield sse({"type": "price_not_found"})

            yield "data: [DONE]\n\n"

        except Exception as e:
            logging.error(f"Error en manual_update SSE {session_id}: {e}", exc_info=True)
            yield sse({"type": "error", "message": "Error interno"})
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


# ─── Offers ───────────────────────────────────────────────────────────────────

_CA_API_BASE      = "https://servicios-compra-agil.mercadopublico.cl/v1/compra-agil/solicitud"
_CA_PO_BASE       = "https://www.mercadopublico.cl/PurchaseOrder/Modules/PO/DetailsPurchaseOrder.aspx"
_CA_BUSCADOR_BASE = "https://buscador.mercadopublico.cl/ficha"

@app.get("/api/offers/{session_id}")
async def get_offers_endpoint(session_id: str, _: str = Depends(require_api_key)):
    ficha = agent.get_ficha(session_id)
    if not ficha.get("tipo_equipo"):
        return {"offers": []}

    price_data = await asyncio.to_thread(price_service.estimate, ficha)
    p25 = price_data.get("p25") if price_data else None
    p75 = price_data.get("p75") if price_data else None

    rows = await asyncio.to_thread(price_service.get_offer_rows, ficha, 30, p25, p75)
    if not rows:
        return {"offers": []}

    token = await asyncio.to_thread(price_service.get_token)
    unique_reqs = list(dict.fromkeys(r["codigo_requerimiento"] for r in rows if r["codigo_requerimiento"]))
    # req_code -> {oferta_id -> {"oc_codes": [...], "razon_social": str|None}}
    req_oferta_map: dict[str, dict[int, dict]] = {req: {} for req in unique_reqs}
    ca_map: dict[str, bool] = {req: False for req in unique_reqs}

    if token and unique_reqs:
        sem = asyncio.Semaphore(5)

        async def fetch_req(req_code: str):
            async with sem:
                try:
                    async with httpx.AsyncClient(timeout=8.0, verify=False) as client:
                        resp = await client.get(
                            f"{_CA_API_BASE}/{req_code}?size=20&page=0",
                            headers={"Authorization": f"Bearer {token}"},
                        )
                    if resp.status_code == 200:
                        ofertas = (resp.json().get("payload") or {}).get("ofertas") or []
                        oferta_data: dict[int, dict] = {}
                        for o in ofertas:
                            oid = o.get("id")
                            if oid is None:
                                continue
                            oc_codes = list(dict.fromkeys(
                                oc["code"]
                                for oc in (o.get("ordenesCompra") or [])
                                if oc.get("code")
                            ))
                            oferta_data[int(oid)] = {
                                "oc_codes":    oc_codes,
                                "razon_social": o.get("razonSocial"),
                            }
                        return req_code, oferta_data, True
                except Exception as e:
                    logging.warning(f"[fetch_oc] {req_code}: {e}")
            return req_code, {}, False

        results = await asyncio.gather(*[fetch_req(r) for r in unique_reqs])
        req_oferta_map = {req: data for req, data, _ in results}
        ca_map = {req: ok for req, _, ok in results}

    offers = []
    for row in rows:
        req = row["codigo_requerimiento"]
        oferta_id = row.get("id_oferta_aquiles")
        oferta_info = req_oferta_map.get(req, {}).get(oferta_id, {}) if oferta_id else {}
        oc_codes = oferta_info.get("oc_codes", [])
        offers.append({
            **row,
            "oc_codes":    oc_codes,
            "oc_urls":     [f"{_CA_PO_BASE}?CodigoOC={c}" for c in oc_codes],
            "ca_url":      f"{_CA_BUSCADOR_BASE}?code={req}",
            "ca_available": bool(ca_map.get(req)),
            "razon_social": oferta_info.get("razon_social"),
        })
    return {"offers": offers}


# ─── Reset ────────────────────────────────────────────────────────────────────

@app.post("/api/reset/{session_id}")
async def reset_endpoint(session_id: str, _: str = Depends(require_api_key)):
    agent.reset_session(session_id)
    _session_histories.pop(session_id, None)
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
