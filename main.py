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

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv

from agents.attribute_matcher import AttributeMatcher
from agents.ficha_agent import FichaAgent, FILLABLE_ATTRIBUTES
from services.price_service import PriceService

load_dotenv()
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

matcher = AttributeMatcher()
price_service = PriceService()
agent: FichaAgent = None

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
    logging.info("Iniciando servidor — calentando índices FAISS...")
    matcher.warm()
    agent = FichaAgent(matcher)
    logging.info("Servidor listo")
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
            yield sse({"type": "thinking"})

            result = {}
            async for event_type, event_data in agent.stream_process_message(session_id, content):
                if event_type == "chunk":
                    yield sse({"type": "assistant_chunk", "delta": event_data})
                elif event_type == "done":
                    result = event_data

            yield sse({"type": "assistant_done"})

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


# ─── Reset ────────────────────────────────────────────────────────────────────

@app.post("/api/reset/{session_id}")
async def reset_endpoint(session_id: str, _: str = Depends(require_api_key)):
    agent.reset_session(session_id)
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
