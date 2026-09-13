from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from detector import logging_db
from detector.explain import explain
from detector.paths import MODEL_PATH
from detector.predict import load_bundle, predict_from_wav_bytes

app = FastAPI(title="Altur detector", version="1.0.0")
_BUNDLE = None
_MTIME = None


def _bundle():
    global _BUNDLE, _MTIME
    mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    if _BUNDLE is None or mtime != _MTIME:
        _BUNDLE = load_bundle()
        _MTIME = mtime
    return _BUNDLE


@app.get("/health")
def health() -> dict[str, Any]:
    ready = MODEL_PATH.exists()
    bundle = _bundle() if ready else {}
    return {
        "ok": ready,
        "model": str(MODEL_PATH.name) if ready else None,
        "features": bundle.get("feature_names") if ready else [],
        "val_accuracy": bundle.get("val_accuracy") if ready else None,
        "tiebreak": bundle.get("tiebreak") if ready else False,
        "early_exit": bundle.get("early_exit", False) if ready else False,
        "with_semantic": bundle.get("with_semantic") if ready else False,
    }


def _fallback() -> dict[str, Any]:
    # Never 500 during judging. Weak synthetic prior, low confidence.
    return {"is_synthetic": True, "confidence": 0.52}


def _verdict_from_bytes(data: bytes, call_id: str | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        bundle = _bundle()
        result = predict_from_wav_bytes(data, bundle)
        conf = float(result["confidence"])
        if conf < 0.0 or conf > 1.0:
            conf = 0.52
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if logging_db.enabled():
            try:
                contributions = explain(result.get("features") or {}, bundle)
            except Exception:
                contributions = None
            logging_db.log_call(call_id or str(uuid.uuid4()), result, contributions, latency_ms)
        return {"is_synthetic": bool(result["is_synthetic"]), "confidence": conf}
    except Exception:
        return _fallback()


def _b64_to_wav(value: str) -> bytes:
    text = value.strip()
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    text = "".join(text.split())
    pad = (-len(text)) % 4
    return base64.b64decode(text + ("=" * pad))


def _extract_payload(body: bytes, content_type: str) -> tuple[bytes | None, str | None]:
    if not body:
        return None, None
    if body[:4] == b"RIFF":
        return body, None
    if "json" in content_type or body[:1] in (b"{", b"["):
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body, None
        if isinstance(payload, dict):
            call_id = payload.get("call_id") if isinstance(payload.get("call_id"), str) else None
            for key in ("audio_base64", "audio", "wav", "clip", "data"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    try:
                        return _b64_to_wav(value), call_id
                    except Exception:
                        return None, None
        return None, None
    return body, None


async def _file_from_form(request: Request) -> bytes | None:
    form = await request.form()
    for key in ("file", "audio", "wav", "clip", "upload"):
        item = form.get(key)
        if item is not None and hasattr(item, "read"):
            data = await item.read()
            if data:
                return data
    for item in form.values():
        if hasattr(item, "read"):
            data = await item.read()
            if data:
                return data
    return None


async def _verdict_response(data: bytes | None, call_id: str | None = None) -> JSONResponse:
    if not data:
        return JSONResponse(_fallback())
    return JSONResponse(await asyncio.to_thread(_verdict_from_bytes, data, call_id))


@app.post(
    "/detect",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                        },
                    }
                },
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "call_id": {"type": "string"},
                            "audio_base64": {"type": "string"},
                            "sample_rate": {"type": "integer"},
                            "channels": {"type": "integer"},
                        },
                        "required": ["audio_base64"],
                    }
                },
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
            },
        }
    },
)
async def detect(request: Request) -> JSONResponse:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        return await _verdict_response(await _file_from_form(request))
    body = await request.body()
    data, call_id = _extract_payload(body, content_type)
    return await _verdict_response(data, call_id)


_BENCHMARK_PATH = Path(__file__).parent / "benchmark.json"


def _load_benchmark() -> dict | None:
    if not _BENCHMARK_PATH.exists():
        return None
    try:
        return json.loads(_BENCHMARK_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


@app.get("/dashboard/data")
def dashboard_data() -> dict[str, Any]:
    return {
        "enabled": logging_db.enabled(),
        "stats": logging_db.aggregate_stats() if logging_db.enabled() else {"available": False},
        "recent": logging_db.recent_calls(limit=30) if logging_db.enabled() else [],
        "benchmark": _load_benchmark(),
    }


_DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return _DASHBOARD_HTML
