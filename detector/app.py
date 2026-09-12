from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse

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
        "early_exit": bundle.get("early_exit", True) if ready else False,
    }


def _fallback() -> dict[str, Any]:
    # Never 500 during judging. Weak synthetic prior, low confidence.
    return {"is_synthetic": True, "confidence": 0.52}


def _verdict_from_bytes(data: bytes) -> dict[str, Any]:
    try:
        result = predict_from_wav_bytes(data, _bundle())
        return {"is_synthetic": result["is_synthetic"], "confidence": result["confidence"]}
    except Exception:
        return _fallback()


def _extract_payload(body: bytes, content_type: str) -> bytes | None:
    if not body:
        return None
    if body[:4] == b"RIFF":
        return body
    if "json" in content_type or body[:1] in (b"{", b"["):
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body
        if isinstance(payload, dict):
            for key in ("audio_base64", "audio", "wav", "clip", "data"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    return value.encode("ascii")
        return None
    return body


@app.post("/detect")
async def detect(request: Request) -> JSONResponse:
    body = await request.body()
    content_type = request.headers.get("content-type", "")
    data = _extract_payload(body, content_type)
    if data is None:
        return JSONResponse(_fallback())
    return JSONResponse(_verdict_from_bytes(data))


@app.post("/detect/upload")
async def detect_upload(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read()
    return JSONResponse(_verdict_from_bytes(data))
