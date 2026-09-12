from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from detector.paths import transcript_path, vosk_model_dir

_MODEL = None


def load_transcript(anon_id: str) -> dict | None:
    path = transcript_path(anon_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_transcript(anon_id: str, payload: dict) -> Path:
    path = transcript_path(anon_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def transcribe_audio(audio: np.ndarray, sr: int) -> dict:
    """ASR per channel with Vosk. Optional — dialogue scoring does not need this."""
    model = _vosk_model()
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    caller = _recognize(model, audio[:, 0], sr)
    agent = _recognize(model, audio[:, 1], sr)
    return {"caller": caller, "agent": agent, "engine": "vosk"}


def _vosk_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from vosk import Model
    except ImportError as exc:
        raise RuntimeError("vosk is not installed. pip install vosk") from exc
    path = vosk_model_dir()
    if path is None:
        raise RuntimeError("Download a Vosk Spanish model into models/vosk-model-small-es-0.42/")
    _MODEL = Model(str(path))
    return _MODEL


def _recognize(model, samples: np.ndarray, sr: int) -> str:
    from vosk import KaldiRecognizer, SetLogLevel

    SetLogLevel(-1)
    rec = KaldiRecognizer(model, sr)
    rec.SetWords(False)
    pcm = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16).tobytes()
    chunk = 4000
    parts: list[str] = []
    for i in range(0, len(pcm), chunk):
        if rec.AcceptWaveform(pcm[i : i + chunk]):
            parts.append(json.loads(rec.Result()).get("text", ""))
    parts.append(json.loads(rec.FinalResult()).get("text", ""))
    return " ".join(p for p in parts if p).strip()
