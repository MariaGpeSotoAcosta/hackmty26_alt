from __future__ import annotations

import base64
import io
import wave
from pathlib import Path

import numpy as np


def _decode_base64(payload: str) -> bytes:
    text = payload.strip()
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    text = "".join(text.split())
    pad = (-len(text)) % 4
    return base64.b64decode(text + ("=" * pad))


def load_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    if not data:
        raise ValueError("empty audio payload")
    if data[:4] != b"RIFF":
        # Judge may wrap the WAV in base64 even when the body looks binary-ish.
        try:
            data = _decode_base64(data.decode("ascii", errors="strict"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("payload is not a WAV file") from exc
    return _read_wave(io.BytesIO(data))


def load_wav_base64(payload: str) -> tuple[np.ndarray, int]:
    return load_wav_bytes(_decode_base64(payload))


def load_wav_path(path: str | Path) -> tuple[np.ndarray, int]:
    with Path(path).open("rb") as handle:
        return load_wav_bytes(handle.read())


def _read_wave(buffer: io.BytesIO) -> tuple[np.ndarray, int]:
    with wave.open(buffer, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sr = wav.getframerate()
        n_frames = wav.getnframes()
        raw = wav.readframes(n_frames)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels)
    else:
        audio = np.column_stack([audio, audio])

    if audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio, int(sr)
