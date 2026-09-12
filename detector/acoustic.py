from __future__ import annotations

import numpy as np

ACOUSTIC_FEATURES = [
    "ac_zcr",
    "ac_flatness",
    "ac_centroid",
    "ac_rms_cv",
    "ac_f0_std",
]


def extract_acoustic_features(audio: np.ndarray, sr: int, turns: list[dict]) -> dict[str, float]:
    """Cheap caller-only cues. Used as optional support, not the main signal."""
    caller = audio[:, 0] if audio.ndim > 1 else audio
    voiced = _caller_voiced(caller, sr, turns)
    if voiced.size < sr // 4:
        return {name: 0.0 for name in ACOUSTIC_FEATURES}

    hop = max(1, int(sr * 0.02))
    n = len(voiced) // hop
    frames = voiced[: n * hop].reshape(n, hop)
    window = np.hanning(hop)
    specs = np.abs(np.fft.rfft(frames * window, axis=1)) + 1e-12
    freqs = np.fft.rfftfreq(hop, 1.0 / sr)

    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0)
    geo = np.exp(np.mean(np.log(specs), axis=1))
    arith = np.mean(specs, axis=1)
    flatness = float(np.mean(geo / arith))
    centroid = float(np.mean(np.sum(specs * freqs, axis=1) / np.sum(specs, axis=1)))
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    rms_cv = float(np.std(rms) / (np.mean(rms) + 1e-12))
    f0_std = _f0_std(voiced, sr)
    return {
        "ac_zcr": float(zcr),
        "ac_flatness": flatness,
        "ac_centroid": centroid / max(sr / 2.0, 1.0),
        "ac_rms_cv": rms_cv,
        "ac_f0_std": f0_std,
    }


def _caller_voiced(caller: np.ndarray, sr: int, turns: list[dict]) -> np.ndarray:
    chunks = []
    for turn in turns:
        if turn["channel"] != 0:
            continue
        start = max(0, int(turn["start"] * sr))
        end = min(len(caller), int(turn["end"] * sr))
        if end > start:
            chunks.append(caller[start:end])
    return np.concatenate(chunks) if chunks else caller


def _f0_std(samples: np.ndarray, sr: int) -> float:
    """Very small autocorrelation F0 tracker; std near 0 often means TTS."""
    hop = int(sr * 0.02)
    win = int(sr * 0.04)
    min_lag = max(1, int(sr / 400))
    max_lag = max(min_lag + 1, int(sr / 70))
    f0s: list[float] = []
    for start in range(0, len(samples) - win, hop):
        frame = samples[start : start + win]
        if np.sqrt(np.mean(frame * frame)) < 0.01:
            continue
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode="full")[win - 1 :]
        if corr[0] <= 1e-8:
            continue
        region = corr[min_lag:max_lag]
        if region.size == 0:
            continue
        lag = int(np.argmax(region)) + min_lag
        if corr[lag] / corr[0] < 0.3:
            continue
        f0s.append(sr / lag)
    return float(np.std(f0s)) if len(f0s) > 3 else 0.0
