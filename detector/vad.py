from __future__ import annotations

import numpy as np

FRAME_MS = 20
MIN_SPEECH_S = 0.20
MERGE_GAP_S = 0.30
ABS_THR = 0.005
REL_K = 4.5
HANGOVER_FRAMES = 4

# Tuned per channel via grid search against turns/*.json.
# Caller has more hesitation/silence than the agent, so a higher relative
# threshold and shorter hangover measure it more accurately.
CHANNEL_REL_K = {0: 6.0, 1: 8.0}
CHANNEL_HANGOVER_FRAMES = {0: 2, 1: 2}


def vad_channel(
    samples: np.ndarray,
    sr: int,
    *,
    frame_ms: int = FRAME_MS,
    min_speech_s: float = MIN_SPEECH_S,
    merge_gap_s: float = MERGE_GAP_S,
    abs_thr: float = ABS_THR,
    rel_k: float = REL_K,
    hangover_frames: int = HANGOVER_FRAMES,
) -> list[dict]:
    """Energy VAD tuned to match the official turns (min 0.2s, merge 0.3s)."""
    hop = max(1, int(sr * frame_ms / 1000))
    n_frames = len(samples) // hop
    if n_frames < 3:
        return []

    frames = samples[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    q20 = float(np.percentile(rms, 20))
    quiet = rms[rms <= q20]
    noise = float(np.median(quiet)) if quiet.size else float(np.median(rms))
    thr = max(abs_thr, noise * rel_k)
    speech = rms > thr

    with_hang = speech.copy()
    last_on = -10_000
    for i, flag in enumerate(speech):
        if flag:
            last_on = i
        if i - last_on <= hangover_frames:
            with_hang[i] = True

    raw: list[list[float]] = []
    i = 0
    while i < n_frames:
        if not with_hang[i]:
            i += 1
            continue
        j = i + 1
        while j < n_frames and with_hang[j]:
            j += 1
        raw.append([i * hop / sr, j * hop / sr])
        i = j

    merged: list[list[float]] = []
    for start, end in raw:
        if merged and start - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [
        {"start": round(start, 4), "end": round(end, 4)}
        for start, end in merged
        if end - start >= min_speech_s
    ]


def turns_from_audio(audio: np.ndarray, sr: int) -> list[dict]:
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    turns: list[dict] = []
    for channel in (0, 1):
        seg_iter = vad_channel(
            audio[:, channel],
            sr,
            rel_k=CHANNEL_REL_K[channel],
            hangover_frames=CHANNEL_HANGOVER_FRAMES[channel],
        )
        for seg in seg_iter:
            turns.append({"channel": channel, **seg})
    turns.sort(key=lambda t: (t["start"], t["channel"]))
    return turns
