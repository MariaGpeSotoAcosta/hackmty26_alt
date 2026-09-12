from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from detector.acoustic import extract_acoustic_features
from detector.semantic import extract_semantic_features, empty_semantic_features
from detector.vad import turns_from_audio

DIALOGUE_FEATURES = [
    "duration_s",
    "n_caller",
    "n_agent",
    "turn_caller_mean",
    "turn_caller_std",
    "turn_caller_cv",
    "caller_speech_ratio",
    "agent_speech_ratio",
    "barge_in",
    "barge_rate",
    "agent_barge",
    "overlap_s",
    "overlap_rate",
    "lat_mean",
    "lat_med",
    "lat_p90",
    "lat_std",
    "lat_cv",
    "first_latency",
    "silence_fill",
    "silence_fill_rate",
    "caller_gap_mean",
]


def load_official_turns(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["turns"]


def extract_dialogue_features(turns: list[dict], duration_s: float | None = None) -> dict[str, float]:
    caller = _channel(turns, 0)
    agent = _channel(turns, 1)
    if duration_s is None:
        last = max((t["end"] for t in turns), default=0.0)
        duration_s = max(last, 1.0)
    duration_s = max(float(duration_s), 1.0)

    caller_durs = [t["end"] - t["start"] for t in caller]
    agent_durs = [t["end"] - t["start"] for t in agent]
    n_caller = len(caller)
    n_agent = len(agent)

    latencies = _response_latencies(caller, agent)
    lat_mean = _mean(latencies)
    lat_std = _std(latencies)
    barges = _barge_count(caller, agent)
    agent_barges = _barge_count(agent, caller)
    overlap_s = _overlap_seconds(caller, agent)
    fills = _silence_fills(caller, agent)
    caller_gaps = _same_speaker_gaps(caller)

    turn_mean = _mean(caller_durs)
    turn_std = _std(caller_durs)
    return {
        "duration_s": duration_s,
        "n_caller": float(n_caller),
        "n_agent": float(n_agent),
        "turn_caller_mean": turn_mean,
        "turn_caller_std": turn_std,
        "turn_caller_cv": turn_std / turn_mean if turn_mean > 1e-6 else 0.0,
        "caller_speech_ratio": sum(caller_durs) / duration_s,
        "agent_speech_ratio": sum(agent_durs) / duration_s,
        "barge_in": float(barges),
        "barge_rate": barges / n_caller if n_caller else 0.0,
        "agent_barge": float(agent_barges),
        "overlap_s": overlap_s,
        "overlap_rate": overlap_s / duration_s,
        "lat_mean": lat_mean,
        "lat_med": _percentile(latencies, 50),
        "lat_p90": _percentile(latencies, 90),
        "lat_std": lat_std,
        "lat_cv": lat_std / lat_mean if lat_mean > 1e-6 else 0.0,
        "first_latency": latencies[0] if latencies else 0.0,
        "silence_fill": float(fills),
        "silence_fill_rate": fills / n_caller if n_caller else 0.0,
        "caller_gap_mean": _mean(caller_gaps),
    }


def extract_all_features(
    turns: list[dict],
    duration_s: float | None = None,
    *,
    audio: np.ndarray | None = None,
    sr: int | None = None,
    transcript: dict | None = None,
    with_semantic: bool = False,
    with_acoustic: bool = False,
) -> dict[str, float]:
    feats = extract_dialogue_features(turns, duration_s)
    if with_semantic:
        feats.update(extract_semantic_features(transcript) if transcript else empty_semantic_features())
    if with_acoustic:
        if audio is None or sr is None:
            raise ValueError("audio and sr are required for acoustic features")
        feats.update(extract_acoustic_features(audio, sr, turns))
    return feats


def features_from_audio(
    audio: np.ndarray,
    sr: int,
    *,
    transcript: dict | None = None,
    with_semantic: bool = False,
    with_acoustic: bool = False,
) -> tuple[dict[str, float], list[dict]]:
    turns = turns_from_audio(audio, sr)
    duration_s = len(audio) / max(sr, 1)
    feats = extract_all_features(
        turns,
        duration_s,
        audio=audio,
        sr=sr,
        transcript=transcript,
        with_semantic=with_semantic,
        with_acoustic=with_acoustic,
    )
    return feats, turns


def vectorize(feat_dict: dict[str, float], names: list[str]) -> np.ndarray:
    return np.array([float(feat_dict.get(name, 0.0)) for name in names], dtype=np.float64)


def _channel(turns: Iterable[dict], channel: int) -> list[dict]:
    return sorted(
        [t for t in turns if t["channel"] == channel],
        key=lambda t: t["start"],
    )


def _overlap_seconds(caller: list[dict], agent: list[dict]) -> float:
    total = 0.0
    for c in caller:
        for a in agent:
            start = max(c["start"], a["start"])
            end = min(c["end"], a["end"])
            if end > start:
                total += end - start
    return total


def _barge_count(speaker: list[dict], other: list[dict]) -> int:
    count = 0
    for turn in speaker:
        for other_turn in other:
            if other_turn["start"] < turn["start"] < other_turn["end"] - 0.05:
                count += 1
                break
    return count


def _response_latencies(caller: list[dict], agent: list[dict]) -> list[float]:
    if not agent:
        return []
    latencies: list[float] = []
    for turn in caller:
        previous = [a for a in agent if a["end"] <= turn["start"] + 0.05]
        if not previous:
            continue
        last = max(previous, key=lambda a: a["end"])
        lat = turn["start"] - last["end"]
        if lat >= -0.05:
            latencies.append(max(lat, 0.0))
    return latencies


def _silence_fills(caller: list[dict], agent: list[dict], gap_s: float = 1.5) -> int:
    """Caller speaks again after waiting for an agent that does not come back."""
    fills = 0
    for prev, curr in zip(caller, caller[1:]):
        gap = curr["start"] - prev["end"]
        if gap < gap_s:
            continue
        agent_in_gap = any(
            prev["end"] - 0.05 < a["start"] < curr["start"] + 0.05 for a in agent
        )
        if not agent_in_gap:
            fills += 1
    return fills


def _same_speaker_gaps(segs: list[dict]) -> list[float]:
    return [b["start"] - a["end"] for a, b in zip(segs, segs[1:]) if b["start"] >= a["end"]]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if len(values) > 1 else 0.0


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0
