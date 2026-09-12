from __future__ import annotations

import numpy as np
import pandas as pd

from detector.features import features_from_audio, load_official_turns
from detector.io_wav import load_wav_path
from detector.paths import MANIFEST, audio_path, turns_path
from detector.predict import classify_features, load_bundle
from detector.vad import turns_from_audio


def _mask(turns: list[dict], channel: int, duration: float, hop: float = 0.01) -> np.ndarray:
    n = max(1, int(duration / hop))
    mask = np.zeros(n, dtype=bool)
    for turn in turns:
        if turn["channel"] != channel:
            continue
        a = max(0, int(turn["start"] / hop))
        b = min(n, int(turn["end"] / hop))
        mask[a:b] = True
    return mask


def iou(pred: list[dict], gold: list[dict], channel: int, duration: float) -> float:
    p = _mask(pred, channel, duration)
    g = _mask(gold, channel, duration)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / union)


def main() -> None:
    df = pd.read_csv(MANIFEST)
    ious = {0: [], 1: []}
    for _, row in df.iterrows():
        anon_id = row["anon_id"]
        wav = audio_path(anon_id)
        tpath = turns_path(anon_id)
        if not wav.exists() or not tpath.exists():
            continue
        audio, sr = load_wav_path(wav)
        pred = turns_from_audio(audio, sr)
        gold = load_official_turns(tpath)
        duration = len(audio) / sr
        for ch in (0, 1):
            ious[ch].append(iou(pred, gold, ch, duration))

    print(f"IoU caller mean={np.mean(ious[0]):.3f} med={np.median(ious[0]):.3f}")
    print(f"IoU agent  mean={np.mean(ious[1]):.3f} med={np.median(ious[1]):.3f}")

    try:
        bundle = load_bundle()
    except FileNotFoundError:
        print("no trained model; skip val accuracy")
        return

    val = df[df["split"] == "val"]
    y_true, y_pred = [], []
    for _, row in val.iterrows():
        audio, sr = load_wav_path(audio_path(row["anon_id"]))
        feats, _ = features_from_audio(
            audio,
            sr,
            with_semantic=bool(bundle.get("with_semantic")),
            with_acoustic=bool(bundle.get("with_acoustic")),
        )
        out = classify_features(feats, bundle)
        y_true.append(row["label"] == "synthetic")
        y_pred.append(out["is_synthetic"])
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    print(f"val accuracy with VAD model: {acc:.3f}  ({int(acc * len(val))}/{len(val)})")


if __name__ == "__main__":
    main()
