from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from detector.features import features_from_audio, load_official_turns, extract_all_features, vectorize
from detector.io_wav import load_wav_bytes, load_wav_path
from detector.paths import MODEL_PATH, audio_path, turns_path
from detector.transcribe import load_transcript, transcribe_audio


def load_bundle(model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Train first: python -m detector.train --from-wav  (missing {model_path})")
    return joblib.load(model_path)


def _confidence(p_synth: float, is_synthetic: bool) -> float:
    p = float(np.clip(p_synth, 1e-6, 1 - 1e-6))
    return float(p if is_synthetic else 1.0 - p)


def classify_features(feat_dict: dict[str, float], bundle: dict | None = None) -> dict:
    bundle = bundle or load_bundle()
    names = bundle["feature_names"]
    x = vectorize(feat_dict, names).reshape(1, -1)
    model = bundle["model"]
    p_synth = float(model.predict_proba(x)[0, 1])
    is_synthetic = p_synth >= 0.5
    return {
        "is_synthetic": bool(is_synthetic),
        "confidence": round(_confidence(p_synth, is_synthetic), 4),
        "p_synthetic": round(p_synth, 4),
    }


def predict_from_wav_bytes(data: bytes, bundle: dict | None = None) -> dict:
    bundle = bundle or load_bundle()
    audio, sr = load_wav_bytes(data)
    return _predict_audio(audio, sr, bundle, transcript=None, run_asr=False)


def predict_from_path(anon_or_path: str, *, from_wav: bool = True, bundle: dict | None = None) -> dict:
    bundle = bundle or load_bundle()
    path = Path(anon_or_path)
    anon_id = path.stem if path.suffix == ".wav" else anon_or_path
    wav = path if path.suffix == ".wav" else audio_path(anon_id)
    transcript = load_transcript(anon_id) if bundle.get("with_semantic") else None

    if from_wav:
        audio, sr = load_wav_path(wav)
        return _predict_audio(audio, sr, bundle, transcript=transcript, run_asr=False)

    turns = load_official_turns(turns_path(anon_id))
    feats = extract_all_features(
        turns,
        transcript=transcript,
        with_semantic=bool(bundle.get("with_semantic")),
        with_acoustic=False,
    )
    return classify_features(feats, bundle)


def _predict_audio(
    audio: np.ndarray,
    sr: int,
    bundle: dict,
    transcript: dict | None,
    run_asr: bool,
) -> dict:
    if bundle.get("with_semantic") and transcript is None and run_asr:
        try:
            transcript = transcribe_audio(audio, sr)
        except Exception:
            transcript = None
    feats, _ = features_from_audio(
        audio,
        sr,
        transcript=transcript,
        with_semantic=bool(bundle.get("with_semantic")),
        with_acoustic=bool(bundle.get("with_acoustic")),
    )
    return classify_features(feats, bundle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("anon_id")
    parser.add_argument("--wav", action="store_true", default=True)
    parser.add_argument("--turns", action="store_true", help="Use official turns JSON instead of VAD")
    args = parser.parse_args()
    result = predict_from_path(args.anon_id, from_wav=not args.turns)
    print(result)


if __name__ == "__main__":
    main()
