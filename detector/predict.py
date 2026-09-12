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


def _model_p(model, feat_dict: dict[str, float], names: list[str]) -> float:
    x = vectorize(feat_dict, names).reshape(1, -1)
    return float(model.predict_proba(x)[0, 1])


def classify_features(feat_dict: dict[str, float], bundle: dict | None = None) -> dict:
    bundle = bundle or load_bundle()
    p_synth = _model_p(bundle["model"], feat_dict, bundle["feature_names"])
    p_dialogue = p_synth
    p_acoustic = None
    used_tiebreak = False
    ac_model = bundle.get("acoustic_model")
    ac_names = bundle.get("acoustic_feature_names") or []
    if ac_model is not None and bundle.get("tiebreak") and ac_names and all(n in feat_dict for n in ac_names):
        p_acoustic = _model_p(ac_model, feat_dict, ac_names)
        lo = float(bundle.get("tiebreak_lo", 0.35))
        hi = float(bundle.get("tiebreak_hi", 0.65))
        if lo <= p_dialogue <= hi:
            p_synth = 0.5 * p_dialogue + 0.5 * p_acoustic
            used_tiebreak = True
    is_synthetic = p_synth >= 0.5
    return {
        "is_synthetic": bool(is_synthetic),
        "confidence": round(_confidence(p_synth, is_synthetic), 4),
        "p_synthetic": round(p_synth, 4),
        "p_dialogue": round(p_dialogue, 4),
        "p_acoustic": None if p_acoustic is None else round(p_acoustic, 4),
        "tiebreak": used_tiebreak,
    }


def predict_from_wav_bytes(data: bytes, bundle: dict | None = None) -> dict:
    bundle = bundle or load_bundle()
    audio, sr = load_wav_bytes(data)
    run_asr = bool(bundle.get("with_semantic"))
    return _predict_audio(audio, sr, bundle, transcript=None, run_asr=run_asr)


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


def _need_acoustic(bundle: dict) -> bool:
    if bundle.get("with_acoustic"):
        return True
    return bool(bundle.get("tiebreak") and bundle.get("acoustic_model"))


def _classify_clip(
    audio: np.ndarray,
    sr: int,
    bundle: dict,
    transcript: dict | None,
) -> dict:
    feats, _ = features_from_audio(
        audio,
        sr,
        transcript=transcript,
        with_semantic=bool(bundle.get("with_semantic")),
        with_acoustic=_need_acoustic(bundle),
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

    early_s = float(bundle.get("early_exit_s", 90.0))
    early_lo = float(bundle.get("early_exit_lo", 0.05))
    early_hi = float(bundle.get("early_exit_hi", 0.95))
    n_early = int(early_s * sr)
    use_early = bool(bundle.get("early_exit", False)) and audio.shape[0] > n_early + sr
    if use_early:
        prefix = _classify_clip(audio[:n_early], sr, bundle, transcript)
        p = float(prefix["p_synthetic"])
        if p <= early_lo or p >= early_hi:
            prefix["early_exit"] = True
            return prefix

    result = _classify_clip(audio, sr, bundle, transcript)
    result["early_exit"] = False
    return result


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
