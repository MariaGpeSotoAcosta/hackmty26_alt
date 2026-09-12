from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from detector.acoustic import ACOUSTIC_FEATURES
from detector.features import (
    DIALOGUE_FEATURES,
    extract_all_features,
    features_from_audio,
    load_official_turns,
    vectorize,
)
from detector.io_wav import load_wav_path
from detector.paths import FEATURES_CSV, MANIFEST, MODEL_PATH, MODELS_DIR, audio_path, turns_path
from detector.predict import classify_features
from detector.semantic import SEMANTIC_FEATURES
from detector.transcribe import load_transcript


def feature_names(with_semantic: bool, with_acoustic: bool) -> list[str]:
    names = list(DIALOGUE_FEATURES)
    if with_semantic:
        names.extend(SEMANTIC_FEATURES)
    if with_acoustic:
        names.extend(ACOUSTIC_FEATURES)
    return names


def extract_dataset(from_wav: bool, with_semantic: bool, with_acoustic: bool) -> pd.DataFrame:
    df = pd.read_csv(MANIFEST)
    rows: list[dict] = []
    # Always pull acoustic columns on WAV so we can train a grey-zone tiebreaker
    # without putting those features in the main dialogue model.
    extract_acoustic = from_wav or with_acoustic
    for i, row in df.iterrows():
        anon_id = row["anon_id"]
        transcript = load_transcript(anon_id) if with_semantic else None
        wav = audio_path(anon_id)
        audio = sr = None
        if from_wav or extract_acoustic:
            if not wav.exists():
                print(f"skip missing audio {anon_id}")
                continue
            audio, sr = load_wav_path(wav)
            feats, _ = features_from_audio(
                audio,
                sr,
                transcript=transcript,
                with_semantic=with_semantic,
                with_acoustic=extract_acoustic,
            )
        else:
            tpath = turns_path(anon_id)
            if not tpath.exists():
                print(f"skip missing turns {anon_id}")
                continue
            turns = load_official_turns(tpath)
            feats = extract_all_features(
                turns,
                float(row["duration_s"]),
                audio=audio,
                sr=sr,
                transcript=transcript,
                with_semantic=with_semantic,
                with_acoustic=with_acoustic,
            )
        feats["anon_id"] = anon_id
        feats["label"] = row["label"]
        feats["split"] = row["split"]
        feats["view"] = "vad" if from_wav or extract_acoustic else "turns"
        rows.append(feats)
        if (i + 1) % 25 == 0:
            print(f"features {i + 1}/{len(df)}")
    return pd.DataFrame(rows)


def _official_turns_train_rows(with_semantic: bool) -> pd.DataFrame:
    """Second view of train only. Val and /detect stay on VAD."""
    df = pd.read_csv(MANIFEST)
    rows: list[dict] = []
    for row in df[df["split"] == "train"].itertuples():
        tpath = turns_path(row.anon_id)
        if not tpath.exists():
            continue
        transcript = load_transcript(row.anon_id) if with_semantic else None
        feats = extract_all_features(
            load_official_turns(tpath),
            float(row.duration_s),
            transcript=transcript,
            with_semantic=with_semantic,
            with_acoustic=False,
        )
        feats["anon_id"] = row.anon_id
        feats["label"] = row.label
        feats["split"] = "train"
        feats["view"] = "turns"
        rows.append(feats)
    return pd.DataFrame(rows)


def _vad_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "view" not in frame.columns:
        return frame
    return frame[frame["view"] != "turns"].copy()


def _split_xy(frame: pd.DataFrame, names: list[str], split: str) -> tuple[np.ndarray, np.ndarray]:
    part = frame[frame["split"] == split]
    x = np.vstack([vectorize(row, names) for row in part.to_dict(orient="records")])
    y = (part["label"].to_numpy() == "synthetic").astype(int)
    return x, y


def _logreg() -> Pipeline:
    clf = LogisticRegression(
        C=0.4,
        class_weight="balanced",
        max_iter=800,
        solver="lbfgs",
    )
    calibrated = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
    return Pipeline([("scaler", StandardScaler()), ("clf", calibrated)])


def _hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3,
        min_samples_leaf=15,
        learning_rate=0.06,
        max_iter=120,
        l2_regularization=0.2,
        random_state=42,
    )


def _report(name: str, y_true: np.ndarray, proba: np.ndarray) -> float:
    pred = (proba >= 0.5).astype(int)
    print(f"\n=== {name} ===")
    print(confusion_matrix(y_true, pred))
    print(classification_report(y_true, pred, target_names=["human", "synthetic"], digits=3))
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, proba)
        print(f"AUC {auc:.3f}")
    acc = float((pred == y_true).mean())
    print(f"accuracy {acc:.3f}")
    return acc


def train(
    from_wav: bool,
    with_semantic: bool,
    with_acoustic: bool,
    *,
    dual_view: bool = False,
    reuse_csv: bool = False,
) -> Path:
    print(
        f"extracting features from_wav={from_wav} semantic={with_semantic} "
        f"acoustic={with_acoustic} dual_view={dual_view} reuse_csv={reuse_csv}"
    )
    if reuse_csv and FEATURES_CSV.exists():
        frame = pd.read_csv(FEATURES_CSV)
        if "view" not in frame.columns:
            frame["view"] = "vad" if from_wav else "turns"
        print(f"reused {FEATURES_CSV}  rows={len(frame)}")
    else:
        frame = extract_dataset(from_wav, with_semantic, with_acoustic)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        vad_out = _vad_rows(frame)
        vad_out.to_csv(FEATURES_CSV, index=False)

    if dual_view:
        extra = _official_turns_train_rows(with_semantic)
        print(f"dual-view: +{len(extra)} official-turns train rows")
        frame = pd.concat([frame, extra], ignore_index=True)

    names = feature_names(with_semantic, with_acoustic)
    x_train, y_train = _split_xy(frame, names, "train")
    x_val, y_val = _split_xy(_vad_rows(frame), names, "val")
    print("train", x_train.shape, "val", x_val.shape, "(val = VAD only)")

    candidates = [("logreg", _logreg()), ("hgb", _hgb())]
    scored: list[tuple[str, object, float, float]] = []
    for name, model in candidates:
        model.fit(x_train, y_train)
        train_acc = _report(f"{name} TRAIN", y_train, _proba(model, x_train))
        val_acc = _report(f"{name} VAL", y_val, _proba(model, x_val))
        scored.append((name, model, train_acc, val_acc))

    # Hidden set is speaker-disjoint. A 1-call val win is not worth a memorizing tree.
    logreg = next(s for s in scored if s[0] == "logreg")
    others = [s for s in scored if s[0] != "logreg"]
    best_name, best_model, train_acc, val_acc = logreg
    for name, model, tr_acc, va_acc in others:
        gap = tr_acc - va_acc
        if va_acc >= val_acc + 0.03 and gap < 0.08:
            best_name, best_model, train_acc, val_acc = name, model, tr_acc, va_acc
    print(f"\nselected {best_name}  train={train_acc:.3f} val={val_acc:.3f}")
    if best_name == "logreg":
        print("kept logreg: trees that jump to ~100% train do not survive unseen voices")

    bundle = {
        "model": best_model,
        "feature_names": names,
        "from_wav": from_wav,
        "dual_view": dual_view,
        "with_semantic": with_semantic,
        "with_acoustic": with_acoustic,
        "model_name": best_name,
        "val_accuracy": val_acc,
        "train_accuracy": train_acc,
        "tiebreak": False,
        "early_exit": False,
    }
    _attach_acoustic_tiebreak(bundle, frame, val_acc)
    # Judge path is VAD. Dual-view train acc is not comparable to the old 91.5%.
    vad_train_acc = _score_split(frame, bundle, "train")
    bundle["train_accuracy"] = vad_train_acc
    print(f"VAD-only train acc={vad_train_acc:.3f}  (what /detect sees)")
    joblib.dump(bundle, MODEL_PATH)
    print("wrote", MODEL_PATH)
    _print_logreg_weights(best_model, names)
    _dump_error_ids(frame, bundle)
    return MODEL_PATH


def _attach_acoustic_tiebreak(bundle: dict, frame: pd.DataFrame, dialogue_val: float) -> None:
    if not all(col in frame.columns for col in ACOUSTIC_FEATURES):
        return
    vad = _vad_rows(frame)
    if not all(col in vad.columns for col in ACOUSTIC_FEATURES):
        return
    x_train, y_train = _split_xy(vad, ACOUSTIC_FEATURES, "train")
    x_val, y_val = _split_xy(vad, ACOUSTIC_FEATURES, "val")
    ac_model = _logreg()
    ac_model.fit(x_train, y_train)
    ac_val = _report("acoustic-only VAL", y_val, _proba(ac_model, x_val))

    lo, hi = 0.35, 0.65
    trial = dict(bundle)
    trial["acoustic_model"] = ac_model
    trial["acoustic_feature_names"] = list(ACOUSTIC_FEATURES)
    trial["tiebreak"] = True
    trial["tiebreak_lo"] = lo
    trial["tiebreak_hi"] = hi
    y_true, y_hat, n_flip = [], [], 0
    val = _vad_rows(frame)
    val = val[val["split"] == "val"]
    for row in val.to_dict(orient="records"):
        gold = row["label"] == "synthetic"
        off = classify_features(row, {**trial, "tiebreak": False})
        on = classify_features(row, trial)
        y_true.append(gold)
        y_hat.append(on["is_synthetic"])
        if off["is_synthetic"] != on["is_synthetic"]:
            n_flip += 1
    blend_acc = float(np.mean(np.array(y_true) == np.array(y_hat)))
    print(f"\ntiebreak VAL acc={blend_acc:.3f}  dialogue={dialogue_val:.3f}  acoustic={ac_val:.3f}  flipped={n_flip}")
    bundle["acoustic_model"] = ac_model
    bundle["acoustic_feature_names"] = list(ACOUSTIC_FEATURES)
    bundle["tiebreak_lo"] = lo
    bundle["tiebreak_hi"] = hi
    # Strict > : an exact tie (this val: 0.944 vs 0.944) is not a reason to
    # ship acoustics. Hidden voices lose recall under that extra model.
    if blend_acc > dialogue_val:
        bundle["tiebreak"] = True
        bundle["val_accuracy"] = blend_acc
        print("enabled acoustic tiebreak on grey-zone dialogue scores")
    else:
        bundle["tiebreak"] = False
        print("disabled acoustic tiebreak: it did not beat dialogue-only val")


def _print_logreg_weights(model, names: list[str]) -> None:
    clf = model
    if isinstance(model, Pipeline):
        clf = model.named_steps.get("clf", model)
    if isinstance(clf, CalibratedClassifierCV):
        coefs = []
        for est in getattr(clf, "calibrated_classifiers_", []):
            inner = getattr(est, "estimator", None) or getattr(est, "base_estimator", None)
            if inner is not None and hasattr(inner, "coef_"):
                coefs.append(inner.coef_.ravel())
        if not coefs:
            return
        weights = np.mean(coefs, axis=0)
        ranked = sorted(zip(names, weights), key=lambda kv: -abs(kv[1]))
        print("\nlogreg weights (synthetic +):")
        for name, weight in ranked[:12]:
            print(f"  {weight:+.3f}  {name}")


def _proba(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    scores = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-scores))


def _score_split(frame: pd.DataFrame, bundle: dict, split: str) -> float:
    part = _vad_rows(frame)
    part = part[part["split"] == split]
    y_true, y_hat = [], []
    for row in part.to_dict(orient="records"):
        out = classify_features(row, bundle)
        y_true.append(row["label"] == "synthetic")
        y_hat.append(out["is_synthetic"])
    return float(np.mean(np.array(y_true) == np.array(y_hat)))


def _dump_error_ids(frame: pd.DataFrame, bundle: dict) -> None:
    val = _vad_rows(frame)
    val = val[val["split"] == "val"]
    wrong = []
    for row in val.to_dict(orient="records"):
        out = classify_features(row, bundle)
        gold = row["label"] == "synthetic"
        if out["is_synthetic"] != gold:
            wrong.append({"anon_id": row["anon_id"], "label": row["label"]})
    path = MODELS_DIR / "val_errors.json"
    path.write_text(json.dumps(wrong, indent=2), encoding="utf-8")
    print(f"val errors ({len(wrong)}): {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-wav", action="store_true", help="VAD on WAV (production / judge path)")
    parser.add_argument("--with-semantic", action="store_true")
    parser.add_argument("--with-acoustic", action="store_true")
    parser.add_argument(
        "--dual-view",
        action="store_true",
        help="Also train on official turns for train calls; val and /detect stay VAD",
    )
    parser.add_argument(
        "--reuse-csv",
        action="store_true",
        help="Reuse models/dialogue_features.csv instead of re-extracting WAVs",
    )
    args = parser.parse_args()
    train(
        args.from_wav,
        args.with_semantic,
        args.with_acoustic,
        dual_view=args.dual_view,
        reuse_csv=args.reuse_csv,
    )


if __name__ == "__main__":
    main()
