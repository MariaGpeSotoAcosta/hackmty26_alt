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
    for i, row in df.iterrows():
        anon_id = row["anon_id"]
        transcript = load_transcript(anon_id) if with_semantic else None
        wav = audio_path(anon_id)
        audio = sr = None
        if from_wav or with_acoustic:
            if not wav.exists():
                print(f"skip missing audio {anon_id}")
                continue
            audio, sr = load_wav_path(wav)
            feats, _ = features_from_audio(
                audio,
                sr,
                transcript=transcript,
                with_semantic=with_semantic,
                with_acoustic=with_acoustic,
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
        rows.append(feats)
        if (i + 1) % 25 == 0:
            print(f"features {i + 1}/{len(df)}")
    return pd.DataFrame(rows)


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


def train(from_wav: bool, with_semantic: bool, with_acoustic: bool) -> Path:
    print(
        f"extracting features from_wav={from_wav} semantic={with_semantic} acoustic={with_acoustic}"
    )
    frame = extract_dataset(from_wav, with_semantic, with_acoustic)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FEATURES_CSV, index=False)

    names = feature_names(with_semantic, with_acoustic)
    x_train, y_train = _split_xy(frame, names, "train")
    x_val, y_val = _split_xy(frame, names, "val")
    print("train", x_train.shape, "val", x_val.shape)

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
        "with_semantic": with_semantic,
        "with_acoustic": with_acoustic,
        "model_name": best_name,
        "val_accuracy": val_acc,
        "train_accuracy": train_acc,
    }
    joblib.dump(bundle, MODEL_PATH)
    print("wrote", MODEL_PATH)
    _print_logreg_weights(best_model, names)
    _dump_error_ids(frame, names, best_model)
    return MODEL_PATH


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


def _dump_error_ids(frame: pd.DataFrame, names: list[str], model) -> None:
    val = frame[frame["split"] == "val"].copy()
    x, y = _split_xy(val, names, "val")
    pred = (_proba(model, x) >= 0.5).astype(int)
    wrong = val.loc[pred != y, ["anon_id", "label"]]
    path = MODELS_DIR / "val_errors.json"
    path.write_text(json.dumps(wrong.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(f"val errors ({len(wrong)}): {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-wav", action="store_true", help="VAD on WAV (production / judge path)")
    parser.add_argument("--with-semantic", action="store_true")
    parser.add_argument("--with-acoustic", action="store_true")
    args = parser.parse_args()
    train(args.from_wav, args.with_semantic, args.with_acoustic)


if __name__ == "__main__":
    main()
