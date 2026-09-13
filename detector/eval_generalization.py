from __future__ import annotations

"""Leave-one-voice-cluster-out validation.

Random train/val splits only guarantee unseen *callers*, not unseen
*voices/engines* (the manifest promises the judges' hidden set brings both).
Two calls can share the same underlying synthetic voice with different
invented personal data on top, so a random split can hide poor generalization
to a genuinely new voice.

This groups the synthetic calls into acoustic-similarity clusters (a proxy
for "same voice/engine") and holds each cluster out completely during
training, instead of holding out random rows. Use this before trusting a
val-accuracy bump from any new feature.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from detector.acoustic import ACOUSTIC_FEATURES
from detector.features import DIALOGUE_FEATURES, vectorize
from detector.train import extract_dataset


def _logreg() -> Pipeline:
    clf = LogisticRegression(C=0.4, class_weight="balanced", max_iter=800, solver="lbfgs")
    return Pipeline([("scaler", StandardScaler()), ("clf", CalibratedClassifierCV(clf, method="sigmoid", cv=3))])


def leave_one_voice_cluster_out(
    X: np.ndarray,
    y: np.ndarray,
    cluster_features: np.ndarray,
    model_factory=_logreg,
    n_clusters: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Hold out each synthetic-voice cluster fully; report accuracy/recall per cluster."""
    synth_idx = np.where(y == 1)[0]
    human_idx = np.where(y == 0)[0]

    scaler = StandardScaler()
    cf = scaler.fit_transform(cluster_features[synth_idx])
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(cf)

    human_folds = [te for _, te in StratifiedKFold(n_splits=n_clusters, shuffle=True, random_state=seed).split(
        human_idx, np.zeros(len(human_idx))
    )]

    rows = []
    for c in range(n_clusters):
        held_synth = synth_idx[labels == c]
        train_synth = synth_idx[labels != c]
        held_human = human_idx[human_folds[c]]
        train_human = np.setdiff1d(human_idx, held_human)

        train_idx = np.concatenate([train_synth, train_human])
        test_idx = np.concatenate([held_synth, held_human])

        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        acc = float((pred == y[test_idx]).mean())
        synth_recall = float((model.predict(X[held_synth]) == 1).mean()) if len(held_synth) else float("nan")

        rows.append({
            "cluster": c,
            "n_synth_held": len(held_synth),
            "n_human_held": len(held_human),
            "accuracy": acc,
            "synth_recall": synth_recall,
        })
    return pd.DataFrame(rows)


class GatedAcousticModel:
    """Dialogue model by default; only consults acoustic when dialogue confidence is 0.50-0.65."""

    def __init__(self, n_dialogue: int, low: float = 0.35, high: float = 0.65):
        self.n_dialogue = n_dialogue
        self.low = low
        self.high = high

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GatedAcousticModel":
        self.dialogue_model = _logreg().fit(X[:, : self.n_dialogue], y)
        self.acoustic_model = _logreg().fit(X[:, self.n_dialogue :], y)
        return self

    def _proba(self, X: np.ndarray) -> np.ndarray:
        p_d = self.dialogue_model.predict_proba(X[:, : self.n_dialogue])[:, 1]
        p_a = self.acoustic_model.predict_proba(X[:, self.n_dialogue :])[:, 1]
        ambiguous = (p_d >= self.low) & (p_d <= self.high)
        p_final = p_d.copy()
        p_final[ambiguous] = p_a[ambiguous]
        return p_final

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._proba(X) >= 0.5).astype(int)


def main() -> None:
    print("extracting dialogue + acoustic features from WAV (production VAD)...")
    frame = extract_dataset(from_wav=True, with_semantic=False, with_acoustic=True)
    y = (frame["label"].to_numpy() == "synthetic").astype(int)

    X_dialogue = np.vstack([vectorize(row, DIALOGUE_FEATURES) for row in frame.to_dict(orient="records")])
    X_acoustic = np.vstack([vectorize(row, ACOUSTIC_FEATURES) for row in frame.to_dict(orient="records")])
    X_combo = np.concatenate([X_dialogue, X_acoustic], axis=1)

    print(f"\n{len(frame)} llamadas  (synthetic={int(y.sum())}  human={int((1 - y).sum())})")

    print("\n=== Produccion actual: solo dialogo (21 features), clusters de voz por acustica ===")
    res_dialogue = leave_one_voice_cluster_out(X_dialogue, y, X_acoustic)
    print(res_dialogue.to_string(index=False))
    print(f"recall promedio en voz oculta: {res_dialogue.synth_recall.mean():.3f}")

    print("\n=== Candidato: dialogo + acustica combinados, mismos clusters ===")
    res_combo = leave_one_voice_cluster_out(X_combo, y, X_acoustic)
    print(res_combo.to_string(index=False))
    print(f"recall promedio en voz oculta: {res_combo.synth_recall.mean():.3f}")

    print("\n=== Candidato: acustica solo como desempate (confianza dialogo 0.50-0.65) ===")
    n_dialogue = X_dialogue.shape[1]
    gated_factory = lambda: GatedAcousticModel(n_dialogue)
    res_gated = leave_one_voice_cluster_out(X_combo, y, X_acoustic, model_factory=gated_factory)
    print(res_gated.to_string(index=False))
    print(f"recall promedio en voz oculta: {res_gated.synth_recall.mean():.3f}")

    print("\n=== Comparacion ===")
    print(f"solo dialogo        -> accuracy media: {res_dialogue.accuracy.mean():.3f}  recall medio: {res_dialogue.synth_recall.mean():.3f}")
    print(f"dialogo+acust (todo)-> accuracy media: {res_combo.accuracy.mean():.3f}  recall medio: {res_combo.synth_recall.mean():.3f}")
    print(f"dialogo+acust (gate)-> accuracy media: {res_gated.accuracy.mean():.3f}  recall medio: {res_gated.synth_recall.mean():.3f}")


if __name__ == "__main__":
    main()
