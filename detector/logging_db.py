from __future__ import annotations

"""Logs every /detect call to Tiger Data (TimescaleDB) for the dashboard.

Configured via the TIGER_DATA_URL env var (a normal Postgres connection
string — Tiger Data is Postgres-compatible). If it's unset, or the database
is unreachable, logging is a silent no-op: /detect must never fail or slow
down because the dashboard's database had a hiccup.
"""

import csv
import os
from contextlib import contextmanager
from typing import Any, Iterator

from detector.paths import MANIFEST

_DB_URL = os.environ.get("TIGER_DATA_URL")
_SCHEMA_READY = False

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS calls (
    time             TIMESTAMPTZ NOT NULL DEFAULT now(),
    call_id          TEXT NOT NULL,
    is_synthetic     BOOLEAN NOT NULL,
    confidence       DOUBLE PRECISION NOT NULL,
    p_synthetic      DOUBLE PRECISION NOT NULL,
    tiebreak         BOOLEAN NOT NULL DEFAULT false,
    early_exit       BOOLEAN NOT NULL DEFAULT false,
    latency_ms       DOUBLE PRECISION NOT NULL,
    features         JSONB NOT NULL,
    contributions    JSONB
);
SELECT create_hypertable('calls', 'time', if_not_exists => TRUE);
"""


def enabled() -> bool:
    return bool(_DB_URL)


@contextmanager
def _conn() -> Iterator[Any]:
    """A fresh, short-lived connection per call.

    No pooling on purpose: this dashboard is low-traffic, and a plain
    connect-per-call avoids any chance of a stale/reused connection or
    cursor carrying state across requests (seen in the wild after the
    server churned through many restarts).
    """
    import psycopg

    with psycopg.connect(_DB_URL, autocommit=True) as conn:
        yield conn


def ensure_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY or not enabled():
        return
    with _conn() as conn:
        conn.execute(_SCHEMA_SQL)
    _SCHEMA_READY = True


def log_call(
    call_id: str,
    result: dict,
    contributions: list[dict] | None,
    latency_ms: float,
) -> None:
    """Fire-and-forget: never raises, never blocks /detect on failure."""
    if not enabled():
        return
    try:
        import json

        ensure_schema()
        with _conn() as conn:
            conn.execute(
                """
                INSERT INTO calls
                    (call_id, is_synthetic, confidence, p_synthetic, tiebreak,
                     early_exit, latency_ms, features, contributions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    call_id,
                    bool(result.get("is_synthetic")),
                    float(result.get("confidence", 0.0)),
                    float(result.get("p_synthetic", 0.0)),
                    bool(result.get("tiebreak", False)),
                    bool(result.get("early_exit", False)),
                    float(latency_ms),
                    json.dumps(result.get("features") or {}),
                    json.dumps(contributions) if contributions is not None else None,
                ),
            )
    except Exception:
        pass


def recent_calls(limit: int = 50) -> list[dict]:
    if not enabled():
        return []
    try:
        with _conn() as conn:
            rows = conn.execute(
                """
                SELECT time, call_id, is_synthetic, confidence, p_synthetic,
                       tiebreak, early_exit, latency_ms, features, contributions
                FROM calls ORDER BY time DESC LIMIT %s
                """,
                (limit,),
            ).fetchall()
        cols = ["time", "call_id", "is_synthetic", "confidence", "p_synthetic",
                "tiebreak", "early_exit", "latency_ms", "features", "contributions"]
        return [dict(zip(cols, r)) for r in rows]
    except Exception:
        return []


_LABELS: dict[str, str] | None = None


def _manifest_labels() -> dict[str, str]:
    global _LABELS
    if _LABELS is not None:
        return _LABELS
    labels: dict[str, str] = {}
    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                labels[row["anon_id"]] = row["label"]
    _LABELS = labels
    return labels


def _auc(scores: list[float], labels: list[int]) -> float | None:
    pairs = sorted(zip(scores, labels))
    ranks, i = {}, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        for k in range(i, j):
            ranks[k] = (i + j + 1) / 2
        i = j
    pos = [ranks[k] for k, (_, y) in enumerate(pairs) if y == 1]
    n_pos, n_neg = len(pos), len(pairs) - len(pos)
    if not n_pos or not n_neg:
        return None
    return (sum(pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _calibration(probs: list[float], ys: list[int], n_bins: int = 5) -> list[dict]:
    bins = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        idx = [k for k, p in enumerate(probs) if (p >= lo and p < hi) or (hi == 1.0 and p == 1.0)]
        if not idx:
            continue
        pred = [probs[k] for k in idx]
        actual = [ys[k] for k in idx]
        bins.append(
            {
                "bucket_lo": lo,
                "bucket_hi": hi,
                "n": len(idx),
                "mean_predicted": sum(pred) / len(pred),
                "actual_rate_synthetic": sum(actual) / len(actual),
            }
        )
    return bins


def _score_labeled(rows: list[tuple]) -> dict[str, Any]:
    """rows: (call_id, is_synthetic, confidence, p_synthetic, latency_ms)."""
    gold = _manifest_labels()
    labeled = []
    for call_id, is_synth, conf, p_synth, _lat in rows:
        label = gold.get(call_id)
        if label not in ("human", "synthetic"):
            continue
        labeled.append(
            {
                "label": label,
                "is_synthetic": bool(is_synth),
                "confidence": float(conf),
                "p_synthetic": float(p_synth),
            }
        )
    n_lab = len(labeled)
    if not n_lab:
        return {
            "labeled": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "tpr_synthetic": None,
            "tnr_human": None,
            "auc": None,
            "brier": None,
            "calibration": [],
        }
    tp = sum(1 for r in labeled if r["label"] == "synthetic" and r["is_synthetic"])
    tn = sum(1 for r in labeled if r["label"] == "human" and not r["is_synthetic"])
    n_syn = sum(1 for r in labeled if r["label"] == "synthetic")
    n_hum = sum(1 for r in labeled if r["label"] == "human")
    tpr = tp / n_syn if n_syn else None
    tnr = tn / n_hum if n_hum else None
    probs = [r["p_synthetic"] for r in labeled]
    ys = [1 if r["label"] == "synthetic" else 0 for r in labeled]
    return {
        "labeled": n_lab,
        "accuracy": (tp + tn) / n_lab,
        "tpr_synthetic": tpr,
        "tnr_human": tnr,
        "balanced_accuracy": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
        "auc": _auc(probs, ys),
        "brier": sum((p - y) ** 2 for p, y in zip(probs, ys)) / n_lab,
        "calibration": _calibration(probs, ys),
    }


def aggregate_stats(window_minutes: int = 240) -> dict:
    if not enabled():
        return {"available": False}
    try:
        with _conn() as conn:
            total, n_synth, avg_conf, avg_latency, p95_latency, max_latency = conn.execute(
                """
                SELECT count(*), sum(is_synthetic::int), avg(confidence),
                       avg(latency_ms), percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms),
                       max(latency_ms)
                FROM calls WHERE time > now() - (%s || ' minutes')::interval
                """,
                (window_minutes,),
            ).fetchone()
            buckets = conn.execute(
                """
                SELECT time_bucket('1 minute', time) AS bucket,
                       count(*), avg(confidence), avg(latency_ms)
                FROM calls WHERE time > now() - (%s || ' minutes')::interval
                GROUP BY bucket ORDER BY bucket
                """,
                (window_minutes,),
            ).fetchall()
            scored = conn.execute(
                """
                SELECT call_id, is_synthetic, confidence, p_synthetic, latency_ms
                FROM calls WHERE time > now() - (%s || ' minutes')::interval
                """,
                (window_minutes,),
            ).fetchall()
        out = {
            "available": True,
            "total": total or 0,
            "n_synthetic": n_synth or 0,
            "avg_confidence": float(avg_conf) if avg_conf is not None else None,
            "avg_latency_ms": float(avg_latency) if avg_latency is not None else None,
            "p95_latency_ms": float(p95_latency) if p95_latency is not None else None,
            "max_latency_ms": float(max_latency) if max_latency is not None else None,
            "timeline": [
                {
                    "time": b[0].isoformat(),
                    "count": b[1],
                    "avg_confidence": float(b[2]) if b[2] is not None else None,
                    "avg_latency_ms": float(b[3]) if b[3] is not None else None,
                }
                for b in buckets
            ],
        }
        out.update(_score_labeled(list(scored)))
        return out
    except Exception:
        return {"available": False}
