from __future__ import annotations

"""Logs every /detect call to Tiger Data (TimescaleDB) for the dashboard.

Configured via the TIGER_DATA_URL env var (a normal Postgres connection
string — Tiger Data is Postgres-compatible). If it's unset, or the database
is unreachable, logging is a silent no-op: /detect must never fail or slow
down because the dashboard's database had a hiccup.
"""

import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

_DB_URL = os.environ.get("TIGER_DATA_URL")
_POOL = None
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


def _get_pool():
    global _POOL
    if _POOL is None:
        from psycopg_pool import ConnectionPool

        _POOL = ConnectionPool(_DB_URL, min_size=1, max_size=4, kwargs={"autocommit": True})
    return _POOL


@contextmanager
def _conn() -> Iterator[Any]:
    pool = _get_pool()
    with pool.connection() as conn:
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


def aggregate_stats(window_minutes: int = 240) -> dict:
    if not enabled():
        return {"available": False}
    try:
        with _conn() as conn:
            total, n_synth, avg_conf, avg_latency, p95_latency = conn.execute(
                """
                SELECT count(*), sum(is_synthetic::int), avg(confidence),
                       avg(latency_ms), percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
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
        return {
            "available": True,
            "total": total or 0,
            "n_synthetic": n_synth or 0,
            "avg_confidence": float(avg_conf) if avg_conf is not None else None,
            "avg_latency_ms": float(avg_latency) if avg_latency is not None else None,
            "p95_latency_ms": float(p95_latency) if p95_latency is not None else None,
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
    except Exception:
        return {"available": False}
