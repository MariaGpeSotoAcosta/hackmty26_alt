# Synthetic Caller Detector

Detects whether the **caller** on a Mexican-Spanish bank line is a person or an autonomous ASR → LLM → TTS stack.

HackMTY 2026 — Altur Challenge.

```json
POST /detect  →  {"is_synthetic": true, "confidence": 0.87}
```

**Live dashboard:** [http://45.32.199.144:8000/dashboard](http://45.32.199.144:8000/dashboard)

## Problem

Recorded calls sit between a caller (channel 0) and the bank’s agent (channel 1). Some callers are people. Others are synthetic systems that listen, generate a reply, and speak it back on the same number.

The task is a single WAV in, a boolean out: is channel 0 synthetic?

An acoustic classifier can latch onto the fingerprint of one voice engine. Hidden evaluation uses **new speakers on the same engine**, so a spectral shortcut does not travel. The detector has to score **how the caller converses**, not what they sound like.

## Core idea

A synthetic caller is a conversational system, not a voice.

It hears, thinks, and speaks **in series**. ASR, the language model, and TTS each add time. That shows up on the clock of the dialogue: longer waits after the agent finishes, paragraph-length turns, and a metronomic regularity even when the pipeline is fast (`lat_cv`).

Those patterns do not depend on which TTS produced channel 0.

## Approach

```
Stereo 8 kHz WAV  (ch0 = caller, ch1 = agent)
    → decode to float32
    → per-channel energy VAD
    → 22 dialogue-timing features
    → standardized logistic regression + Platt calibration
    → is_synthetic = (p ≥ 0.5)
    → confidence = max(p, 1 − p)
```

| Stage | What it does |
| --- | --- |
| **VAD** | Independent energy segmentation per channel (RMS, 20 ms). Adaptive threshold vs. estimated noise. Caller `rel_k` 6.0, agent `rel_k` 8.0 — the caller’s level varies more. After VAD the waveform is discarded. |
| **Features** | 22 timings: conversation size, response latency (including `lat_cv`), barge-in / overlap, silence handling. Definitions and weights: [`docs/FEATURES.md`](docs/FEATURES.md). |
| **Classifier** | `LogisticRegression` (`C=0.4`, balanced classes) on standardized inputs, Platt-calibrated (sigmoid, 3 folds). Linear boundary over timings — not a memorization of callers. |
| **Dual-view train** | Each **train** call is seen twice with the same label: VAD on the WAV, and official `turns/` intervals. Val and `/detect` use **VAD only**. |

`/detect` always returns HTTP 200. Unreadable audio gets a weak prior `{true, 0.52}` so a malformed body never becomes a 500.

## Results

Official judge client (`scripts/check_endpoint.py`) against the deployed `/detect`, **full val split** (71 calls, speaker-disjoint from train):

| Metric | Value |
| --- | --- |
| Answered / format errors | 71 / 0 |
| Accuracy | 0.944 (67/71) |
| Balanced accuracy | 0.945 |
| TPR (synthetic) | 0.971 |
| TNR (human) | 0.919 |
| AUC | 0.986 |
| Brier score | 0.053 |
| Mean end-to-end latency | 0.94 s |
| Max end-to-end latency | 2.0 s |

End-to-end latency is full round-trip to the public endpoint (network + decode + inference). Feature extraction + logistic regression on a full-length call is roughly 60–100 ms. Judge timeout is 30 s.

The operating point (`p ≥ 0.5`) favors recall on synthetic callers: a missed bot costs more than extra friction on a patient human.

## How it works

1. Decode the stereo WAV to two `float32` channels in \([-1, 1]\) (`detector/io_wav.py`).
2. Segment each channel into speech intervals (`detector/vad.py`).
3. From caller vs. agent geometry, compute the 22 features (`detector/features.py`).
4. Score with `models/dialogue_model.joblib` (`detector/predict.py`).
5. FastAPI serves `/detect`, `/health`, and `/dashboard` (`detector/app.py`). Optional Timescale logging: [`deploy/tiger-data.md`](deploy/tiger-data.md).

## Run it

Python 3.9+. Unzip `altur-challenge-audio.zip` into `audio/` at the repo root.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The trained model ships in `models/dialogue_model.joblib`. To retrain:

```bash
python -m detector.train --from-wav --dual-view
```

Serve:

```bash
python -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

For a public slot use `--host 0.0.0.0`, or the Vultr systemd unit in [`deploy/README.md`](deploy/README.md).

```bash
python -m detector.test_endpoint call_0e1e2f29bfdc
python scripts/check_endpoint.py --url http://localhost:8000/detect --split val --n 20
```

## API

Judge request:

```json
{"call_id": "...", "audio_base64": "<full WAV file, base64>", "sample_rate": 8000, "channels": 2}
```

Also accepted: multipart `file`, or a raw `RIFF` body (up to ~5 MB).

Response, always HTTP 200:

```json
{"is_synthetic": true, "confidence": 0.87}
```

`is_synthetic` is a required JSON boolean. `confidence` ∈ [0, 1] is optional for the judge; it is always sent (AUC, calibration, ties). A timeout, a non-200, or a non-boolean `is_synthetic` counts as wrong.

| Route | Purpose |
| --- | --- |
| `POST /detect` | Verdict |
| `GET /health` | Model loaded, 22 feature names, val accuracy |
| `GET /dashboard` | Live verdicts — [http://45.32.199.144:8000/dashboard](http://45.32.199.144:8000/dashboard) |

## Dataset

| Path | Contents |
| --- | --- |
| `manifest.csv` | `anon_id`, `label` (`human` / `synthetic`), `split`, `duration_s` |
| `audio/<id>.wav` | Stereo 8 kHz PCM. Ch0 = caller, ch1 = agent. Not in git. |
| `turns/<id>.json` | Official speech intervals. Train only — never sent by the judge, never read at inference. |

Human callers volunteered, used invented personal data, and knew the call was recorded. HackMTY 2026 only; do not redistribute.

## Repository

```
detector/     Production package: VAD, features, train, /detect, dashboard
docs/         Feature definitions and logistic weights
deploy/       Vultr + systemd
models/       dialogue_model.joblib
scripts/      Official judge client
```
