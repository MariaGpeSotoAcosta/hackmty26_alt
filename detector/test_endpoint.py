from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.request

from detector.paths import audio_path


def main() -> None:
    parser = argparse.ArgumentParser(description="POST /detect with a local WAV")
    parser.add_argument("anon_id")
    parser.add_argument("--url", default="http://127.0.0.1:8000/detect")
    args = parser.parse_args()

    wav = audio_path(args.anon_id)
    if not wav.exists():
        raise SystemExit(f"missing {wav}")
    payload = {"audio_base64": base64.b64encode(wav.read_bytes()).decode("ascii")}
    req = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            print(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach {args.url}. Start: python -m uvicorn detector.app:app --port 8000\n{exc}") from exc


if __name__ == "__main__":
    main()
