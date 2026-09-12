from __future__ import annotations

import argparse

import pandas as pd

from detector.io_wav import load_wav_path
from detector.paths import MANIFEST, audio_path, transcript_path
from detector.transcribe import load_transcript, save_transcript, transcribe_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache Vosk transcripts under models/transcripts/")
    parser.add_argument("--ids", nargs="*", help="Only these anon_id values")
    parser.add_argument("--print-text", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(MANIFEST)
    if args.ids:
        df = df[df["anon_id"].isin(args.ids)]

    for i, row in df.iterrows():
        anon_id = row["anon_id"]
        if transcript_path(anon_id).exists() and not args.force:
            payload = load_transcript(anon_id)
            print(f"[skip] {anon_id}")
        else:
            wav = audio_path(anon_id)
            if not wav.exists():
                print(f"[miss] {anon_id}")
                continue
            audio, sr = load_wav_path(wav)
            payload = transcribe_audio(audio, sr)
            save_transcript(anon_id, payload)
            print(f"[ok]   {anon_id}")
        if args.print_text and payload:
            print("  caller:", (payload.get("caller") or "")[:240])
            print("  agent: ", (payload.get("agent") or "")[:240])


if __name__ == "__main__":
    main()
