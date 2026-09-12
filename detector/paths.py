from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "manifest.csv"
TURNS_DIR = ROOT / "turns"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "dialogue_model.joblib"
FEATURES_CSV = MODELS_DIR / "dialogue_features.csv"
TRANSCRIPTS_DIR = MODELS_DIR / "transcripts"

_AUDIO_CANDIDATES = (
    ROOT / "audio",
    ROOT.parent / "altur-challenge-audio" / "audio",
)


def audio_dir() -> Path:
    for path in _AUDIO_CANDIDATES:
        if path.is_dir():
            return path
    return _AUDIO_CANDIDATES[0]


def audio_path(anon_id: str) -> Path:
    return audio_dir() / f"{anon_id}.wav"


def turns_path(anon_id: str) -> Path:
    return TURNS_DIR / f"{anon_id}.json"


def transcript_path(anon_id: str) -> Path:
    return TRANSCRIPTS_DIR / f"{anon_id}.json"


def vosk_model_dir() -> Path | None:
    if not MODELS_DIR.is_dir():
        return None
    matches = sorted(MODELS_DIR.glob("vosk-model*"))
    for path in matches:
        if path.is_dir() and ((path / "am").exists() or (path / "conf").exists()):
            return path
    return matches[0] if matches else None
