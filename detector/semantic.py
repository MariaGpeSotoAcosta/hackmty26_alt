from __future__ import annotations

import re

SEMANTIC_FEATURES = [
    "filler_rate",
    "casual_count",
    "refusal_count",
    "trap_then_refuse",
    "formal_count",
    "trap_then_invent",
    "trap_q_count",
    "repeat_prompt_count",
    "digit_token_count",
    "caller_chars",
]

_FILLERS = (
    "este",
    "eee",
    "eeeeste",
    "mmm",
    "em",
    "eh",
    "pues",
    "bueno",
    "o sea",
    "la verdad",
)
_CASUAL = (
    "oye",
    "qué tal",
    "que tal",
    "hola",
    "ajá",
    "aja",
    "okey",
    "okay",
    "a ver",
    "no manches",
    "órale",
    "orale",
)
_REFUSALS = (
    "no tengo",
    "no se",
    "no sé",
    "no tengo eso",
    "no me acuerdo",
    "no cuento",
    "no existe",
    "no entiendo",
    "no me suena",
    "no sé de qué",
    "no se de que",
    "cuál es eso",
    "cual es eso",
    "no manejo",
    "eso no",
)
_FORMAL = (
    "por supuesto",
    "correcto",
    "así es",
    "asi es",
    "con gusto",
    "claro que sí",
    "claro que si",
    "el número es",
    "el numero es",
    "mi número",
    "mi numero",
    "por supuesto que",
    "desde luego",
    "afirmativo",
)
_TRAPS = (
    "folio",
    "clabe",
    "nip de",
    "token",
    "código de verificación que no",
    "codigo de verificacion que no",
    "sucursal que no",
    "número de contrato interno",
    "numero de contrato interno",
    "clave interbancaria",
    "referencia inexistente",
)
_REPEATS = (
    "repít",
    "repit",
    "otra vez",
    "de nuevo",
    "confirme",
    "confirma",
    "vuelva a",
    "me lo puede repetir",
    "dígamelo otra",
    "digamelo otra",
)

_DIGIT_RE = re.compile(r"\b\d{2,}\b")


def empty_semantic_features() -> dict[str, float]:
    return {name: 0.0 for name in SEMANTIC_FEATURES}


def extract_semantic_features(transcript: dict | None) -> dict[str, float]:
    if not transcript:
        return empty_semantic_features()
    caller = _norm(transcript.get("caller") or transcript.get("channel_0") or "")
    agent = _norm(transcript.get("agent") or transcript.get("channel_1") or "")
    if not caller and not agent:
        return empty_semantic_features()

    words = max(len(caller.split()), 1)
    filler = _count(caller, _FILLERS)
    casual = _count(caller, _CASUAL)
    refuse = _count(caller, _REFUSALS)
    formal = _count(caller, _FORMAL)
    traps = _count(agent, _TRAPS)
    repeats = _count(agent, _REPEATS)
    digits = len(_DIGIT_RE.findall(caller))
    trap_then_refuse = 1.0 if traps and refuse else 0.0
    trap_then_invent = 1.0 if traps and digits >= 2 and not refuse else 0.0
    return {
        "filler_rate": filler / words,
        "casual_count": float(casual),
        "refusal_count": float(refuse),
        "trap_then_refuse": trap_then_refuse,
        "formal_count": float(formal),
        "trap_then_invent": trap_then_invent,
        "trap_q_count": float(traps),
        "repeat_prompt_count": float(repeats),
        "digit_token_count": float(digits),
        "caller_chars": float(len(caller)),
    }


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").split())


def _count(text: str, phrases: tuple[str, ...]) -> int:
    return sum(text.count(_norm(p)) for p in phrases)
