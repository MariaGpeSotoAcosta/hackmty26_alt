import soundfile as sf
import json

from turns_constructors import detectar_turnos_desde_audio

anon_id = "call_0181ce113ebe"
audio, sr = sf.read(f"audio/{anon_id}.wav")
canal_caller = audio[:, 0]
canal_agente = audio[:, 1]

with open(f"turns/{anon_id}.json") as f:
    turns_reales = json.load(f)["turns"]

print(f"Turnos reales: {len(turns_reales)}\n")

for gap in [0.3, 0.5, 0.8, 1.0, 1.5]:
    turnos_detectados = detectar_turnos_desde_audio(canal_caller, canal_agente, sr, merge_gap=gap)
    print(f"merge_gap={gap}: {len(turnos_detectados)} turnos detectados")