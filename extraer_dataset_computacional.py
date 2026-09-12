import pandas as pd
import json
import numpy as np
import time

def extraer_features_conv(turns_path, duracion_total):
    with open(turns_path) as f:
        data = json.load(f)

    turns = sorted(data["turns"], key=lambda t: t["start"])
    turnos_caller = [t for t in turns if t["channel"] == 0]

    n_turnos_caller = len(turnos_caller)
    duraciones_caller = [t["end"] - t["start"] for t in turnos_caller]
    dur_mean = np.mean(duraciones_caller) if duraciones_caller else 0
    dur_std = np.std(duraciones_caller) if len(duraciones_caller) > 1 else 0

    latencias = []
    interrupciones_caller = 0
    interrupciones_agente = 0

    for i in range(len(turns) - 1):
        actual = turns[i]
        siguiente = turns[i + 1]
        if actual["channel"] != siguiente["channel"]:
            gap = siguiente["start"] - actual["end"]
            if gap >= 0:
                if actual["channel"] == 1 and siguiente["channel"] == 0:
                    latencias.append(gap)
            else:
                if siguiente["channel"] == 0:
                    interrupciones_caller += 1
                else:
                    interrupciones_agente += 1

    latencia_mean = np.mean(latencias) if latencias else 0
    latencia_std = np.std(latencias) if len(latencias) > 1 else 0
    tiempo_hablado_total = sum(t["end"] - t["start"] for t in turns)
    ratio_silencio = 1 - (tiempo_hablado_total / duracion_total) if duracion_total > 0 else 0

    return np.array([
        n_turnos_caller, dur_mean, dur_std,
        latencia_mean, latencia_std,
        interrupciones_caller, interrupciones_agente,
        ratio_silencio
    ])

df = pd.read_csv("manifest.csv")

X_conv_train, X_conv_val = [], []
inicio = time.time()

for i, row in df.iterrows():
    turns_path = f"turns/{row['anon_id']}.json"
    try:
        features = extraer_features_conv(turns_path, row["duration_s"])
    except Exception as e:
        print(f"Error con {row['anon_id']}: {e}")
        features = np.zeros(8)  # fallback si algun archivo falla, para no romper el orden

    if row["split"] == "train":
        X_conv_train.append(features)
    else:
        X_conv_val.append(features)

    if (i + 1) % 50 == 0:
        print(f"Procesadas {i+1}/{len(df)} ({time.time()-inicio:.1f}s)")

X_conv_train = np.array(X_conv_train)
X_conv_val = np.array(X_conv_val)

print("X_conv_train:", X_conv_train.shape, "X_conv_val:", X_conv_val.shape)

np.save("X_conv_train.npy", X_conv_train)
np.save("X_conv_val.npy", X_conv_val)
print("Guardado")