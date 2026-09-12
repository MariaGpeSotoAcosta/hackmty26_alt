import json
import numpy as np

def extraer_features_conv(turns_path, duracion_total):
    with open(turns_path) as f:
        data = json.load(f)

    turns = sorted(data["turns"], key=lambda t: t["start"])

    turnos_caller = [t for t in turns if t["channel"] == 0]
    turnos_agente = [t for t in turns if t["channel"] == 1]

    # 1. Numero de turnos
    n_turnos_caller = len(turnos_caller)

    # 2. Duracion de los turnos del caller
    duraciones_caller = [t["end"] - t["start"] for t in turnos_caller]
    dur_mean = np.mean(duraciones_caller) if duraciones_caller else 0
    dur_std = np.std(duraciones_caller) if len(duraciones_caller) > 1 else 0

    # 3. Latencia de respuesta: tiempo entre que el agente termina y el caller empieza
    #    (usamos turns ordenados cronologicamente para encontrar pares consecutivos)
    latencias = []
    interrupciones_caller = 0  # caller empieza a hablar ANTES de que el agente termine
    interrupciones_agente = 0  # agente empieza a hablar ANTES de que el caller termine

    for i in range(len(turns) - 1):
        actual = turns[i]
        siguiente = turns[i + 1]

        if actual["channel"] != siguiente["channel"]:  # cambio de hablante
            gap = siguiente["start"] - actual["end"]

            if gap >= 0:
                # Solo contamos como "latencia de respuesta" cuando el caller responde al agente
                if actual["channel"] == 1 and siguiente["channel"] == 0:
                    latencias.append(gap)
            else:
                # gap negativo = el siguiente empezo antes de que el actual terminara = interrupcion/overlap
                if siguiente["channel"] == 0:
                    interrupciones_caller += 1
                else:
                    interrupciones_agente += 1

    latencia_mean = np.mean(latencias) if latencias else 0
    latencia_std = np.std(latencias) if len(latencias) > 1 else 0

    # 4. Ratio de silencio: tiempo total hablado vs duracion total de la llamada
    tiempo_hablado_total = sum(t["end"] - t["start"] for t in turns)
    ratio_silencio = 1 - (tiempo_hablado_total / duracion_total) if duracion_total > 0 else 0

    return np.array([
        n_turnos_caller,
        dur_mean, dur_std,
        latencia_mean, latencia_std,
        interrupciones_caller,
        interrupciones_agente,
        ratio_silencio
    ])

# --- Prueba con tu audio real ---
features = extraer_features_conv("turns/call_092aef8d1243.json", duracion_total=170.22)
nombres = ["n_turnos_caller", "dur_mean", "dur_std", "latencia_mean", "latencia_std",
           "interrupciones_caller", "interrupciones_agente", "ratio_silencio"]

for nombre, valor in zip(nombres, features):
    print(f"{nombre}: {valor:.3f}")