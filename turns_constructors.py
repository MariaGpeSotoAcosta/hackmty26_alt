import librosa
import numpy as np

def detectar_turnos_desde_audio(canal_caller, canal_agente, sr, top_db=30, merge_gap=0.5):
    """
    merge_gap: si dos segmentos del MISMO canal estan separados por menos
    de esta cantidad de segundos, se fusionan en un solo turno.
    """
    def procesar_canal(audio_canal, channel_id):
        intervalos = librosa.effects.split(audio_canal, top_db=top_db)
        if len(intervalos) == 0:
            return []

        # Convertir a segundos
        segmentos = [(start / sr, end / sr) for start, end in intervalos]

        # Fusionar segmentos cercanos
        fusionados = [segmentos[0]]
        for start, end in segmentos[1:]:
            ultimo_inicio, ultimo_fin = fusionados[-1]
            if start - ultimo_fin <= merge_gap:
                # Muy cerca del anterior -> extender el turno anterior
                fusionados[-1] = (ultimo_inicio, end)
            else:
                # Suficientemente separado -> es un turno nuevo
                fusionados.append((start, end))

        return [{"channel": channel_id, "start": s, "end": e} for s, e in fusionados]

    turns = procesar_canal(canal_caller, 0) + procesar_canal(canal_agente, 1)
    turns.sort(key=lambda t: t["start"])
    return turns