# Análisis del detector y plan de mejoras

Resumen de cómo funciona el sistema actual (`detector/`), qué tan confiables son sus números, y qué se cambió. Todo lo de este documento se corrió y verificó, no es teoría. Estado: puntos 1, 2 y 5 implementados; punto 3 (semántica) en evaluación; punto 4 (acústica) evaluado y **descartado con evidencia**.

**Actualización — se integró con `origin/main` (commits de la compañera `9f0b7d1`/`fa144b8`)**: agregó dual-view training (entrena con VAD + turnos oficiales del mismo train, val sigue solo VAD) y early-exit a 90s. También encontró y apagó el desempate acústico por su cuenta, **coincidiendo de forma independiente** con lo que `eval_generalization.py` ya había demostrado aquí. Combinando su dual-view con el VAD tuneado por canal (sección 2): **94.4% val** (4 errores), mejor que el 93.0% que ella tenía sola con el VAD viejo. El desempate acústico se reactivó automáticamente por un empate exacto (0.944 vs 0.944) en su lógica de entrenamiento; se desactivó a mano porque un empate no es mejora y `eval_generalization.py` ya mostró que ese desempate cuesta recall en voces ocultas (0.890 vs 0.907 solo-diálogo) — se corrió de nuevo tras el merge para confirmar que el número no cambió.

---

## 1. Cómo funciona hoy

```
WAV estéreo 8 kHz (canal 0 = caller, canal 1 = agente)
    → VAD por energía (20 ms/frame, umbral por canal) reconstruye los turnos de habla
    → 21 features de diálogo (timing entre caller y agente)
    → regresión logística calibrada
    → {"is_synthetic": true, "confidence": 0.87}
```

El sistema **no usa el timbre de la voz**. Usa *cómo* conversa el caller: cuánto tarda en responder, si interrumpe, si se queda callado, qué tan regulares son sus turnos. Esto es deliberado: el set de jueces trae voces/motores TTS nuevos, y un clasificador de timbre no tiene por qué generalizar a eso (ver sección 4).

**Accuracy actual: 91.5% en val** (65/71 llamadas), tras el ajuste de VAD de la sección 2. Reproducido con `python -m detector.train --from-wav`.

### Las 21 features (todas vienen de los turnos, no del audio en sí)

| Grupo | Features |
|---|---|
| Cuántos turnos y qué tan largos | `n_caller`, `n_agent`, `turn_caller_mean/std/cv` |
| Latencia de respuesta | `lat_mean`, `lat_med`, `lat_p90`, `lat_std`, `first_latency` |
| Interrupciones | `barge_in`, `barge_rate`, `agent_barge` |
| Habla simultánea | `overlap_s`, `overlap_rate` |
| Silencios | `silence_fill`, `silence_fill_rate`, `caller_gap_mean` |
| Proporciones | `caller_speech_ratio`, `agent_speech_ratio`, `duration_s` |

Las que más pesan en la decisión: `agent_speech_ratio` (↓ = más sintético), `lat_med`/`lat_mean` (↑ = más sintético — el bot ASR+LLM tarda más en responder), `turn_caller_mean` (turnos largos = más sintético), `lat_std` (↓ = más sintético — los humanos son más irregulares).

---

## 2. VAD tuneado por canal — IMPLEMENTADO

El servidor **nunca recibe** el JSON de turnos oficial — solo audio. El VAD (`detector/vad.py`) tiene que reconstruir los turnos por su cuenta, y ese paso mete error.

**Antes de tocar nada**, validación contra `turns/*.json` (IoU, intersección sobre unión) con los parámetros originales (un solo umbral para ambos canales):

| Canal | IoU media (antes) |
|---|---|
| Caller | 0.895 |
| Agente | 0.962 |

El VAD medía peor al caller — tiene más silencios/vacilación que confunden al detector de energía por umbral único.

**Cambio hecho**: grid search de `rel_k` (umbral relativo al ruido) y `hangover_frames` **por canal**, optimizando IoU contra los turnos reales:

```python
# detector/vad.py
CHANNEL_REL_K = {0: 6.0, 1: 8.0}          # antes: 4.5 para ambos
CHANNEL_HANGOVER_FRAMES = {0: 2, 1: 2}     # antes: 4 para ambos
```

**Resultado, medido de nuevo con `python -m detector.eval_vad`**:

| Canal | IoU media (antes → después) |
|---|---|
| Caller | 0.895 → **0.918** |
| Agente | 0.962 → **0.981** |

**Impacto en el modelo real** (reentrenado con `--from-wav` para no desajustar train/serve): **88.7% → 91.5% val** (63/71 → 65/71). Confirmado también en `/health` del servidor corriendo.

---

## 3. Auditoría de las llamadas que el modelo aún falla — HECHO

Con el VAD mejorado, `val_errors.json` bajó de 8 a **6** llamadas. Hipótesis antes de medir: "son las llamadas donde el VAD mide peor los turnos." Se cruzó el IoU individual contra el resto de val — **no se cumplió**: el IoU de las que fallan es prácticamente igual al de las que acierta. El VAD no es la causa de estos 6 errores puntuales.

Se comparó cada llamada fallida contra la mediana de features de **su propia clase real**, y sí apareció un patrón claro:

| Llamadas humanas que el modelo marca como sintéticas | Por qué |
|---|---|
| `call_569ffb0869eb`, `call_678ee1dd2242`, `call_6971b2685c1d` | Latencia de respuesta muy por encima de lo normal en humanos (2-3× la mediana) y casi no interrumpen (`barge_in` 0-1 vs. mediana humana de 4) — son personas atípicamente pacientes/formales, se comportan como el patrón que el modelo asocia a bot. |

| Llamadas sintéticas que el modelo marca como humanas | Por qué |
|---|---|
| `call_6058d9c5c82f`, `call_a3a1f51a4bf9`, `call_e3f555297277` | Latencia por debajo de lo normal en sintéticos (más rápida que la mediana, cercana al rango humano) — son bots con un pipeline ASR+LLM inusualmente rápido. |

**Conclusión**: son casos de comportamiento genuinamente ambiguo dentro de su propia clase, no ruido de medición ni error del VAD. No hay una corrección barata aquí — subir esto requeriría una señal adicional (semántica o acústica) que capture algo más que timing.

---

## 4. Clasificador acústico (MFCC/timbre) — EVALUADO Y DESCARTADO

### 4.1 Motivo de la duda inicial
Se probó un modelo alterno (RandomForest/LogReg sobre 84 features MFCC + delta + espectrales, sin turnos, directo del audio) que llegó a **100% en un val split aleatorio**.

Se investigó si era un atajo trivial (ej. 2-3 voces TTS reusadas) y **no se encontró** ese mecanismo: diversidad interna similar entre clases (ratio 1.13x, sin cuasi-duplicados), y las features más importantes (`zcr_std`, `flatness_std` — variabilidad espectral) son consistentes con literatura real de anti-spoofing, no ruido.

### 4.2 El problema real: dos niveles de "nunca visto"
El reto distingue explícitamente:

> *Splits*: train/val son speaker-disjoint (ningún **caller** se repite). El set de jueces trae **callers y voces** que no aparecen en ninguno de los dos splits.

Train y val pueden compartir la misma voz/motor TTS con distintos callers encima. Un split aleatorio (o k-fold normal) no prueba generalización a una **voz nueva**, solo a un **caller nuevo** con una voz ya vista.

### 4.3 Herramienta permanente: `detector/eval_generalization.py` — NUEVO
Se agregó al paquete `detector/` un validador reutilizable: agrupa las llamadas sintéticas por similitud acústica (proxy de "motor/voz") y deja cada cluster completo fuera del entrenamiento (leave-one-voice-cluster-out), en vez de un split aleatorio. Corre en segundos con `python -m detector.eval_generalization`.

### 4.4 Resultado: tres configuraciones probadas bajo esa validación más dura

| Configuración | Accuracy media | Recall sintético medio | Peor cluster |
|---|---|---|---|
| **Solo diálogo (producción actual)** | **0.887** | **0.907** | 0.800 |
| Diálogo + acústica mezcladas siempre | 0.829 | 0.707 | 0.313 |
| Diálogo + acústica solo como desempate (confianza 0.50–0.65) | 0.880 | 0.890 | — |

**Hallazgo importante que no se sabía antes de hoy**: el modelo de diálogo actual, solo con las 21 features de comportamiento, **ya es notablemente más robusto** a voces nunca vistas que cualquier variante que incluya acústica — incluso la versión "solo como desempate" queda ligeramente peor, no mejor. Mezclar acústica sin condición cae en picada (recall 0.313 en el peor cluster).

**Decisión: no se integra acústica a producción.** La recomendación original (sección 5, punto 4 de la versión previa de este documento) queda descartada con evidencia, no solo por precaución teórica.

---

## 5. Semántica (Vosk ASR) — en evaluación

Se instaló `vosk` y el modelo `vosk-model-small-es-0.42`, y se transcribió el dataset completo (`python -m detector.transcribe_dataset`, cachea en `models/transcripts/`). Calidad de transcripción razonable en español a 8 kHz para este propósito (no perfecta, pero capta contenido).

Pendiente de reportar en este documento: correr `python -m detector.train --from-wav --with-semantic` y validar con `detector/eval_generalization.py` (agregando semántica al set de features evaluado) antes de decidir si sube a producción — la misma regla que se le aplicó a la acústica.

---

## 6. Reglas para cualquier feature nueva de aquí en adelante

1. Un val accuracy alto en split aleatorio **no es suficiente evidencia** de que generaliza a voces/motores nuevos — el reto promete traer eso al set de jueces.
2. Antes de subir cualquier feature a producción, correr `python -m detector.eval_generalization` (o extenderlo para incluir la nueva feature) y comparar accuracy/recall contra la producción actual bajo leave-one-voice-cluster-out, no solo contra el val normal.
3. Si la nueva feature no mejora bajo esa prueba más dura, no entra — aunque el val aleatorio se vea bien (ver sección 4).

---

## Cómo reproducir estos análisis

```powershell
python -m detector.train --from-wav          # reentrena y confirma 91.5% val
python -m detector.eval_vad                   # IoU del VAD vs turns oficiales (0.918 / 0.981)
python -m detector.eval_generalization        # leave-one-voice-cluster-out: dialogo vs acustica
python -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
python -m detector.test_endpoint call_0e1e2f29bfdc   # esperado: human
python -m detector.test_endpoint call_4d8129939686   # esperado: synthetic
```
