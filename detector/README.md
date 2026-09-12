# Detector Altur — implementación

Clasifica si el **caller** (canal 0) de una llamada bancaria es **humano** o **sintético** (ASR + LLM + voz). El canal 1 es el agente del banco.

El set de los jueces es speaker-disjoint y puede traer voces/engines nuevos. Por eso el núcleo **no** es un clasificador de timbre: es **cómo conversa** el caller (latencia, interrupciones, silencio, regularidad de turnos). El PDF del reto pide originalidad más allá de un clasificador de audio de estantería, y profundidad en una señal bien hecha.

---

## Estado actual

Producción: **VAD sobre el WAV → 21 features de diálogo → logística calibrada**.

| Pipeline | Train | Val (71 callers nuevos) |
| --- | --- | --- |
| **Logística (lo que sirve `/detect`)** | 90.8% | **88.7%** (33/37 human, 30/34 synth), AUC 0.975 |
| HistGradientBoosting (descartado) | 99.6% | 90.1% — memoriza, no va al hidden set |

`confidence` es `max(p, 1-p)`: qué tan seguro está del veredicto, no un sello de que acertó. La probabilidad viene de Platt (CV en train).

Errores actuales en val (8): `models/val_errors.json`.

---

## Cómo funciona

```
WAV estéreo 8 kHz
    → separar canal 0 (caller) y canal 1 (agente)
    → VAD por energía (20 ms), un turno = habla continua ≥ 0.2 s
    → features de diálogo (timing entre ambos)
    → logística
    → {"is_synthetic": true, "confidence": 0.87}
```

El juez **no manda** `turns/*.json`. Esos JSON solo sirven en desarrollo. En inferencia reconstruimos los turnos con VAD. El modelo de producción se entrenó **con ese mismo VAD** (`--from-wav`) para no tener desajuste train/serve.

IoU del VAD vs turns oficiales: ~0.90 en caller, ~0.96 en agente.

Señales que más empujan a *synthetic*: latencia mediana/media alta, turnos del caller más largos. A *human*: latencia irregular (`lat_std`) y más habla del agente. En el dataset, humanas contestan ~2 s e interrumpen más; sintéticos esperan ~3–5 s (ASR+LLM) y son más regulares.

Semántica (Vosk) y acústica ligera **existen en código** pero **no** entran a `/detect` hasta que val suba. El endpoint no corre ASR.

Si el WAV no se puede leer: `is_synthetic: true`, `confidence: 0.52` (nunca 500).

---

## Archivos nuevos

Todo el sistema vive en `detector/` más artefactos en `models/`.

### Paquete `detector/`

| Archivo | Qué hace |
| --- | --- |
| `paths.py` | Rutas: `manifest.csv`, `turns/`, `audio/`, `models/`. Busca el audio también en un zip hermano si no está en la raíz. |
| `io_wav.py` | Lee WAV 8 kHz estéreo desde path, bytes o base64 (`RIFF` o JSON). Normaliza a float32, 2 canales. |
| `vad.py` | VAD por energía, 20 ms, hangover, merge 0.3 s, mínimo 0.2 s. `turns_from_audio` arma la lista `{channel, start, end}`. |
| `features.py` | Features de diálogo + `extract_all_features` (une semántica/acústica si se piden). `features_from_audio` = VAD + features. |
| `acoustic.py` | Features baratas solo del caller voiced: ZCR, flatness, centroide, CV de RMS, std de F0. **No está en el modelo actual.** |
| `semantic.py` | Features de texto: muletillas, rechazos (“no tengo eso”), formalidad tipo LLM, trampas del agente, dígitos. **No está en el modelo actual.** |
| `transcribe.py` | ASR por canal con **Vosk** (español). Whisper se cae en Python 3.14 / Windows. |
| `transcribe_dataset.py` | Recorre el manifest, transcribe y cachea `models/transcripts/<anon_id>.json`. |
| `train.py` | Extrae features en **train**, mide en **val**. Flags: `--from-wav`, `--with-semantic`, `--with-acoustic`. Elige logística salvo que un árbol gane ≥3 pts en val **y** no overfittee. Guarda el `.joblib` y `val_errors.json`. |
| `predict.py` | Carga el bundle y clasifica desde WAV o desde turns oficiales (`--turns`, solo debug). |
| `app.py` | FastAPI: `POST /detect` (jueces), `POST /detect/upload` (Swagger), `GET /health`. |
| `test_endpoint.py` | Prueba `/detect` sin pegar base64: lee el WAV local y lo manda como JSON. |
| `eval_vad.py` | IoU del VAD vs turns oficiales + accuracy en val con el modelo guardado. |
| `__init__.py` | Marca el directorio como paquete. |
| `README.md` | Este documento. |

### Artefactos en `models/`

| Archivo | Qué es |
| --- | --- |
| `dialogue_model.joblib` | Modelo de producción (logística + nombres de features + flags). |
| `dialogue_features.csv` | Una fila por llamada, para inspeccionar. |
| `val_errors.json` | Las 8 llamadas de val que hoy falla. |
| `transcripts/` | Cache de ASR (se genera; no commitear). |
| `vosk-model-small-es-0.42/` | Modelo ASR (se descarga; no commitear). |

`requirements.txt` lista dependencias. `.gitignore` excluye `audio/`, zips, Vosk, transcripciones y `__pycache__`.

---

## Features de diálogo (las que sí usa el modelo)

Del timing caller (0) vs agente (1):

- `barge_in` / `barge_rate` — el caller empieza mientras el agente habla
- `overlap_s` / `overlap_rate` — segundos a la vez
- `lat_mean`, `lat_med`, `lat_p90`, `lat_std`, `first_latency` — demora en contestar
- `n_caller`, `n_agent`, `turn_caller_mean` / `std` / `cv` — cuántos turnos y qué tan largos/regulares
- `silence_fill` / `silence_fill_rate` — el caller vuelve a hablar si el agente no responde
- `caller_speech_ratio`, `agent_speech_ratio`, `caller_gap_mean`, `duration_s`

---

## Cómo correrlo

```powershell
python -m pip install -r requirements.txt
python -m detector.train --from-wav
python -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

Si 8000 está ocupado, usa `--port 8001`. Confirma que `/health` traiga `"model": "dialogue_model.joblib"` y `"val_accuracy": 0.887…`. Si solo ves `{"status":"ok"}`, es **otro** proceso en ese puerto.

### API (lo que evalúan)

| Ruta | Para quién |
| --- | --- |
| `POST /detect` | Jueces. JSON `{"audio_base64":"<wav>"}`, o el WAV crudo (`RIFF...`). También acepta campos `audio`, `wav`, `clip`. |
| `POST /detect/upload` | Nosotros. En [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (o `:8001/docs`): **Try it out → Choose File**. Un WAV de 5 MB no se pega en `/detect`. |
| `GET /health` | ¿Está vivo el modelo bueno? |

Respuesta:

```json
{"is_synthetic": true, "confidence": 0.87}
```

Probar el **mismo** `POST /detect` de los jueces, sin Swagger:

```powershell
python -m detector.test_endpoint call_0e1e2f29bfdc
python -m detector.test_endpoint call_4d8129939686
```

| ID | Manifest | Esperado |
| --- | --- | --- |
| `call_0e1e2f29bfdc` | human, val | `is_synthetic: false` (~0.93) |
| `call_4d8129939686` | synthetic, train | `is_synthetic: true` (~0.95) |

Clasificar sin HTTP: `python -m detector.predict call_0e1e2f29bfdc --wav`.

### Variantes (no son producción hasta que val mejore)

```powershell
python -m detector.eval_vad
python -m detector.transcribe_dataset
python -m detector.train --from-wav --with-semantic
python -m detector.train --from-wav --with-acoustic
```

Vosk (solo semántica), si falta:

```powershell
curl -L -o models/vosk-model-small-es-0.42.zip https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
tar -xf models/vosk-model-small-es-0.42.zip -C models
```

---

## Train vs val vs hidden

- **Train** (282): aprende. No decide si el modelo es bueno.
- **Val** (71): callers distintos. Métrica honesta de ahora.
- **Hidden test**: voces y personas nuevas. Por eso no hay speaker-ID ni un árbol al 100% train.

---

## Qué falta

1. **Semántica en producción.** El código está (`semantic.py`, `transcribe.py`). Falta transcribir el dataset, leer a mano si Vosk small-es entiende “no tengo eso” en 8 kHz, y subir `--with-semantic` **solo si val gana ≥2 puntos**. Si no, se queda para la demo (“aquí el humano se niega; el bot inventa”).
2. **Acústica como desempate, no como rama.** `acoustic.py` existe. Meterla en *todas* las llamadas puede pelear con el diálogo. El plan es usarla solo si `confidence` está en ~0.50–0.65.
3. **Early-exit / latencia.** Criterio de los jueces. Hoy se procesa la llamada entera (~150 s). Falta decidir a los 30–45 s si la confianza ya es alta.
4. **Auditoría de las 8 de val.** ¿VAD roto o el clasificador? Si varias son umbral de VAD, se gana más ahí que sumando features.
5. **Un solo servidor.** En esta máquina el 8000 a veces es otra API. Para el juicio: un uvicorn, `/health` con `dialogue_model.joblib`.
6. **Demo de 15 min.** Lado a lado: humana que pisa al agente / sintética que espera 4 s / pregunta fantasma. El benchmark corre el número; nosotros vendemos la señal.

