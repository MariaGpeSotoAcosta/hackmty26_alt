# Detector Altur — implementación

Clasifica si el **caller** (canal 0) de una llamada bancaria es **humano** o **sintético** (ASR + LLM + voz). El canal 1 es el agente del banco.

El set de los jueces es speaker-disjoint y puede traer voces/engines nuevos. Por eso el núcleo **no** es un clasificador de timbre: es **cómo conversa** el caller (latencia, interrupciones, silencio, regularidad de turnos). El PDF del reto pide originalidad más allá de un clasificador de audio de estantería, y profundidad en una señal bien hecha.

---

## Estado actual

Producción: **VAD → diálogo → logística**. El train ve dos cortes de la misma llamada (VAD + `turns/` oficiales); val y `/detect` solo VAD. El desempate acústico se apagó: bajaba val. En `/detect`, si a los 90 s `P` ya está ≤0.05 o ≥0.95, se responde sin esperar el resto (cortes más agresivos bajaban val). No se calcula acústica ni ASR.

| Pipeline | Train (VAD) | Val VAD (71) |
| --- | --- | --- |
| Diálogo, solo VAD | 90.8% | 88.7% |
| + desempate acústico | 91.5% | 91.5% (6 errores) |
| **+ dual-view (VAD + turns en train)** | **90.4%** (255/282, no 99%) | **93.0%** CSV (5 errores) |
| HistGradientBoosting (descartado) | ~99% | 95.8% (overfit) |

`confidence` es `max(p, 1-p)`. La probabilidad viene de Platt (CV en train).

Errores de val en el camino de `/detect` (`models/val_errors.json`): `569ffb`, `678ee1`, `6971b2` (humanas) y `6058d9` (sintético). Un early-exit a 45 s con umbrales flojos bajaba val a 84%; por eso son 90 s y 0.05/0.95.

---

## Cómo funciona

```
WAV estéreo 8 kHz
    → (opcional) primeros 90 s → VAD → logística
    → si P ≤ 0.05 o P ≥ 0.95: responder
    → si no: llamada entera → VAD → logística
    → {"is_synthetic": true, "confidence": 0.87}
```

El juez **no manda** `turns/*.json`. Esos JSON solo sirven en desarrollo. En inferencia reconstruimos los turnos con VAD. El modelo de producción se entrenó **con ese mismo VAD** (`--from-wav`) para no tener desajuste train/serve.

IoU del VAD vs turns oficiales: ~0.90 en caller, ~0.96 en agente.

Señales que más empujan a *synthetic*: latencia mediana/media alta, turnos del caller más largos. A *human*: latencia irregular (`lat_std`) y más habla del agente. En el dataset, humanas contestan ~2 s e interrumpen más; sintéticos esperan ~3–5 s (ASR+LLM) y son más regulares.

Semántica (Vosk) existe en código pero **no** entra a `/detect`. El endpoint no corre ASR. La acústica no se calcula: el desempate está apagado.

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
| `val_errors.json` | Errores de val del bundle actual. |
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
python -m detector.train --from-wav --dual-view
python -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

Si 8000 está ocupado, mata el `python` que lo tiene y no lances un segundo. `/health` del bundle congelado:

```json
{"ok": true, "model": "dialogue_model.joblib", "val_accuracy": 0.929…, "tiebreak": false, "early_exit": true}
```

Si solo ves `{"status":"ok"}`, es **otro** proceso. No reentrenar para el juicio. Si hace falta: `--from-wav --dual-view --reuse-csv`.

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

| ID | Rol en demo | Manifest | `/detect` (este joblib) |
| --- | --- | --- | --- |
| `call_0e1e2f29bfdc` | humana que encaja | human, val | `false` ~0.95 |
| `call_4d8129939686` | bot que espera | synthetic, train | `true` ~1.00 |
| `call_2d4374d86df7` | VAD lo parte; latencia 13 s | synthetic, val | `true` ~1.00 |
| `call_6058d9c5c82f` | bot rápido (~2 s); el diálogo falla | synthetic, val | `false` ~0.70 |

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

- **Train VAD** (282): **90.4%**. Logística, no un árbol al 99%. Dual-view en train no dispara overfit.
- **Val** (71): ~93% en CSV; con early-exit de serve, 67/71. n=71 es ruido (±6 pts). Dual-view vs el bundle anterior: arregló 2, rompió 1 (`6058`); McNemar no es significativo. **No citar 93% como hidden.**
- **Hidden**: otras voces/engines. El método es diálogo, no el porcentaje.

`turns/` solo en train (segunda vista). `/detect` nunca los lee.

---

## Juicio (modelo congelado)

1. Un uvicorn en 8000. Comprobar `/health` (arriba).
2. Demo, no más train: humana (`0e1e2f`) / bot lento (`4d8129`) / `2d4374` (arreglado por latencia, no por voz) / `6058d9` (bot a ~2 s; el diálogo se equivoca a propósito, sin `if`).
3. Semántica y acústica **fuera** del endpoint (lentas o 63% solas). Early-exit ya está: 90 s y solo si `P` ≤0.05 o ≥0.95.

