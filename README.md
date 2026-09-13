# Altur Challenge — Detector de Caller Sintético (HackMTY 2026)

Sistema que recibe una llamada telefónica (WAV estéreo, banco + cliente) y decide si quien llama es una **persona real** o un **caller sintético** (ASR + LLM + voz generada) haciéndose pasar por una. Construido para el reto Altur de HackMTY 2026.

```json
POST /detect  →  {"is_synthetic": true, "confidence": 0.87}
```

## Índice

- [El reto](#el-reto)
- [La solución](#la-solución)
- [Por qué timing y no voz](#por-qué-timing-y-no-voz)
- [Resultados](#resultados)
- [Stack tecnológico](#stack-tecnológico)
- [Estructura del repo](#estructura-del-repo)
- [Instalación y uso](#instalación-y-uso)
- [API](#api)
- [Deploy y dashboard](#deploy-y-dashboard)
- [Qué se probó y se descartó](#qué-se-probó-y-se-descartó)
- [Términos del dataset](#términos-del-dataset)

---

## El reto

Llamadas grabadas entre un caller y el agente de IA de atención a clientes de un banco, en español mexicano. En algunas llamadas el caller es una persona real; en otras es una IA autónoma (reconocimiento de voz + modelo de lenguaje + voz sintética) marcando al mismo número. La tarea: dado un WAV, decidir si el caller es humano o sintético.

| Archivo | Contenido |
| --- | --- |
| `manifest.csv` | Una fila por llamada: `anon_id`, `label` (`human`/`synthetic`), `split` (`train`/`val`), `duration_s`. |
| `audio/<anon_id>.wav` | Estéreo, 8 kHz, 16-bit PCM. Canal 0 = caller (el que se clasifica). Canal 1 = agente. |
| `turns/<anon_id>.json` | Segmentos de habla por canal, derivados automáticamente del audio; usados como punto de partida, no como entrada de inferencia. |

`train` y `val` son *speaker-disjoint* (ningún caller se repite entre splits). El set de jueces trae **callers y voces que no aparecen en ninguno de los dos splits** — esa es la condición que más shapeó las decisiones técnicas de este proyecto (ver [Por qué timing y no voz](#por-qué-timing-y-no-voz)).

El audio no se distribuye en este repo (`.gitignore`); se descarga aparte como `altur-challenge-audio.zip` y se descomprime en `audio/`.

## La solución

El sistema **no analiza el timbre ni el contenido de la voz**. Analiza *cómo el caller conversa* con el agente: cuánto tarda en responder, qué tan largos y regulares son sus turnos, si interrumpe, si hablan a la vez, y qué hace en los silencios. Esa señal es independiente de qué motor TTS o voz produzca el audio, así que generaliza a voces nunca vistas — que es exactamente lo que trae el set de jueces.

```
WAV estéreo 8 kHz (canal 0 = caller, canal 1 = agente)
    → decodificar a dos canales float32                (detector/io_wav.py)
    → VAD por energía, umbral y hangover por canal      (detector/vad.py)
    → 22 features de timing de diálogo                  (detector/features.py)
    → regresión logística estandarizada + calibración   (detector/train.py, predict.py)
        Platt (sigmoide, 3 folds)
    → {"is_synthetic": p ≥ 0.5, "confidence": max(p, 1-p)}
```

Servido con FastAPI (`detector/app.py`) detrás de un endpoint `POST /detect` que nunca debe fallar durante el juicio: cualquier excepción cae a una respuesta por defecto (`is_synthetic: true, confidence: 0.52`) en vez de un 500.

El detalle completo de las 22 features, sus pesos y la interpretación de cada una está en [`detector/README.md`](detector/README.md); la bitácora de qué se midió, qué mejoró y qué se descartó con evidencia está en [`ANALISIS_Y_MEJORAS.md`](ANALISIS_Y_MEJORAS.md).

## Por qué timing y no voz

La primera versión de este proyecto (los scripts sueltos en la raíz: `extraer_dataset_completo.py`, `entrenar_modelo.py`, `test_feature.py`) fue un clasificador acústico clásico — MFCC + features espectrales del canal del caller, sobre un RandomForest. Llegó a **100% en un split aleatorio de validación**, lo cual encendió una alerta en vez de una celebración.

Se investigó si era un atajo trivial (p. ej. 2-3 voces TTS reusadas) y no se encontró ese mecanismo. El problema real es más sutil: el reto distingue dos niveles de "nunca visto". `train`/`val` son *speaker-disjoint*, pero pueden compartir la misma voz/motor TTS con distinto caller encima — un split aleatorio no prueba generalización a una **voz nueva**, solo a un **caller nuevo con una voz ya vista**. El set de jueces sí trae voces nuevas.

Para medir eso de verdad se construyó `detector/eval_generalization.py`: agrupa las llamadas sintéticas por similitud acústica (proxy de motor/voz) y deja cada clúster completo fuera del entrenamiento (*leave-one-voice-cluster-out*) en vez de un split aleatorio. Bajo esa prueba más dura:

| Configuración | Accuracy media | Recall sintético medio | Peor clúster |
| --- | --- | --- | --- |
| **Solo diálogo (producción actual)** | **0.887** | **0.907** | 0.800 |
| Diálogo + acústica mezcladas siempre | 0.829 | 0.707 | 0.313 |
| Diálogo + acústica solo como desempate | 0.880 | 0.890 | — |

El modelo de solo-diálogo, con 22 features de comportamiento y sin tocar el timbre, es notablemente más robusto a voces nunca vistas que cualquier variante que mezcle acústica. Por eso el acústico se descartó de producción — con evidencia medida, no por precaución teórica. El detalle completo está en la sección 4 de [`ANALISIS_Y_MEJORAS.md`](ANALISIS_Y_MEJORAS.md).

Una vía de semántica (transcribir con Vosk y usar el contenido de lo dicho) quedó explorada pero en evaluación bajo la misma regla — no entra a producción sin pasar `eval_generalization`.

## Resultados

| Iteración | Accuracy en val | Notas |
| --- | --- | --- |
| Prototipo acústico (MFCC + RandomForest) | 100% (split aleatorio) | Descartado: no sobrevive `eval_generalization` (ver arriba) |
| VAD con umbral único para ambos canales | 88.7% (63/71) | Baseline del modelo de diálogo |
| VAD tuneado por canal (`rel_k`, hangover) | 91.5% (65/71) | El caller vacila más que el agente; un solo umbral lo medía peor |
| + dual-view training (VAD + turnos oficiales en train) | 94.4% (67/71) | Reconciliado tras integrar cambios en paralelo de una compañera de equipo |
| **Estado actual del modelo en el repo** | **95.8% (68/71)** | `models/dialogue_model.joblib`, 3 errores — los 3 son humanos atípicamente pacientes/formales que el modelo confunde con bot |

Bajo la validación más dura de generalización a voces nunca vistas (`eval_generalization.py`, no el split normal), el modelo de producción mide **88.7% accuracy / 90.7% recall sintético** en el peor caso por clúster de voz — el número relevante para lo que realmente evalúa el juicio.

Reproducible con:

```bash
python -m detector.train --from-wav --dual-view   # reentrena y confirma el accuracy
python -m detector.eval_vad                         # IoU del VAD vs turns/ oficiales
python -m detector.eval_generalization              # leave-one-voice-cluster-out
```

## Stack tecnológico

| Capa | Herramienta | Uso |
| --- | --- | --- |
| Lenguaje | Python 3.9–3.14 | Probado en 3.9.6 y en 3.14.7 (última estable); sin dependencias que rompan en la nueva |
| API | FastAPI + Uvicorn | Sirve `POST /detect`, `GET /health`, `GET /dashboard` |
| ML | scikit-learn (`LogisticRegression`, `StandardScaler`, `CalibratedClassifierCV`) | Clasificador de diálogo |
| Datos | NumPy, Pandas, SciPy, joblib | Extracción de features y persistencia del modelo |
| Audio | Lectura WAV con la librería estándar (`detector/io_wav.py`) | Sin dependencias pesadas de audio en el path de producción |
| ASR (experimental) | Vosk (`vosk-model-small-es-0.42`) | Transcripción para la vía semántica, en evaluación, no en producción |
| Base de datos | PostgreSQL + TimescaleDB (Tiger Data / Timescale Cloud) | Log de cada veredicto para el dashboard, opcional |
| Deploy | Vultr (Ubuntu 22.04) + systemd | `deploy/setup.sh` deja `/detect` público con reinicio automático |
| Prototipado descartado | librosa, soundfile | Solo en los scripts exploratorios de la raíz (acústico, no producción) |

## Estructura del repo

```
detector/                  Paquete de producción
├── app.py                 FastAPI: /detect, /health, /dashboard
├── io_wav.py               Decodifica WAV → float32 estéreo
├── vad.py                  VAD por energía, umbral/hangover por canal
├── features.py             Las 22 features de timing de diálogo
├── train.py                Entrena y calibra el modelo (--from-wav, --dual-view, ...)
├── predict.py               Carga el modelo y corre inferencia end-to-end
├── explain.py               Contribución de cada feature a un veredicto
├── eval_vad.py               IoU del VAD contra turns/ oficiales
├── eval_generalization.py    Validación leave-one-voice-cluster-out
├── acoustic.py / semantic.py / transcribe*.py   Vías exploradas, no en producción
├── logging_db.py            Log de veredictos en TimescaleDB (opcional)
├── dashboard.html            Dashboard en vivo servido por /dashboard
└── test_endpoint.py           Cliente para probar /detect contra el manifest

deploy/                     Cómo dejarlo público
├── setup.sh                 Systemd + venv en un servidor Vultr limpio
├── README.md                 Pasos de deploy
└── tiger-data.md             Cómo conectar el dashboard a TimescaleDB

scripts/
├── check_endpoint.py         Cliente oficial del juez
└── example_server.py         Ejemplo mínimo de servidor /detect

models/                     Modelo entrenado y artefactos de evaluación
├── dialogue_model.joblib     Pipeline + nombres de features + corte 0.5 (trackeado en git)
├── dialogue_features.csv     Cache de features extraídas
└── val_errors.json            Llamadas de val donde el modelo falla, para auditoría

manifest.csv, turns/, hackmty26-altur-challenge.pdf     Dataset y enunciado del reto
audio/                       Audio del dataset (no trackeado, se descarga aparte)

extraer_dataset_completo.py, entrenar_modelo.py,
test_feature.py, testaudio.py, exploreman.py           Prototipo acústico original, no producción

ANALISIS_Y_MEJORAS.md        Bitácora de qué se midió y qué se descartó, con evidencia
```

## Instalación y uso

Requiere Python 3.9+ (probado también con Python 3.14, el más reciente disponible vía Homebrew: `brew install python@3.14`).

```bash
git clone https://github.com/MariaGpeSotoAcosta/hackmty26_alt.git
cd hackmty26_alt
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Descomprimir `altur-challenge-audio.zip` (Releases) en la raíz para tener `audio/`.

Entrenar el modelo (regenera `models/dialogue_model.joblib`):

```bash
python -m detector.train --from-wav --dual-view
```

Levantar el servidor:

```bash
python -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

Probar contra el dataset local:

```bash
python -m detector.test_endpoint call_0e1e2f29bfdc   # esperado: human
python scripts/check_endpoint.py --url http://localhost:8000/detect --split val --n 20
```

## API

`POST /detect` acepta tres formatos de cuerpo:

**JSON** (el que usa el juez):

```json
{"call_id": "...", "audio_base64": "<WAV completo en base64>", "sample_rate": 8000, "channels": 2}
```

**Multipart** (`file=<wav>`), o el WAV crudo (`Content-Type: application/octet-stream`, cuerpo `RIFF...`).

Respuesta, siempre HTTP 200:

```json
{"is_synthetic": true, "confidence": 0.87}
```

`is_synthetic` es obligatorio; `confidence` en `[0, 1]` es opcional pero siempre se manda (mejora AUC/calibración del lado del juez). Timeout de 30 s. Un payload no leíble como WAV responde con un prior débil (`is_synthetic: true, confidence: 0.52`) en vez de fallar.

`GET /health` reporta si el modelo está cargado, sus features y el accuracy de val con el que se entrenó. `GET /dashboard` sirve el panel en vivo (ver abajo).

## Deploy y dashboard

- [`deploy/README.md`](deploy/README.md) — instancia en Vultr, systemd, firewall, cómo actualizar en caliente.
- [`deploy/tiger-data.md`](deploy/tiger-data.md) — conectar `/detect` a una base TimescaleDB (Tiger Data) para que cada veredicto (features, confianza, latencia, por qué decidió eso) quede registrado y visible en `/dashboard`. Es opcional: sin `TIGER_DATA_URL` configurado, `/detect` funciona igual, solo no persiste nada. No se guarda audio ni transcripción, solo los números de comportamiento ya derivados.

## Qué se probó y se descartó

Resumen ejecutable de la bitácora completa en [`ANALISIS_Y_MEJORAS.md`](ANALISIS_Y_MEJORAS.md):

1. **VAD tuneado por canal** — implementado. El caller vacila más que el agente; un umbral relativo (`rel_k`) y hangover distintos por canal subieron el IoU contra `turns/` oficiales de 0.895/0.962 a 0.918/0.981, y el accuracy en val de 88.7% a 91.5%.
2. **Auditoría de errores residuales** — hecho. Los errores que quedan no son culpa del VAD (su IoU es igual al de los aciertos); son casos de comportamiento genuinamente ambiguo dentro de su propia clase (humanos atípicamente pacientes, o bots con pipeline inusualmente rápido).
3. **Clasificador acústico (MFCC/timbre)** — evaluado y descartado con evidencia (ver [Por qué timing y no voz](#por-qué-timing-y-no-voz)).
4. **Semántica (Vosk ASR)** — transcrito el dataset completo, en evaluación bajo la misma regla que la acústica: no sube a producción sin superar `eval_generalization.py`.

## Términos del dataset

Los callers humanos participaron voluntariamente, sabían que la llamada era para probar una IA, y usaron datos personales inventados. No intentar identificar a nadie. El dataset es exclusivo de HackMTY 2026; no redistribuir.
