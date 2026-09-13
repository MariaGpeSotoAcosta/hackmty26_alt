# Aura3000 — Devpost Submission Copy

---

## Elevator Pitch — EN (171/200 chars)

> Aura3000 catches AI callers impersonating humans on bank lines by how they talk — turn-taking, pauses, interruptions — not their voice. Works on voices it has never heard.

## Elevator Pitch — ES (182/200 caracteres)

> Aura3000 detecta callers de IA que se hacen pasar por humanos en llamadas bancarias por cómo hablan — turnos, pausas, interrupciones — no por su voz. Funciona con voces nunca vistas.

---

## Project Story — EN

### Inspiration
Banks are starting to get calls from autonomous AI agents — ASR + LLM + synthetic voice — impersonating real customers. Altur Challenge (HackMTY 2026) gave us real bank calls, human and synthetic, and asked us to tell them apart from audio alone — even on voices we'd never seen.

### What it does
Given a stereo call (channel 0 = caller, channel 1 = agent), Aura3000 returns a verdict and confidence:

\\( P(\text{synthetic} \mid x) = \sigma(\mathbf{w}^\top x + b), \qquad \text{is\_synthetic} = \mathbb{1}[P \geq 0.5] \\)

It never inspects timbre or words — only *how* the caller converses: response latency, turn regularity, interruptions, overlap, silence-filling. That signal is voice-engine-agnostic.

### How we built it
Energy-based VAD (tuned per channel) reconstructs speech turns from the raw WAV, 22 dialogue-timing features are extracted, and a calibrated logistic regression scores them. Served via FastAPI (`POST /detect`), deployed on Vultr + systemd, with verdicts logged to TimescaleDB for a live dashboard.

### Challenges we ran into
Our first acoustic model (MFCC + RandomForest) hit:

$$\text{accuracy}_{\text{random split}} = 1.00$$

— a red flag, not a win: the judging set brings **unseen voices**, and a random split can't test that. We built `eval_generalization.py` (leave-one-voice-cluster-out) and measured:

| Model | Acc. | Synth. recall | Worst cluster |
|---|---|---|---|
| Dialogue only | \\(0.887\\) | \\(0.907\\) | \\(0.800\\) |
| + acoustic (mixed) | \\(0.829\\) | \\(0.707\\) | \\(0.313\\) |

Acoustic features cut from production — on evidence, not intuition. We also caught and disabled an acoustic tiebreak that silently re-activated on an exact accuracy tie during a merge with a teammate's parallel work.

### Accomplishments
\\(0.958\\) validation accuracy with zero use of voice timbre; a standing rule (`eval_generalization.py`) that any future feature must survive before shipping; a `/detect` endpoint that has never hard-failed.

### What we learned
A perfect score on an easy test is a warning sign. Matching the *evaluation* to the real threat model — unseen voices — mattered more than any single feature.

### What's next
A semantic path (Vosk ASR transcripts) is being held to the same bar: no production without beating `eval_generalization.py`. Longer term: the same behavioral approach applied to other phone fraud vectors (insurance, SIM-swap, benefits).

---

## Project Story — ES

### Inspiración
Los bancos empiezan a recibir llamadas de agentes de IA autónomos — ASR + LLM + voz sintética — que se hacen pasar por clientes reales. El reto Altur (HackMTY 2026) nos dio llamadas reales, humanas y sintéticas, y pidió distinguirlas solo con el audio — incluso con voces nunca vistas.

### Qué hace
Con una llamada estéreo (canal 0 = caller, canal 1 = agente), Aura3000 regresa un veredicto y una confianza:

\\( P(\text{sintético} \mid x) = \sigma(\mathbf{w}^\top x + b), \qquad \text{is\_synthetic} = \mathbb{1}[P \geq 0.5] \\)

Nunca analiza timbre ni palabras — solo *cómo* conversa el caller: latencia de respuesta, regularidad de turnos, interrupciones, traslape, relleno de silencios. Esa señal es independiente del motor de voz.

### Cómo lo construimos
Un VAD por energía (ajustado por canal) reconstruye los turnos de habla del WAV crudo, se extraen 22 features de timing, y una regresión logística calibrada los puntúa. Servido con FastAPI (`POST /detect`), desplegado en Vultr + systemd, con veredictos guardados en TimescaleDB para un dashboard en vivo.

### Retos que enfrentamos
Nuestro primer modelo acústico (MFCC + RandomForest) llegó a:

$$\text{accuracy}_{\text{split aleatorio}} = 1.00$$

— una alerta, no un triunfo: el set de jueces trae **voces nunca vistas**, y un split aleatorio no prueba eso. Construimos `eval_generalization.py` (leave-one-voice-cluster-out) y medimos:

| Modelo | Acc. | Recall sintético | Peor clúster |
|---|---|---|---|
| Solo diálogo | \\(0.887\\) | \\(0.907\\) | \\(0.800\\) |
| + acústica (mezclada) | \\(0.829\\) | \\(0.707\\) | \\(0.313\\) |

La acústica se sacó de producción — con evidencia, no intuición. También detectamos y desactivamos un desempate acústico que se reactivó solo por un empate exacto al integrar el trabajo en paralelo de una compañera.

### Logros
\\(0.958\\) de accuracy en validación sin usar timbre de voz; una regla permanente (`eval_generalization.py`) que toda feature nueva debe superar antes de subir a producción; un endpoint `/detect` que nunca ha fallado en duro.

### Qué aprendimos
Un puntaje perfecto en una prueba fácil es una señal de alerta, no de éxito. Ajustar la *evaluación* a la amenaza real — voces nunca vistas — importó más que cualquier feature nueva.

### Qué sigue
Una vía semántica (transcripciones con Vosk) está sujeta a la misma regla: no sube a producción sin superar `eval_generalization.py`. A futuro: aplicar el mismo enfoque de comportamiento a otros vectores de fraude telefónico (seguros, SIM-swap, verificación de beneficios).
