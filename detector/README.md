# Detector Altur

Clasifica si el **caller** (canal 0) de una llamada bancaria en español mexicano es **humano** o **sintético**. El caller sintético es una pila ASR + modelo de lenguaje + voz. El canal 1 es el agente del banco.

La decisión no usa el timbre ni el texto. Usa **cómo se engancha el caller con el agente**: cuánto espera, qué tan largos y regulares son sus turnos, si corta al agente, si hablan a la vez, y qué hace cuando el otro se calla. Eso es independiente de qué motor de voz produzca el canal 0.

```
WAV estéreo 8 kHz
    → decodificar a dos canales float32
    → VAD de energía por canal
    → 21 features de diálogo
    → logística calibrada (p = P(sintético))
    → is_synthetic = (p ≥ 0.5)
    → confidence = max(p, 1 − p)
```

---

## Entrada de audio

El endpoint `POST /detect` recibe un WAV estéreo 8 kHz: cuerpo `RIFF` crudo, o JSON con el audio en base64 (`audio_base64`, `audio`, `wav` o `clip`). `POST /detect/upload` hace lo mismo con un archivo.

`io_wav` normaliza a `float32` en \([-1, 1]\) y fuerza dos canales. Canal 0 = caller (quien se clasifica). Canal 1 = agente (contexto: a qué está respondiendo el caller).

Si el payload no se puede leer como WAV, la respuesta es `is_synthetic: true` y `confidence: 0.52`.

`turns/*.json` no entra a inferencia. En desarrollo existe como segunda vista de los mismos intervalos `{channel, start, end}`.

---

## VAD: de muestras a turnos

Cada canal se segmenta por separado. Un turno es un intervalo continuo de habla `{start, end}` en segundos.

1. **RMS por frame.** Se parte el canal en ventanas de 20 ms (160 muestras a 8 kHz). De cada ventana se toma la raíz de la energía media.
2. **Umbral adaptativo.** El ruido de fondo es la mediana de los frames por debajo del percentil 20 de RMS. El umbral es `max(0.005, ruido × rel_k)`. Todo frame por encima cuenta como habla.
3. **Hangover.** Tras el último frame de habla se siguen marcando como habla los siguientes `hangover` frames, para no cortar finales de palabra.
4. **Fusión.** Huecos ≤ 0.3 s entre dos segmentos se unen (una vacilación corta no parte el turno).
5. **Duración mínima.** Se descartan segmentos de menos de 0.2 s (clicks, ruido).

Caller y agente no comparten `rel_k`. El caller vacila y baja la voz; el agente es más estable y más alto.

| Canal | `rel_k` | Hangover |
|---|---|---|
| 0 caller | 6.0 | 2 frames (40 ms) |
| 1 agente | 8.0 | 2 frames (40 ms) |

Los dos listados se mezclan y se ordenan por `start`. A partir de ahí el audio ya no se usa: solo queda la geometría temporal entre canales.

---

## Cómo se construye cada feature

Hay 21 números. Todas salen de los turnos del caller \(C\) y del agente \(A\). \(T\) es la duración de la llamada.

### Tamaño de la conversación

| Feature | Definición |
|---|---|
| `duration_s` | Duración del WAV, \(T\). |
| `n_caller` | Número de turnos del caller. |
| `n_agent` | Número de turnos del agente. |
| `turn_caller_mean` | Media de las duraciones \(e_i - s_i\) del caller. |
| `turn_caller_std` | Desviación de esas duraciones. |
| `turn_caller_cv` | `std / mean` (0 si no hay habla). Regularidad relativa del largo de turno. |
| `caller_speech_ratio` | Suma de duraciones del caller / \(T\). |
| `agent_speech_ratio` | Suma de duraciones del agente / \(T\). |

### Latencia de respuesta

Para cada turno del caller se busca el turno del agente que **terminó justo antes** (con 50 ms de holgura). La latencia es `start_caller − end_agente`, recortada a ≥ 0. Un turno que empieza casi encima del agente no aporta latencia (eso es barge-in). Si el caller habla primero y no hay agente previo, ese turno no entra.

| Feature | Definición |
|---|---|
| `lat_mean` | Media de esas esperas. |
| `lat_med` | Mediana. Menos sensible a un solo silencio largo. |
| `lat_p90` | Percentil 90. La espera larga típica, no la extrema. |
| `lat_std` | Dispersión de las esperas. |
| `first_latency` | La primera latencia de la llamada (saludo / primer dato). |

### Interrupciones y overlap

Un **barge-in** del caller es un turno cuyo `start` cae *dentro* de un turno del agente (el agente aún no había terminado, con 50 ms de margen). `agent_barge` es lo simétrico: el agente corta al caller.

| Feature | Definición |
|---|---|
| `barge_in` | Conteos de veces que el caller interrumpe. |
| `barge_rate` | `barge_in / n_caller`. |
| `agent_barge` | Veces que el agente interrumpe al caller. |
| `overlap_s` | Segundos en los que ambos canales tienen un turno a la vez (intersección de intervalos). |
| `overlap_rate` | `overlap_s / T`. |

### Silencios

| Feature | Definición |
|---|---|
| `silence_fill` | El caller vuelve a hablar después de ≥ 1.5 s de su propio silencio **y** el agente no metió un turno en ese hueco. Es “rellenar” cuando el otro no retoma. |
| `silence_fill_rate` | `silence_fill / n_caller`. |
| `caller_gap_mean` | Media de los huecos entre turnos consecutivos del caller (`start_{i+1} − end_i`, solo si no se solapan). |

---

## Qué separa humano de sintético

El caller sintético no es “una voz rara”. Es un sistema que **oye, piensa y habla** en serie. Eso deja huella en el reloj, no en el espectro.

- **Espera más.** Entre el fin del agente y su respuesta hay cola de ASR + LLM. Humanos del dataset contestan cerca de 2 s; sintéticos suelen irse a 3–5 s. Por eso `lat_med` y `lat_mean` altos empujan a sintético.
- **Habla en bloques más largos.** El LLM entrega un párrafo; la persona corta, asiente, pregunta. `turn_caller_mean` alto → sintético.
- **Es más metrónomo.** La misma pila tarda parecido cada turno. Un humano a veces dispara y a veces se queda pensando: `lat_std` alto → humano.
- **Deja más aire al agente.** Si el caller es un bot que suelta respuestas largas y espera el siguiente prompt, el agente ocupa menos fracción de la llamada. `agent_speech_ratio` alto → humano. `caller_speech_ratio` alto también apunta a humano: la persona se mete más a menudo, no solo en monólogos.
- **Interrumpe distinto.** El barge-in humano es frecuente e irregular. El bot suele esperar a que el agente termine; cuando el agente lo pisa (`agent_barge`), a veces es porque el bot no cede el canal.

Los pesos de la logística (signo **+** = hacia sintético, tras estandarizar) ordenan así la separación:

| Peso | Feature | Hacia sintético cuando… | Hacia humano cuando… |
|---|---|---|---|
| −1.68 | `agent_speech_ratio` | El agente ocupa poco de la llamada | El agente habla una fracción grande |
| +1.60 | `lat_med` | La espera típica es larga | Contesta pronto |
| −1.47 | `caller_speech_ratio` | El caller ocupa poco (pocos metidos) | El caller se mete más en el tiempo total |
| +1.41 | `lat_mean` | La espera promedio es larga | Igual que `lat_med`, media |
| +1.15 | `turn_caller_mean` | Turnos largos tipo párrafo | Turnos cortos |
| +0.74 | `turn_caller_std` | Largos muy variables *y* ya controlando la media | Turnos de tamaño parecido y cortos |
| −0.63 | `lat_std` | Esperas casi iguales (reloj de pila) | Esperas irregulares |
| −0.53 | `lat_p90` | (residual: con media/mediana altas, un p90 extra no suma bot) | Cola de esperas muy desigual |
| −0.52 | `duration_s` | Llamadas más cortas | Llamadas más largas |
| +0.51 | `barge_rate` | Interrumpe una fracción alta de *sus* turnos, en el patrón del bot | — |
| −0.45 | `overlap_s` | Poco habla a la vez | Más solape |
| +0.44 | `agent_barge` | El agente lo pisa seguido | El caller cede menos de esa forma |

Las que **más separan** son las tres primeras: fracción de habla del agente, mediana de latencia, y fracción de habla del caller. El resto afina regularidad e interrupciones. `n_caller`, `n_agent`, `turn_caller_cv`, `first_latency`, `barge_in` crudo, `overlap_rate` y los silence-fills pesan menos; el modelo las tiene porque describen la misma geometría, no porque cada una corte sola.

Hay callers humanos muy pacientes (latencia alta, casi sin barge-in) y bots con pila rápida (latencia ~2 s). Esos se parecen en este espacio y el modelo los puede cruzar.

---

## Clasificador

Las 21 features se estandarizan (`StandardScaler`) y entran a una **regresión logística** (`C=0.4`, clases balanceadas, `lbfgs`). Encima hay **calibración sigmoide** (Platt, 3 folds sobre train): el número que sale es una probabilidad, no solo un score lineal.

- `p ≥ 0.5` → `is_synthetic: true`
- `confidence = p` si es sintético, `1 − p` si es humano

Se usa logística y no un boosting de árboles: la frontera es una combinación de timings, no una memorización de callers.

### Dual-view en train

Cada llamada de **train** entra dos veces, misma etiqueta:

1. Features del VAD sobre el WAV (la vista de inferencia).
2. Features de los intervalos oficiales en `turns/` (la misma conversación, segmentada sin el error del VAD).

Val e inferencia usan **solo** la vista VAD. El JSON oficial enseña al modelo la geometría “limpia” de la conversación; el VAD le enseña la geometría ruidosa que va a ver siempre. Las dos vistas son el mismo fenómeno (timing), no un segundo sensor (voz o texto).

El artefacto es `models/dialogue_model.joblib`: pipeline, nombres de las 21 features y el corte en 0.5.
