# Dialogue-timing features

22 numbers, all from caller turns \(C\), agent turns \(A\), and call duration \(T\). After VAD, the waveform is discarded; only interval geometry remains. Implemented in `detector/features.py`. Weights below are the trained logistic coefficients after standardization (**+** toward synthetic).

## VAD

Each channel is segmented on its own. A turn is a continuous speech interval `{start, end}` in seconds (`detector/vad.py`).

1. **RMS per frame.** 20 ms windows (160 samples at 8 kHz).
2. **Adaptive threshold.** Background noise is the median of frames below the 20th percentile of RMS. Threshold is `max(0.005, noise × rel_k)`. Frames above it are speech.
3. **Hangover.** After the last speech frame, the next `hangover` frames stay marked as speech so word endings are not cut.
4. **Merge.** Gaps ≤ 0.3 s between segments are joined (a short hesitation is not a new turn).
5. **Minimum duration.** Segments shorter than 0.2 s (clicks, noise) are dropped.

Caller and agent do not share `rel_k`. The caller hesitates and drops in level; the agent is louder and more stable.

| Channel | `rel_k` | Hangover |
| --- | --- | --- |
| 0 caller | 6.0 | 2 frames (40 ms) |
| 1 agent | 8.0 | 2 frames (40 ms) |

The two lists are merged and sorted by `start`.

## Conversation size

| Feature | Definition |
| --- | --- |
| `duration_s` | WAV duration, \(T\). |
| `n_caller` | Number of caller turns. |
| `n_agent` | Number of agent turns. |
| `turn_caller_mean` | Mean caller turn length \(e_i - s_i\). |
| `turn_caller_std` | Std of those lengths. |
| `turn_caller_cv` | `std / mean` (0 if no speech). Relative regularity of turn length. |
| `caller_speech_ratio` | Sum of caller durations / \(T\). |
| `agent_speech_ratio` | Sum of agent durations / \(T\). |

## Response latency

For each caller turn, find the agent turn that **ended just before** it (50 ms slack). Latency is `start_caller − end_agent`, clipped to ≥ 0. A turn that starts on top of the agent does not contribute latency (that is barge-in). If the caller speaks first and there is no prior agent turn, that turn is skipped.

| Feature | Definition |
| --- | --- |
| `lat_mean` | Mean of those waits. |
| `lat_med` | Median. Less sensitive to one long silence. |
| `lat_p90` | 90th percentile. The typical long wait, not the extreme. |
| `lat_std` | Dispersion of the waits. |
| `lat_cv` | `std / mean` of the waits. A fast bot is still a metronome; `lat_std` alone misses it because the waits are short. |
| `first_latency` | First latency of the call (greeting / first datum). |

## Interruptions and overlap

A caller **barge-in** is a turn whose `start` falls *inside* an agent turn (the agent had not finished, 50 ms margin). `agent_barge` is the symmetric case: the agent cuts the caller.

| Feature | Definition |
| --- | --- |
| `barge_in` | Times the caller interrupts. |
| `barge_rate` | `barge_in / n_caller`. |
| `agent_barge` | Times the agent interrupts the caller. |
| `overlap_s` | Seconds both channels have a turn at once (interval intersection). |
| `overlap_rate` | `overlap_s / T`. |

## Silences

| Feature | Definition |
| --- | --- |
| `silence_fill` | The caller speaks again after ≥ 1.5 s of their own silence **and** the agent did not take a turn in that gap. Filling when the other side does not resume. |
| `silence_fill_rate` | `silence_fill / n_caller`. |
| `caller_gap_mean` | Mean gap between consecutive caller turns (`start_{i+1} − end_i`, only if they do not overlap). |

## What the weights say

The synthetic caller hears, thinks, and speaks in series. That leaves a mark on the clock:

- **Waits longer.** Between the agent’s end and the reply there is an ASR + LLM queue. Humans in this set answer near 2 s; synthetics often sit at 3–5 s. High `lat_med` / `lat_mean` push synthetic.
- **Speaks in longer blocks.** The LLM delivers a paragraph; a person cuts, nods, asks. High `turn_caller_mean` → synthetic.
- **More metronomic.** The same stack takes a similar time every turn, slow or fast. Low `lat_cv` → synthetic.
- **Leaves more air for the agent.** If the caller dumps long replies and waits for the next prompt, the agent occupies less of the call. High `agent_speech_ratio` → human. High `caller_speech_ratio` also points human: the person jumps in more often, not only in monologues.
- **Interrupts differently.** Human barge-in is frequent and irregular. The bot usually waits for the agent to finish; when the agent steps on it (`agent_barge`), it is often because the bot does not yield the channel.

| Weight | Feature | Toward synthetic when… | Toward human when… |
| --- | --- | --- | --- |
| −1.64 | `agent_speech_ratio` | The agent occupies little of the call | The agent speaks a large fraction |
| −1.40 | `caller_speech_ratio` | The caller occupies little | The caller takes more of the total time |
| +1.28 | `lat_mean` | Average wait is long | Answers soon |
| +1.16 | `turn_caller_mean` | Long, paragraph-like turns | Short turns |
| +1.14 | `lat_med` | Typical wait is long | Answers soon |
| −1.00 | `lat_cv` | Waits almost the same relative length (stack clock) | Some short, some very long |
| +0.80 | `turn_caller_std` | Highly variable lengths *after* controlling the mean | Turns similar and short |
| −0.70 | `lat_p90` | Residual: once mean/median are high, extra p90 does not add “bot” | A very uneven wait tail |
| +0.59 | `barge_rate` | Interrupts a high fraction of *its* turns, in the bot pattern | — |
| −0.48 | `duration_s` | Shorter calls | Longer calls |
| −0.48 | `overlap_s` | Little simultaneous speech | More overlap |
| +0.47 | `agent_barge` | The agent steps on the caller often | The caller yields less that way |

The strongest splitters are speech fraction on each side, mean/median wait, turn length, and `lat_cv`. The rest refine interruptions. `n_caller`, `n_agent`, `turn_caller_cv`, `lat_std`, `first_latency`, raw `barge_in`, `overlap_rate`, and the silence-fills weigh less; they stay because they describe the same geometry, not because each one cuts alone.

`lat_cv` catches bots with a fast stack (waits ~2 s but almost equal). What remains are very patient humans: they wait 5–19 s as if there were an ASR queue, even if irregularly. In this space they look like a slow bot, and the model can cross them.
