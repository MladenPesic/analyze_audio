# Football Anthem Pitch Analysis — Technical Documentation

**Project:** "How in-tune are football fans with their anthems?"

This document describes the audio analysis pipeline used to quantify the pitch accuracy of football crowds singing national anthems. It is intended for stakeholders evaluating the project's findings, possibilities, limitations, and risks.

---

## 1. Project Overview

The pipeline takes a YouTube clip of fans singing a national anthem at a football match, extracts the crowd's vocal pitch over time, compares it against a reference MIDI of the intended melody, and produces quantitative metrics describing how accurately the crowd sang. Multiple anthems are scored using the same methodology, then ranked.

The headline output is a comparative scoreboard:

> *Welsh fans hit ~37% of notes within ±50 cents of correct pitch; Italian fans hit ~31%. Welsh crowds drift slightly flat (-13.5¢), Italian crowds do not.*

---

## 2. Pipeline Architecture

The system runs in five sequential stages. Each stage's output is the next stage's input. Intermediate artifacts are cached so individual stages can be re-run without recomputing earlier ones.

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 1. Acquisition  │ → │ 2. Vocal         │ → │ 3. Pitch         │
│    YouTube      │    │    Separation    │    │    Extraction    │
│    → WAV file   │    │    → vocals.wav  │    │    → pitch.csv   │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                                         ↓
┌────────────────────────┐    ┌─────────────────────────────────┐
│ 5. Comparison &        │ ← │ 4. Reference Alignment & Scoring │
│    Visualization       │    │    Crowd CSV vs. MIDI melody    │
│    → ranked scoreboard │    │    → deviation metrics, plot    │
└────────────────────────┘    └─────────────────────────────────┘
```

Each stage is implemented as a standalone Python module that can be invoked independently or chained via the `run_anthem.py` driver. A configuration file (`anthems.py`) registers all anthems with their metadata; adding a new anthem requires no code changes, only a config entry.

---

## 3. Stage-by-Stage Breakdown

### Stage 1 — Audio Acquisition

**Module:** `download.py`
**Library:** `yt-dlp`

**What it does.** Downloads the audio stream of a YouTube video, optionally trimmed to a specific time window (e.g., just the anthem segment), and converts the result to a standard WAV file.

**Why this approach.** YouTube hosts the most comprehensive collection of stadium anthem footage (cup finals, internationals, fan recordings). `yt-dlp` is the industry-standard tool for programmatic YouTube extraction and handles authentication, format selection, and quality choices reliably. Trimming at download-time prevents wasting storage and downstream compute on irrelevant minutes of broadcast.

**Output:** `data/raw/<video_id>.wav` — typically 60–80 seconds of stereo WAV at 44.1 kHz or 48 kHz.

**Dependencies:** Requires FFmpeg installed locally for audio extraction and trimming.

---

### Stage 2 — Vocal Isolation

**Module:** `separate.py`
**Library:** `demucs` (Hybrid Transformer Demucs, fine-tuned variant `htdemucs_ft`)

**What it does.** Uses a deep learning model to separate the raw broadcast audio into "vocals" and "everything else" (band, PA system, commentary, ambient stadium noise).

**Why this matters.** The crowd is rarely the dominant element in a broadcast mix. PA systems play orchestral arrangements; commentators talk over anthems; bands play accompaniment. Without separation, pitch analysis would track whichever element was loudest — usually not the crowd. Demucs allows us to attenuate non-vocal content so that subsequent pitch detection focuses on what the fans actually sang.

**Why htdemucs_ft specifically.** Demucs ships several model variants. `htdemucs_ft` (fine-tuned Hybrid Transformer) is the most accurate for vocal isolation and is the recommended default for music-style content. It is slower than the base model but produces noticeably cleaner stems on stadium audio.

**Output:** `data/stems/htdemucs_ft/<video_id>/vocals.wav` and a complementary `no_vocals.wav`.

**Resource cost.** ~2–3 minutes of CPU time per minute of input audio. GPU acceleration (CUDA) reduces this by ~5×.

**Important limitation.** Demucs was trained on close-miked studio vocals. A 70,000-person unison crowd in a reverberant stadium is far outside its training distribution. Quality of separation depends heavily on source material — see Section 6.

---

### Stage 3 — Pitch Extraction

**Module:** `analyze.py`
**Library:** `librosa`

**What it does.** Reads the isolated vocals WAV, then uses `librosa.pyin()` to estimate the dominant fundamental frequency (F0) at each time frame. The result is a frame-by-frame array of pitches in Hertz, plus voicing flags indicating whether each frame contains a confident pitch.

**Why pyin.** `pyin` (probabilistic YIN) is the de facto standard for monophonic pitch tracking. It uses a Hidden Markov Model over candidate frequencies to produce smooth, robust pitch trajectories. Alternatives exist (CREPE, SwiPe, autocorrelation) but pyin offers the best accuracy/dependency tradeoff for this use case.

**Why this works on a crowd.** Strictly speaking, pyin is designed for one singer. A crowd is many singers. But because the dominant note in the spectrum is what pyin tracks, and because most fans sing approximately the same melody, pyin returns the *centroid* of the crowd's pitch distribution — which is exactly the right quantity for this analysis. We measure where the *typical* fan was singing, not any individual voice.

**Conversion to musical units.** The Hertz output is converted to MIDI note numbers (`librosa.hz_to_midi`) and then to deviations in *cents* (1 cent = 1/100 of a semitone) from the nearest equal-tempered note. Cents are the standard musical unit for measuring small pitch errors.

**Output:** `data/results/<video_id>_pitch.csv` — one row per frame at ~43 frames/second, columns for time, F0, MIDI value, and cent deviation.

---

### Stage 4 — Reference Alignment and Scoring

**Modules:** `extract_melody.py`, `align_score.py`
**Libraries:** `mido` (MIDI parsing), `librosa.sequence.dtw` (alignment), `numpy`

This is the most algorithmically substantive stage. It is where "the crowd sang some pitches" becomes "the crowd was X% accurate against the *intended* melody."

**Step 4a — Extract the reference melody.** A MIDI file of the anthem (typically obtained from public archives such as freemidi.org or BitMidi) is parsed with `mido`. MIDI files contain multiple tracks (e.g., melody, alto, bass, piano accompaniment); we identify the melody track via heuristics (monophonic, vocal range, narrow span) and extract a sequence of (start time, duration, pitch) tuples. This sequence is then "rendered" into a frame-by-frame pitch array sampled at the same rate as the crowd recording, so the two can be directly compared.

**Step 4b — Dynamic Time Warping (DTW).** Crowds do not sing at the MIDI's metronome tempo. They speed up at exciting moments, hold climactic notes longer, breathe at unscripted points. DTW solves this: given two sequences representing the same musical content at different speeds, it finds the optimal non-linear time alignment between them. We use `librosa.sequence.dtw` with a cost matrix of absolute pitch differences in semitones.

**Step 4c — Transposition search.** Crowds sing at whatever key is comfortable for them, which often differs from the MIDI's key. We brute-force every transposition from -12 to +12 semitones, run DTW for each, and select the transposition with the lowest alignment cost — this finds the key the crowd actually sang in. A tie-breaker favors transpositions closest to zero (the canonical key) to avoid octave-equivalent ambiguity.

**Step 4d — Per-note scoring.** Once aligned, we compute the pitch deviation between the crowd and the reference at every aligned frame. For fairness, we then collapse consecutive frames belonging to the same reference note into one score per note (a 3-second held note shouldn't dominate the average just because it's long). The resulting note-level deviations are octave-wrapped so that singing one octave off (a common harmony behaviour in crowds) is not penalized as a 1200-cent error.

**Output:** `data/results/<video_id>_aligned.png` (visual report) plus printed metrics.

---

### Stage 5 — Comparison and Visualization

**Module:** `compare.py`
**Libraries:** `pandas`, `matplotlib`

**What it does.** Loops over all anthems registered in `anthems.py`, runs Stage 4 scoring for each, builds a ranked summary table, and produces comparison charts.

**Outputs:**
- `data/results/comparison.csv` — ranked table of all anthems by accuracy
- `data/results/comparison.png` — horizontal bar charts comparing pitch accuracy and sharp/flat tendency across anthems

---

## 4. Library Stack Summary

| Library | Stage | Role |
|---|---|---|
| `yt-dlp` | 1 | YouTube audio extraction with trim support |
| FFmpeg (system tool) | 1, 2 | Audio format conversion, codec handling |
| `demucs` | 2 | Deep learning vocal source separation |
| `torch`, `torchaudio`, `torchcodec` | 2 | Underlying ML framework for Demucs |
| `librosa` | 3, 4 | Pitch tracking (`pyin`), DTW alignment, audio I/O |
| `numpy` | 3, 4 | Array math, MIDI/cent conversions |
| `soundfile` | 3 | Robust WAV reading |
| `mido` | 4 | MIDI file parsing |
| `pandas` | 3, 5 | CSV I/O, tabular reports |
| `matplotlib` | 4, 5 | Visualization |

---

## 5. Key Metrics and Their Interpretation

The pipeline produces several metrics. Each captures a different aspect of singing accuracy.

| Metric | What it measures | How to read it |
|---|---|---|
| **Notes within ±50¢ (%)** | Fraction of intended notes the crowd hit accurately (within a quarter-tone) | The headline "in-tune-ness" score. Higher is better. |
| **Notes within ±100¢ (%)** | Fraction of notes within a full semitone of correct | A more forgiving metric — captures notes that are "in the right neighborhood" |
| **Median \|deviation\|** | Typical magnitude of pitch error per note (in cents) | Lower is better. Robust to outliers. |
| **Mean offset (cents)** | Average signed deviation. Negative = flat, positive = sharp. | Reveals systematic bias. Crowds famously drift flat over time. |
| **Best transposition** | Semitone shift between MIDI's key and the crowd's actual singing key | Diagnostic. Confirms the alignment found a sensible match. |

**Reference benchmarks for the headline metric (notes within ±50¢):**

| Singer type | Approx. score |
|---|---|
| Professional opera singer | ~95% |
| Trained choir | ~80% |
| Karaoke night enthusiast | ~50% |
| Welsh fans, "Hen Wlad Fy Nhadau" | ~37% |
| Italian fans, "Il Canto degli Italiani" | ~31% |
| Unfocused crowd, half-singing along | ~15% |

---

## 6. Limitations and Risks

This section is critical for stakeholders. The pipeline produces real numbers, but they should be interpreted with the following caveats.

### 6.1 Source quality dominates everything

The single largest determinant of result quality is the YouTube source clip. A broadcast feed with heavy commentary or PA system bleed produces unusable separation; a fan-recorded clip from inside the stadium produces clean data. **Approximately 50% of project effort, for any new anthem, is spent finding a usable source clip.** Risk: results from low-quality clips look superficially valid but are dominated by separation artifacts.

**Mitigation in place:** A diagnostic script (`check_wav.py`) validates the vocal-stem peak/RMS levels after Stage 2, flagging clips where separation produced inadequate signal.

### 6.2 MIDI availability is patchy

Reference MIDIs exist for major nations' anthems but are sparse for smaller footballing nations. Quality also varies — some are clean melody-only transcriptions, others are dense orchestral arrangements where the melody track must be hunted for.

**Implication for project scope:** A full World Cup study (32 nations) is unlikely to be feasible due to MIDI availability alone. A focused study (6–12 anthems with reliable references) is realistic.

**Mitigation in place:** An auto-detector (`inspect_midi.py`) scores each MIDI track on melody-likelihood (monophony, vocal range, note density) and previews top candidates so the user can confirm visually.

### 6.3 Algorithmic measurement noise

The pipeline involves multiple approximations: pyin's pitch grid, Demucs' separation residuals, DTW's alignment choices. The combined measurement uncertainty has not been formally quantified. An informal estimate: results are likely accurate to ±3–5 percentage points on the headline metric, but this assumes a clean source clip.

**Risk:** Two anthems whose scores differ by less than ~5 points may not be distinguishable — the difference could be measurement noise rather than real performance variation.

**Recommended mitigation (not yet implemented):** Run the pipeline on multiple clips of the same anthem from different matches to empirically estimate clip-to-clip variance.

### 6.4 The "right note" problem

The current scoring asks: "did the crowd hit the *correct* note at this moment?" This is the right question. But the pipeline measures the *centroid* of the crowd's pitch distribution. If half the crowd sings the melody and half sings a harmony note (e.g., a fifth above), the centroid lands somewhere in between — neither correct.

This is a known limitation of analyzing collective singing rather than individual voices. Findings should be framed as measuring *how the typical crowd member sings*, not what any specific person sings.

### 6.5 Pyin confidence on crowd vocals

`pyin`'s internal confidence metric (`voiced_prob`) is unreliable on crowd vocals — typically reports ~1% confidence even on clearly-pitched singing. We use the binary `voiced_flag` instead. This works in practice but means we cannot use confidence-weighted statistics.

### 6.6 Copyright and ToS

YouTube downloads are technically against YouTube's terms of service. For academic research this is generally tolerated, but redistributing the downloaded audio is not advisable. The project should not publish raw audio files, only analytical results.

### 6.7 Cultural/contextual confounds

Differences between anthems are not purely about "in-tune-ness." Welsh and Italian crowds sing different songs, in different venues, at different match types, with different cultural norms around participation. Comparisons should be framed carefully:

> *"Welsh fans at Principality Stadium during Six Nations rugby sing more accurately than Italian fans at Euro 2016"*

is defensible. Stripping it down to "Welsh fans are better singers than Italian fans" is not.

---

## 7. What This Pipeline Can and Cannot Do

### Can do

- Quantify the average pitch accuracy of a crowd recording with reasonable precision
- Compare anthems against each other on a consistent methodology
- Detect systematic biases (flat drift, sharp tendency) in crowd singing
- Identify which sections of an anthem the crowd struggles with (frame-level data is preserved in the CSV)
- Scale to additional anthems with minimal code changes (config-driven)

### Cannot do

- Distinguish individual fans from the crowd centroid
- Measure rhythm or timing accuracy (only pitch)
- Assess emotional quality, volume, or "atmosphere"
- Reliably analyze clips where commentary or PA systems dominate the mix
- Provide absolute "ground truth" — all numbers are pipeline-relative comparisons

---

## 8. Reproducibility

Every result is reproducible from three inputs: the YouTube URL, the trim window, and the MIDI reference. These are stored in `anthems.py`. Re-running `python run_anthem.py <country>` and `python compare.py` regenerates the entire output from scratch.

The pipeline is deterministic given identical inputs. Demucs, pyin, and DTW have no randomness in their core algorithms. The only source of run-to-run variation is YouTube serving slightly different audio versions (rare but possible).

---

## 9. Current State and Next Steps

**Completed (as of this writing):**
- Full pipeline operational end-to-end
- Two anthems scored: Wales (37.3%) and Italy (31.1%)
- Comparison framework producing ranked scoreboards

**Recommended next steps before scaling:**
1. **Repeatability check** — run a second clip of the Welsh anthem to estimate clip-to-clip variance. Until variance is bounded, score differences cannot be confidently interpreted.
2. **Expand to 4–6 anthems** — sufficient sample for meaningful ranking. Suggested candidates: France (La Marseillaise), England (God Save the King), Argentina, Brazil.
3. **Document one detailed case study** — produce a richer per-anthem report (with section-level analysis) for one anthem as a reference for what the deeper analysis can reveal.

**Possible future extensions:**
- Section-level scoring (verse vs. chorus accuracy)
- Crowd-engagement metric (how much of the time does the crowd actually sing?)
- Intonation drift over time within a single anthem
- Comparison of the same anthem across different venues or eras

---

## Appendix — Project Files

| File | Purpose |
|---|---|
| `anthems.py` | Configuration registry of all anthems with URLs, trims, MIDI paths |
| `download.py` | Stage 1: YouTube → WAV |
| `separate.py` | Stage 2: WAV → isolated vocals |
| `analyze.py` | Stage 3: vocals → pitch CSV |
| `extract_melody.py` | Stage 4a: MIDI → reference pitch sequence |
| `align_score.py` | Stage 4b–d: DTW alignment + scoring |
| `compare.py` | Stage 5: cross-anthem scoreboard |
| `run_anthem.py` | End-to-end pipeline driver |
| `visualize.py` | Standalone per-anthem report generator |
| `inspect_midi.py` | Diagnostic: identify melody track in new MIDI files |
| `check_wav.py` | Diagnostic: validate audio file integrity |
