# Anthem Rendition Analysis

Which nation sang its anthem best at this year's FIFA World Cup?

This pipeline takes the official anthem-ceremony broadcast clip for each
nation, isolates the singing, and scores the rendition on four measurable,
reference-free components. No sheet music or MIDI reference is required --
every measure is derived from the clip itself and validated against a
random-noise baseline.

## How the scoring works

Each clip is analyzed in ~3-second windows. Four components are measured per
window and aggregated with the median (robust to soloist/camera artifacts):

| Component | Question it answers | Range |
|---|---|---|
| `intonation` / `intonation_R` | How tightly does the pitch sit on stable notes, measured against the crowd's *own* tuning (tuning-invariant)? | % / 0-1 |
| `activity` | Is there real melodic movement, or a monotone drone? | bits |
| `stability` | Are notes held cleanly, or smeared? | 0-1 |
| `participation` | How much of the clip is confident, audible singing? | 0-1 |

No single component is trustworthy alone (a drone is "perfectly in tune"; noise
is "highly active") -- the composite requires several at once, which is what
isolates genuine singing. Every run prints each component **next to its
random-noise baseline**, so a score is always read relative to chance.

The final ranking (`aggregate_rank.py`) normalizes components across nations,
blends them with stated weights, and attaches a within-clip **bootstrap 95%
confidence interval** to every nation. Pairs whose intervals overlap are
explicitly reported as statistically indistinguishable.

## Quickstart

```bash
pip install -r requirements.txt
# ffmpeg must be installed (or set FFMPEG_PATH to its location)

# 1. Register a clip in anthems.py (URL, trim window, dataset="tournament")
# 2. Run the pipeline for it:
python run_anthem.py <anthem_key>            # e.g. python run_anthem.py wales
python run_anthem.py --all                   # process the whole registry
python run_anthem.py <key> --force           # recompute cached stages

# 3. Once all tournament clips are scored, build the ranking:
python aggregate_rank.py "data/results/*_windows.csv"
```

Pipeline stages (each cached): **download** (yt-dlp, trimmed to the anthem) ->
**separate** (Demucs `htdemucs_ft` vocal isolation; slow on CPU) -> **pitch**
(librosa `pyin` F0 track) -> **score** (`score_rendition.py`).

## Repository layout

```
anthems.py            clip registry; dataset = "tournament" | "calibration"
run_anthem.py         driver: download -> separate -> pitch -> score
download.py           YouTube acquisition (yt-dlp + ffmpeg)
separate.py           Demucs vocal isolation
analyze.py            pyin pitch extraction (descriptive grid stats only)
score_rendition.py    THE scorer: 4 components + noise baselines + QC flags
aggregate_rank.py     cross-nation ranking with bootstrap CIs
check_wav.py          diagnostic: audio sanity check
visualize.py          diagnostic: pitch-track plots
legacy/               retired MIDI-reference experiments (not in the pipeline)
data/raw, data/stems  audio (gitignored -- regenerable from anthems.py)
data/results/         pitch CSVs, per-window CSVs, plots (tracked)
```

## Methodology rules

- **One official clip per nation**, all from the same tournament, registered
  with `dataset: "tournament"`. Clips marked `"calibration"` were used to
  develop the method and are excluded from the ranking.
- Clips flagged by the soloist QC warning (a lone miked voice scores far
  cleaner than a crowd) must be manually reviewed before inclusion.
- Claims are scoped to what is measured: *"the most in-tune and committed
  rendition in the official ceremony clip"* -- not "the best singers" in any
  absolute sense.

## Known limitations

- Demucs and pyin are trained/designed for studio audio; stadium recordings
  are out-of-distribution, so the vocal stem is best-effort, not perfect.
- One clip per nation means per-broadcast luck (camera/mix choices) is part of
  the measurement; the bootstrap CI captures within-clip variability only.
- pyin tracks the dominant voice, roughly the loudest subgroup of the crowd,
  not the average of every singer.