# align_score_v2.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
from collections import defaultdict
from extract_melody import extract_melody_track, melody_to_pitch_series


def dtw_align(crowd_seq, ref_seq):
    cost = np.abs(crowd_seq[:, None] - ref_seq[None, :]).astype(np.float64)
    D, wp = librosa.sequence.dtw(C=cost, subseq=False, backtrack=True)
    return wp[::-1], D[-1, -1]


def search_best_transposition(crowd_seq, ref_seq, range_semitones=12):
    results = []
    for shift in range(-range_semitones, range_semitones + 1):
        path, cost = dtw_align(crowd_seq, ref_seq + shift)
        results.append((shift, cost / len(path), path))
        print(f"  transposition {shift:+d}: normalized cost = {cost / len(path):.3f}")

    # Find the minimum cost
    min_cost = min(r[1] for r in results)

    # All transpositions within 5% of the best are considered "tied"
    near_best = [r for r in results if r[1] <= min_cost * 1.05]

    # Among ties, prefer the one closest to 0 (canonical key)
    return min(near_best, key=lambda r: abs(r[0]))


def wrap_to_octave(cents):
    """Reduce deviations modulo 1200¢, mapping to [-600, +600]."""
    return ((np.asarray(cents) + 600) % 1200) - 600


def score_per_reference_note(crowd_seq, ref_seq_shifted, path):
    """
    For each reference frame, gather all crowd frames DTW matched to it,
    take the median crowd pitch, compute one deviation.
    Then collapse consecutive identical reference frames into note-level scores.
    """
    # Group: ref_idx -> list of crowd pitches matched to it
    matches = defaultdict(list)
    for ci, ri in path:
        matches[ri].append(crowd_seq[ci])

    # Per-frame deviations
    frame_devs = []
    for ri, pitches in matches.items():
        crowd_median = np.median(pitches)
        ref_pitch = ref_seq_shifted[ri]
        frame_devs.append((ri, (crowd_median - ref_pitch) * 100.0))

    frame_devs.sort()
    # Collapse runs of identical reference pitch into single note events
    note_devs = []
    last_pitch = None
    accum = []
    for ri, dev in frame_devs:
        p = ref_seq_shifted[ri]
        if last_pitch is None or p == last_pitch:
            accum.append(dev)
        else:
            note_devs.append(np.median(accum))
            accum = [dev]
        last_pitch = p
    if accum:
        note_devs.append(np.median(accum))
    return np.array(note_devs)


def main(video_id, midi_path, melody_track=1):
    crowd_df = pd.read_csv(f"data/results/{video_id}_pitch.csv")
    crowd_times_full = crowd_df["time_s"].values
    crowd_midi_full = crowd_df["midi"].values
    crowd_fps = 1.0 / np.median(np.diff(crowd_times_full))

    crowd_valid = ~np.isnan(crowd_midi_full)
    crowd_seq = crowd_midi_full[crowd_valid]
    crowd_times = crowd_times_full[crowd_valid]
    print(f"Crowd: {len(crowd_seq)} valid frames at {crowd_fps:.2f} fps")

    notes = extract_melody_track(midi_path, track_index=melody_track)
    _, ref_full = melody_to_pitch_series(notes, sr_frames_per_sec=crowd_fps)
    ref_valid = ~np.isnan(ref_full)
    ref_seq = ref_full[ref_valid]
    print(f"Reference: {len(ref_seq)} valid frames\n")

    print("Searching for best key alignment...")
    best_shift, best_cost, best_path = search_best_transposition(
        crowd_seq, ref_seq, range_semitones=12
    )
    print(f"\n→ Best transposition: {best_shift:+d} semitones (cost {best_cost:.3f})")

    ref_shifted = ref_seq + best_shift

    # --- Frame-level deviations (raw and octave-wrapped) ---
    frame_devs_raw = np.array([
        (crowd_seq[ci] - ref_shifted[ri]) * 100.0 for ci, ri in best_path
    ])
    frame_devs_wrapped = wrap_to_octave(frame_devs_raw)

    # --- Note-level deviations (octave-wrapped) ---
    note_devs = score_per_reference_note(crowd_seq, ref_shifted, best_path)
    note_devs_wrapped = wrap_to_octave(note_devs)

    print(f"\n=== Scoring ===")
    print(f"\nFrame-level (raw, octave-naive):")
    print(f"  Median |deviation|:        {np.median(np.abs(frame_devs_raw)):.1f}¢")
    print(f"  Within ±50¢:               {np.mean(np.abs(frame_devs_raw)<50)*100:.1f}%")

    print(f"\nFrame-level (octave-wrapped — fairer):")
    print(f"  Median |deviation|:        {np.median(np.abs(frame_devs_wrapped)):.1f}¢")
    print(f"  Within ±50¢:               {np.mean(np.abs(frame_devs_wrapped)<50)*100:.1f}%")
    print(f"  Within ±100¢:              {np.mean(np.abs(frame_devs_wrapped)<100)*100:.1f}%")

    print(f"\nNote-level (one score per reference note, octave-wrapped):")
    print(f"  N notes:                   {len(note_devs_wrapped)}")
    print(f"  Median |deviation|:        {np.median(np.abs(note_devs_wrapped)):.1f}¢")
    print(f"  Mean |deviation|:          {np.mean(np.abs(note_devs_wrapped)):.1f}¢")
    print(f"  Mean offset (sharp/flat):  {np.mean(note_devs_wrapped):+.1f}¢")
    print(f"  Notes within ±50¢:         {np.mean(np.abs(note_devs_wrapped)<50)*100:.1f}%")
    print(f"  Notes within ±100¢:        {np.mean(np.abs(note_devs_wrapped)<100)*100:.1f}%")

    # --- Plots ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    crowd_plot_t = [crowd_times[ci] for ci, _ in best_path]
    crowd_plot_p = [crowd_seq[ci] for ci, _ in best_path]
    ref_plot_p = [ref_shifted[ri] for _, ri in best_path]
    axes[0].scatter(crowd_plot_t, crowd_plot_p, s=4, alpha=0.5,
                    label="crowd (sang)", color="#1f77b4")
    axes[0].scatter(crowd_plot_t, ref_plot_p, s=4, alpha=0.5,
                    label=f"reference, transposed {best_shift:+d}",
                    color="#d62728")
    axes[0].set(xlabel="Time (s)", ylabel="MIDI pitch",
                title="Crowd vs. time-warped intended melody")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].hist(frame_devs_wrapped, bins=60, range=(-600, 600),
                 edgecolor="black", color="#1f77b4", alpha=0.8)
    axes[1].axvline(0, color="green", ls="--", lw=2, label="correct")
    axes[1].axvline(-50, color="gray", ls=":", lw=1)
    axes[1].axvline(50, color="gray", ls=":", lw=1, label="±50¢")
    axes[1].set(xlabel="Cents (octave-wrapped)", ylabel="Frame count",
                title="Frame-level deviation (every DTW match counts)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].hist(note_devs_wrapped, bins=40, range=(-600, 600),
                 edgecolor="black", color="#2ca02c", alpha=0.8)
    axes[2].axvline(0, color="green", ls="--", lw=2, label="correct")
    axes[2].axvline(-50, color="gray", ls=":", lw=1)
    axes[2].axvline(50, color="gray", ls=":", lw=1, label="±50¢")
    axes[2].set(xlabel="Cents (octave-wrapped)", ylabel="Note count",
                title="Note-level deviation (one score per intended note — fairer)")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path(f"data/results/{video_id}_aligned.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main(
        video_id="ZwdZOHm8r-Y",
        midi_path=Path("data/reference/hen_wlad_fy_nhadau.mid"),
        melody_track=1,
    )