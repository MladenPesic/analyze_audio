# inspect_midi.py
"""Inspect a MIDI file's tracks and visualize candidate melody tracks."""
import sys
import mido
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from extract_melody import extract_melody_track, melody_to_pitch_series


def analyze_track(track):
    """Return note count, pitch range, and overlap ratio for a track."""
    abs_tick, opens, notes = 0, {}, []
    for msg in track:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            opens[msg.note] = abs_tick
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in opens:
                notes.append((opens.pop(msg.note), abs_tick, msg.note))
    if not notes:
        return None

    pitches = [p for _, _, p in notes]
    overlaps = sum(
        1 for i, (s1, e1, _) in enumerate(notes)
        for s2, e2, _ in notes[i+1:]
        if s2 < e1 and e2 > s1
    )
    return {
        "n_notes": len(notes),
        "median": int(np.median(pitches)),
        "min": min(pitches),
        "max": max(pitches),
        "range": max(pitches) - min(pitches),
        "mono_pct": int(100 * (1 - min(overlaps / len(notes), 1.0))),
    }


def score_melody_likelihood(f):
    """Heuristic: monophonic + vocal range + narrow span = melody."""
    s = f["mono_pct"] * 0.3
    s += 20 if 55 <= f["median"] <= 80 else 0
    s += 15 if f["range"] <= 24 else (5 if f["range"] <= 36 else 0)
    s -= 10 if f["n_notes"] > 200 else (10 if f["n_notes"] < 20 else 0)
    return s


def inspect(midi_path: Path):
    mid = mido.MidiFile(str(midi_path))
    print(f"\n{midi_path.name}")
    print(f"Tracks: {len(mid.tracks)}, ticks/beat: {mid.ticks_per_beat}, "
          f"length: {mid.length:.1f}s\n")
    print(f"{'idx':>3} {'name':<25} {'notes':>6} {'med':>5} {'range':>8} {'mono':>5} {'score':>6}")
    print("-" * 70)

    candidates = []
    for i, track in enumerate(mid.tracks):
        feat = analyze_track(track)
        if feat is None:
            print(f"{i:>3} {track.name[:25]:<25} (no notes)")
            continue
        score = score_melody_likelihood(feat)
        candidates.append((i, score, feat))
        print(f"{i:>3} {track.name[:25]:<25} {feat['n_notes']:>6} "
              f"{feat['median']:>5} {feat['min']:>3}-{feat['max']:<3} "
              f"{feat['mono_pct']:>4}% {score:>6.1f}")

    if not candidates:
        print("\nNo notes found in any track.")
        return

    best_idx = max(candidates, key=lambda c: c[1])[0]
    print(f"\n→ Best guess for melody: Track {best_idx}")

    # Plot the top 3 candidates side by side for visual confirmation
    top3 = sorted(candidates, key=lambda c: -c[1])[:3]
    fig, axes = plt.subplots(len(top3), 1, figsize=(14, 3*len(top3)), squeeze=False)
    for ax, (idx, _, _) in zip(axes.flat, top3):
        notes = extract_melody_track(midi_path, track_index=idx)
        times, pitches = melody_to_pitch_series(notes)
        valid = ~np.isnan(pitches)
        ax.scatter(times[valid], pitches[valid], s=4, c="#1f77b4")
        ax.set_title(f"Track {idx}: {mid.tracks[idx].name or '(unnamed)'} "
                     f"— {(notes and len(notes)) or 0} notes")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("MIDI pitch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = Path(f"data/results/{midi_path.stem}_tracks.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"Saved track preview: {out}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_midi.py <path-to-midi>")
        sys.exit(1)
    inspect(Path(sys.argv[1]))