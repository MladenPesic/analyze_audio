# extract_melody.py
import mido
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_melody_track(midi_path: Path, track_index: int = 1):
    """
    Extract a list of (start_time_seconds, duration_seconds, midi_pitch) tuples
    from a single MIDI track.
    """
    mid = mido.MidiFile(str(midi_path))
    track = mid.tracks[track_index]

    # Build tempo map — tempo changes are in track 0 typically, but mido can
    # iterate the whole file and give us seconds directly.
    # Easiest: use mido.MidiFile.__iter__ which yields msgs with absolute time
    # in seconds, but only for the merged file. So we walk the chosen track manually.

    # First, build a tempo map from the meta track (track 0)
    tempo_changes = []  # list of (tick, tempo_us_per_beat)
    abs_tick = 0
    for msg in mid.tracks[0]:
        abs_tick += msg.time
        if msg.type == "set_tempo":
            tempo_changes.append((abs_tick, msg.tempo))
    if not tempo_changes:
        tempo_changes = [(0, 500000)]  # default 120 BPM

    def tick_to_seconds(target_tick: int) -> float:
        seconds = 0.0
        last_tick = 0
        last_tempo = tempo_changes[0][1]
        for tick, tempo in tempo_changes:
            if tick >= target_tick:
                break
            seconds += mido.tick2second(tick - last_tick, mid.ticks_per_beat, last_tempo)
            last_tick = tick
            last_tempo = tempo
        seconds += mido.tick2second(target_tick - last_tick, mid.ticks_per_beat, last_tempo)
        return seconds

    # Walk the chosen track, tracking note_on / note_off pairs
    notes = []
    open_notes = {}  # pitch -> start_tick
    abs_tick = 0
    for msg in track:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            open_notes[msg.note] = abs_tick
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in open_notes:
                start_tick = open_notes.pop(msg.note)
                start_s = tick_to_seconds(start_tick)
                end_s = tick_to_seconds(abs_tick)
                notes.append((start_s, end_s - start_s, msg.note))
    notes.sort()
    return notes


def melody_to_pitch_series(notes, sr_frames_per_sec: float = 86.13):
    """
    Convert (start, duration, pitch) tuples into a frame-by-frame MIDI pitch
    array, sampled at the same frame rate librosa uses.

    librosa.pyin with frame_length=2048 at sr=22050 → hop=512 → ~43.07 fps default,
    but pyin's default returns 1 frame per ~256 samples, so check times_like output.
    Default for our pipeline: sr=22050, hop_length=512 → 22050/512 ≈ 43.07 fps.
    """
    if not notes:
        return np.array([]), np.array([])

    total_duration = max(start + dur for start, dur, _ in notes)
    n_frames = int(np.ceil(total_duration * sr_frames_per_sec))
    times = np.arange(n_frames) / sr_frames_per_sec
    pitches = np.full(n_frames, np.nan)

    for start, dur, pitch in notes:
        i_start = int(start * sr_frames_per_sec)
        i_end   = int((start + dur) * sr_frames_per_sec)
        pitches[i_start:i_end] = pitch
    return times, pitches


if __name__ == "__main__":
    midi_path = Path("data/reference/hen_wlad_fy_nhadau.mid")  # adjust if needed

    notes = extract_melody_track(midi_path, track_index=1)
    print(f"Extracted {len(notes)} notes from track 1")
    print(f"First 5 notes (start_s, dur_s, midi):")
    for n in notes[:5]:
        print(f"  start={n[0]:6.2f}s  dur={n[1]:5.2f}s  midi={n[2]}")
    print(f"Last note ends at: {notes[-1][0] + notes[-1][1]:.2f}s")

    times, pitches = melody_to_pitch_series(notes)
    valid = ~np.isnan(pitches)
    print(f"\nPitch series: {len(pitches)} frames, {valid.sum()} with notes")

    # Plot for visual sanity check
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(times[valid], pitches[valid], s=4)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("MIDI pitch")
    ax.set_title("Reference melody — Track 1 of MIDI file")
    ax.grid(alpha=0.3)
    out_path = Path("data/results/reference_melody.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.show()