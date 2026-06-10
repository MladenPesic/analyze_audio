# extract_melody.py
"""
Build a REFERENCE PITCH SERIES for an anthem.

A "reference" in this project is the triple (times, pitches, note_ids):
    times    : float seconds, one value per frame, on the SAME frame grid as
               the crowd recording produced by analyze.py
    pitches  : MIDI pitch sounding at that frame, or NaN during rests
    note_ids : an integer that is CONSTANT within one note and CHANGES at every
               note boundary -- including between two consecutive notes that
               happen to share the same pitch -- or NaN during rests

Why note_ids matters
--------------------
The scorer collapses per-frame deviations into one score per note. The old code
decided "is this still the same note?" by comparing pitch *values*, so repeated
notes (and same-pitch notes on either side of a rest) were silently merged.
On the real references this merged ~20% of Welsh notes and ~30% of Italian
notes -- and because the loss differs per anthem, it biased the comparison
between them. Collapsing by note_id instead of pitch value removes that bug.

Reference sources
-----------------
Today the reference comes from a MIDI file (reference_from_midi). A future
reference_from_audio() -- built for nations with no MIDI -- will return the
SAME triple from a clean recording, so nothing downstream has to change.
"""
import mido
import numpy as np

# Pitch frame grid. MUST stay in sync with analyze.py.
# librosa.pyin defaults hop_length = frame_length // 4 = 2048 // 4 = 512,
# so at SR = 22050 the frame rate is 22050 / 512 = 43.07 fps  (NOT 86.13).
SR = 22050
HOP = 512
FRAMES_PER_SEC = SR / HOP


def extract_melody_track(midi_path, track_index=1):
    """
    Return a list of (start_seconds, duration_seconds, midi_pitch) tuples for a
    single MIDI track, sorted by start time. (This logic was already correct.)
    """
    mid = mido.MidiFile(str(midi_path))

    # Build a tempo map from the meta track (track 0) so ticks -> real seconds.
    tempo_changes = []  # (abs_tick, microseconds_per_beat)
    abs_tick = 0
    for msg in mid.tracks[0]:
        abs_tick += msg.time
        if msg.type == "set_tempo":
            tempo_changes.append((abs_tick, msg.tempo))
    if not tempo_changes:
        tempo_changes = [(0, 500000)]  # default 120 BPM

    def tick_to_seconds(target_tick):
        seconds, last_tick, last_tempo = 0.0, 0, tempo_changes[0][1]
        for tick, tempo in tempo_changes:
            if tick >= target_tick:
                break
            seconds += mido.tick2second(tick - last_tick, mid.ticks_per_beat, last_tempo)
            last_tick, last_tempo = tick, tempo
        seconds += mido.tick2second(target_tick - last_tick, mid.ticks_per_beat, last_tempo)
        return seconds

    notes, open_notes, abs_tick = [], {}, 0
    for msg in mid.tracks[track_index]:
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


def melody_to_pitch_series(notes, fps=FRAMES_PER_SEC):
    """
    Render (start, duration, pitch) notes onto a frame grid at `fps`.

    Returns (times, pitches, note_ids). pitches and note_ids are NaN during
    rests. Each note receives a unique integer id (its order in the melody) so
    two consecutive notes of the same pitch remain distinguishable.

    Changes from the original:
      * fps default is the correct 43.07 (was a misleading 86.13)
      * uses round() and guarantees >= 1 frame per note (short notes never vanish)
      * also returns note_ids
    """
    if not notes:
        return np.array([]), np.array([]), np.array([])

    total_duration = max(start + dur for start, dur, _ in notes)
    n_frames = int(np.ceil(total_duration * fps))
    times = np.arange(n_frames) / fps
    pitches = np.full(n_frames, np.nan)
    note_ids = np.full(n_frames, np.nan)

    for idx, (start, dur, pitch) in enumerate(notes):
        i0 = int(round(start * fps))
        i1 = int(round((start + dur) * fps))
        i1 = max(i1, i0 + 1)      # never collapse a short note to zero frames
        i1 = min(i1, n_frames)    # stay in bounds
        pitches[i0:i1] = pitch
        note_ids[i0:i1] = idx

    return times, pitches, note_ids


def reference_from_midi(midi_path, track_index=1, fps=FRAMES_PER_SEC):
    """
    Adapter: MIDI file -> standard reference triple (times, pitches, note_ids).

    This is the seam the rest of the pipeline depends on. To support nations
    with no MIDI, add a reference_from_audio() that returns the same triple;
    the scorer will not need to change.
    """
    notes = extract_melody_track(midi_path, track_index)
    return melody_to_pitch_series(notes, fps=fps)


if __name__ == "__main__":
    notes = extract_melody_track("data/reference/canada.mid", track_index=1)
    times, pitches, note_ids = melody_to_pitch_series(notes)
    distinct = len(np.unique(note_ids[~np.isnan(note_ids)]))
    print(f"{len(notes)} notes -> {distinct} distinct note_ids preserved")
    print(f"grid: {FRAMES_PER_SEC:.2f} fps, {len(times)} frames, {times[-1]:.1f}s")


    # Plot for visual sanity check
    # fig, ax = plt.subplots(figsize=(14, 4))
    # ax.scatter(times[valid], pitches[valid], s=4)
    # ax.set_xlabel("Time (s)"); ax.set_ylabel("MIDI pitch")
    # ax.set_title("Reference melody — Track 1 of MIDI file")
    # ax.grid(alpha=0.3)
    # out_path = Path("data/results/reference_melody.png")
    # plt.savefig(out_path, dpi=120, bbox_inches="tight")
    # print(f"\nSaved: {out_path}")
    # plt.show()