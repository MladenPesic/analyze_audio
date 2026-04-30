# visualize.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from pathlib import Path

def visualize(video_id: str, anthem_name: str = ""):
    """Generate the standard 3-panel report for a given analysis."""
    csv_path = Path(f"data/results/{video_id}_pitch.csv")
    df = pd.read_csv(csv_path)
    df_voiced = df.dropna(subset=["f0_hz"]).copy()

    if df_voiced.empty:
        print(f"No voiced frames in {csv_path}. Skipping.")
        return

    # Compute summary metrics
    cents = df_voiced["cents_deviation"].values
    metrics = {
        "voiced_pct": 100 * len(df_voiced) / len(df),
        "mean_cents": float(np.mean(cents)),
        "abs_mean_cents": float(np.mean(np.abs(cents))),
        "std_cents": float(np.std(cents)),
        "pct_within_25": float(np.mean(np.abs(cents) < 25) * 100),
    }

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1.5], hspace=0.45, wspace=0.25)

    title = anthem_name or video_id
    fig.suptitle(f"Crowd Pitch Analysis — {title}", fontsize=14, fontweight="bold")

    # 1. F0 over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(df_voiced["time_s"], df_voiced["f0_hz"], s=3, alpha=0.6, c="#1f77b4")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("F0 (Hz)")
    ax1.set_title("Dominant crowd pitch over time")
    ax1.grid(alpha=0.3)

    # 2. MIDI contour with note labels on y-axis
    ax2 = fig.add_subplot(gs[1, :])
    ax2.scatter(df_voiced["time_s"], df_voiced["midi"], s=3, alpha=0.5,
                c="#1f77b4", label="measured")
    ax2.scatter(df_voiced["time_s"], df_voiced["nearest_midi"], s=3, alpha=0.4,
                c="#d62728", label="nearest semitone")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Note")
    ax2.set_title("Melodic contour (notes the crowd sang)")

    # Convert MIDI ticks to note names for readability
    midi_min = int(np.floor(df_voiced["midi"].min())) - 1
    midi_max = int(np.ceil(df_voiced["midi"].max())) + 1
    ticks = list(range(midi_min, midi_max + 1, 2))
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([librosa.midi_to_note(m) for m in ticks])
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    # 3. Deviation histogram
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.hist(cents, bins=40, edgecolor="black", color="#1f77b4", alpha=0.8)
    ax3.axvline(0, color="green", linestyle="--", lw=2, label="perfect")
    ax3.axvline(metrics["mean_cents"], color="orange", linestyle="--", lw=2,
                label=f"mean = {metrics['mean_cents']:.1f}¢")
    ax3.set_xlabel("Deviation from nearest semitone (cents)")
    ax3.set_ylabel("Frame count")
    ax3.set_title("Pitch deviation distribution")
    ax3.legend(); ax3.grid(alpha=0.3)

    # 4. Metrics box
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis("off")
    text = (
        f"Voiced frames:         {metrics['voiced_pct']:.1f}%\n\n"
        f"Mean deviation:        {metrics['mean_cents']:+.1f}¢\n"
        f"Mean |deviation|:      {metrics['abs_mean_cents']:.1f}¢\n"
        f"Spread (std):          {metrics['std_cents']:.1f}¢\n\n"
        f"Within ±25¢:           {metrics['pct_within_25']:.1f}%"
    )
    ax4.text(0.05, 0.95, text, family="monospace", fontsize=11,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="gray"))
    ax4.set_title("Summary metrics")

    out_path = Path(f"data/results/{video_id}_report.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    visualize("ZwdZOHm8r-Y", anthem_name="Hen Wlad Fy Nhadau (Wales)")