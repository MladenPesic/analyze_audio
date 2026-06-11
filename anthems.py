# anthems.py
"""Registry of anthem clips.

Each entry is a plain dict with four fields:
    country, url, start, end          (start/end as "HH:MM:SS")

Paste a CLEAN YouTube URL (no &list=... playlist junk).

dataset: top-level sets which group a clip belongs to. Only clips under
TOURNAMENT enter the final ranking; CALIBRATION clips are for method
development and are excluded by aggregate_rank.py.
"""

CALIBRATION = {
    "wales":  {"country": "Wales",  "url": "https://www.youtube.com/watch?v=ZwdZOHm8r-Y", "start": "00:00:10", "end": "00:01:20"},
    "italy":  {"country": "Italy",  "url": "https://www.youtube.com/watch?v=eRKiAiOa7pU", "start": "00:00:19", "end": "00:01:23"},
    "canada": {"country": "Canada", "url": "https://www.youtube.com/watch?v=Rqd3y_QThNo", "start": "00:00:16", "end": "00:01:19"},

    # WC 2022 validation batch -- rename keys/countries and set trims after watching:
    "country1": {"country": "Ecuador", "url": "https://www.youtube.com/watch?v=k0-jutsp2Zk", "start": "00:00:00", "end": "00:01:22"},
    "country2": {"country": "Qatar", "url": "https://www.youtube.com/watch?v=qoBoU0IPaBc", "start": "00:00:00", "end": "00:01:24"},
    "country3": {"country": "England", "url": "https://www.youtube.com/watch?v=6jpkzoKVktM", "start": "00:00:00", "end": "00:00:45"},
    "country4": {"country": "Iran", "url": "https://www.youtube.com/watch?v=cwhwtKxg1Ds", "start": "00:00:00", "end": "00:00:55"},
    "country5": {"country": "Senegal", "url": "https://www.youtube.com/watch?v=6gIzfPgYah4", "start": "00:00:00", "end": "00:00:56"},
    "country6": {"country": "Ecuador2", "url": "https://www.youtube.com/watch?v=IWoiVQ_jd5w", "start": "00:00:46", "end": "00:01:37"}
}

TOURNAMENT = {
    # 2026 FIFA World Cup -- the real 48-nation dataset. Add entries like:
    # "brazil": {"country": "Brazil", "url": "https://www.youtube.com/watch?v=XXXX", "start": "00:00:12", "end": "00:01:25"},
}

# Merge into one registry, tagging each with its dataset.
ANTHEMS = {}
for _key, _e in CALIBRATION.items():
    ANTHEMS[_key] = {**_e, "dataset": "calibration"}
for _key, _e in TOURNAMENT.items():
    ANTHEMS[_key] = {**_e, "dataset": "tournament"}