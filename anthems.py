# anthems.py
"""Registry of all anthems being analyzed. Add new entries as you process them."""

ANTHEMS = {
    "wales": {
        "country": "Wales",
        "anthem_name": "Hen Wlad Fy Nhadau",
        "youtube_url": "https://www.youtube.com/watch?v=ZwdZOHm8r-Y",
        "video_id": "ZwdZOHm8r-Y",
        "trim_start": "00:00:10",
        "trim_end": "00:01:20",
        "midi_path": "data/reference/hen_wlad_fy_nhadau.mid",
        "melody_track": 1,
        "venue": "Principality Stadium",
        "context": "Six Nations rugby vs. England",
    },
    "italy": {
        "country": "Italy",
        "anthem_name": "Il Canto degli Italiani",
        "youtube_url": "https://www.youtube.com/watch?v=eRKiAiOa7pU",
        "video_id": "eRKiAiOa7pU",
        "trim_start": "00:00:19",
        "trim_end": "00:01:23",
        "midi_path": "data/reference/italy.mid",
        "melody_track": 2,
        "venue": "Euro 2016",
        "context": "Euro 2016",
    },
    "canada": {
    "country": "Canada",
    "anthem_name": "O Canada",
    "youtube_url": "https://www.youtube.com/watch?v=Rqd3y_QThNo",
    "video_id": "Rqd3y_QThNo",
    "trim_start": "00:00:16",
    "trim_end":   "00:01:19",
    "midi_path": "data/reference/canada.mid",
    "melody_track": 1,
    "venue": "Qatar (WC 2022)",
    "context": "FIFA World Cup 2022"
}}