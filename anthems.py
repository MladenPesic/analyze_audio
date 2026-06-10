# anthems.py
"""Registry of anthem clips.

Every clip the project touches is declared here. The `dataset` field enforces
the methodology:

    "tournament"   Official anthem clip from THIS YEAR'S FIFA World Cup.
                   Exactly one per nation. ONLY these enter the final ranking.

    "calibration"  Clips used while developing and validating the method
                   (any event, any year). Never included in the ranking.

Required fields per entry:
    country, anthem_name, youtube_url, video_id, trim_start, trim_end, dataset
Optional metadata:
    venue, context
"""

ANTHEMS = {
    # ------------------------------------------------------------------
    # CALIBRATION CLIPS (method development only -- excluded from ranking)
    # ------------------------------------------------------------------
    "wales": {
        "country": "Wales",
        "anthem_name": "Hen Wlad Fy Nhadau",
        "youtube_url": "https://www.youtube.com/watch?v=ZwdZOHm8r-Y",
        "video_id": "ZwdZOHm8r-Y",
        "trim_start": "00:00:10",
        "trim_end": "00:01:20",
        "venue": "Principality Stadium",
        "context": "Six Nations rugby vs. England",
        "dataset": "calibration",
    },
    "italy": {
        "country": "Italy",
        "anthem_name": "Il Canto degli Italiani",
        "youtube_url": "https://www.youtube.com/watch?v=eRKiAiOa7pU",
        "video_id": "eRKiAiOa7pU",
        "trim_start": "00:00:19",
        "trim_end": "00:01:23",
        "venue": "Euro 2016",
        "context": "Euro 2016",
        "dataset": "calibration",
    },
    "canada": {
        "country": "Canada",
        "anthem_name": "O Canada",
        "youtube_url": "https://www.youtube.com/watch?v=Rqd3y_QThNo",
        "video_id": "Rqd3y_QThNo",
        "trim_start": "00:00:16",
        "trim_end": "00:01:19",
        "venue": "Qatar (WC 2022)",
        "context": "FIFA World Cup 2022",
        "dataset": "calibration",
    },

    # ------------------------------------------------------------------
    # TOURNAMENT CLIPS (this year's FIFA World Cup -- the actual dataset)
    # Add all 48 nations here as clips are collected.
    # ------------------------------------------------------------------
}