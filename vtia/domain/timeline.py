"""Wall-clock time of a video, from the file-naming convention of the capture protocol.

    P<point>_<YYYYMMDD>_<HHMM>.mp4          session starting at HH:MM
    P<point>_<YYYYMMDD>_<HHMM>_s<SSSS>.mp4  clip cut SSSS seconds into that session
"""
import re
from datetime import datetime, timedelta

_NAME = re.compile(r"^P[^_]+_(\d{8})_(\d{4})(?:_s(\d+))?$")


def video_start(stem: str) -> datetime | None:
    """Local start time encoded in a file name (without extension), or None if it doesn't follow the convention."""
    m = _NAME.match(stem)
    if not m:
        return None
    try:
        start = datetime.strptime(m[1] + m[2], "%Y%m%d%H%M")
    except ValueError:  # e.g. month 13
        return None
    return start + timedelta(seconds=int(m[3] or 0))


def clock(start: datetime, frame: int, fps: float) -> datetime:
    return start + timedelta(seconds=frame / fps)
