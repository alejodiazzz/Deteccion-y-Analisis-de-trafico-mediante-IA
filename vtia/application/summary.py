"""Use case: aggregate a VideoAnalysis into the figures a mobility report shows."""
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from statistics import mean

from vtia.application.process_video import VideoAnalysis
from vtia.domain.classes import RoadClass
from vtia.domain.counting import count_by
from vtia.domain.timeline import clock


@dataclass
class Summary:
    duration_s: float
    start: datetime | None
    total: int                                        # line crossings
    by_class: list[tuple[RoadClass, int, float]]      # class, count, share of total (largest first)
    by_line: list[tuple[str, str, RoadClass, int]]    # line, direction, class, count
    interval_s: int
    by_interval: list[tuple[str, int]]                # interval label (clock or elapsed), crossings
    peak: tuple[str, int] | None                      # busiest interval
    dwell: list[tuple[str, int, float, float]]        # zone, visits, mean seconds, max seconds
    alerts_by_kind: list[tuple[str, int]]
    tracks: int
    mean_confidence: float


def auto_interval(duration_s: float) -> int:
    """Bins that give a readable number of bars: 10 s for clips, 1 min up to half an hour, then 15 min."""
    return 10 if duration_s <= 120 else 60 if duration_s <= 1800 else 900


def summarize(a: VideoAnalysis, interval_s: int | None = None) -> Summary:
    duration = a.frames / a.fps
    interval = interval_s or auto_interval(duration)
    total = len(a.events)
    classes = Counter(e.road_class for e in a.events)

    bins = Counter(int(e.frame / a.fps // interval) for e in a.events)
    n_bins = max(1, -(-int(duration) // interval))  # ceil, at least one bin
    by_interval = [(_label(a, i * interval, interval), bins[i]) for i in range(n_bins)]
    peak = max(by_interval, key=lambda b: b[1]) if total else None

    visits_by_zone: dict[str, list[float]] = {}
    for v in a.visits:
        visits_by_zone.setdefault(v.zone, []).append((v.exit_frame - v.enter_frame + 1) / a.fps)

    return Summary(
        duration_s=duration,
        start=a.start,
        total=total,
        by_class=[(c, n, n / total) for c, n in classes.most_common()],
        by_line=[(*key, n) for key, n in sorted(count_by(a.events).items())],
        interval_s=interval,
        by_interval=by_interval,
        peak=peak,
        dwell=[(zone, len(s), mean(s), max(s)) for zone, s in sorted(visits_by_zone.items())],
        alerts_by_kind=Counter(x.kind for x in a.alerts).most_common(),
        tracks=a.tracks,
        mean_confidence=a.mean_confidence,
    )


def _label(a: VideoAnalysis, seconds: int, interval: int) -> str:
    """Wall-clock time when known (HH:MM:SS for bins under a minute), else elapsed m:ss."""
    if not a.start:
        return f"{seconds // 60}:{seconds % 60:02d}"
    t = clock(a.start, round(seconds * a.fps), a.fps)
    return f"{t:%H:%M:%S}" if interval < 60 else f"{t:%H:%M}"
