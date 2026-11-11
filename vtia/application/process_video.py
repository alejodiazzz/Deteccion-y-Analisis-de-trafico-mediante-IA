"""Use case: analyse a video from one camera point (counts, zone visits, alerts)."""
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from vtia.application.ports import Detector, Tracker
from vtia.domain.alerts import AlertRules
from vtia.domain.counting import LineCounter
from vtia.domain.models import Alert, CrossingEvent, Detection, PointConfig, Visit
from vtia.domain.zones import ZoneMonitor

# Called once per frame with (frame index, tracked detections, crossings, alerts), e.g. to draw them.
FrameObserver = Callable[[int, list[Detection], list[CrossingEvent], list[Alert]], None]


@dataclass
class VideoAnalysis:
    events: list[CrossingEvent]
    visits: list[Visit]
    alerts: list[Alert]
    frames: int
    fps: float
    start: datetime | None   # wall-clock start, if known
    tracks: int              # distinct tracked objects
    mean_confidence: float   # over every tracked box: a rough quality signal for the report


def process_video(source: Path, detector: Detector, tracker: Tracker, point: PointConfig, fps: float,
                  start: datetime | None = None, on_frame: FrameObserver | None = None) -> VideoAnalysis:
    counter = LineCounter(list(point.lines))
    zones = ZoneMonitor(point.zones, gap_frames=round(fps))  # 1 s unseen inside a zone = it left
    rules = AlertRules(point, fps, start)
    events: list[CrossingEvent] = []
    alerts: list[Alert] = []
    track_ids: set[int] = set()
    confidence, boxes, frames = 0.0, 0, 0
    for frame, tracked in enumerate(tracker.track(detector.detect(source), fps)):
        crossings = counter.update(tracked)
        zones.update(frame, tracked)
        new_alerts = rules.update(frame, crossings, zones)
        events += crossings
        alerts += new_alerts
        track_ids.update(d.track_id for d in tracked)
        confidence += sum(d.confidence for d in tracked)
        boxes += len(tracked)
        if on_frame:
            on_frame(frame, tracked, crossings, new_alerts)
        frames = frame + 1
    return VideoAnalysis(events, zones.finish(), alerts, frames, fps, start, len(track_ids),
                         confidence / boxes if boxes else 0.0)
