"""Use case: propose a camera point from the first minutes of a video."""
from collections.abc import Callable, Iterator
from itertools import islice
from pathlib import Path

from vtia.application.ports import Detector, Tracker
from vtia.domain.models import Detection, PointConfig
from vtia.domain.scene import suggest_point


def suggest_from_video(source: Path, detector: Detector, tracker: Tracker, fps: float, seconds: float,
                       on_frame: Callable[[int], None] | None = None) -> PointConfig:
    """Tracks the first `seconds` of `source` and proposes lines and zones from the trajectories."""
    frames = islice(detector.detect(source), max(1, round(fps * seconds)))
    return suggest_point(_report(tracker.track(frames, fps), on_frame), fps)


def _report(frames, on_frame) -> Iterator[list[Detection]]:
    for i, frame in enumerate(frames):
        if on_frame:
            on_frame(i + 1)
        yield frame
