"""ByteTrack or BoT-SORT from Roboflow's `trackers` package (same interface for both).

The same tracker runs after every detector so detector comparisons stay fair.
"""
from collections.abc import Iterable, Iterator
from dataclasses import replace

import numpy as np
import supervision as sv
from trackers import BoTSORTTracker, ByteTrackTracker

from vtia.domain.models import Detection

ALGORITHMS = ("bytetrack", "botsort")


class MotTracker:
    def __init__(self, algorithm: str = "bytetrack", track_thresh: float = 0.6, min_conf: float = 0.1):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"unknown tracker {algorithm!r}, expected one of {ALGORITHMS}")
        self.algorithm = algorithm
        # Calibration knob, parameterised as in the ByteTrack paper: boxes >= track_thresh are
        # "high confidence"; a new track needs >= track_thresh + 0.1. 0.6 = library defaults.
        self.track_thresh = track_thresh
        self.min_conf = min_conf  # low-score boxes still help keep existing tracks alive

    def _new(self, frame_rate: float):
        kwargs = dict(frame_rate=frame_rate, high_conf_det_threshold=self.track_thresh,
                      track_activation_threshold=self.track_thresh + 0.1)
        if self.algorithm == "botsort":
            # Fixed cameras: no camera-motion compensation (it would also need the frames).
            return BoTSORTTracker(enable_cmc=False, **kwargs)
        return ByteTrackTracker(**kwargs)

    def track(self, frames: Iterable[list[Detection]], frame_rate: float = 30.0) -> Iterator[list[Detection]]:
        tracker = self._new(frame_rate)  # class-agnostic: class flicker can't break a track
        for frame in frames:
            dets = [d for d in frame if d.confidence >= self.min_conf]
            out = tracker.update(sv.Detections(
                xyxy=np.array([d.box for d in dets], dtype=float).reshape(-1, 4),
                confidence=np.array([d.confidence for d in dets], dtype=float),
                class_id=np.arange(len(dets)),  # carries the index back to our Detection
            ))
            if len(out) == 0:
                yield []
                continue
            yield [replace(dets[i], track_id=int(t)) for i, t in zip(out.class_id, out.tracker_id) if t != -1]
