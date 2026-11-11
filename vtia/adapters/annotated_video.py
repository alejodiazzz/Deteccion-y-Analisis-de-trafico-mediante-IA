"""Writes a copy of the input video with tracks, lines, zones, counts and alerts drawn on it.

Used as the per-frame observer of process_video: it reads the source video in
step with the pipeline, so nothing is held in memory.
"""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from vtia.adapters.video_io import open_writer
from vtia.domain.classes import RoadClass
from vtia.domain.models import Alert, CrossingEvent, Detection, PointConfig

COLORS = {  # BGR
    RoadClass.PEDESTRIAN: (0, 200, 255),
    RoadClass.BICYCLE: (255, 200, 0),
    RoadClass.MOTORCYCLE: (255, 0, 200),
    RoadClass.CAR: (0, 220, 0),
    RoadClass.BUS: (0, 140, 255),
    RoadClass.TRUCK: (60, 60, 255),
}
ZONE_COLOR = (255, 255, 0)
ALERT_COLOR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_WIDTH = 1920  # 4K input is written at 1080p: plenty to watch, a quarter of the size


class AnnotatedVideo:
    def __init__(self, source: Path, out: Path, point: PointConfig):
        self.cap = cv2.VideoCapture(str(source))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale = min(1.0, MAX_WIDTH / w)
        size = (round(w * self.scale), round(h * self.scale))
        self.writer = open_writer(out, fps, size)
        self.point = point
        self.counts: Counter = Counter()      # (line, direction) -> crossings so far
        self.flash: dict[str, int] = {}       # line -> frame until which it is highlighted
        self.flash_frames = round(fps / 3)
        self.recent: list[tuple[int, Alert]] = []  # (frame until which it is shown, alert)
        self.alert_frames = round(fps * 3)
        self.thick = max(2, round(h / 540))   # line/box thickness and font size follow resolution
        self.font = h / 1400

    def __call__(self, frame_index: int, tracked: list[Detection], events: list[CrossingEvent],
                 alerts: list[Alert]) -> None:
        ok, img = self.cap.read()
        if not ok:
            return
        for e in events:
            self.counts[(e.line, e.direction)] += 1
            self.flash[e.line] = frame_index + self.flash_frames
        self.recent = [(until, a) for until, a in self.recent if until >= frame_index]
        self.recent += [(frame_index + self.alert_frames, a) for a in alerts]
        alerted = {a.track_id for _, a in self.recent}

        for zone in self.point.zones:
            pts = np.array(zone.polygon, dtype=np.int32)
            cv2.polylines(img, [pts], True, ZONE_COLOR, self.thick)
            self._label(img, zone.name, tuple(pts[0]), ZONE_COLOR)
        for line in self.point.lines:
            lit = self.flash.get(line.name, -1) >= frame_index
            cv2.line(img, _pt(line.start), _pt(line.end), (0, 255, 0) if lit else (0, 0, 255), self.thick * (3 if lit else 2))
        for d in tracked:
            color = ALERT_COLOR if d.track_id in alerted else COLORS[d.road_class]
            x1, y1, x2, y2 = map(round, d.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, self.thick * (2 if d.track_id in alerted else 1))
            cv2.circle(img, _pt(d.anchor), self.thick * 2, color, -1)  # the point that is counted
            self._label(img, f"{d.road_class} #{d.track_id}", (x1, y1), color)
        self._panel(img)
        if self.scale < 1:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self.writer.write(img)

    def close(self) -> None:
        self.cap.release()
        self.writer.release()

    def _panel(self, img) -> None:
        rows = [(f"{line.name} {direction}: {self.counts[(line.name, direction)]}", (255, 255, 255))
                for line in self.point.lines for direction in (line.to_right, line.to_left)]
        rows += [(f"ALERT {a.kind}: {a.detail}", (80, 80, 255)) for _, a in self.recent[-4:]]
        scale = self.font * 1.2
        (w, h), _ = cv2.getTextSize(max((r for r, _ in rows), key=len), FONT, scale, self.thick)
        step = round(h * 1.8)
        cv2.rectangle(img, (0, 0), (w + step, step * len(rows) + step // 2), (0, 0, 0), -1)
        for i, (row, color) in enumerate(rows):
            cv2.putText(img, row, (step // 2, step * (i + 1)), FONT, scale, color, self.thick, cv2.LINE_AA)

    def _label(self, img, text: str, corner: tuple[int, int], color) -> None:
        """Text on a filled tag in the given colour, sitting on top of `corner`."""
        scale = self.font * 1.1
        (w, h), base = cv2.getTextSize(text, FONT, scale, self.thick)
        x, y = int(corner[0]), int(corner[1])
        cv2.rectangle(img, (x, y - h - base), (x + w, y), color, -1)
        cv2.putText(img, text, (x, y - base // 2), FONT, scale, (0, 0, 0), self.thick, cv2.LINE_AA)


def _pt(p) -> tuple[int, int]:
    return round(p[0]), round(p[1])
