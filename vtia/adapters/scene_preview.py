"""Draws a camera point over a frame, to review a suggested (or hand-made) point by eye."""
from pathlib import Path

import cv2
import numpy as np

from vtia.domain.models import PointConfig

ROAD, WALK, LINE = (255, 160, 0), (0, 165, 255), (0, 255, 0)  # BGR: blue carriageways, orange walks
FONT = cv2.FONT_HERSHEY_SIMPLEX


def read_frame(video: Path, frame: int = 0) -> np.ndarray:
    """One BGR frame of the video, at full resolution (the pixels the camera-point YAML uses)."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read frame {frame} of {video}")
    return img


def draw_point(video: Path, point: PointConfig, out: Path, frame: int = 0) -> Path:
    img = read_frame(video, frame)
    thick = max(2, img.shape[1] // 640)
    overlay = img.copy()
    for z in point.zones:
        pts = np.array([[round(x), round(y)] for x, y in z.polygon], np.int32)
        color = WALK if z.name.startswith("peatonal") else ROAD
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, thick)
    img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
    for ln in point.lines:
        a, b = (round(ln.start[0]), round(ln.start[1])), (round(ln.end[0]), round(ln.end[1]))
        cv2.line(img, a, b, LINE, thick * 2)
        # Arrow toward the allowed side: the right-hand side walking start -> end (y grows down).
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = max(1.0, (dx * dx + dy * dy) ** 0.5)
        side = 1 if ln.allowed in (None, ln.to_right) else -1
        reach = min(0.25 * norm, 15 * thick) * side  # short: an arrow, not another line
        tip = (mid[0] - dy / norm * reach, mid[1] + dx / norm * reach)
        cv2.arrowedLine(img, (round(mid[0]), round(mid[1])), (round(tip[0]), round(tip[1])), LINE, thick, tipLength=0.3)
        _label(img, ln.name, b, thick)
    for z in point.zones:
        x = min(p[0] for p in z.polygon)
        y = min(p[1] for p in z.polygon)
        _label(img, z.name, (round(x), round(y)), thick)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return out


def _label(img, text: str, at: tuple[int, int], thick: int) -> None:
    scale = thick / 3
    (w, h), base = cv2.getTextSize(text, FONT, scale, thick)
    x = min(max(0, at[0]), img.shape[1] - w)
    y = min(max(h + base, at[1]), img.shape[0] - base)
    cv2.rectangle(img, (x, y - h - base), (x + w, y + base // 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y - base // 2), FONT, scale, (255, 255, 255), max(1, thick // 2), cv2.LINE_AA)
