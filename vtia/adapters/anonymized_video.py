"""Applies the blur to the video pixels, and builds a contact sheet to check the result by eye."""
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np

from vtia.adapters.video_io import open_writer
from vtia.domain.models import Box


def blur_video(source: Path, out: Path, regions: Iterable[list[Box]],
               progress: Callable[[int], None] | None = None) -> int:
    """Writes `out` at the source resolution with every region blurred. No audio (voices are personal data too).
    `regions` must yield one list per frame, in order. Returns the frames written."""
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_writer(out, fps, size)
    frames = 0
    try:
        for boxes in regions:
            ok, img = cap.read()
            if not ok:
                break
            for b in boxes:
                _blur(img, b)
            writer.write(img)
            frames += 1
            if progress:
                progress(frames)
    finally:
        cap.release()
        writer.release()
    return frames


def contact_sheet(video: Path, out: Path, n: int = 12, width: int = 480) -> None:
    """n frames spread over the video, tiled 4 per row, for a quick human check of the anonymisation."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if ok:
            tile = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])))
            cv2.putText(tile, f"frame {i}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
    cap.release()
    if not tiles:
        return
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)  # pad the last row
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(out), np.vstack(rows))


def _blur(img: np.ndarray, box: Box) -> None:
    """Pixelate, then smooth: irreversible, and less jarring than black boxes."""
    h, w = img.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(box[2]) + 1), min(h, int(box[3]) + 1)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    small = cv2.resize(roi, (max(1, (x2 - x1) // 8), max(1, (y2 - y1) // 8)), interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(coarse, (0, 0), sigmaX=max(2.0, (x2 - x1) / 12))
