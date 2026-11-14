"""Use case: where to blur, frame by frame, to anonymise a video before it leaves the server PC."""
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from vtia.application.ports import Detector, PrivateFinder
from vtia.domain.evaluation import iou
from vtia.domain.models import Box, Detection
from vtia.domain.privacy import Found, cabin_search, plate_search, private_regions

REUSE_IOU = 0.3  # a vehicle this similar to one seen at the last search is taken as the same one


def regions_per_frame(source: Path, detector: Detector, hold_frames: int = 3, *,
                      finder: PrivateFinder | None = None, frame_at: Callable[[int], Any] | None = None,
                      refresh_every: int = 4) -> Iterator[list[Box]]:
    """Private regions of each frame, plus those of the previous `hold_frames` frames: a person or
    vehicle the detector misses for a frame or two stays blurred instead of flashing into view.

    With a `finder` (plate/face models), near vehicles are searched every `refresh_every` frames
    only, on the pixels from `frame_at(i)`; in between, what was found follows its vehicle box.
    Vehicles not matched to a search fall back to the fixed bands until the next one."""
    recent: deque[list[Box]] = deque(maxlen=hold_frames + 1)
    last: list[tuple[Box, Found]] = []  # vehicle box at the last search + what it held, relative to it
    for i, frame in enumerate(detector.detect(source)):
        found = None
        if finder is not None:
            if i % refresh_every == 0:
                found = _search(finder, frame_at(i), frame)
                last = [(d.box, _to_relative(f, d.box)) for d, f in zip(frame, found) if f.plates or f.faces]
            else:
                found = [_reuse(d.box, last) for d in frame]
        recent.append(private_regions(frame, found))
        yield [r for regions in recent for r in regions]


def _search(finder: PrivateFinder, image: Any, frame: Sequence[Detection]) -> list[Found]:
    plate_idx = [k for k, d in enumerate(frame) if plate_search(d)]
    cabins = [(k, cabin_search(frame[k])) for k in range(len(frame))]
    cabins = [(k, a) for k, a in cabins if a is not None]
    if not plate_idx and not cabins:
        return [Found() for _ in frame]
    plates, faces = finder.find(image, [frame[k].box for k in plate_idx], [a for _, a in cabins])
    found = [Found() for _ in frame]
    for k, boxes in zip(plate_idx, plates):
        found[k] = Found(plates=tuple(boxes))
    for (k, _), boxes in zip(cabins, faces):
        found[k] = Found(plates=found[k].plates, faces=tuple(boxes))
    return found


def _reuse(box: Box, last: list[tuple[Box, Found]]) -> Found:
    best = max(last, key=lambda bf: iou(box, bf[0]), default=None)
    if best is None or iou(box, best[0]) < REUSE_IOU:
        return Found()
    return _to_absolute(best[1], box)


def _to_relative(f: Found, ref: Box) -> Found:
    return Found(tuple(_rel(b, ref) for b in f.plates), tuple(_rel(b, ref) for b in f.faces))


def _to_absolute(f: Found, ref: Box) -> Found:
    return Found(tuple(_abs(b, ref) for b in f.plates), tuple(_abs(b, ref) for b in f.faces))


def _rel(b: Box, ref: Box) -> Box:
    w, h = ref[2] - ref[0], ref[3] - ref[1]
    return (b[0] - ref[0]) / w, (b[1] - ref[1]) / h, (b[2] - ref[0]) / w, (b[3] - ref[1]) / h


def _abs(b: Box, ref: Box) -> Box:
    w, h = ref[2] - ref[0], ref[3] - ref[1]
    return ref[0] + b[0] * w, ref[1] + b[1] * h, ref[0] + b[2] * w, ref[1] + b[3] * h

