"""Which parts of a frame must be blurred before a video leaves the server PC (Ley 1581 de 2012).

Heads, plates and windshields are first located from the road-user boxes the detector already
finds, which also works when faces and plates are a few pixels wide. On near vehicles, optional
plate/face models (`Found`) narrow that down to the real plate and the driver's face.
"""
from collections.abc import Sequence
from dataclasses import dataclass

from vtia.domain.classes import RoadClass
from vtia.domain.models import Box, Detection

# Fixed proportions of each box: the fallback when no plate/face model is used or it finds nothing,
# so privacy comes first. Deep enough for high cameras, where the box includes the roof and the plate
# sits at ~70 % of its height (piloto_autopista, 2026-11-14); narrow enough that a near vehicle stays
# < 60 % blurred even then (blurring the object itself biases evaluation on anonymized video).
HEAD = 0.28              # top share of a person's box
PLATE_BAND = {           # bottom share of a vehicle's box, and share of its width (centred)
    RoadClass.CAR: (0.40, 0.60),
    RoadClass.BUS: (0.30, 0.55),
    RoadClass.TRUCK: (0.30, 0.55),
    RoadClass.MOTORCYCLE: (0.45, 0.70),
}
# Windshield band of large cars/buses/trucks (top, bottom share of the box; side inset share):
# drivers are not boxed as persons. Below CABIN_MIN_PX of box height the faces inside are a few
# pixels wide: not blurred.
CABIN = (0.15, 0.45)
CABIN_INSET = 0.15
CABIN_MIN_PX = 250
MARGIN = 0.08            # widen each region by this share of its size; anonymize.py also merges
                         # recent frames, which absorbs most box jitter


# Refinement: a plate model and a face model look inside near vehicles, so only the real plate and
# the driver's face get blurred. Whatever they miss falls back to the bands above (privacy first).
PLATE_MIN_PX = 60        # vehicle box height from which the plate model is asked; below, the band is tiny
CABIN_SEARCH = 0.65      # top share of a near vehicle where drivers' faces are looked for
FOUND_MARGIN = 0.25      # found plates/faces are small: a wider share absorbs box jitter


@dataclass(frozen=True)
class Found:
    """What the plate/face models saw inside one detection, in frame pixels. Empty = band."""
    plates: tuple[Box, ...] = ()
    faces: tuple[Box, ...] = ()


def plate_search(d: Detection) -> bool:
    return d.road_class in PLATE_BAND and d.box[3] - d.box[1] >= PLATE_MIN_PX


def cabin_search(d: Detection) -> Box | None:
    """Where to look for drivers' faces, or None when they are too small to identify."""
    x1, y1, x2, y2 = d.box
    if d.road_class in PLATE_BAND and d.road_class != RoadClass.MOTORCYCLE and y2 - y1 >= CABIN_MIN_PX:
        return x1, y1, x2, y1 + (y2 - y1) * CABIN_SEARCH
    return None


def private_regions(detections: Sequence[Detection], found: Sequence[Found] | None = None) -> list[Box]:
    regions = []
    for i, d in enumerate(detections):
        f = found[i] if found else Found()
        x1, y1, x2, y2 = d.box
        w, h = x2 - x1, y2 - y1
        if d.road_class == RoadClass.PEDESTRIAN:  # riders are boxed as persons by COCO models too
            regions.append(_grow((x1, y1, x2, y1 + h * HEAD)))
        elif d.road_class in PLATE_BAND:
            if f.plates:
                regions.extend(_grow(p, FOUND_MARGIN) for p in f.plates)
            else:
                depth, width = PLATE_BAND[d.road_class]
                cx = (x1 + x2) / 2
                regions.append(_grow((cx - w * width / 2, y2 - h * depth, cx + w * width / 2, y2)))
            if cabin_search(d):
                if f.faces:
                    regions.extend(_grow(p, FOUND_MARGIN) for p in f.faces)
                else:
                    top, bottom = CABIN
                    regions.append(_grow((x1 + w * CABIN_INSET, y1 + h * top, x2 - w * CABIN_INSET, y1 + h * bottom)))
    return regions


def _grow(b: Box, margin: float = MARGIN) -> Box:
    dx, dy = (b[2] - b[0]) * margin, (b[3] - b[1]) * margin
    return b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy
