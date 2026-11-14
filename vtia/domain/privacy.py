"""Which parts of a frame must be blurred before a video leaves the server PC (Ley 1581 de 2012).

No face or plate model: heads and plates are located from the road-user boxes the
detector already finds, which also works when faces and plates are a few pixels wide.
"""
from collections.abc import Iterable

from vtia.domain.classes import RoadClass
from vtia.domain.models import Box, Detection

# ponytail: fixed proportions of each box, not a face/plate detector. Kept tight on purpose:
# blurring most of a vehicle hides the object itself and biases evaluation on anonymized video
# (tests cap a near vehicle at < 50 % blurred). Add a dedicated plate model only if real clips
# show plates escaping.
HEAD = 0.28              # top share of a person's box
PLATE_BAND = {           # bottom share of a vehicle's box, and share of its width (centred)
    RoadClass.CAR: (0.22, 0.55),
    RoadClass.BUS: (0.18, 0.50),
    RoadClass.TRUCK: (0.18, 0.50),
    RoadClass.MOTORCYCLE: (0.30, 0.60),
}
# Windshield band of large cars/buses/trucks (top, bottom share of the box; side inset share):
# drivers are not boxed as persons. Below CABIN_MIN_PX of box height the faces inside are a few
# pixels wide: not blurred.
CABIN = (0.15, 0.45)
CABIN_INSET = 0.15
CABIN_MIN_PX = 250
MARGIN = 0.08            # widen each region by this share of its size; anonymize.py also merges
                         # recent frames, which absorbs most box jitter


def private_regions(detections: Iterable[Detection]) -> list[Box]:
    regions = []
    for d in detections:
        x1, y1, x2, y2 = d.box
        w, h = x2 - x1, y2 - y1
        if d.road_class == RoadClass.PEDESTRIAN:  # riders are boxed as persons by COCO models too
            regions.append(_grow((x1, y1, x2, y1 + h * HEAD)))
        elif d.road_class in PLATE_BAND:
            depth, width = PLATE_BAND[d.road_class]
            cx = (x1 + x2) / 2
            regions.append(_grow((cx - w * width / 2, y2 - h * depth, cx + w * width / 2, y2)))
            if d.road_class != RoadClass.MOTORCYCLE and h >= CABIN_MIN_PX:
                top, bottom = CABIN
                regions.append(_grow((x1 + w * CABIN_INSET, y1 + h * top, x2 - w * CABIN_INSET, y1 + h * bottom)))
    return regions


def _grow(b: Box) -> Box:
    dx, dy = (b[2] - b[0]) * MARGIN, (b[3] - b[1]) * MARGIN
    return b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy
