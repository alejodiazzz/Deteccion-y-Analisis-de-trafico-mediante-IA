"""Clean-up of raw detector output before tracking or scoring."""
from dataclasses import replace

from vtia.domain.classes import RoadClass
from vtia.domain.evaluation import area, intersection, iou
from vtia.domain.models import Detection


def suppress_duplicates(detections: list[Detection], max_iou: float = 0.7) -> list[Detection]:
    """Class-agnostic NMS: drops a box overlapping a more confident one by more than `max_iou`.

    Detectors often report the same object twice with different classes (car + truck);
    the tracker then sees two objects, which causes ID switches and double counts.
    """
    # ponytail: O(n^2) per frame in pure Python; fine for a few hundred boxes, use torchvision.ops.nms if not.
    kept: list[Detection] = []
    for d in sorted(detections, key=lambda d: -d.confidence):
        if all(iou(d.box, k.box) <= max_iou for k in kept):
            kept.append(d)
    return kept


TWO_WHEELERS = (RoadClass.BICYCLE, RoadClass.MOTORCYCLE)


def merge_riders(detections: list[Detection], min_overlap: float = 0.4) -> list[Detection]:
    """Experiment E2: COCO models box a cyclist or motorcyclist as a separate "person", while the
    project counts rider + vehicle as one object. A person rides a bicycle/motorcycle when at
    least `min_overlap` of the person's box overlaps it and the person's feet (anchor) fall
    within its width; the person is dropped and the vehicle box grows to cover both.
    """
    vehicles = [i for i, d in enumerate(detections) if d.road_class in TWO_WHEELERS]
    riders: dict[int, list] = {}
    dropped = set()
    for j, p in enumerate(detections):
        if p.road_class != RoadClass.PEDESTRIAN or not vehicles or area(p.box) == 0:
            continue
        i = max(vehicles, key=lambda i: intersection(p.box, detections[i].box))
        v = detections[i].box
        if intersection(p.box, v) >= min_overlap * area(p.box) and v[0] <= p.anchor[0] <= v[2]:
            riders.setdefault(i, []).append(p.box)  # a parrillero is a second rider of the same motorcycle
            dropped.add(j)
    return [replace(d, box=_union(d.box, *riders[i])) if i in riders else d
            for i, d in enumerate(detections) if i not in dropped]


def _union(*boxes) -> tuple[float, float, float, float]:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
