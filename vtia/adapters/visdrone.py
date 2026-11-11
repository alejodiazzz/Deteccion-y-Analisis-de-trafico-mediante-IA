"""VisDrone2019-MOT reader: folders of frames + one annotation file per sequence.

Annotation row: frame, object_id, left, top, width, height, score, category, truncation, occlusion
(frames start at 1; score 0 = not evaluated).
"""
import csv
from pathlib import Path

from vtia.application.evaluate import AnnotatedSequence
from vtia.domain.classes import RoadClass
from vtia.domain.models import GroundTruth

# VisDrone convention, like COCO's: a rider is a "people" box separate from the bicycle/motor box.
VISDRONE_TO_ROAD = {
    1: RoadClass.PEDESTRIAN,  # pedestrian
    2: RoadClass.PEDESTRIAN,  # people (not standing/walking)
    3: RoadClass.BICYCLE,
    4: RoadClass.CAR,
    5: RoadClass.CAR,         # van
    6: RoadClass.TRUCK,
    9: RoadClass.BUS,
    10: RoadClass.MOTORCYCLE,  # motor
}
# Everything else (0 ignored region, 7/8 tricycles, 11 others) becomes an ignore region.


def load_visdrone(root: Path) -> list[AnnotatedSequence]:
    root = Path(root)
    sequences = []
    for images in sorted((root / "sequences").iterdir()):
        n = sum(1 for _ in images.glob("*.jpg"))
        truths = [[] for _ in range(n)]
        ignore = [[] for _ in range(n)]
        with open(root / "annotations" / f"{images.name}.txt", encoding="utf-8") as f:
            for row in csv.reader(f):
                frame, obj, x, y, w, h, score, cat = (int(v) for v in row[:8])
                i = frame - 1
                if not 0 <= i < n:
                    continue
                box = (x, y, x + w, y + h)
                if score == 0 or cat not in VISDRONE_TO_ROAD:
                    ignore[i].append(box)
                else:
                    truths[i].append(GroundTruth(i, obj, VISDRONE_TO_ROAD[cat], box))
        sequences.append(AnnotatedSequence(images.name, images, truths, ignore))
    return sequences
