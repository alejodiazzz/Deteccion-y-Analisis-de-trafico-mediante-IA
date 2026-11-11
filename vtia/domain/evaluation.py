"""Evaluation rules: box overlap, ignore regions, matching and the confusion matrix."""
from collections import Counter
from collections.abc import Iterable, Sequence
from statistics import mean
from typing import TypeVar

from vtia.domain.classes import RoadClass
from vtia.domain.models import Box, Detection, GroundTruth

Boxed = TypeVar("Boxed", Detection, GroundTruth)
BACKGROUND = "background"  # confusion row: a detection that matches no object (false positive)
MISSED = "missed"          # confusion column: an object no detection matched (false negative)


def intersection(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def drop_ignored(items: Iterable[Boxed], ignore: Sequence[Box], min_overlap: float = 0.5) -> list[Boxed]:
    """Removes detections or annotations lying mostly (>= min_overlap of their area) inside an ignore region.

    Ignore regions hold what is out of scope (crowds, tricycles...). Anything there,
    predicted or annotated, is left out of scoring; applying it to both sides keeps
    e.g. a person sitting on an ignored tricycle from counting as a miss.
    """
    def inside(box: Box, region: Box) -> bool:
        size = area(box)
        return size > 0 and intersection(box, region) >= min_overlap * size

    return [x for x in items if not any(inside(x.box, r) for r in ignore)]


def match(truths: Sequence[GroundTruth], detections: Sequence[Detection], min_iou: float = 0.5):
    """Greedy, class-agnostic matching: the most confident detection takes its best free object.

    Class-agnostic on purpose, so a bus detected as a truck shows up as a confusion
    instead of as one miss plus one false positive.
    Returns (pairs, unmatched truths, unmatched detections).
    """
    # ponytail: O(objects x detections) per frame in pure Python; vectorise with numpy if it gets slow.
    free = set(range(len(truths)))
    pairs, extra = [], []
    for d in sorted(detections, key=lambda d: -d.confidence):
        best, best_iou = None, min_iou
        for i in free:
            v = iou(truths[i].box, d.box)
            if v >= best_iou:
                best, best_iou = i, v
        if best is None:
            extra.append(d)
        else:
            free.remove(best)
            pairs.append((truths[best], d))
    return pairs, [truths[i] for i in sorted(free)], extra


def confusion(truths: Sequence[GroundTruth], detections: Sequence[Detection], min_iou: float = 0.5) -> Counter:
    """Counts keyed by (true class or BACKGROUND, predicted class or MISSED)."""
    pairs, missed, extra = match(truths, detections, min_iou)
    counts = Counter((t.road_class, d.road_class) for t, d in pairs)
    counts.update((t.road_class, MISSED) for t in missed)
    counts.update((BACKGROUND, d.road_class) for d in extra)
    return counts


def class_report(counts: Counter, classes: Sequence[RoadClass] = tuple(RoadClass)) -> dict:
    """Per-class precision/recall/F1 plus macro F1, balanced accuracy and micro recall.

    `accuracy` remains as a backwards-compatible alias for micro recall; it is not
    conventional classification accuracy because true negatives are undefined here.

    Precision counts background false positives; recall counts missed objects.
    Classes without annotated objects are left out of the averages.
    """
    per_class = {}
    for c in classes:
        tp = counts[(c, c)]
        predicted = sum(n for (_, p), n in counts.items() if p == c)
        actual = sum(n for (t, _), n in counts.items() if t == c)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}
    present = [m for m in per_class.values() if m["support"]]
    support = sum(m["support"] for m in present)
    micro_recall = sum(counts[(c, c)] for c in classes) / support if support else 0.0
    return {
        "per_class": per_class,
        "macro_f1": mean(m["f1"] for m in present) if present else 0.0,
        "balanced_accuracy": mean(m["recall"] for m in present) if present else 0.0,
        "micro_recall": micro_recall,
        "accuracy": micro_recall,  # legacy alias; not conventional accuracy
    }
