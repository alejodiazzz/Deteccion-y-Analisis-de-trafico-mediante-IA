"""Detection mAP with pycocotools: the standard COCO definition, including AP for small objects."""
import contextlib
import io

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from vtia.domain.classes import RoadClass
from vtia.domain.models import Detection, GroundTruth


def _xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def coco_map(truths: list[list[GroundTruth]], detections: list[list[Detection]]) -> dict:
    """Both lists are aligned per frame (each frame is one COCO image)."""
    cat = {c: i + 1 for i, c in enumerate(RoadClass)}
    images, anns, dets = [], [], []
    for img, (frame_truths, frame_dets) in enumerate(zip(truths, detections)):
        images.append({"id": img})
        for t in frame_truths:
            bbox = _xywh(t.box)
            anns.append({"id": len(anns) + 1, "image_id": img, "category_id": cat[t.road_class],
                         "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0})
        dets += [{"image_id": img, "category_id": cat[d.road_class], "bbox": _xywh(d.box), "score": d.confidence}
                 for d in frame_dets]
    if not dets:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "AP_small": 0.0, "per_class": {c: {"AP50": 0.0, "AP50_95": 0.0} for c in cat}}

    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": i, "name": str(c)} for c, i in cat.items()]}
    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints a lot
        gt.createIndex()
        e = COCOeval(gt, gt.loadRes(dets), "bbox")
        e.params.maxDets = [1, 100, 500]  # dense scenes exceed COCO's default cap of 100 per image
        e.evaluate()
        e.accumulate()
        e.summarize()

    precision = e.eval["precision"]  # [IoU threshold, recall, class, area range, maxDets]

    def ap(k: int, iou_index=slice(None)) -> float:
        p = precision[iou_index, :, k, 0, -1]
        p = p[p > -1]
        return float(p.mean()) if p.size else float("nan")

    return {
        "mAP50": float(e.stats[1]), "mAP50_95": float(e.stats[0]), "AP_small": float(e.stats[3]),
        "per_class": {c: {"AP50": ap(i - 1, 0), "AP50_95": ap(i - 1)} for c, i in cat.items()},
    }
