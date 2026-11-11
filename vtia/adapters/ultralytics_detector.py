"""Ultralytics adapter: YOLO or RT-DETR (picked from the weights name) -> domain detections."""
from collections.abc import Iterator
from pathlib import Path

from ultralytics import RTDETR, YOLO

from vtia.domain.classes import MODEL_LABELS
from vtia.domain.models import Detection


class UltralyticsDetector:
    def __init__(self, weights: str = "yolo26n.pt", conf: float = 0.1, imgsz: int = 640,
                 device: str | None = None):
        w = Path(weights)
        self.weights_path = w.expanduser()
        # Fine-tuned runs all save best.pt/last.pt: prefix the run folder so names (and caches) don't collide.
        self.name = f"{w.parents[1].name}-{w.stem}" if w.stem in ("best", "last") and len(w.parents) > 1 else w.stem
        self.model = (RTDETR if "rtdetr" in w.stem else YOLO)(weights)
        self.conf = conf
        self.imgsz = imgsz  # longest image side fed to the network
        self.device = device  # None -> GPU if available, else CPU
        # Model class id -> project class (COCO or fine-tuned names); other classes are never returned.
        self.class_map = {i: MODEL_LABELS[n] for i, n in self.model.names.items() if n in MODEL_LABELS}

    def detect(self, source: Path) -> Iterator[list[Detection]]:
        results = self.model.predict(
            str(source), stream=True, classes=list(self.class_map), conf=self.conf,
            imgsz=self.imgsz, device=self.device, max_det=300, verbose=False,
        )
        for frame, r in enumerate(results):
            b = r.boxes
            yield [Detection(frame, self.class_map[int(c)], float(s), tuple(xyxy))
                   for c, s, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
