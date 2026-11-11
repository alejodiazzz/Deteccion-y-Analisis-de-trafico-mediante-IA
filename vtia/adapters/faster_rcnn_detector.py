"""torchvision Faster R-CNN (ResNet-50 FPN v2, COCO weights) -> domain detections."""
from collections.abc import Iterator
from pathlib import Path

import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

from vtia.adapters.video_io import read_frames
from vtia.domain.classes import COCO_TO_ROAD
from vtia.domain.models import Detection


class FasterRCNNDetector:
    name = "faster_rcnn"

    def __init__(self, conf: float = 0.1, imgsz: int = 640, device: str | None = None):
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.weights_id = str(weights)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # min_size = max_size = imgsz -> longest side = imgsz, same input size as the Ultralytics models.
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=weights, min_size=imgsz, max_size=imgsz,
            box_score_thresh=conf, box_detections_per_img=300,
        ).eval().to(self.device)
        self.class_map = {i: COCO_TO_ROAD[n] for i, n in enumerate(weights.meta["categories"]) if n in COCO_TO_ROAD}

    @torch.inference_mode()
    def detect(self, source: Path) -> Iterator[list[Detection]]:
        for frame, image in enumerate(read_frames(source)):
            x = torch.from_numpy(image).to(self.device).permute(2, 0, 1).float() / 255
            out = self.model([x])[0]
            yield [Detection(frame, self.class_map[label], score, tuple(box))
                   for box, label, score in zip(out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist())
                   if label in self.class_map]

