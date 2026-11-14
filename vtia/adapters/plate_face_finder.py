"""Plate (YOLO) and face (OpenCV YuNet) models for the anonymizer, run on crops of near vehicles.

Neither was trained on Colombian plates or cameras: check the *_check.jpg contact sheets. Whatever
they miss is still blurred with the fixed bands (vtia/domain/privacy.py).
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from vtia.domain.models import Box

# morsetechlab/yolov11-license-plate-detection, nano, AGPL-3.0 (like Ultralytics itself).
PLATE_URL = ("https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/"
             "license-plate-finetune-v1n.pt")
# OpenCV Zoo YuNet, MIT. The 2026may export has dynamic input sizes, needed by OpenCV 5.
FACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
            "face_detection_yunet_2026may.onnx")
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FACE_UPSCALE_BELOW = 400  # cabin crops smaller than this (px) are doubled: drivers' faces are tiny


def fetch(url: str, folder: Path = MODELS_DIR) -> Path:
    """Downloads a model file once; later runs reuse it."""
    dest = folder / url.rsplit("/", 1)[-1]
    if not dest.exists():
        folder.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, part)
        part.replace(dest)
    return dest


class PlateFaceFinder:
    def __init__(self, plate_conf: float = 0.25, face_conf: float = 0.5, plate_imgsz: int = 320,
                 device: str | None = None):
        self.plates = YOLO(str(fetch(PLATE_URL)))
        self.faces = cv2.FaceDetectorYN.create(str(fetch(FACE_URL)), "", (320, 320), face_conf)
        self.plate_conf, self.plate_imgsz, self.device = plate_conf, plate_imgsz, device

    def find(self, image: np.ndarray | None, plate_areas: list[Box], face_areas: list[Box]
             ) -> tuple[list[list[Box]], list[list[Box]]]:
        if image is None:
            return [[] for _ in plate_areas], [[] for _ in face_areas]
        return self._plates(image, plate_areas), [self._faces(image, a) for a in face_areas]

    def _plates(self, image: np.ndarray, areas: list[Box]) -> list[list[Box]]:
        crops = [_crop(image, a) for a in areas]
        valid = [k for k, (c, _) in enumerate(crops) if c is not None]
        found: list[list[Box]] = [[] for _ in areas]
        if not valid:
            return found
        results = self.plates.predict([crops[k][0] for k in valid], imgsz=self.plate_imgsz, conf=self.plate_conf,
                                      device=self.device, verbose=False)
        for k, r in zip(valid, results):
            ox, oy = crops[k][1]
            found[k] = [(x1 + ox, y1 + oy, x2 + ox, y2 + oy) for x1, y1, x2, y2 in r.boxes.xyxy.tolist()]
        return found

    def _faces(self, image: np.ndarray, area: Box) -> list[Box]:
        crop, (ox, oy) = _crop(image, area)
        if crop is None:
            return []
        s = 2 if max(crop.shape[:2]) < FACE_UPSCALE_BELOW else 1
        if s > 1:
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        self.faces.setInputSize((crop.shape[1], crop.shape[0]))
        _, faces = self.faces.detect(crop)
        if faces is None:
            return []
        return [(ox + x / s, oy + y / s, ox + (x + w) / s, oy + (y + h) / s) for x, y, w, h in faces[:, :4].tolist()]


def _crop(image: np.ndarray, area: Box) -> tuple[np.ndarray | None, tuple[int, int]]:
    h, w = image.shape[:2]
    x1, y1 = max(0, int(area[0])), max(0, int(area[1]))
    x2, y2 = min(w, int(np.ceil(area[2]))), min(h, int(np.ceil(area[3])))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None, (x1, y1)
    return image[y1:y2, x1:x2], (x1, y1)
