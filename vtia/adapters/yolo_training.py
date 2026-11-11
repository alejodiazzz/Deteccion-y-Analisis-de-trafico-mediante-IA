"""Phase 4: fine-tune an Ultralytics model on annotated sequences (CVAT exports or VisDrone)."""
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO

from vtia.adapters.video_io import read_frames
from vtia.application.evaluate import AnnotatedSequence
from vtia.domain.classes import ROAD_TO_COCO, RoadClass

# Class ids of the fine-tuned model follow RoadClass order, but carry COCO's names ("person", not
# "pedestrian"): Ultralytics copies a class's pretrained head row only when the names match, so all six
# start from what COCO learned. Detections map back through MODEL_LABELS ("person" -> pedestrian).
CLASS_ID = {c: i for i, c in enumerate(RoadClass)}
NAMES = [ROAD_TO_COCO[c] for c in RoadClass]
IGNORE_FILL = (114, 114, 114)         # Ultralytics' padding grey: ignore regions become "nothing to learn"


def export_dataset(train: list[AnnotatedSequence], val: list[AnnotatedSequence], out: Path, every: int = 5) -> Path:
    """Writes images/ and labels/ in YOLO format (1 of every `every` frames) and data.yaml; returns its path.
    Ignore regions are painted grey, so the model is neither rewarded nor punished for them."""
    out = Path(out)
    for split, sequences in (("train", train), ("val", val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for seq in sequences:
            for i, rgb in enumerate(read_frames(seq.source)):
                if i % every or i >= len(seq.truths):
                    continue
                img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for x1, y1, x2, y2 in seq.ignore[i]:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), IGNORE_FILL, -1)
                h, w = img.shape[:2]
                stem = f"{seq.name}_{i:06d}"
                cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img)
                lines = []
                for t in seq.truths[i]:
                    x1, y1, x2, y2 = max(0, t.box[0]), max(0, t.box[1]), min(w, t.box[2]), min(h, t.box[3])
                    if x2 > x1 and y2 > y1:
                        lines.append(f"{CLASS_ID[t.road_class]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    data = out / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(out.resolve()), "train": "images/train", "val": "images/val",
                                    "names": dict(enumerate(NAMES))}), encoding="utf-8")
    return data


def train(data: Path, weights: str = "yolo26n.pt", epochs: int = 50, imgsz: int = 640, batch: int = 16,
          project: Path = Path("outputs/train"), name: str = "run", device: str | None = None) -> Path:
    """Fine-tunes from COCO weights with Ultralytics' default augmentation (it never flips vertically,
    as the proposal asks). Returns the best checkpoint."""
    if torch.version.hip:
        # ponytail: ROCm 7.2.1 on Windows can't compile MIOpen's training batch-norm kernel
        # (hiprtc error). PyTorch's own GPU kernels work, just slower; drop this when MIOpen is fixed.
        torch.backends.cudnn.enabled = False
    model = YOLO(weights)
    model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, project=str(Path(project).resolve()),
                name=name, exist_ok=True, device=device, plots=True, verbose=False)
    return Path(model.trainer.best)
