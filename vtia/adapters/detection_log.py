"""Detections recorded while anonymizing, replayed by the analysis of the anonymized copy.

Anonymizing already runs a detector over every frame of the original. When the analysis asks for
the same model, replaying that run saves a whole second pass, and the boxes come from the original
pixels, so the blur cannot bias them. Anything else (another model, a lower confidence, another
video) runs the model as usual.
"""
import json
from collections.abc import Iterator
from pathlib import Path

from vtia.application.ports import Detector
from vtia.domain.classes import RoadClass
from vtia.domain.models import Detection

SUFFIX = ".detections.jsonl"


def log_path(video: Path) -> Path:
    return Path(video).with_suffix(SUFFIX)


class RecordingDetector:
    """Passes detections through and writes them to `path`; the file appears only once complete."""

    def __init__(self, inner: Detector, path: Path):
        self.inner, self.path, self.name = inner, Path(path), inner.name

    def detect(self, source: Path) -> Iterator[list[Detection]]:
        tmp = self.path.with_suffix(".part")
        frames = 0
        try:
            with tmp.open("w", encoding="utf-8") as f:
                for frame in self.inner.detect(source):
                    f.write(json.dumps([[d.road_class.value, d.confidence, *d.box] for d in frame]) + "\n")
                    frames += 1
                    yield frame
            meta = {"model": self.inner.name, "conf": getattr(self.inner, "conf", None),
                    "imgsz": getattr(self.inner, "imgsz", None), "frames": frames}
            self.path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
            tmp.replace(self.path)
        finally:
            tmp.unlink(missing_ok=True)


class _Replay:
    def __init__(self, path: Path, name: str, conf: float):
        self.path, self.name, self.conf = path, name, conf

    def detect(self, source: Path) -> Iterator[list[Detection]]:
        with self.path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                yield [Detection(i, RoadClass(c), s, tuple(box)) for c, s, *box in json.loads(line)
                       if s >= self.conf]


def recorded_detections(video: Path, model: str, conf: float, frames: int, imgsz: int = 640) -> Detector | None:
    """A detector replaying the log next to `video`, or None when it would not equal running `model`."""
    path = log_path(video)
    try:
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    same = (meta.get("model") == Path(model).stem and meta.get("imgsz") == imgsz and meta.get("frames") == frames
            and meta.get("conf") is not None and meta["conf"] <= conf and path.is_file())
    return _Replay(path, meta["model"], conf) if same else None
