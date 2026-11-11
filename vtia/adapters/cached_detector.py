"""Content-addressed detector cache with atomic completion metadata."""
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

from vtia.application.ports import Detector
from vtia.domain.classes import RoadClass
from vtia.domain.models import Detection

CACHE_SCHEMA = 2


def _sha256(path: Path) -> str:
    """Hash a file or a directory of frames (VisDrone sequences)."""
    digest = hashlib.sha256()
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"cannot fingerprint empty or missing source: {path}")
    for item in files:
        if path.is_dir():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        with item.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _model_fingerprint(inner: Detector) -> dict:
    data = {"class": f"{type(inner).__module__}.{type(inner).__qualname__}", "name": inner.name}
    for key in ("conf", "imgsz", "device"):
        if hasattr(inner, key):
            data[key] = getattr(inner, key)
    model = getattr(inner, "model", None)
    weights = getattr(inner, "weights_path", None)
    if weights and not Path(str(weights)).expanduser().is_file():
        weights = getattr(model, "ckpt_path", None) or weights
    weights = weights or getattr(model, "ckpt_path", None) or getattr(inner, "weights", None)
    if hasattr(inner, "weights_id"):
        data["weights_id"] = inner.weights_id
    if weights:
        path = Path(str(weights)).expanduser()
        data["weights"] = {"path": str(path.resolve()), "sha256": _sha256(path)} if path.is_file() else str(weights)
    else:
        # torchvision default weights are identified by adapter and installed package versions.
        try:
            from importlib.metadata import version
            data["torch"] = version("torch")
            data["torchvision"] = version("torchvision")
        except Exception:
            pass
    return data


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class CachedDetector:
    """Run once per exact source/model configuration; incomplete caches are never reused."""
    def __init__(self, inner: Detector, cache_dir: Path):
        self.inner = inner
        self.name = inner.name
        self.dir = Path(cache_dir) / inner.name
        self.frames = 0
        self.seconds = 0.0

    def detect(self, source: Path) -> Iterator[list[Detection]]:
        source = Path(source)
        if not source.exists() or (source.is_dir() and not any(source.iterdir())):
            raise FileNotFoundError(source)
        identity = {"schema": CACHE_SCHEMA, "source_sha256": _sha256(source),
                    "model": _model_fingerprint(self.inner)}
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        cache = self.dir / f"{source.stem}-{signature[:20]}.jsonl"
        meta = cache.with_suffix(".meta.json")
        valid = False
        if cache.is_file() and meta.is_file():
            try:
                stored = json.loads(meta.read_text(encoding="utf-8"))
                valid = stored.get("identity") == identity and stored.get("cache_sha256") == _sha256(cache)
            except (OSError, ValueError, TypeError):
                valid = False
        if not valid:
            self.dir.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            fd, tmp = tempfile.mkstemp(prefix=f".{cache.name}.", suffix=".tmp", dir=self.dir)
            frames = 0
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    for frame in self.inner.detect(source):
                        f.write(json.dumps([[d.road_class.value, d.confidence, *d.box] for d in frame]) + "\n")
                        frames += 1
                    f.flush()
                    os.fsync(f.fileno())
                seconds = time.perf_counter() - start
                os.replace(tmp, cache)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            metadata = {"identity": identity, "frames": frames, "seconds": seconds,
                        "cache_sha256": _sha256(cache)}
            _atomic_write(meta, json.dumps(metadata, sort_keys=True))
        info = json.loads(meta.read_text(encoding="utf-8"))
        self.frames += info["frames"]
        self.seconds += info["seconds"]
        with cache.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                yield [Detection(i, RoadClass(c), conf, tuple(box)) for c, conf, *box in json.loads(line)]
