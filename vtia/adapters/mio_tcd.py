"""MIO-TCD localization set (traffic cameras, Canada/US) exported from Roboflow in YOLO format.

Export layout: data.yaml (with `names`) + train/ valid/ test/, each with images/ and labels/.
License: CC BY-NC-SA 4.0 (non-commercial, share-alike), whatever the Roboflow page says, because it
is a copy of Miovision's set. Cite Luo et al., "MIO-TCD: A new benchmark dataset for vehicle
classification and localization", IEEE TIP 2018. Weights trained on it inherit those terms.
"""
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml

from vtia.adapters.yolo_training import IGNORE_FILL, NAMES
from vtia.domain.classes import RoadClass

# Follows Docs/Guia_Anotacion.md: pickups are `car`, trucks of any size are `truck`.
MIO_TCD_TO_ROAD = {
    "pedestrian": RoadClass.PEDESTRIAN,
    "bicycle": RoadClass.BICYCLE,
    "motorcycle": RoadClass.MOTORCYCLE,
    "car": RoadClass.CAR,
    "pickuptruck": RoadClass.CAR,
    "bus": RoadClass.BUS,
    "articulatedtruck": RoadClass.TRUCK,
    "singleunittruck": RoadClass.TRUCK,
}
# Painted grey like CVAT ignore regions: "work van" is an open case in the guide (cargo van = truck?
# passenger van = car?), and the other two are not one of our classes.
MIO_TCD_IGNORED = {"workvan", "nonmotorizedvehicle", "motorizedvehicle"}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _names(data: dict) -> list[str]:
    names = data["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def _box(values: list[float]) -> tuple[float, float, float, float]:
    """YOLO box (cx cy w h) or Roboflow polygon (x1 y1 x2 y2 ...) -> normalised cx cy w h."""
    if len(values) == 4:
        return tuple(values)
    xs, ys = values[0::2], values[1::2]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)


def convert_export(src: Path, out: Path) -> tuple[Path, Counter]:
    """Rewrites a Roboflow YOLO export with VTIA class ids (RoadClass order) under `out`.
    Returns the new data.yaml and the box count per (split, class); ignored boxes count as "ignore"."""
    src, out = Path(src), Path(out)
    names = _names(yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8")))
    unknown = [n for n in names if _key(n) not in MIO_TCD_TO_ROAD and _key(n) not in MIO_TCD_IGNORED]
    if unknown:
        raise ValueError(f"classes not in MIO-TCD, map them in vtia/adapters/mio_tcd.py first: {unknown}")
    target = {i: (NAMES.index(MIO_TCD_TO_ROAD[_key(n)].value) if _key(n) in MIO_TCD_TO_ROAD else None)
              for i, n in enumerate(names)}

    counts: Counter = Counter()
    splits = {}
    for split, folder in (("train", "train"), ("val", "valid"), ("val", "val"), ("test", "test")):
        images = src / folder / "images"
        if split in splits or not images.is_dir():
            continue
        splits[split] = f"{split}/images"
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        for image in sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            label = src / folder / "labels" / f"{image.stem}.txt"
            rows = label.read_text(encoding="utf-8").split("\n") if label.exists() else []
            keep, ignore = [], []
            for row in rows:
                parts = row.split()
                if len(parts) < 5:
                    continue
                cx, cy, w, h = _box([float(v) for v in parts[1:]])
                cls = target[int(parts[0])]
                if cls is None:
                    ignore.append((cx, cy, w, h))
                    counts[(split, "ignore")] += 1
                else:
                    keep.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    counts[(split, NAMES[cls])] += 1
            dest = out / split / "images" / image.name
            if ignore:
                img = cv2.imread(str(image))
                ih, iw = img.shape[:2]
                for cx, cy, w, h in ignore:
                    x1, y1 = int((cx - w / 2) * iw), int((cy - h / 2) * ih)
                    x2, y2 = int((cx + w / 2) * iw), int((cy + h / 2) * ih)
                    cv2.rectangle(img, (x1, y1), (x2, y2), IGNORE_FILL, -1)
                cv2.imwrite(str(dest), img)
            else:
                shutil.copy2(image, dest)
            (out / split / "labels" / f"{image.stem}.txt").write_text("\n".join(keep), encoding="utf-8")
    if "train" not in splits or "val" not in splits:
        raise ValueError(f"{src} needs train/images and valid/images (Roboflow export, YOLO format)")
    data = out / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(out.resolve()), **splits, "names": dict(enumerate(NAMES))}),
                    encoding="utf-8")
    return data, counts
