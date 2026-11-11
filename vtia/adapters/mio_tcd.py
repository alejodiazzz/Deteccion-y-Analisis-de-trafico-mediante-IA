"""MIO-TCD localization set (traffic cameras, Canada/US) -> YOLO dataset with VTIA's six classes.

Two sources:
- Official release (https://tcd.miovision.com/challenge/dataset.html): MIO-TCD-Localization/ with
  train/<id>.jpg and gt_train.csv (rows: id,label,x1,y1,x2,y2 in pixels, no header). The test labels
  are not public, so the validation split is carved out of train.
- A Roboflow export in YOLO format (data.yaml + train/ valid/ test/). Check it first: the copy
  thesis-ltv3y/mio_tcd-f3kk4 has a single box per image, which teaches the model that the other
  vehicles are background.

License: CC BY-NC-SA 4.0 (non-commercial, share-alike), also for copies. Cite Luo et al., "MIO-TCD: A new
benchmark dataset for vehicle classification and localization", IEEE TIP 2018. Weights trained on it
inherit those terms.
"""
import csv
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml
from PIL import Image

from vtia.adapters.yolo_training import CLASS_ID, IGNORE_FILL, NAMES
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
Box = tuple[float, float, float, float]  # normalised cx, cy, w, h


def _key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _road_id(name: str) -> int | None:
    """VTIA class id, None for the ignored classes; ValueError for names MIO-TCD doesn't have."""
    key = _key(name)
    if key in MIO_TCD_TO_ROAD:
        return CLASS_ID[MIO_TCD_TO_ROAD[key]]
    if key in MIO_TCD_IGNORED:
        return None
    raise ValueError(f"class not in MIO-TCD, map it in vtia/adapters/mio_tcd.py first: {name!r}")


def _link_or_copy(src: Path, dest: Path) -> None:
    try:
        os.symlink(src.resolve(), dest)
    except OSError:  # Windows without symlink rights
        shutil.copy2(src, dest)


def _write(image: Path, boxes: list[tuple[int | None, Box]], out: Path, split: str, counts: Counter) -> None:
    """One image + its YOLO label file; boxes of ignored classes (id None) are painted grey."""
    keep = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls, (cx, cy, w, h) in boxes if cls is not None]
    ignore = [box for cls, box in boxes if cls is None]
    for cls, _ in boxes:
        counts[(split, "ignore" if cls is None else list(RoadClass)[cls].value)] += 1
    dest = out / split / "images" / image.name
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if ignore:
        img = cv2.imread(str(image))
        ih, iw = img.shape[:2]
        for cx, cy, w, h in ignore:
            cv2.rectangle(img, (int((cx - w / 2) * iw), int((cy - h / 2) * ih)),
                          (int((cx + w / 2) * iw), int((cy + h / 2) * ih)), IGNORE_FILL, -1)
        cv2.imwrite(str(dest), img)
    else:
        _link_or_copy(image, dest)
    (out / split / "labels" / f"{image.stem}.txt").write_text("\n".join(keep), encoding="utf-8")


def _data_yaml(out: Path, splits: list[str]) -> Path:
    data = out / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(out.resolve()), **{s: f"{s}/images" for s in splits},
                                    "names": dict(enumerate(NAMES))}), encoding="utf-8")
    return data


def _mkdirs(out: Path, splits) -> None:
    for split in splits:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)


# --- official release ------------------------------------------------------------------------

def convert_official(root: Path, out: Path, sample: int | None = None, val_fraction: float = 0.1,
                     seed: int = 0) -> tuple[Path, Counter]:
    """MIO-TCD-Localization/ (train/ + gt_train.csv) -> YOLO dataset under `out`.
    `sample` keeps that many random training images (the full set is ~110k). The validation split is a
    random `val_fraction` of the images: MIO-TCD has no camera ids, so near-identical frames of one camera
    can fall on both sides and flatter the validation mAP. Returns data.yaml and boxes per (split, class)."""
    root, out = Path(root), Path(out)
    if not (root / "gt_train.csv").exists():
        root = root / "MIO-TCD-Localization"
    rows = defaultdict(list)
    with open(root / "gt_train.csv", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 6:
                rows[row[0].strip()].append((_road_id(row[1].strip()), [float(v) for v in row[2:6]]))
    ids = sorted(rows)
    rng = random.Random(seed)
    rng.shuffle(ids)
    if sample:
        ids = ids[:sample]
    n_val = max(1, round(len(ids) * val_fraction))
    _mkdirs(out, ("train", "val"))
    counts: Counter = Counter()
    for i, image_id in enumerate(ids):
        image = root / "train" / f"{image_id}.jpg"
        if not image.exists():
            continue
        with Image.open(image) as im:
            iw, ih = im.size
        boxes = []
        for cls, (x1, y1, x2, y2) in rows[image_id]:
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(iw), x2), min(float(ih), y2)
            if x2 > x1 and y2 > y1:
                boxes.append((cls, ((x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih, (x2 - x1) / iw, (y2 - y1) / ih)))
        _write(image, boxes, out, "val" if i < n_val else "train", counts)
    return _data_yaml(out, ["train", "val"]), counts


# --- Roboflow export (YOLO format) -----------------------------------------------------------

def _names(data: dict) -> list[str]:
    names = data["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def _box(values: list[float]) -> Box:
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
    target = {i: _road_id(n) for i, n in enumerate(names)}
    counts: Counter = Counter()
    splits = []
    for split, folder in (("train", "train"), ("val", "valid"), ("val", "val"), ("test", "test")):
        images = src / folder / "images"
        if split in splits or not images.is_dir():
            continue
        splits.append(split)
        _mkdirs(out, (split,))
        for image in sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            label = src / folder / "labels" / f"{image.stem}.txt"
            rows = label.read_text(encoding="utf-8").split("\n") if label.exists() else []
            boxes = [(target[int(p[0])], _box([float(v) for v in p[1:]])) for p in (r.split() for r in rows)
                     if len(p) >= 5]
            _write(image, boxes, out, split, counts)
    if "train" not in splits or "val" not in splits:
        raise ValueError(f"{src} needs train/images and valid/images (Roboflow export, YOLO format)")
    return _data_yaml(out, splits), counts


def convert(src: Path, out: Path, sample: int | None = None) -> tuple[Path, Counter]:
    """Official release if `src` has gt_train.csv (directly or in MIO-TCD-Localization/), else Roboflow."""
    src = Path(src)
    if (src / "gt_train.csv").exists() or (src / "MIO-TCD-Localization" / "gt_train.csv").exists():
        return convert_official(src, out, sample=sample)
    if sample:
        raise ValueError("--sample only applies to the official release")
    return convert_export(src, out)
