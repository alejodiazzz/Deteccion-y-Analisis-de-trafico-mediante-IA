import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from vtia.adapters.mio_tcd import convert, convert_export, convert_official
from vtia.adapters.yolo_training import IGNORE_FILL, NAMES

ROBOFLOW_NAMES = ["Articulated_truck", "Bicycle", "Bus", "Car", "Motorcycle", "Motorized_vehicle",
                  "Non-motorized_vehicle", "Pedestrian", "Pickup_truck", "Single_unit_truck", "Work_van"]


def export(root: Path, names=ROBOFLOW_NAMES) -> None:
    root.mkdir()
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names, "nc": len(names)}), encoding="utf-8")
    for folder in ("train", "valid"):
        (root / folder / "images").mkdir(parents=True)
        (root / folder / "labels").mkdir(parents=True)
        cv2.imwrite(str(root / folder / "images" / "a.jpg"), np.zeros((100, 200, 3), np.uint8))
        (root / folder / "labels" / "a.txt").write_text(
            "8 0.25 0.5 0.1 0.2\n"           # pickup -> car
            "0 0.5 0.5 0.2 0.2\n"            # articulated truck -> truck
            "7 0.1 0.1 0.2 0.1 0.2 0.3\n"    # pedestrian as a polygon
            "10 0.8 0.5 0.2 0.4\n", encoding="utf-8")  # work van -> ignore


class ConvertExportTest(unittest.TestCase):
    def test_classes_are_remapped_and_ambiguous_ones_painted_grey(self):
        with tempfile.TemporaryDirectory() as d:
            export(Path(d) / "src")
            data, counts = convert_export(Path(d) / "src", Path(d) / "out")
            spec = yaml.safe_load(data.read_text(encoding="utf-8"))
            self.assertEqual(spec["names"], dict(enumerate(NAMES)))
            self.assertEqual((spec["train"], spec["val"]), ("train/images", "val/images"))
            rows = (Path(d) / "out" / "val" / "labels" / "a.txt").read_text(encoding="utf-8").split("\n")
            self.assertEqual([int(r.split()[0]) for r in rows],
                             [NAMES.index("car"), NAMES.index("truck"), NAMES.index("pedestrian")])
            self.assertEqual(rows[2].split()[1:], ["0.150000", "0.200000", "0.100000", "0.200000"])
            img = cv2.imread(str(Path(d) / "out" / "train" / "images" / "a.jpg"))
            self.assertEqual(tuple(int(v) for v in img[50, 160]), IGNORE_FILL)  # inside the work van
            self.assertEqual(tuple(int(v) for v in img[50, 10]), (0, 0, 0))
            self.assertEqual(counts[("train", "ignore")], 1)
            self.assertEqual(counts[("val", "car")], 1)

    def test_unknown_classes_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            export(Path(d) / "src", names=ROBOFLOW_NAMES + ["tram"])
            with self.assertRaisesRegex(ValueError, "tram"):
                convert_export(Path(d) / "src", Path(d) / "out")


def official(root: Path, n_images: int = 10) -> None:
    """MIO-TCD-Localization/ like the real one: train/<id>.jpg + gt_train.csv (id,label,x1,y1,x2,y2)."""
    loc = root / "MIO-TCD-Localization"
    (loc / "train").mkdir(parents=True)
    rows = []
    for i in range(n_images):
        image_id = f"{i:08d}"
        cv2.imwrite(str(loc / "train" / f"{image_id}.jpg"), np.zeros((100, 200, 3), np.uint8))
        rows += [f"{image_id},pickup_truck,10,10,30,30", f"{image_id},car,40,10,80,50",
                 f"{image_id},work_van,150,40,190,90", f"{image_id},pedestrian,190,0,260,20"]
    (loc / "gt_train.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


class ConvertOfficialTest(unittest.TestCase):
    def test_every_box_of_an_image_is_kept_and_val_is_carved_from_train(self):
        with tempfile.TemporaryDirectory() as d:
            official(Path(d) / "src")
            data, counts = convert(Path(d) / "src", Path(d) / "out")  # finds MIO-TCD-Localization/ inside
            spec = yaml.safe_load(data.read_text(encoding="utf-8"))
            self.assertEqual((spec["train"], spec["val"]), ("train/images", "val/images"))
            out = Path(d) / "out"
            self.assertEqual(len(list((out / "val" / "labels").glob("*.txt"))), 1)
            self.assertEqual(len(list((out / "train" / "labels").glob("*.txt"))), 9)
            rows = (out / "train" / "labels" / "00000003.txt").read_text(encoding="utf-8").split("\n")
            self.assertEqual(rows, [f"{NAMES.index('car')} 0.100000 0.200000 0.100000 0.200000",
                                    f"{NAMES.index('car')} 0.300000 0.300000 0.200000 0.400000",
                                    f"{NAMES.index('pedestrian')} 0.975000 0.100000 0.050000 0.200000"])  # clipped
            img = cv2.imread(str(out / "train" / "images" / "00000003.jpg"))
            self.assertEqual(tuple(int(v) for v in img[60, 170]), IGNORE_FILL)  # the work van
            self.assertEqual(counts[("train", "car")] + counts[("val", "car")], 20)
            self.assertEqual(counts[("train", "ignore")] + counts[("val", "ignore")], 10)

    def test_sample_keeps_n_images(self):
        with tempfile.TemporaryDirectory() as d:
            official(Path(d) / "src")
            convert_official(Path(d) / "src", Path(d) / "out", sample=4)
            n = sum(len(list((Path(d) / "out" / s / "labels").glob("*.txt"))) for s in ("train", "val"))
            self.assertEqual(n, 4)


if __name__ == "__main__":
    unittest.main()
