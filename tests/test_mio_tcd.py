import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from vtia.adapters.mio_tcd import convert_export
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


if __name__ == "__main__":
    unittest.main()
