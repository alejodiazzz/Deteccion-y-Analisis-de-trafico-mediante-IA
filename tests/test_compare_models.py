"""The comparison use case, with fake detectors and a pass-through tracker.

Run: python -m unittest discover -s tests
"""
import csv
import tempfile
import unittest
from pathlib import Path

from vtia.adapters.comparison_files import clear_results, write_comparison
from vtia.application.compare_models import ManualCount, compare_models
from vtia.application.evaluate import AnnotatedSequence
from vtia.domain.classes import RoadClass as C
from vtia.domain.models import Detection, GroundTruth, Line, PointConfig

POINT = PointConfig((Line("main", (0, 100), (200, 100), "down", "up"),), ())
BOX_ABOVE, BOX_BELOW = (40, 60, 60, 80), (40, 120, 60, 140)  # anchors at y=80 and y=140


class Fake:
    """Detector + tracker: one car crossing the line, reported as `cls`."""

    def __init__(self, name, cls=C.CAR, frames=10, seconds=1.0):
        self.name, self.cls, self.frames, self.seconds = name, cls, frames, seconds

    def detect(self, source):
        yield [Detection(0, self.cls, 0.9, BOX_ABOVE, 1)]
        yield [Detection(1, self.cls, 0.9, BOX_BELOW, 1)]

    def track(self, frames, frame_rate=30.0):
        return frames


def build_for(**classes):
    def build(name):
        fake = Fake(name, classes.get(name, C.CAR))
        return fake, fake
    return build


def truth():
    frames = [[GroundTruth(0, 1, C.CAR, BOX_ABOVE)], [GroundTruth(1, 1, C.CAR, BOX_BELOW)]]
    return AnnotatedSequence("clip", Path("clip.mp4"), frames, [[], []])


def fake_map(truths, dets):
    return {"mAP50": 0.5, "mAP50_95": 0.25, "per_class": {c: {"AP50": 0.5, "AP50_95": 0.25} for c in C}}


def fake_mot(sequences):
    return {"HOTA": 0.4, "MOTA": 0.3, "IDF1": 0.2, "IDSW": 1}


def run(models=("a", "b"), build=None, **kwargs):
    return compare_models(Path("clip.mp4"), POINT, 30.0, models, build or build_for(), **kwargs)


class CompareModelsTest(unittest.TestCase):
    def test_without_reference_there_is_no_winner_and_no_evaluation(self):
        c = run()
        self.assertEqual([m.crossings for m in c.models], [1, 1])
        self.assertEqual(c.models[0].counts_by_class, {C.CAR: 1})
        self.assertIsNone(c.models[0].evaluation)
        self.assertEqual((c.verdict.best_by_annotations, c.verdict.best_by_counts), ((), ()))
        self.assertEqual(c.models[0].fps, 10.0)

    def test_annotations_pick_the_model_that_classifies_correctly(self):
        c = run(build=build_for(a=C.TRUCK, b=C.CAR), truth=truth(), map_metric=fake_map, mot_metric=fake_mot)
        self.assertEqual(c.verdict.best_by_annotations, ("b",))
        self.assertEqual(c.models[1].evaluation.confusion[(C.CAR, C.CAR)], 2)

    def test_manual_counts_per_class_pick_the_closest_model(self):
        c = run(build=build_for(a=C.TRUCK, b=C.CAR), manual=[ManualCount(C.CAR, 1)])
        self.assertEqual(c.models[0].counting["overall"]["mae"], 1)
        self.assertEqual(c.models[1].counting["overall"]["mae"], 0)
        self.assertEqual(c.verdict.best_by_counts, ("b",))

    def test_manual_counts_per_line_and_direction(self):
        c = run(models=("a",), manual=[ManualCount(C.CAR, 3, "main", "down"), ManualCount(C.CAR, 0, "main", "up")])
        self.assertEqual([(r["manual"], r["automatic"]) for r in c.models[0].counting["records"]], [(3, 1), (0, 0)])

    def test_a_failing_model_is_reported_and_the_others_continue(self):
        def build(name):
            if name == "bad":
                raise RuntimeError("weights not found")
            fake = Fake(name)
            return fake, fake

        c = run(models=("bad", "good"), build=build, manual=[ManualCount(C.CAR, 1)])
        self.assertEqual(c.models[0].error, "weights not found")
        self.assertIsNone(c.models[1].error)
        self.assertEqual(c.verdict.best_by_counts, ("good",))

    def test_reports_each_model_as_it_starts(self):
        seen = []
        run(on_model=lambda i, n, name: seen.append((i, n, name)))
        self.assertEqual(seen, [(1, 2, "a"), (2, 2, "b")])


class ComparisonFilesTest(unittest.TestCase):
    def test_writes_summary_and_tables_of_each_evaluated_model(self):
        c = run(truth=truth(), map_metric=fake_map, mot_metric=fake_mot)
        with tempfile.TemporaryDirectory() as tmp:
            write_comparison(Path(tmp), c)
            with (Path(tmp) / "summary.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["model"] for r in rows], ["a", "b"])
            self.assertEqual(rows[0]["mAP50"], "0.5")
            self.assertTrue((Path(tmp) / "a" / "confusion.csv").is_file())
            self.assertTrue((Path(tmp) / "b" / "per_class.csv").is_file())

    def test_clearing_keeps_only_the_detection_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "detections").mkdir()
            (root / "detections" / "cached.jsonl").write_text("x")
            (root / "yolo").mkdir()
            (root / "yolo" / "confusion.csv").write_text("x")
            (root / "summary.csv").write_text("x")
            clear_results(root)
            self.assertEqual([p.name for p in root.iterdir()], ["detections"])
            self.assertTrue((root / "detections" / "cached.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
