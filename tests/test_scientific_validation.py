import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from vtia.adapters.cached_detector import CachedDetector
from vtia.adapters.cvat_mot import read_annotations
from vtia.adapters.scientific_report import write_scientific_pdf
from vtia.application.evaluate import AnnotatedSequence, evaluate
from vtia.domain.classes import RoadClass
from vtia.domain.counting_evaluation import count_error, evaluate_count_rows
from vtia.domain.dataset_validation import read_split_manifest, select_training_sequences
from vtia.domain.models import Detection, GroundTruth
from vtia.domain.splits import validate_split_assignment, parse_clip
from vtia.domain.robustness import summarize_robustness


class CountEvaluationTest(unittest.TestCase):
    def test_zero_manual_count_has_no_percentage(self):
        result = count_error(0, 3)
        self.assertEqual(result["absolute_difference"], 3)
        self.assertIsNone(result["error_percent"])

    def test_mae_rmse_and_group_breakdown(self):
        result = evaluate_count_rows([
            {"manual": "10", "automatic": "8", "class": "car", "video": "v1"},
            {"manual": "0", "automatic": "2", "class": "car", "video": "v2"},
        ])
        self.assertEqual(result["overall"]["mae"], 2)
        self.assertAlmostEqual(result["overall"]["rmse"], 2)
        self.assertIn("car", result["breakdown"]["class"])
        self.assertIsNone(result["records"][1]["error_percent"])

    def test_negative_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            count_error(-1, 0)

    def test_unknown_class_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown class"):
            evaluate_count_rows([{"manual": "1", "automatic": "1", "class": "lorry"}])


class SplitManifestTest(unittest.TestCase):
    def test_manifest_rejects_duplicate_clip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "splits.csv"
            path.write_text("clip,split\nP1_20260101_0700_s0001,fold1\nP1_20260101_0700_s0001,test\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_split_manifest(path)

    def test_session_cannot_cross_partitions(self):
        names = ["P1_20260101_0700_s0001", "P1_20260101_0700_s0300"]
        issues = validate_split_assignment(names, {names[0]: "test", names[1]: "fold1"}, folds=1)
        self.assertTrue(any("spans partitions" in issue for issue in issues))

    def test_manifest_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "splits.csv"
            path.write_text("clip,split\nP1_20260101_0700_s0001,fold1\n")
            sidecar = path.with_suffix(".manifest.json")
            sidecar.write_text(json.dumps({"assignment_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                read_split_manifest(path)

    def test_training_selection_never_returns_fixed_test_sequences(self):
        class Seq:
            def __init__(self, name): self.name = name
        names = ["P1_20260101_0700", "P1_20260102_0700", "P1_20260103_0700"]
        assignment = {names[0]: "test", names[1]: "fold1", names[2]: "fold2"}
        train, validation = select_training_sequences([Seq(n) for n in names], assignment, 1)
        self.assertEqual([s.name for s in train], [names[2]])
        self.assertEqual([s.name for s in validation], [names[1]])
        self.assertNotIn(names[0], [s.name for s in train + validation])

    def test_capture_time_bands_follow_protocol_boundaries(self):
        self.assertEqual(parse_clip("P1_20260101_0630").stratum, "P1/morning")
        self.assertEqual(parse_clip("P1_20260101_0900").stratum, "P1/other")
        self.assertEqual(parse_clip("P1_20260101_1130").stratum, "P1/midday")
        self.assertEqual(parse_clip("P1_20260101_1630").stratum, "P1/afternoon")
        self.assertEqual(parse_clip("P1_20260101_1830").stratum, "P1/other")


class CvatValidationTest(unittest.TestCase):
    def _archive(self, root, rows, labels="pedestrian\n"):
        path = root / "clip.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("gt/gt.txt", rows)
            z.writestr("gt/labels.txt", labels)
        return path

    @mock.patch("vtia.adapters.cvat_mot.video_info", return_value=(25.0, 3))
    def test_sequence_keeps_the_frame_rate_of_its_video(self, _):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            seq = read_annotations(self._archive(root, "1,1,10,10,20,30,1,1,1\n"), root / "clip.mp4")
        self.assertEqual(seq.fps, 25.0)

    @mock.patch("vtia.adapters.cvat_mot.video_info", return_value=(25.0, 3))
    def test_valid_mot_row_is_loaded(self, _):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = self._archive(root, "1,1,10,10,20,30,1,1,1\n")
            seq = read_annotations(archive, root / "clip.mp4")
            self.assertEqual(len(seq.truths[0]), 1)
            self.assertEqual(seq.truths[0][0].road_class, RoadClass.PEDESTRIAN)

    @mock.patch("vtia.adapters.cvat_mot.video_info", return_value=(25.0, 3))
    def test_out_of_range_frame_is_rejected_not_silently_dropped(self, _):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = self._archive(root, "4,1,10,10,20,30,1,1,1\n")
            with self.assertRaisesRegex(ValueError, "outside video range"):
                read_annotations(archive, root / "clip.mp4")

    @mock.patch("vtia.adapters.cvat_mot.video_info", return_value=(25.0, 3))
    def test_blank_label_line_is_rejected(self, _):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = self._archive(root, "1,1,10,10,20,30,1,1,1\n", "pedestrian\n\ncar\n")
            with self.assertRaisesRegex(ValueError, "blank lines"):
                read_annotations(archive, root / "clip.mp4")

    @mock.patch("vtia.adapters.cvat_mot.video_info", return_value=(25.0, 3))
    def test_invalid_box_is_rejected(self, _):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = self._archive(root, "1,1,10,10,0,30,1,1,1\n")
            with self.assertRaisesRegex(ValueError, "width/height"):
                read_annotations(archive, root / "clip.mp4")


class RobustnessTest(unittest.TestCase):
    def test_insufficient_conditions_are_explicit(self):
        per_sequence = {f"s{i}": {"detection": {"mAP50": 0.5}, "tracking": {"HOTA": 0.4},
                                  "classes": {"macro_f1": 0.6}} for i in range(2)}
        rows = [{"sequence": f"s{i}", "lighting": "low"} for i in range(2)]
        result = summarize_robustness(per_sequence, rows, min_sequences=3)
        self.assertEqual(result["conditions"]["lighting"]["low"]["status"], "insufficient_evidence")

    def test_supported_condition_averages_metrics(self):
        per_sequence = {f"s{i}": {"detection": {"mAP50": 0.5 + i * 0.1}, "tracking": {"HOTA": 0.4},
                                  "classes": {"macro_f1": 0.6}} for i in range(3)}
        rows = [{"sequence": f"s{i}", "lighting": "good"} for i in range(3)]
        result = summarize_robustness(per_sequence, rows, min_sequences=3)
        self.assertEqual(result["conditions"]["lighting"]["good"]["status"], "supported")
        self.assertAlmostEqual(result["conditions"]["lighting"]["good"]["metrics_mean"]["mAP50"], 0.6)


class PerSequenceEvaluationTest(unittest.TestCase):
    def test_evaluation_returns_metrics_per_sequence(self):
        class Detector:
            name = "oracle"
            def detect(self, source):
                yield [Detection(0, RoadClass.CAR, 0.9, (0, 0, 10, 10), 1)]
        class Tracker:
            def track(self, frames, frame_rate=30.0):
                yield from frames
        seq = AnnotatedSequence("s1", Path("s1.mp4"),
                                [[GroundTruth(0, 1, RoadClass.CAR, (0, 0, 10, 10))]], [[]])
        result = evaluate([seq], Detector(), Tracker(),
                          lambda truths, detections: {"mAP50": 1.0, "mAP50_95": 1.0},
                          lambda sequences: {"HOTA": 1.0, "MOTA": 1.0, "IDF1": 1.0})
        self.assertIn("s1", result.per_sequence)
        self.assertEqual(result.per_sequence["s1"]["ground_truth_instances"], 1)


class ScientificPdfTest(unittest.TestCase):
    def test_pdf_report_is_written(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "report.pdf"
            write_scientific_pdf(path, "Test report", {"dataset": "synthetic"}, [["m", "1"]], ["metric", "value"])
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 500)


class _FakeDetector:
    name = "fake"
    conf = 0.1
    imgsz = 640
    device = "cpu"
    def __init__(self):
        self.calls = 0
    def detect(self, source):
        self.calls += 1
        yield [Detection(0, RoadClass.CAR, 0.9, (1, 2, 3, 4))]


class CacheTest(unittest.TestCase):
    def test_cache_is_reused_and_invalidated_when_source_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "clip.mp4"
            source.write_bytes(b"video-a")
            inner = _FakeDetector()
            cache = CachedDetector(inner, root / "cache")
            self.assertEqual(len(list(cache.detect(source))), 1)
            self.assertEqual(len(list(cache.detect(source))), 1)
            self.assertEqual(inner.calls, 1)
            source.write_bytes(b"video-b")
            self.assertEqual(len(list(cache.detect(source))), 1)
            self.assertEqual(inner.calls, 2)

    def test_frame_directory_is_supported_and_hashed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "sequence"
            source.mkdir()
            frame = source / "000001.jpg"
            frame.write_bytes(b"frame-a")
            inner = _FakeDetector()
            cache = CachedDetector(inner, root / "cache")
            list(cache.detect(source))
            list(cache.detect(source))
            self.assertEqual(inner.calls, 1)
            frame.write_bytes(b"frame-b")
            list(cache.detect(source))
            self.assertEqual(inner.calls, 2)


if __name__ == "__main__":
    unittest.main()
