import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from vtia.application.evaluate import AnnotatedSequence
from vtia.cli import _sequences, main


class ScientificCliTest(unittest.TestCase):
    def _manifest(self, root):
        names = [f"P1_202610{day:02d}_0700_s0001" for day in range(1, 7)]
        parts = ["test", "fold1", "fold2", "fold3", "fold4", "fold5"]
        path = root / "splits.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["clip", "split"])
            writer.writerows(zip(names, parts))
        return path, names, parts

    def test_cvat_evaluation_requires_explicit_partition_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = Namespace(visdrone=None, cvat=root, videos=root, sequences=None, splits=None,
                             partition=None, command_name="evaluate")
            with self.assertRaisesRegex(SystemExit, "requires --splits"):
                _sequences(args)

    def test_calibration_rejects_fixed_test_partition(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, names, parts = self._manifest(root)
            seq = AnnotatedSequence(names[0], root / "v.mp4", [[]], [[]])
            args = Namespace(visdrone=None, cvat=root, videos=root, sequences=None, splits=manifest,
                             partition="test", command_name="calibrate")
            with mock.patch("vtia.adapters.cvat_mot.load_cvat_dir", return_value=[seq]):
                with self.assertRaisesRegex(SystemExit, "cannot be used for calibration"):
                    _sequences(args)

    def test_evaluation_selects_only_the_requested_partition(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, names, parts = self._manifest(root)
            test_seq = AnnotatedSequence(names[0], root / "test.mp4", [[]], [[]])
            fold_seq = AnnotatedSequence(names[1], root / "fold.mp4", [[]], [[]])
            args = Namespace(visdrone=None, cvat=root, videos=root, sequences=None, splits=manifest,
                             partition="test", command_name="evaluate")
            with mock.patch("vtia.adapters.cvat_mot.load_cvat_dir", return_value=[test_seq, fold_seq]):
                selected = _sequences(args)
            self.assertEqual([s.name for s in selected], [names[0]])

    def test_explicit_sequence_from_wrong_partition_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, names, parts = self._manifest(root)
            seq = AnnotatedSequence(names[1], root / "v.mp4", [[]], [[]])
            args = Namespace(visdrone=None, cvat=root, videos=root, sequences=[names[1]], splits=manifest,
                             partition="test", command_name="evaluate")
            with mock.patch("vtia.adapters.cvat_mot.load_cvat_dir", return_value=[seq]):
                with self.assertRaisesRegex(SystemExit, "explicitly requested"):
                    _sequences(args)

    def test_cli_help_is_available_without_model_dependencies(self):
        with self.assertRaises(SystemExit) as caught:
            main(["--help"])
        self.assertEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
