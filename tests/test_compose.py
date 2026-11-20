"""Composition root: what a run leaves in its output folder and what it hands the tracker.

Run: python -m unittest discover -s tests
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vtia.application.compare_models import Comparison
from vtia.application.process_video import VideoAnalysis
from vtia.compose import analyze_file, compare_file, prelabel_file
from vtia.domain.model_selection import Disagreement, Verdict

POINT = Path(__file__).resolve().parents[1] / "configs" / "points" / "sample_vehicles.yaml"


class AnalyzeFileTest(unittest.TestCase):
    def test_run_without_video_removes_the_annotated_video_of_a_previous_run(self):
        analysis = VideoAnalysis([], [], [], frames=10, fps=30.0, start=None, tracks=0, mean_confidence=0.0)
        with tempfile.TemporaryDirectory() as tmp:
            stale = Path(tmp) / "clip" / "annotated.mp4"
            stale.parent.mkdir()
            stale.write_bytes(b"video from an earlier analysis")
            # Only the model loading and the frame loop are replaced; outputs are really written.
            with patch("vtia.compose.make_detector"), patch("vtia.compose.process_video", return_value=analysis):
                run = analyze_file(Path("clip.mp4"), POINT, out=Path(tmp), save_video=False)
            self.assertFalse(stale.exists())
            self.assertTrue((run.out / "report.pdf").exists())


class PrelabelFileTest(unittest.TestCase):
    def test_prelabel_gives_the_tracker_the_video_frame_rate(self):
        class Spy:
            def track(self, frames, frame_rate):
                self.frame_rate = frame_rate
                return []

        spy = Spy()
        with tempfile.TemporaryDirectory() as tmp, \
                patch("vtia.compose.video_info", return_value=(15.0, 1)), \
                patch("vtia.compose.make_detector"), \
                patch("vtia.compose.make_tracker", return_value=spy), \
                patch("vtia.adapters.cvat_mot.write_preannotations", return_value=0):
            prelabel_file(Path("clip.mp4"), Path(tmp))
        self.assertEqual(spy.frame_rate, 15.0)


class CompareFileTest(unittest.TestCase):
    def manifest(self, root, split):
        path = root / "splits.csv"
        path.write_text(f"clip,split\nclip,{split}\n", encoding="utf-8")
        return path

    def test_annotations_of_a_test_partition_clip_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp, patch("vtia.compose.compare_models") as compare:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "partición de prueba"):
                compare_file(Path("clip.mp4"), POINT, annotations=root / "a.zip", out=root,
                             splits=self.manifest(root, "test"))
            compare.assert_not_called()

    def test_test_partition_clip_can_be_compared_without_annotations(self):
        comparison = Comparison([], Verdict((), (), None), Disagreement({}, None))
        with tempfile.TemporaryDirectory() as tmp, patch("vtia.compose.compare_models", return_value=comparison):
            root = Path(tmp)
            run = compare_file(Path("clip.mp4"), POINT, out=root, splits=self.manifest(root, "test"))
            self.assertEqual(run.out, root / "clip" / "comparison")
            self.assertTrue((run.out / "summary.csv").is_file())

    def test_earlier_results_are_removed_but_the_cache_stays(self):
        comparison = Comparison([], Verdict((), (), None), Disagreement({}, None))
        with tempfile.TemporaryDirectory() as tmp, patch("vtia.compose.compare_models", return_value=comparison):
            root = Path(tmp)
            target = root / "clip" / "comparison"
            (target / "detections").mkdir(parents=True)
            (target / "old_model").mkdir()
            compare_file(Path("clip.mp4"), POINT, out=root, splits=root / "none.csv")
            self.assertFalse((target / "old_model").exists())
            self.assertTrue((target / "detections").exists())


if __name__ == "__main__":
    unittest.main()


class AnonymizeRunTest(unittest.TestCase):
    """Anonymizing after the analysis: people are found on the original, the blur lands on the annotated video."""

    @staticmethod
    def noise_video(path: Path, size: tuple[int, int], frames: int = 3) -> None:
        import numpy as np
        from vtia.adapters.video_io import open_writer
        rng = np.random.default_rng(0)
        writer = open_writer(path, 10.0, size)
        for _ in range(frames):
            writer.write(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
        writer.release()

    def run_on(self, tmp: Path, annotated: bool):
        from vtia.compose import Run, anonymize_run
        from vtia.domain.classes import RoadClass
        from vtia.domain.models import Detection

        original = tmp / "clip.mp4"
        self.noise_video(original, (400, 200))
        out = tmp / "out"
        out.mkdir()
        if annotated:  # written at half the size, as AnnotatedVideo does with 4K input
            self.noise_video(out / "annotated.mp4", (200, 100))
        seen = []

        class Detector:  # one person in the left half of the original
            name, conf, imgsz = "yolo26s.pt", 0.1, 640

            def detect(self, source):
                seen.append(Path(source))
                for i in range(3):
                    yield [Detection(i, RoadClass.PEDESTRIAN, 0.9, (0.0, 0.0, 200.0, 200.0))]

        analysis = VideoAnalysis([], [], [], frames=3, fps=10.0, start=None, tracks=0, mean_confidence=0.0)
        run = Run(analysis, None, out, 1.0, original)
        with patch("vtia.compose.make_detector", return_value=Detector()), patch("vtia.compose.PlateFaceFinder"):
            blurred, sheet = anonymize_run(run)
        return original, blurred, sheet, seen

    def test_the_annotated_video_is_blurred_with_regions_found_on_the_original(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            original, blurred, sheet, seen = self.run_on(Path(tmp), annotated=True)
            self.assertEqual(seen, [original])  # never detects on the drawn boxes of the annotated video
            self.assertEqual(blurred.name, "annotated_anonimizado.mp4")
            self.assertTrue(sheet.exists())
            ok, img = cv2.VideoCapture(str(blurred)).read()
            self.assertTrue(ok)
            self.assertEqual(img.shape[:2], (100, 200))
            # The head (top 28 % of the person box) scaled by 0.5 is smooth; the right half keeps its noise.
            self.assertLess(img[2:24, 5:95].std(), img[40:95, 110:195].std() / 2)

    def test_without_an_annotated_video_the_analysed_video_is_blurred(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, blurred, _, _ = self.run_on(Path(tmp), annotated=False)
            self.assertEqual(blurred.name, "clip_anonimizado.mp4")
            self.assertTrue(original.exists())  # the original is never touched

    def test_a_new_analysis_removes_the_blurred_video_of_the_previous_one(self):
        analysis = VideoAnalysis([], [], [], frames=10, fps=30.0, start=None, tracks=0, mean_confidence=0.0)
        with tempfile.TemporaryDirectory() as tmp:
            stale = Path(tmp) / "clip" / "annotated_anonimizado.mp4"
            stale.parent.mkdir()
            stale.write_bytes(b"blurred copy of an earlier analysis")
            with patch("vtia.compose.make_detector"), patch("vtia.compose.process_video", return_value=analysis):
                analyze_file(Path("clip.mp4"), POINT, out=Path(tmp), save_video=False)
            self.assertFalse(stale.exists())
