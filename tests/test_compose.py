"""Composition root: what a run leaves in its output folder and what it hands the tracker.

Run: python -m unittest discover -s tests
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vtia.application.process_video import VideoAnalysis
from vtia.compose import analyze_file, prelabel_file

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


if __name__ == "__main__":
    unittest.main()
