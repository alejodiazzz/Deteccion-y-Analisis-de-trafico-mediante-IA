import tempfile
import unittest
from pathlib import Path

from vtia.adapters.detection_log import RecordingDetector, recorded_detections
from vtia.domain.classes import RoadClass as C
from vtia.domain.models import Detection


class Fake:
    name, conf, imgsz = "yolo26s", 0.1, 640

    def detect(self, source):
        yield [Detection(0, C.CAR, 0.9, (0, 0, 10, 10)), Detection(0, C.PEDESTRIAN, 0.12, (5, 5, 8, 9))]
        yield []


class DetectionLogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = Path(self.tmp.name) / "clip.mp4"
        self.log = Path(self.tmp.name) / "clip.detections.jsonl"
        list(RecordingDetector(Fake(), self.log).detect(Path("raw.mp4")))

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_model_replays_the_recorded_frames(self):
        replay = recorded_detections(self.video, "yolo26s.pt", conf=0.1, frames=2)
        frames = list(replay.detect(self.video))
        self.assertEqual([len(f) for f in frames], [2, 0])
        self.assertEqual(frames[0][0], Detection(0, C.CAR, 0.9, (0, 0, 10, 10)))

    def test_a_higher_confidence_filters_the_replay(self):
        frames = list(recorded_detections(self.video, "yolo26s.pt", conf=0.5, frames=2).detect(self.video))
        self.assertEqual([len(f) for f in frames], [1, 0])

    def test_no_replay_when_it_would_differ_from_running_the_model(self):
        self.assertIsNone(recorded_detections(self.video, "yolo26n.pt", conf=0.1, frames=2))   # other model
        self.assertIsNone(recorded_detections(self.video, "yolo26s.pt", conf=0.05, frames=2))  # lower conf
        self.assertIsNone(recorded_detections(self.video, "yolo26s.pt", conf=0.1, frames=3))   # other video
        self.assertIsNone(recorded_detections(Path(self.tmp.name) / "x.mp4", "yolo26s.pt", 0.1, 2))

    def test_an_interrupted_run_leaves_no_log(self):
        log = Path(self.tmp.name) / "cut.detections.jsonl"
        stream = RecordingDetector(Fake(), log).detect(Path("raw.mp4"))
        next(stream)
        stream.close()
        self.assertFalse(log.exists())
