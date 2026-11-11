"""Zones, dwell time, alert rules, file-name clock and point config.

Run: python -m unittest discover -s tests
"""
import tempfile
import unittest
from datetime import datetime, time
from pathlib import Path

from vtia.adapters.yaml_config import load_point
from vtia.application.process_video import process_video
from vtia.domain.alerts import AlertRules, in_window
from vtia.domain.classes import RoadClass as C
from vtia.domain.models import CrossingEvent, Detection, Line, PointConfig, Restriction, Zone
from vtia.domain.timeline import video_start
from vtia.domain.zones import ZoneMonitor, inside

SQUARE = ((0, 0), (100, 0), (100, 100), (0, 100))
FPS = 10


def at(x, y, track=1, cls=C.PEDESTRIAN, frame=0):
    """Detection whose anchor (bottom-centre) is (x, y)."""
    return Detection(frame, cls, 0.9, (x - 5, y - 20, x + 5, y), track)


def feed(monitor, rules, frames):
    """frames: list of per-frame detection lists. Returns every alert raised."""
    alerts = []
    for f, dets in enumerate(frames):
        monitor.update(f, dets)
        alerts += rules.update(f, [], monitor)
    return alerts


class ZoneTest(unittest.TestCase):
    def test_point_in_polygon(self):
        self.assertTrue(inside((50, 50), SQUARE))
        self.assertFalse(inside((150, 50), SQUARE))
        l_shape = ((0, 0), (100, 0), (100, 40), (40, 40), (40, 100), (0, 100))  # concave
        self.assertFalse(inside((70, 70), l_shape))
        self.assertTrue(inside((20, 70), l_shape))

    def test_visit_closes_after_the_gap_and_keeps_its_dwell(self):
        m = ZoneMonitor([Zone("z", SQUARE)], gap_frames=3)
        for f in range(10):
            m.update(f, [at(50, 50)])        # inside for frames 0..9
        for f in range(10, 15):
            m.update(f, [at(500, 50)])       # outside
        v = m.visits[0]
        self.assertEqual((v.zone, v.track_id, v.enter_frame, v.exit_frame), ("z", 1, 0, 9))

    def test_short_dropout_does_not_split_a_visit(self):
        m = ZoneMonitor([Zone("z", SQUARE)], gap_frames=3)
        for f in [0, 1, 2, 4, 5]:            # tracker lost it in frame 3
            m.update(f, [at(50, 50)])
        self.assertEqual([(v.enter_frame, v.exit_frame) for v in m.finish()], [(0, 5)])

    def test_occupancy_leaves_out_objects_seen_outside_the_zone(self):
        m = ZoneMonitor([Zone("z", SQUARE)], gap_frames=30)
        m.update(0, [at(50, 50, track=1)])
        m.update(1, [at(500, 500, track=1), at(50, 50, track=2)])
        self.assertEqual(m.occupancy("z"), 1)

    def test_occupancy_keeps_an_object_the_tracker_lost_briefly(self):
        m = ZoneMonitor([Zone("z", SQUARE)], gap_frames=30)
        m.update(0, [at(50, 50)])
        m.update(1, [])
        self.assertEqual(m.occupancy("z"), 1)

    def test_object_that_steps_out_and_back_is_inside_again_in_one_visit(self):
        m = ZoneMonitor([Zone("z", SQUARE)], gap_frames=30)
        m.update(0, [at(50, 50)])
        m.update(1, [at(500, 500)])
        m.update(2, [at(50, 50)])
        self.assertEqual(m.occupancy("z"), 1)
        self.assertEqual([(v.enter_frame, v.exit_frame) for v in m.finish()], [(0, 2)])


class AlertTest(unittest.TestCase):
    def test_forbidden_class_fires_once_after_half_a_second(self):
        zone = Zone("roadway", SQUARE, forbidden=frozenset({C.PEDESTRIAN}))
        rules = AlertRules(PointConfig((), (zone,)), FPS)
        alerts = feed(ZoneMonitor([zone], 5), rules, [[at(50, 50)]] * 20)
        self.assertEqual([(a.kind, a.frame) for a in alerts], [("forbidden_in_zone", 4)])  # 5 frames = 0.5 s

    def test_long_dwell(self):
        zone = Zone("stop", SQUARE, max_dwell_s=1)
        alerts = feed(ZoneMonitor([zone], 5), AlertRules(PointConfig((), (zone,)), FPS), [[at(50, 50, cls=C.CAR)]] * 30)
        self.assertEqual([(a.kind, a.frame) for a in alerts], [("long_dwell", 10)])  # 11 frames > 1 s

    def test_crowding_fires_again_only_after_clearing(self):
        zone = Zone("plaza", SQUARE, max_occupancy=2)
        two = [at(20, 20, track=1), at(60, 60, track=2)]
        frames = [two] * 5 + [[]] * 10 + [two] * 5
        alerts = feed(ZoneMonitor([zone], 3), AlertRules(PointConfig((), (zone,)), FPS), frames)
        self.assertEqual([(a.kind, a.frame) for a in alerts], [("crowding", 0), ("crowding", 15)])

    def test_people_passing_one_at_a_time_are_not_a_crowd(self):
        zone = Zone("plaza", SQUARE, max_occupancy=2)
        frames = [[at(50, 50, track=1)],
                  [at(500, 500, track=1), at(50, 50, track=2)],
                  [at(500, 500, track=2), at(50, 50, track=3)]]
        alerts = feed(ZoneMonitor([zone], 30), AlertRules(PointConfig((), (zone,)), FPS), frames)
        self.assertEqual(alerts, [])

    def test_process_video_gives_the_tracker_the_video_frame_rate(self):
        class Source:
            name = "fake"

            def detect(self, source):
                return iter([[]])

        class Spy:
            def track(self, frames, frame_rate):
                self.frame_rate = frame_rate
                return frames

        spy = Spy()
        process_video(Path("clip.mp4"), Source(), spy, PointConfig((), ()), fps=15.0)
        self.assertEqual(spy.frame_rate, 15.0)

    def test_wrong_way_and_restricted_hours(self):
        line = Line("main", (0, 100), (200, 100), "down", "up", allowed="down")
        config = PointConfig((line,), (), (Restriction(C.TRUCK, ((time(6), time(9)),)),))
        rules = AlertRules(config, FPS, start=datetime(2026, 10, 15, 7, 30))
        crossings = [CrossingEvent(0, 1, C.CAR, "main", "up"), CrossingEvent(0, 2, C.TRUCK, "main", "down"),
                     CrossingEvent(0, 3, C.CAR, "main", "down")]
        alerts = rules.update(0, crossings, ZoneMonitor([], 3))
        self.assertEqual([(a.kind, a.track_id) for a in alerts], [("wrong_way", 1), ("restricted_hours", 2)])

    def test_time_windows_can_cross_midnight(self):
        self.assertTrue(in_window(time(23), time(22), time(5)))
        self.assertTrue(in_window(time(4, 59), time(22), time(5)))
        self.assertFalse(in_window(time(12), time(22), time(5)))
        self.assertFalse(in_window(time(9), time(6), time(9)))  # end excluded


class ClockTest(unittest.TestCase):
    def test_start_time_from_file_name(self):
        self.assertEqual(video_start("P1_20261015_0705"), datetime(2026, 10, 15, 7, 5))
        self.assertEqual(video_start("P2_20261015_0705_s0312"), datetime(2026, 10, 15, 7, 10, 12))
        self.assertIsNone(video_start("vehicles"))
        self.assertIsNone(video_start("P1_20261315_0705"))  # month 13


class PointConfigTest(unittest.TestCase):
    def test_sample_configs_load(self):
        for path in Path("configs/points").glob("*.yaml"):
            self.assertTrue(load_point(path).lines, path)

    def test_allowed_must_be_one_of_the_directions(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.yaml"
            bad.write_text("lines:\n  - {name: a, start: [0, 0], end: [1, 0], to_right: in, to_left: out, allowed: up}\n")
            with self.assertRaises(ValueError):
                load_point(bad)


if __name__ == "__main__":
    unittest.main()
