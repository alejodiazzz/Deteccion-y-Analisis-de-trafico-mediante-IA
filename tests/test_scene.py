import unittest

from vtia.domain.classes import RoadClass as C
from vtia.domain.counting import LineCounter
from vtia.domain.models import Detection
from vtia.domain.scene import suggest_point
from vtia.domain.zones import inside


def track(tid, cls, start, end, frames=30, size=(50, 60), first=0):
    """Detections of one object moving in a straight line (anchor = bottom-centre of the box)."""
    w, h = size
    out = []
    for k in range(frames):
        x = start[0] + (end[0] - start[0]) * k / (frames - 1)
        y = start[1] + (end[1] - start[1]) * k / (frames - 1)
        out.append(Detection(first + k, cls, 0.9, (x - w / 2, y - h, x + w / 2, y), tid))
    return out


def by_frame(*tracks):
    frames = {}
    for t in tracks:
        for d in t:
            frames.setdefault(d.frame, []).append(d)
    return [frames.get(i, []) for i in range(max(frames) + 1)]


def two_way_road(n=5):
    down = [track(i, C.CAR, (300 + 10 * (i % 3), 100), (300 + 10 * (i % 3), 700), first=8 * i) for i in range(n)]
    up = [track(100 + i, C.CAR, (600 - 10 * (i % 3), 700), (600 - 10 * (i % 3), 100), first=8 * i) for i in range(n)]
    return down, up


class SuggestLinesTest(unittest.TestCase):
    def test_one_line_per_flow_counting_its_own_direction_as_allowed(self):
        down, up = two_way_road()
        point = suggest_point(by_frame(*down, *up))
        self.assertEqual(len(point.lines), 2)
        counter = LineCounter(list(point.lines))
        events = [e for frame in by_frame(*down, *up) for e in counter.update(frame)]
        self.assertEqual(len(events), 10)  # every car crosses exactly one line
        for line in point.lines:
            crossing = [e for e in events if e.line == line.name]
            self.assertEqual(len(crossing), 5)
            self.assertTrue(all(e.direction == line.allowed for e in crossing))
        self.assertEqual({line.allowed for line in point.lines}, {"toward_camera", "away_from_camera"})

    def test_parked_vehicles_and_thin_flows_get_no_line(self):
        down, _ = two_way_road()
        parked = [track(50 + i, C.CAR, (800, 400), (802, 401)) for i in range(5)]
        few = [track(70 + i, C.CAR, (100, 300), (500, 300)) for i in range(2)]  # only 2 tracks: noise
        point = suggest_point(by_frame(*down, *parked, *few))
        self.assertEqual(len(point.lines), 1)

    def test_no_moving_vehicles_no_lines(self):
        self.assertEqual(suggest_point(by_frame(track(1, C.PEDESTRIAN, (10, 10), (20, 20)))).lines, ())


class SuggestZonesTest(unittest.TestCase):
    def test_a_carriageway_zone_per_flow_covers_its_vehicles(self):
        down, up = two_way_road()
        point = suggest_point(by_frame(*down, *up))
        roads = [z for z in point.zones if z.name.startswith("calzada")]
        self.assertEqual(len(roads), 2)
        for flow in (down, up):
            anchors = [d.anchor for t in flow for d in t[3:-3]]
            self.assertTrue(any(all(inside(p, z.polygon) for p in anchors) for z in roads))

    def test_pedestrians_walking_in_one_place_make_one_zone(self):
        down, _ = two_way_road()
        people = [track(200 + i, C.PEDESTRIAN, (900, 300 + 15 * i), (960, 320 + 15 * i), size=(20, 50))
                  for i in range(4)]
        point = suggest_point(by_frame(*down, *people))
        walks = [z for z in point.zones if z.name.startswith("peatonal")]
        self.assertEqual(len(walks), 1)
        self.assertTrue(inside((930, 340), walks[0].polygon))
        self.assertFalse(inside((300, 400), walks[0].polygon))  # not on the road


class DumpPointTest(unittest.TestCase):
    def test_a_suggested_point_is_written_as_yaml_that_loads_back(self):
        import tempfile
        from pathlib import Path

        from vtia.adapters.yaml_config import dump_point, load_point
        down, up = two_way_road()
        point = suggest_point(by_frame(*down, *up))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.yaml"
            path.write_text(dump_point(point, "Sugerido\nrevisar"), encoding="utf-8")
            self.assertTrue(path.read_text(encoding="utf-8").startswith("# Sugerido\n# revisar\n"))
            self.assertEqual(load_point(path), point)
        sample = load_point(Path("configs/points/sample_vehicles.yaml"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.yaml"
            path.write_text(dump_point(sample), encoding="utf-8")
            self.assertEqual(load_point(path), sample)  # rules and restrictions survive too


class RealTrafficTest(unittest.TestCase):
    """Cases found on piloto_autopista (2026-11-15)."""

    def test_adjacent_lanes_of_one_direction_are_one_flow(self):
        lanes = [track(i, C.CAR, (300 + 35 * (i % 3), 700), (300 + 35 * (i % 3), 100), first=5 * i) for i in range(9)]
        self.assertEqual(len(suggest_point(by_frame(*lanes)).lines), 1)

    def test_separate_carriageways_of_one_direction_are_two_flows(self):
        roads = [track(i, C.CAR, (x, 700), (x, 100), first=5 * i) for i, x in enumerate([200, 210, 220, 700, 710, 720])]
        self.assertEqual(len(suggest_point(by_frame(*roads)).lines), 2)

    def test_zigzagging_tracks_make_no_flow(self):
        down, _ = two_way_road()
        zigzag = []
        for i in range(5):  # jittering boxes in a jam: back and forth, little net progress
            pts = [(400 + (60 if k % 2 else 0), 300 + k) for k in range(20)]
            zigzag.append([Detection(k, C.CAR, 0.9, (x - 25, y - 60, x + 25, y), 300 + i) for k, (x, y) in enumerate(pts)])
        self.assertEqual(len(suggest_point(by_frame(*down, *zigzag)).lines), 1)

    def test_a_small_flow_next_to_a_big_one_is_noise(self):
        down = [track(i, C.CAR, (300, 100), (300, 700), first=3 * i) for i in range(40)]
        cross = [track(100 + i, C.CAR, (100, 400), (700, 420)) for i in range(3)]  # e.g. lane changes / ID swaps
        self.assertEqual(len(suggest_point(by_frame(*down, *cross)).lines), 1)

    def test_fast_people_are_riders_not_pedestrians(self):
        down, _ = two_way_road()
        riders = [track(400 + i, C.PEDESTRIAN, (300, 100), (300, 700), size=(20, 50), first=4 * i) for i in range(4)]
        point = suggest_point(by_frame(*down, *riders), fps=30)
        self.assertEqual([z for z in point.zones if z.name.startswith("peatonal")], [])

    def test_scattered_people_do_not_chain_into_one_zone(self):
        down, _ = two_way_road()
        loners = [track(500 + i, C.PEDESTRIAN, (100 * i, 900), (100 * i + 5, 905), size=(20, 50)) for i in range(8)]
        point = suggest_point(by_frame(*down, *loners), fps=30)
        self.assertEqual([z for z in point.zones if z.name.startswith("peatonal")], [])

    def test_jittering_boxes_still_count_as_a_straight_trip(self):
        jam = []
        for i in range(5):  # slow traffic: +-8 px of box noise every frame on a 300 px trip
            pts = [(300 + (8 if k % 2 else -8), 700 - 5 * k) for k in range(60)]
            jam.append([Detection(4 * i + k, C.CAR, 0.9, (x - 25, y - 60, x + 25, y), 600 + i)
                        for k, (x, y) in enumerate(pts)])
        self.assertEqual(len(suggest_point(by_frame(*jam)).lines), 1)

    def test_people_stuck_in_traffic_are_riders(self):
        down, _ = two_way_road()
        stuck = [track(700 + i, C.PEDESTRIAN, (310, 400 + 20 * i), (312, 410 + 20 * i), size=(20, 50)) for i in range(4)]
        point = suggest_point(by_frame(*down, *stuck), fps=30)
        self.assertEqual([z for z in point.zones if z.name.startswith("peatonal")], [])

    def test_people_crossing_the_road_make_a_crosswalk_zone(self):
        down, _ = two_way_road()
        crossing = [track(800 + i, C.PEDESTRIAN, (250, 400 + 10 * i), (370, 400 + 10 * i), frames=90, size=(20, 50))
                    for i in range(4)]
        point = suggest_point(by_frame(*down, *crossing), fps=30)
        self.assertEqual(len([z for z in point.zones if z.name.startswith("peatonal")]), 1)


class LabelTest(unittest.TestCase):
    def test_a_diagonal_road_is_named_by_depth(self):
        from vtia.domain.scene import _label
        self.assertEqual(_label((-0.8, -0.6)), "away_from_camera")   # climbing to the upper left
        self.assertEqual(_label((0.7, 0.71)), "toward_camera")
        self.assertEqual(_label((0.95, -0.31)), "left_to_right")      # nearly level: by side
