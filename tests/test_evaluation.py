"""Checks the evaluation module against known answers.

Run: python -m unittest discover -s tests
"""
import unittest
from collections import Counter
from pathlib import Path

from vtia.adapters.mot_tracker import MotTracker
from vtia.adapters.coco_map import coco_map
from vtia.adapters.trackeval_metrics import mot_metrics
from vtia.application.evaluate import AnnotatedSequence, evaluate
from vtia.domain.classes import RoadClass as C
from vtia.domain.evaluation import BACKGROUND, MISSED, class_report, confusion, drop_ignored
from vtia.domain.models import Detection, GroundTruth
from vtia.domain.postprocess import suppress_duplicates

ORDER = [C.PEDESTRIAN, C.BICYCLE, C.MOTORCYCLE, C.CAR, C.BUS, C.TRUCK]


def proposal_matrix() -> Counter:
    """Illustrative confusion matrix of the proposal (§10): rows true, columns predicted + 'no detectado'."""
    rows = [[52, 3, 2, 1, 0, 0, 2], [4, 28, 6, 1, 0, 0, 1], [1, 5, 35, 2, 0, 0, 2],
            [0, 0, 2, 48, 2, 2, 1], [0, 0, 0, 1, 15, 3, 1], [0, 0, 0, 2, 4, 13, 1]]
    counts = Counter()
    for t, row in zip(ORDER, rows):
        for p, n in zip([*ORDER, MISSED], row):
            counts[(t, p)] = n
    return counts


def scene(frames=6):
    """A car driving right and a pedestrian walking down, annotated in every frame."""
    return [[GroundTruth(f, 1, C.CAR, (10 + 10 * f, 100, 60 + 10 * f, 140)),
             GroundTruth(f, 2, C.PEDESTRIAN, (200, 50 + 5 * f, 220, 100 + 5 * f))] for f in range(frames)]


class Oracle:
    """Detector + tracker in one: returns the annotations themselves, with their true ids.

    Isolates the evaluation code from any real model, so a perfect input must score perfectly.
    """
    name = "oracle"

    def __init__(self, truths, skip=()):
        self.truths, self.skip = truths, set(skip)  # skip: (frame, object_id) pairs to leave out

    def detect(self, source):
        for f, frame in enumerate(self.truths):
            yield [Detection(f, t.road_class, 0.9, t.box, t.object_id)
                   for t in frame if (f, t.object_id) not in self.skip]

    def track(self, frames, frame_rate=30.0):
        self.frame_rate = frame_rate
        return frames  # ids already set


def run(truths, skip=()):
    seq = AnnotatedSequence("synthetic", Path("."), truths, [[] for _ in truths])
    oracle = Oracle(truths, skip)
    return evaluate([seq], oracle, oracle, coco_map, mot_metrics)


class ClassReportTest(unittest.TestCase):
    def test_reproduces_the_proposal_table(self):
        r = class_report(proposal_matrix(), ORDER)
        got = [round(r["per_class"][c][k], 3) for c in ORDER for k in ("precision", "recall", "f1")]
        expected = [0.912, 0.867, 0.889, 0.778, 0.700, 0.737, 0.778, 0.778, 0.778,
                    0.873, 0.873, 0.873, 0.714, 0.750, 0.732, 0.722, 0.650, 0.684]
        self.assertEqual(got, expected)
        self.assertAlmostEqual(r["macro_f1"], 0.782, places=3)
        self.assertAlmostEqual(r["accuracy"], 191 / 240)

    def test_background_detections_lower_precision_only(self):
        counts = proposal_matrix()
        counts[(BACKGROUND, C.BUS)] = 21
        r = class_report(counts, ORDER)["per_class"][C.BUS]
        self.assertAlmostEqual(r["precision"], 15 / 42)
        self.assertAlmostEqual(r["recall"], 0.75)


class MatchingTest(unittest.TestCase):
    def test_wrong_class_is_a_confusion_not_a_miss(self):
        t = [GroundTruth(0, 1, C.BUS, (0, 0, 100, 50))]
        d = [Detection(0, C.TRUCK, 0.8, (2, 0, 100, 50)), Detection(0, C.CAR, 0.5, (300, 300, 340, 330))]
        self.assertEqual(confusion(t, d), Counter({(C.BUS, C.TRUCK): 1, (BACKGROUND, C.CAR): 1}))

    def test_detections_inside_ignore_regions_are_dropped(self):
        d = [Detection(0, C.CAR, 0.9, (10, 10, 20, 20)), Detection(0, C.CAR, 0.9, (100, 100, 120, 120))]
        self.assertEqual(drop_ignored(d, [(0, 0, 50, 50)]), d[1:])

    def test_annotations_inside_ignore_regions_are_not_misses(self):
        truths = scene()
        seq = AnnotatedSequence("synthetic", Path("."), truths, [[(190, 0, 230, 400)] for _ in truths])  # covers the pedestrian
        oracle = Oracle(truths, skip={(f, 2) for f in range(6)})  # and the detector never sees it
        r = evaluate([seq], oracle, oracle, coco_map, mot_metrics)
        self.assertEqual(r.confusion, Counter({(C.CAR, C.CAR): 6}))
        self.assertAlmostEqual(r.tracking["MOTA"], 1.0)


class EvaluateTest(unittest.TestCase):
    def test_perfect_detector_scores_perfectly(self):
        r = run(scene())
        self.assertEqual(r.confusion, Counter({(C.CAR, C.CAR): 6, (C.PEDESTRIAN, C.PEDESTRIAN): 6}))
        self.assertAlmostEqual(r.detection["mAP50"], 1.0)
        self.assertAlmostEqual(r.tracking["MOTA"], 1.0)
        self.assertAlmostEqual(r.tracking["IDF1"], 1.0)
        self.assertEqual(r.tracking["IDSW"], 0)

    def test_one_missed_object(self):
        r = run(scene(), skip={(3, 1)})  # the car disappears in frame 3 only
        self.assertEqual(r.confusion[(C.CAR, MISSED)], 1)
        self.assertAlmostEqual(r.classes["per_class"][C.CAR]["recall"], 5 / 6)
        self.assertAlmostEqual(r.tracking["MOTA"], 1 - 1 / 12)
        self.assertEqual(r.tracking["IDSW"], 0)


class MotTrackerAdapterTest(unittest.TestCase):
    def test_returns_the_input_boxes_with_stable_ids(self):
        truths = scene(8)
        frames = list(MotTracker().track(Oracle(truths).detect(None)))
        # `trackers` confirms a new track on its 2nd frame, so frame 0 comes back empty.
        ids = {}
        for frame_truths, frame in zip(truths[1:], frames[1:]):
            self.assertEqual(sorted(d.box for d in frame), sorted(t.box for t in frame_truths))
            for d in frame:
                ids.setdefault(d.road_class, set()).add(d.track_id)
        self.assertEqual([len(v) for v in ids.values()], [1, 1])  # one id per object, all along
        self.assertNotEqual(ids[C.CAR], ids[C.PEDESTRIAN])

    def test_a_lost_object_is_remembered_for_the_same_time_at_any_frame_rate(self):
        # Seen, then 20 empty frames, then seen again. 20 frames are 0.67 s at 30 fps but 1.33 s
        # at 15 fps, and the tracker remembers lost objects for about 1 s.
        box = (100, 100, 160, 140)
        seen = [[Detection(f, C.CAR, 0.9, box)] for f in range(5)]
        frames = seen + [[] for _ in range(20)] + seen

        def ids(frame_rate):
            return {d.track_id for frame in MotTracker().track(frames, frame_rate) for d in frame}

        self.assertEqual(len(ids(30.0)), 1)
        self.assertEqual(len(ids(15.0)), 2)

    def test_evaluation_gives_the_tracker_the_frame_rate_of_each_sequence(self):
        truths = scene(2)
        seq = AnnotatedSequence("synthetic", Path("."), truths, [[] for _ in truths], fps=12.5)
        oracle = Oracle(truths)
        evaluate([seq], oracle, oracle, coco_map, mot_metrics)
        self.assertEqual(oracle.frame_rate, 12.5)



class SuppressDuplicatesTest(unittest.TestCase):
    def test_keeps_the_most_confident_of_overlapping_boxes_of_any_class(self):
        car = Detection(0, C.CAR, 0.9, (0, 0, 100, 50))
        truck_twin = Detection(0, C.TRUCK, 0.6, (2, 0, 100, 52))
        neighbour = Detection(0, C.CAR, 0.5, (60, 0, 160, 50))  # IoU 0.25 with car: a different object
        self.assertEqual(suppress_duplicates([truck_twin, neighbour, car]), [car, neighbour])


if __name__ == "__main__":
    unittest.main()
