"""Anonymisation regions, rider merge, session split and the CVAT round trip.

Run: python -m unittest discover -s tests
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vtia.adapters.cvat_mot import read_annotations, write_preannotations
from vtia.application.anonymize import regions_per_frame
from vtia.domain.classes import RoadClass as C
from vtia.domain.models import Detection
from vtia.domain.postprocess import merge_riders
from vtia.domain.privacy import private_regions
from vtia.domain.splits import parse_clip, split


def det(cls, box, frame=0, track=None, conf=0.9):
    return Detection(frame, cls, conf, box, track)


class PrivacyTest(unittest.TestCase):
    def test_head_of_a_person_and_plate_band_of_a_car(self):
        head, plate = private_regions([det(C.PEDESTRIAN, (100, 100, 140, 200)), det(C.CAR, (0, 300, 200, 400))])
        self.assertLess(head[1], 100)                       # starts above the box (margin)
        self.assertLess(head[3], 140)                       # covers the top of the body only
        self.assertTrue(plate[1] > 300 and plate[3] > 400)  # bottom part of the car
        self.assertAlmostEqual((plate[0] + plate[2]) / 2, 100)  # centred on the car

    def test_cabin_of_large_vehicles_only(self):
        self.assertEqual(len(private_regions([det(C.TRUCK, (0, 0, 300, 400))])), 2)  # plate + windshield
        self.assertEqual(len(private_regions([det(C.TRUCK, (0, 0, 60, 80))])), 1)    # far away: plate only

    def test_bicycles_have_nothing_to_blur(self):
        self.assertEqual(private_regions([det(C.BICYCLE, (0, 0, 50, 50))]), [])

    def test_regions_are_held_for_a_few_frames(self):
        class Flicker:  # detector that sees the person only in frame 0
            name = "flicker"
            def detect(self, source):
                yield [det(C.PEDESTRIAN, (0, 0, 10, 40))]
                yield from ([] for _ in range(4))
        counts = [len(r) for r in regions_per_frame(Path("."), Flicker(), hold_frames=2)]
        self.assertEqual(counts, [1, 1, 1, 0, 0])


class RidersTest(unittest.TestCase):
    def test_rider_and_parrillero_fold_into_the_motorcycle(self):
        moto = det(C.MOTORCYCLE, (100, 150, 200, 250))
        rider, parrillero = det(C.PEDESTRIAN, (120, 60, 170, 220)), det(C.PEDESTRIAN, (150, 70, 195, 225))
        walker = det(C.PEDESTRIAN, (400, 100, 440, 250))
        out = merge_riders([moto, rider, parrillero, walker])
        self.assertEqual([d.road_class for d in out], [C.MOTORCYCLE, C.PEDESTRIAN])
        self.assertEqual(out[0].box, (100, 60, 200, 250))  # grown to cover both riders
        self.assertEqual(out[1], walker)

    def test_person_beside_a_parked_bicycle_is_kept(self):
        bike, person = det(C.BICYCLE, (100, 150, 200, 250)), det(C.PEDESTRIAN, (205, 80, 245, 250))
        self.assertEqual(merge_riders([bike, person]), [bike, person])


class SplitTest(unittest.TestCase):
    NAMES = [f"P{p}_202610{d:02d}_{hh}00_s{s:04d}" for p in (1, 2) for d in (14, 15, 16, 17)
             for hh in ("07", "12", "17") for s in (0, 300)]

    def test_sessions_never_straddle_splits_and_every_stratum_has_test(self):
        parts = split(self.NAMES, folds=3, seed=1)
        by_session = {}
        for name, part in parts.items():
            by_session.setdefault(parse_clip(name).session, set()).add(part)
        self.assertTrue(all(len(p) == 1 for p in by_session.values()))
        strata_in_test = {parse_clip(n).stratum for n, p in parts.items() if p == "test"}
        self.assertEqual(len(strata_in_test), 6)  # 2 points x 3 time bands
        self.assertEqual(parts, split(self.NAMES, folds=3, seed=1))  # deterministic

    def test_bad_names_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_clip("IMG_2041")


class CvatRoundTripTest(unittest.TestCase):
    def test_preannotations_read_back_as_ground_truth(self):
        frames = [[det(C.CAR, (10, 10, 50, 30), f, track=7), det(C.PEDESTRIAN, (100, 10, 110, 40), f, track=8)]
                  for f in range(12)]
        frames[3][0] = det(C.TRUCK, (10, 10, 50, 30), 3, track=7)  # class flicker: the track stays a car
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d) / "clip.zip"
            self.assertEqual(write_preannotations(frames, archive, keyframe_every=5), 2)
            with mock.patch("vtia.adapters.cvat_mot.video_info", return_value=(25.0, 12)):
                seq = read_annotations(archive, Path(d) / "clip.mp4")
        written = [f for f, truths in enumerate(seq.truths) if truths]
        self.assertEqual(written, [0, 5, 10, 11])  # keyframes + last frame; CVAT interpolates the rest
        self.assertEqual({(t.object_id, t.road_class) for t in seq.truths[0]}, {(7, C.CAR), (8, C.PEDESTRIAN)})


if __name__ == "__main__":
    unittest.main()
