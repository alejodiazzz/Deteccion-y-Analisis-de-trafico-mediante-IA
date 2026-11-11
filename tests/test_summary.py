"""Report aggregation. Run: python -m unittest discover -s tests"""
import unittest
from datetime import datetime

from vtia.application.process_video import VideoAnalysis
from vtia.application.summary import auto_interval, summarize
from vtia.domain.classes import RoadClass as C
from vtia.domain.models import Alert, CrossingEvent, Visit

FPS = 10


def analysis(start=None):
    """35 s of video: 3 cars and 1 bus, two of them in the 10-20 s bin; two zone visits; one alert."""
    events = [CrossingEvent(f, i, c, "main", "in") for i, (f, c) in
              enumerate([(5, C.CAR), (120, C.CAR), (150, C.BUS), (310, C.CAR)])]
    visits = [Visit(1, C.CAR, "stop", 0, 9), Visit(2, C.BUS, "stop", 100, 129)]
    alerts = [Alert(150, "wrong_way", "main", 2, C.BUS, "")]
    return VideoAnalysis(events, visits, alerts, frames=350, fps=FPS, start=start, tracks=6, mean_confidence=0.7)


class SummaryTest(unittest.TestCase):
    def test_counts_shares_and_peak(self):
        s = summarize(analysis())
        self.assertEqual(s.total, 4)
        self.assertEqual(s.by_class, [(C.CAR, 3, 0.75), (C.BUS, 1, 0.25)])
        self.assertEqual(s.interval_s, 10)
        self.assertEqual(s.by_interval, [("0:00", 1), ("0:10", 2), ("0:20", 0), ("0:30", 1)])
        self.assertEqual(s.peak, ("0:10", 2))
        self.assertEqual(s.dwell, [("stop", 2, 2.0, 3.0)])  # 1 s and 3 s
        self.assertEqual(s.alerts_by_kind, [("wrong_way", 1)])

    def test_clock_labels_when_the_start_time_is_known(self):
        s = summarize(analysis(start=datetime(2026, 10, 15, 7, 59, 50)))
        self.assertEqual([label for label, _ in s.by_interval], ["07:59:50", "08:00:00", "08:00:10", "08:00:20"])

    def test_interval_grows_with_duration(self):
        self.assertEqual([auto_interval(s) for s in (60, 600, 3600)], [10, 60, 900])


if __name__ == "__main__":
    unittest.main()
