"""Validation of the manual-count CSV.

Run: python -m unittest discover -s tests
"""
import unittest

from vtia.adapters.manual_counts import parse_manual_counts
from vtia.application.compare_models import ManualCount
from vtia.domain.classes import RoadClass as C
from vtia.domain.models import Line, PointConfig

POINT = PointConfig((Line("main", (0, 0), (10, 0), "down", "up"),), ())


class ParseManualCountsTest(unittest.TestCase):
    def test_totals_per_class(self):
        self.assertEqual(parse_manual_counts("class,manual\ncar,5\nbus,0\n", POINT),
                         [ManualCount(C.CAR, 5), ManualCount(C.BUS, 0)])

    def test_per_line_and_direction(self):
        rows = parse_manual_counts("class,manual,line,direction\ncar,5,main,down\ncar,2,main,up\n", POINT)
        self.assertEqual(rows, [ManualCount(C.CAR, 5, "main", "down"), ManualCount(C.CAR, 2, "main", "up")])

    def test_byte_order_mark_is_accepted(self):
        self.assertEqual(parse_manual_counts("﻿class,manual\ncar,1\n", POINT), [ManualCount(C.CAR, 1)])

    def test_an_automatic_column_is_ignored(self):
        self.assertEqual(parse_manual_counts("class,manual,automatic\ncar,5,99\n", POINT), [ManualCount(C.CAR, 5)])

    def test_rejected_files(self):
        cases = {
            "missing column": "class\ncar\n",
            "no rows": "class,manual\n",
            "unknown class": "class,manual\nplane,1\n",
            "not a number": "class,manual\ncar,many\n",
            "negative": "class,manual\ncar,-1\n",
            "duplicate": "class,manual\ncar,1\ncar,2\n",
            "mixed granularity": "class,manual,line,direction\ncar,1,main,down\nbus,1,,\n",
            "line without direction": "class,manual,line\ncar,1,main\n",
            "unknown line": "class,manual,line,direction\ncar,1,other,down\n",
            "unknown direction": "class,manual,line,direction\ncar,1,main,sideways\n",
        }
        for name, text in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                parse_manual_counts(text, POINT)


if __name__ == "__main__":
    unittest.main()
