"""What a comparison may claim, and the cells of a confusion matrix.

Run: python -m unittest discover -s tests
"""
import unittest
from collections import Counter

from vtia.domain.classes import RoadClass as C
from vtia.domain.evaluation import BACKGROUND, MISSED, confusion_cells
from vtia.domain.model_selection import ModelScore, disagreement, select_best


def score(name, f1=None, mae=None, fps=None):
    return ModelScore(name, f1, mae, fps)


class SelectBestTest(unittest.TestCase):
    def test_highest_macro_f1_wins(self):
        verdict = select_best([score("a", f1=0.40), score("b", f1=0.60)])
        self.assertEqual(verdict.best_by_annotations, ("b",))

    def test_macro_f1_within_one_hundredth_ties(self):
        verdict = select_best([score("a", f1=0.50), score("b", f1=0.49), score("c", f1=0.48)])
        self.assertEqual(verdict.best_by_annotations, ("a", "b"))

    def test_lowest_mae_wins(self):
        verdict = select_best([score("a", mae=3.0), score("b", mae=1.0)])
        self.assertEqual(verdict.best_by_counts, ("b",))

    def test_mae_tie_is_strictly_under_half_a_crossing(self):
        verdict = select_best([score("a", mae=1.0), score("b", mae=1.4), score("c", mae=1.5)])
        self.assertEqual(verdict.best_by_counts, ("a", "b"))

    def test_without_reference_data_nobody_wins_but_speed_is_reported(self):
        verdict = select_best([score("a", fps=10.0), score("b", fps=30.0)])
        self.assertEqual(verdict.best_by_annotations, ())
        self.assertEqual(verdict.best_by_counts, ())
        self.assertEqual(verdict.fastest, "b")

    def test_a_failed_model_cannot_win(self):
        verdict = select_best([score("failed"), score("ok", f1=0.1, mae=9.0, fps=1.0)])
        self.assertEqual(verdict.best_by_annotations, ("ok",))
        self.assertEqual(verdict.best_by_counts, ("ok",))
        self.assertEqual(verdict.fastest, "ok")

    def test_no_valid_models(self):
        verdict = select_best([score("a"), score("b")])
        self.assertEqual((verdict.best_by_annotations, verdict.best_by_counts, verdict.fastest), ((), (), None))


class DisagreementTest(unittest.TestCase):
    def test_range_per_class_counts_a_missing_class_as_zero(self):
        d = disagreement({"a": {C.CAR: 10, C.BUS: 2}, "b": {C.CAR: 7}})
        self.assertEqual(d.by_class, {C.BUS: (0, 2), C.CAR: (7, 10)})

    def test_coefficient_of_variation_of_the_totals(self):
        d = disagreement({"a": {C.CAR: 10}, "b": {C.CAR: 30}})
        self.assertAlmostEqual(d.total_cv, 0.5)  # mean 20, population deviation 10

    def test_not_defined_without_two_models_or_with_no_crossings(self):
        self.assertIsNone(disagreement({"a": {C.CAR: 10}}).total_cv)
        self.assertIsNone(disagreement({"a": {}, "b": {}}).total_cv)


class ConfusionCellsTest(unittest.TestCase):
    def cell(self, cells, true, predicted):
        return next(c for c in cells if (c.true, c.predicted) == (true, predicted))

    def test_share_is_of_the_true_row(self):
        cells = confusion_cells(Counter({(C.CAR, C.CAR): 3, (C.CAR, C.TRUCK): 1}))
        self.assertEqual(self.cell(cells, C.CAR, C.CAR).count, 3)
        self.assertEqual(self.cell(cells, C.CAR, C.CAR).share, 0.75)
        self.assertEqual(self.cell(cells, C.CAR, C.TRUCK).share, 0.25)

    def test_missed_objects_are_part_of_their_row(self):
        cells = confusion_cells(Counter({(C.BUS, C.BUS): 1, (C.BUS, MISSED): 1}))
        self.assertEqual(self.cell(cells, C.BUS, MISSED).share, 0.5)

    def test_row_without_objects_has_no_share(self):
        cells = confusion_cells(Counter({(C.CAR, C.CAR): 1}))
        self.assertIsNone(self.cell(cells, C.BUS, C.BUS).share)

    def test_background_by_missed_does_not_exist(self):
        cells = confusion_cells(Counter({(BACKGROUND, C.CAR): 2}))
        self.assertFalse(any((c.true, c.predicted) == (BACKGROUND, MISSED) for c in cells))
        self.assertEqual(self.cell(cells, BACKGROUND, C.CAR).share, 1.0)
        self.assertEqual(len(cells), 7 * 7 - 1)


if __name__ == "__main__":
    unittest.main()
