"""Text and data of the comparison tab, plus a headless render.

Run: python -m unittest discover -s tests
"""
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from streamlit.testing.v1 import AppTest  # noqa: E402

import comparison_view as view  # noqa: E402
from vtia.application.compare_models import Comparison, ModelResult  # noqa: E402
from vtia.compose import ComparisonRun  # noqa: E402
from vtia.domain.classes import RoadClass as C  # noqa: E402
from vtia.domain.evaluation import MISSED  # noqa: E402
from vtia.domain.model_selection import Disagreement, Verdict  # noqa: E402


class TextTest(unittest.TestCase):
    def test_only_video_says_no_winner_is_possible(self):
        self.assertIn("no se puede declarar", view.level_message(False, False))

    def test_counts_allow_a_counting_winner_and_name_what_is_missing(self):
        message = view.level_message(False, True)
        self.assertIn("mejor contando", message)
        self.assertIn("anotaciones", message)

    def test_annotations_mention_the_confusion_matrix(self):
        self.assertIn("matriz de confusión", view.level_message(True, False))

    def test_estimate_uses_the_measured_speed_of_each_model(self):
        self.assertAlmostEqual(view.estimate_seconds(929), 929 * (1 / 65 + 1 / 37 + 1 / 23 + 1 / 13))

    def test_ties_are_reported_as_ties(self):
        lines = view.verdict_lines(Verdict(("a", "b"), (), None))
        self.assertEqual(lines, ["**Mejor detectando y clasificando en este video:** empate entre a y b."])

    def test_a_single_winner_and_speed(self):
        lines = view.verdict_lines(Verdict((), ("a",), "b"))
        self.assertEqual(lines, ["**Mejor contando en este video:** a.", "**Más rápido:** b."])

    def test_no_winner_is_stated(self):
        self.assertEqual(view.verdict_lines(Verdict((), (), None)), ["No hay datos para declarar un mejor modelo."])


class MatrixRowsTest(unittest.TestCase):
    def row(self, rows, true, predicted):
        return next(r for r in rows if (r["Real"], r["Predicho"]) == (true, predicted))

    def test_labels_counts_and_share_in_spanish(self):
        rows = view.matrix_rows(Counter({(C.CAR, C.CAR): 3, (C.CAR, MISSED): 1}))
        cell = self.row(rows, "Automóvil", "No detectado")
        self.assertEqual((cell["Objetos"], cell["Proporción"], cell["Texto"]), (1, 0.25, "1"))

    def test_empty_row_is_marked_without_data(self):
        cell = self.row(view.matrix_rows(Counter()), "Bus", "Bus")
        self.assertIsNone(cell["Proporción"])
        self.assertEqual(cell["Texto"], "s/d")


def _render_for_video_b():
    from pathlib import Path

    import comparison_view

    comparison_view.render(lambda: Path("b.mp4"), "b.mp4", Path("p.yaml"), 0, Path("."))


class RenderTest(unittest.TestCase):
    def test_a_comparison_of_another_video_is_not_shown(self):
        at = AppTest.from_function(_render_for_video_b, default_timeout=60)
        at.session_state["comparison"] = object()  # showing it would raise
        at.session_state["comparison_key"] = ("a.mp4", "p.yaml")
        at.run()
        self.assertFalse(at.exception)

    def test_pressing_compare_shows_the_results_of_this_video(self):
        comparison = Comparison([ModelResult("yolo26n", crossings=3, tracks=2, fps=20.0)],
                                Verdict((), (), "yolo26n"), Disagreement({}, None))
        with tempfile.TemporaryDirectory() as tmp, \
                patch("vtia.compose.compare_file", return_value=ComparisonRun(comparison, Path(tmp))):
            at = AppTest.from_function(_render_for_video_b, default_timeout=60).run()
            at.button(key="cmp-run").click().run()
        self.assertFalse(at.exception)
        self.assertIn("**Más rápido:** yolo26n.", [m.value for m in at.markdown])


if __name__ == "__main__":
    unittest.main()
