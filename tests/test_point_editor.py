"""Hand-drawn camera points: editor shapes <-> YAML layout, names and checks.

Run: python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import point_editor as pe  # noqa: E402
from vtia.adapters.yaml_config import load_point, parse_point  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "configs" / "points"


class PointEditorTest(unittest.TestCase):
    def test_existing_points_survive_the_editor(self):
        for path in sorted(SAMPLES.glob("sample_*.yaml")):
            point = load_point(path)
            shapes, attrs = pe.shapes_from_point(point)
            restrictions = [{"class": r.road_class.value, "hours": [f"{a:%H:%M}-{b:%H:%M}" for a, b in r.windows]}
                            for r in point.restrictions]
            self.assertEqual(parse_point(pe.point_data(shapes, attrs, restrictions)), point, path)

    def test_drawn_line_gets_a_name_and_opposite_direction(self):
        shapes = [{"id": "a", "kind": "line", "pts": [[0, 500.4], [1920, 500]]},
                  {"id": "b", "kind": "zone", "pts": [[0, 0], [10, 0], [10, 10]]}]
        attrs = {"a": {"to_right": "left_to_right", "allowed": "left"}}
        names = pe.default_names(shapes, attrs)
        self.assertEqual(names, {"a": "linea_1", "b": "zona_1"})
        for sid, name in names.items():
            attrs.setdefault(sid, {})["name"] = name
        data = pe.point_data(shapes, attrs)
        self.assertEqual(data["lines"][0], {"name": "linea_1", "start": [0, 500], "end": [1920, 500],
                                            "to_right": "left_to_right", "to_left": "right_to_left",
                                            "allowed": "right_to_left"})
        self.assertNotIn("forbidden", data["zones"][0])  # rules left at 0 are not written
        self.assertEqual(pe.problems(data), [])
        self.assertEqual(len(parse_point(data).zones), 1)

    def test_direction_follows_the_arrow(self):
        self.assertEqual(pe.arrow_direction([[0, 500], [1920, 500]]), "toward_camera")      # arrow points down
        self.assertEqual(pe.arrow_direction([[1920, 500], [0, 500]]), "away_from_camera")   # inverted: up
        self.assertEqual(pe.arrow_direction([[900, 0], [900, 1080]]), "right_to_left")      # arrow points left
        self.assertEqual(pe.arrow_direction([[900, 1080], [900, 0]]), "left_to_right")

    def test_inverting_a_line_turns_its_direction(self):
        a = {}
        pe.sync_direction(a, [[0, 500], [1920, 500]])  # new line: what its arrow shows
        self.assertEqual(a["to_right"], "toward_camera")
        a["to_right"] = "left_to_right"                # chosen by hand: kept while the arrow stays
        pe.sync_direction(a, [[0, 510], [1900, 490]])
        self.assertEqual(a["to_right"], "left_to_right")
        pe.sync_direction(a, [[1920, 500], [0, 500]])  # «Invertir línea»
        self.assertEqual(a["to_right"], "away_from_camera")

    def test_loaded_lines_keep_their_label(self):
        point = load_point(SAMPLES / "sample_vehicles.yaml")
        shapes, attrs = pe.shapes_from_point(point)
        for s in shapes:
            if s["kind"] == "line":
                before = attrs[s["id"]]["to_right"]
                pe.sync_direction(attrs[s["id"]], s["pts"])
                self.assertEqual(attrs[s["id"]]["to_right"], before)

    def test_default_names_skip_taken_ones(self):
        shapes = [{"id": "a", "kind": "line", "pts": []}, {"id": "b", "kind": "line", "pts": []}]
        self.assertEqual(pe.default_names(shapes, {"a": {"name": "linea_1"}}), {"a": "linea_1", "b": "linea_2"})

    def test_problems_in_words(self):
        no_line = pe.point_data([{"id": "z", "kind": "zone", "pts": [[0, 0], [1, 0], [1, 1]]}], {"z": {"name": "z"}})
        self.assertIn("al menos una línea", pe.problems(no_line)[0])
        twice = pe.point_data([{"id": i, "kind": "line", "pts": [[0, 0], [1, 1]]} for i in "ab"],
                              {"a": {"name": "x"}, "b": {"name": "x"}})
        self.assertIn("repetidos: x", pe.problems(twice)[0])

    def test_file_name_is_safe(self):
        self.assertEqual(pe.file_name(" Cruce Norte / día 2 "), "Cruce_Norte_d_a_2")


if __name__ == "__main__":
    unittest.main()
