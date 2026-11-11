# Comparación de modelos — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Comparar los 4 detectores desde el tablero, con métricas, matrices de confusión cuando haya anotaciones y un mejor modelo solo cuando los datos lo permitan.

**Architecture:** Reglas puras en `domain`, orquestación con dependencias inyectadas en `application`, E/S en `adapters`, cableado en `compose` y vista aislada en `app/comparison_view.py`.

**Tech Stack:** Python 3.12, unittest, Streamlit 1.65, Altair 6.3, Ultralytics, torchvision, trackers, TrackEval, pycocotools.

**Spec:** `Docs/superpowers/specs/2026-11-10-comparacion-modelos-design.md`

## Global Constraints

- Empate F1 macro: diferencia ≤ 0,01. Empate MAE: diferencia < 0,5 cruces.
- Matriz: confianza 0,25, IoU 0,5. Caché de detección: confianza 0,01. Tracker: ByteTrack con `default_track_thresh`.
- Orden fijo: `MODELS` de `vtia/compose.py`.
- Código y datos en inglés; lo que lee el usuario, en español.
- No hacer commits ni push salvo petición expresa del usuario.
- Ejecutar los tests con: `python -m unittest discover -s tests`.

---

### Task 1: Reglas de selección y celdas de la matriz

**Files:** Create `vtia/domain/model_selection.py`, `tests/test_model_selection.py`. Modify `vtia/domain/evaluation.py`.

**Interfaces:** Produces `ModelScore(name, macro_f1, count_mae, fps)`, `select_best(scores) -> Verdict(best_by_annotations, best_by_counts, fastest)`, `disagreement(counts_by_model) -> Disagreement(by_class, total_cv)`, `confusion_cells(counts) -> list[ConfusionCell(true, predicted, count, share)]`.

- [ ] **Step 1: Write the failing tests**

```python title="tests/test_model_selection.py"
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
```

- [ ] **Step 2: Run; expect ImportError (feature missing).**

- [ ] **Step 3: Implement**

```python title="vtia/domain/model_selection.py"
"""What a model comparison may claim: who won, who tied, and how much the models disagree."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean, pstdev

from vtia.domain.classes import RoadClass

F1_TIE = 0.01   # macro F1 this close to the best is a tie
MAE_TIE = 0.5   # less than half a crossing from the best is a tie


@dataclass(frozen=True)
class ModelScore:
    """None means not measured: no reference data for it, or the model failed."""
    name: str
    macro_f1: float | None
    count_mae: float | None
    fps: float | None


@dataclass(frozen=True)
class Verdict:
    best_by_annotations: tuple[str, ...]  # empty: nothing can be claimed
    best_by_counts: tuple[str, ...]
    fastest: str | None                   # a speed fact, not an overall winner


@dataclass(frozen=True)
class Disagreement:
    by_class: dict[RoadClass, tuple[int, int]]  # class -> (fewest, most) crossings among the models
    total_cv: float | None                      # coefficient of variation of the total crossings


def select_best(scores: Sequence[ModelScore]) -> Verdict:
    f1 = [s for s in scores if s.macro_f1 is not None]
    mae = [s for s in scores if s.count_mae is not None]
    fps = [s for s in scores if s.fps is not None]
    top_f1 = max((s.macro_f1 for s in f1), default=None)
    low_mae = min((s.count_mae for s in mae), default=None)
    return Verdict(
        # round(): 0.50 - 0.49 is 0.010000000000000009 in floating point
        tuple(s.name for s in f1 if round(top_f1 - s.macro_f1, 9) <= F1_TIE),
        tuple(s.name for s in mae if round(s.count_mae - low_mae, 9) < MAE_TIE),
        max(fps, key=lambda s: s.fps).name if fps else None,
    )


def disagreement(counts_by_model: Mapping[str, Mapping[RoadClass, int]]) -> Disagreement:
    classes = sorted({c for counts in counts_by_model.values() for c in counts})
    by_class = {c: (min(m.get(c, 0) for m in counts_by_model.values()),
                    max(m.get(c, 0) for m in counts_by_model.values())) for c in classes}
    totals = [sum(m.values()) for m in counts_by_model.values()]
    cv = pstdev(totals) / mean(totals) if len(totals) >= 2 and sum(totals) else None
    return Disagreement(by_class, cv)
```

Añadir a `vtia/domain/evaluation.py`, tras `confusion`, la dataclass `ConfusionCell(true, predicted, count, share)` y:

```python
def confusion_cells(counts: Counter, classes: Sequence[RoadClass] = tuple(RoadClass)) -> list[ConfusionCell]:
    """Every cell of the matrix, with its share of the true row (None when the row is empty)."""
    cells = []
    for true in (*classes, BACKGROUND):
        columns = [*classes] if true == BACKGROUND else [*classes, MISSED]
        total = sum(counts[(true, p)] for p in columns)
        cells += [ConfusionCell(true, p, counts[(true, p)], counts[(true, p)] / total if total else None) for p in columns]
    return cells
```

- [ ] **Step 4: Run; expect PASS.**

---

### Task 2: Lectura del conteo manual

**Files:** Create `vtia/adapters/manual_counts.py`, `tests/test_manual_counts.py`, and `ManualCount` in `vtia/application/compare_models.py` (Task 3 completes that file).

**Interfaces:** Produces `parse_manual_counts(text: str, point: PointConfig) -> list[ManualCount]`; `ManualCount(road_class, manual, line=None, direction=None)`.

- [ ] **Step 1: Write the failing tests**

```python title="tests/test_manual_counts.py"
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
```

- [ ] **Step 2: Run; expect ImportError.**

- [ ] **Step 3: Implement**

```python title="vtia/adapters/manual_counts.py"
"""Reads the manual-count CSV of a model comparison. Invalid rows are errors, never dropped."""
import csv
import io

from vtia.application.compare_models import ManualCount
from vtia.domain.classes import RoadClass
from vtia.domain.models import PointConfig


def parse_manual_counts(text: str, point: PointConfig) -> list[ManualCount]:
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    if not {"class", "manual"}.issubset(reader.fieldnames or []):
        raise ValueError("the manual-count CSV needs the columns class and manual")
    directions = {line.name: {line.to_right, line.to_left} for line in point.lines}
    counts: list[ManualCount] = []
    for number, row in enumerate(reader, 2):
        try:
            counts.append(_row(row, directions))
        except ValueError as exc:
            raise ValueError(f"manual-count CSV, line {number}: {exc}") from exc
    if not counts:
        raise ValueError("the manual-count CSV has no rows")
    if len({c.line is None for c in counts}) > 1:
        raise ValueError("every row must give line and direction, or none of them")
    keys = [(c.road_class, c.line, c.direction) for c in counts]
    if len(keys) != len(set(keys)):
        raise ValueError("the manual-count CSV has duplicate rows")
    return counts


def _row(row: dict, directions: dict[str, set[str]]) -> ManualCount:
    road_class = RoadClass((row.get("class") or "").strip())
    manual = int((row.get("manual") or "").strip())
    if manual < 0:
        raise ValueError("counts cannot be negative")
    line = (row.get("line") or "").strip() or None
    direction = (row.get("direction") or "").strip() or None
    if (line is None) != (direction is None):
        raise ValueError("line and direction go together")
    if line is not None and direction not in directions.get(line, ()):
        raise ValueError(f"line {line!r} with direction {direction!r} does not exist in this camera point")
    return ManualCount(road_class, manual, line, direction)
```

- [ ] **Step 4: Run; expect PASS (once Task 3 Step 3 has created the module).**

---

### Task 3: Caso de uso y archivos de salida

**Files:** Create `vtia/application/compare_models.py`, `vtia/adapters/comparison_files.py`, `tests/test_compare_models.py`. Modify `vtia/cli.py` (mover `_write_model_tables`).

**Interfaces:** Consumes Task 1. Produces `compare_models(video, point, fps, models, build, *, truth=None, manual=None, map_metric=None, mot_metric=None, conf=0.25, on_model=None) -> Comparison(models, verdict, disagreement)`; `ModelResult(name, error, counts_by_class, crossings, tracks, fps, evaluation, counting)`; `write_comparison(out, comparison)`, `clear_results(out)`, `write_model_tables(out, name, result)`.

- [ ] **Step 1: Write the failing tests**

```python title="tests/test_compare_models.py"
"""The comparison use case, with fake detectors and a pass-through tracker.

Run: python -m unittest discover -s tests
"""
import csv
import tempfile
import unittest
from pathlib import Path

from vtia.adapters.comparison_files import clear_results, write_comparison
from vtia.application.compare_models import ManualCount, compare_models
from vtia.application.evaluate import AnnotatedSequence
from vtia.domain.classes import RoadClass as C
from vtia.domain.models import Detection, GroundTruth, Line, PointConfig

POINT = PointConfig((Line("main", (0, 100), (200, 100), "down", "up"),), ())
BOX_ABOVE, BOX_BELOW = (40, 60, 60, 80), (40, 120, 60, 140)  # anchors at y=80 and y=140


class Fake:
    """Detector + tracker: one car crossing the line, reported as `cls`."""

    def __init__(self, name, cls=C.CAR, frames=10, seconds=1.0):
        self.name, self.cls, self.frames, self.seconds = name, cls, frames, seconds

    def detect(self, source):
        yield [Detection(0, self.cls, 0.9, BOX_ABOVE, 1)]
        yield [Detection(1, self.cls, 0.9, BOX_BELOW, 1)]

    def track(self, frames, frame_rate=30.0):
        return frames


def build_for(**classes):
    def build(name):
        fake = Fake(name, classes.get(name, C.CAR))
        return fake, fake
    return build


def truth():
    frames = [[GroundTruth(0, 1, C.CAR, BOX_ABOVE)], [GroundTruth(1, 1, C.CAR, BOX_BELOW)]]
    return AnnotatedSequence("clip", Path("clip.mp4"), frames, [[], []])


def fake_map(truths, dets):
    return {"mAP50": 0.5, "mAP50_95": 0.25, "per_class": {c: {"AP50": 0.5, "AP50_95": 0.25} for c in C}}


def fake_mot(sequences):
    return {"HOTA": 0.4, "MOTA": 0.3, "IDF1": 0.2, "IDSW": 1}


def run(models=("a", "b"), build=None, **kwargs):
    return compare_models(Path("clip.mp4"), POINT, 30.0, models, build or build_for(), **kwargs)


class CompareModelsTest(unittest.TestCase):
    def test_without_reference_there_is_no_winner_and_no_evaluation(self):
        c = run()
        self.assertEqual([m.crossings for m in c.models], [1, 1])
        self.assertEqual(c.models[0].counts_by_class, {C.CAR: 1})
        self.assertIsNone(c.models[0].evaluation)
        self.assertEqual((c.verdict.best_by_annotations, c.verdict.best_by_counts), ((), ()))
        self.assertEqual(c.models[0].fps, 10.0)

    def test_annotations_pick_the_model_that_classifies_correctly(self):
        c = run(build=build_for(a=C.TRUCK, b=C.CAR), truth=truth(), map_metric=fake_map, mot_metric=fake_mot)
        self.assertEqual(c.verdict.best_by_annotations, ("b",))
        self.assertEqual(c.models[1].evaluation.confusion[(C.CAR, C.CAR)], 2)

    def test_manual_counts_per_class_pick_the_closest_model(self):
        c = run(build=build_for(a=C.TRUCK, b=C.CAR), manual=[ManualCount(C.CAR, 1)])
        self.assertEqual(c.models[0].counting["overall"]["mae"], 1)
        self.assertEqual(c.models[1].counting["overall"]["mae"], 0)
        self.assertEqual(c.verdict.best_by_counts, ("b",))

    def test_manual_counts_per_line_and_direction(self):
        c = run(models=("a",), manual=[ManualCount(C.CAR, 3, "main", "down"), ManualCount(C.CAR, 0, "main", "up")])
        self.assertEqual([(r["manual"], r["automatic"]) for r in c.models[0].counting["records"]], [(3, 1), (0, 0)])

    def test_a_failing_model_is_reported_and_the_others_continue(self):
        def build(name):
            if name == "bad":
                raise RuntimeError("weights not found")
            fake = Fake(name)
            return fake, fake

        c = run(models=("bad", "good"), build=build, manual=[ManualCount(C.CAR, 1)])
        self.assertEqual(c.models[0].error, "weights not found")
        self.assertIsNone(c.models[1].error)
        self.assertEqual(c.verdict.best_by_counts, ("good",))

    def test_reports_each_model_as_it_starts(self):
        seen = []
        run(on_model=lambda i, n, name: seen.append((i, n, name)))
        self.assertEqual(seen, [(1, 2, "a"), (2, 2, "b")])


class ComparisonFilesTest(unittest.TestCase):
    def test_writes_summary_and_tables_of_each_evaluated_model(self):
        c = run(truth=truth(), map_metric=fake_map, mot_metric=fake_mot)
        with tempfile.TemporaryDirectory() as tmp:
            write_comparison(Path(tmp), c)
            with (Path(tmp) / "summary.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["model"] for r in rows], ["a", "b"])
            self.assertEqual(rows[0]["mAP50"], "0.5")
            self.assertTrue((Path(tmp) / "a" / "confusion.csv").is_file())
            self.assertTrue((Path(tmp) / "b" / "per_class.csv").is_file())

    def test_clearing_keeps_only_the_detection_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "detections").mkdir()
            (root / "detections" / "cached.jsonl").write_text("x")
            (root / "yolo").mkdir()
            (root / "yolo" / "confusion.csv").write_text("x")
            (root / "summary.csv").write_text("x")
            clear_results(root)
            self.assertEqual([p.name for p in root.iterdir()], ["detections"])
            self.assertTrue((root / "detections" / "cached.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run; expect ImportError.**

- [ ] **Step 3: Implement**

```python title="vtia/application/compare_models.py"
"""Use case: run several detectors on one video and say what the available data can show."""
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from vtia.application.evaluate import AnnotatedSequence, EvalResult, MapMetric, MotMetric, evaluate
from vtia.application.ports import Detector, Tracker
from vtia.application.process_video import process_video
from vtia.domain.classes import RoadClass
from vtia.domain.counting import count_by
from vtia.domain.counting_evaluation import evaluate_count_rows
from vtia.domain.model_selection import Disagreement, ModelScore, Verdict, disagreement, select_best
from vtia.domain.models import CrossingEvent, PointConfig


@dataclass(frozen=True)
class ManualCount:
    """Human count of a class, in total or for one line and direction."""
    road_class: RoadClass
    manual: int
    line: str | None = None
    direction: str | None = None


@dataclass
class ModelResult:
    name: str
    error: str | None = None
    counts_by_class: dict[RoadClass, int] = field(default_factory=dict)
    crossings: int = 0
    tracks: int = 0
    fps: float | None = None
    evaluation: EvalResult | None = None
    counting: dict | None = None   # evaluate_count_rows output


@dataclass
class Comparison:
    models: list[ModelResult]
    verdict: Verdict
    disagreement: Disagreement


Build = Callable[[str], tuple[Detector, Tracker]]


def compare_models(video: Path, point: PointConfig, fps: float, models: Sequence[str], build: Build, *,
                   truth: AnnotatedSequence | None = None, manual: list[ManualCount] | None = None,
                   map_metric: MapMetric | None = None, mot_metric: MotMetric | None = None, conf: float = 0.25,
                   on_model: Callable[[int, int, str], None] | None = None) -> Comparison:
    results = []
    for i, model in enumerate(models, 1):
        name = Path(model).stem
        if on_model:
            on_model(i, len(models), name)
        try:
            results.append(_run_one(name, model, video, point, fps, build, truth, manual, map_metric, mot_metric, conf))
        except Exception as e:  # one model failing (weights, memory...) must not lose the others
            results.append(ModelResult(name, error=str(e)))
    valid = [r for r in results if r.error is None]
    scores = [ModelScore(r.name, r.evaluation.classes["macro_f1"] if r.evaluation else None,
                         r.counting["overall"]["mae"] if r.counting else None, r.fps) for r in valid]
    return Comparison(results, select_best(scores), disagreement({r.name: r.counts_by_class for r in valid}))


def _run_one(name, model, video, point, fps, build, truth, manual, map_metric, mot_metric, conf) -> ModelResult:
    detector, tracker = build(model)
    analysis = process_video(video, detector, tracker, point, fps)
    evaluation = evaluate([truth], detector, tracker, map_metric, mot_metric, conf=conf) if truth else None
    seconds = getattr(detector, "seconds", 0)
    return ModelResult(
        name, None, dict(Counter(e.road_class for e in analysis.events)), len(analysis.events), analysis.tracks,
        detector.frames / seconds if seconds else None, evaluation,
        evaluate_count_rows(_count_rows(manual, analysis.events)) if manual else None)


def _count_rows(manual: list[ManualCount], events: list[CrossingEvent]) -> list[dict]:
    by_key = count_by(events)
    rows = []
    for m in manual:
        if m.line is None:
            automatic = sum(n for (_, _, c), n in by_key.items() if c == m.road_class)
        else:
            automatic = by_key[(m.line, m.direction, m.road_class)]
        rows.append({"class": m.road_class.value, "manual": m.manual, "automatic": automatic,
                     "line": m.line, "direction": m.direction})
    return rows
```

```python title="vtia/adapters/comparison_files.py"
"""Files a model comparison leaves on disk."""
import shutil
from pathlib import Path

from vtia.adapters.table_files import markdown_table, write_csv
from vtia.application.compare_models import Comparison, ModelResult
from vtia.application.evaluate import EvalResult
from vtia.domain.classes import RoadClass
from vtia.domain.evaluation import BACKGROUND, MISSED

CACHE_DIR = "detections"
SUMMARY_HEADER = ["model", "error", "crossings", "tracks", "fps_inference", "macro_F1", "mAP50", "mAP50_95",
                  "HOTA", "MOTA", "IDF1", "IDSW", "count_MAE", "count_RMSE"]


def write_model_tables(out: Path, name: str, r: EvalResult) -> str:
    out.mkdir(parents=True, exist_ok=True)
    classes = list(RoadClass)
    per_class_header = ["class", "precision", "recall", "f1", "support", "AP50", "AP50_95"]
    per_class = [[c, m["precision"], m["recall"], m["f1"], m["support"],
                  r.detection["per_class"][c]["AP50"], r.detection["per_class"][c]["AP50_95"]]
                 for c, m in r.classes["per_class"].items()]
    confusion_header = ["true \\ predicted", *classes, MISSED]
    confusion = [[row, *(r.confusion[(row, col)] for col in [*classes, MISSED])] for row in [*classes, BACKGROUND]]
    write_csv(out / "per_class.csv", per_class_header, per_class)
    write_csv(out / "confusion.csv", confusion_header, confusion)
    return (f"## {name}\n\n" + markdown_table(per_class_header, per_class) + "\n"
            + markdown_table(confusion_header, confusion))


def clear_results(out: Path) -> None:
    """Removes an earlier comparison so runs never mix; the detection cache stays."""
    if not out.is_dir():
        return
    for item in out.iterdir():
        if item.name == CACHE_DIR:
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()


def write_comparison(out: Path, comparison: Comparison) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "summary.csv", SUMMARY_HEADER, [_summary_row(m) for m in comparison.models])
    for m in comparison.models:
        if m.evaluation:
            write_model_tables(out / m.name, m.name, m.evaluation)


def _summary_row(m: ModelResult) -> list:
    e, c = m.evaluation, m.counting
    return [m.name, m.error or "", m.crossings, m.tracks, m.fps if m.fps is not None else "",
            *([e.classes["macro_f1"], e.detection["mAP50"], e.detection["mAP50_95"], e.tracking["HOTA"],
               e.tracking["MOTA"], e.tracking["IDF1"], e.tracking["IDSW"]] if e else [""] * 7),
            *([c["overall"]["mae"], c["overall"]["rmse"]] if c else ["", ""])]
```

En `vtia/cli.py`: eliminar `_write_model_tables`, importar `write_model_tables` desde `vtia.adapters.comparison_files` y usarla en `evaluate_cmd`.

- [ ] **Step 4: Run all tests; expect PASS.**

---

### Task 4: Cableado y protección de la partición de prueba

**Files:** Modify `vtia/compose.py`, `tests/test_compose.py`.

**Interfaces:** Produces `compare_file(video, point_path, *, annotations=None, manual_csv=None, models=MODELS, out=Path("outputs/dashboard"), splits=Path("data/splits.csv"), on_model=None) -> ComparisonRun(comparison, out)`.

- [ ] **Step 1: Write the failing tests** — añadir a `tests/test_compose.py`:

```python
class CompareFileTest(unittest.TestCase):
    def manifest(self, root, split):
        path = root / "splits.csv"
        path.write_text(f"clip,split\nclip,{split}\n", encoding="utf-8")
        return path

    def test_annotations_of_a_test_partition_clip_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp, patch("vtia.compose.compare_models") as compare:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "partición de prueba"):
                compare_file(Path("clip.mp4"), POINT, annotations=root / "a.zip", out=root,
                             splits=self.manifest(root, "test"))
            compare.assert_not_called()

    def test_test_partition_clip_can_be_compared_without_annotations(self):
        comparison = Comparison([], Verdict((), (), None), Disagreement({}, None))
        with tempfile.TemporaryDirectory() as tmp, patch("vtia.compose.compare_models", return_value=comparison):
            root = Path(tmp)
            run = compare_file(Path("clip.mp4"), POINT, out=root, splits=self.manifest(root, "test"))
            self.assertEqual(run.out, root / "clip" / "comparison")
            self.assertTrue((run.out / "summary.csv").is_file())

    def test_earlier_results_are_removed_but_the_cache_stays(self):
        comparison = Comparison([], Verdict((), (), None), Disagreement({}, None))
        with tempfile.TemporaryDirectory() as tmp, patch("vtia.compose.compare_models", return_value=comparison):
            root = Path(tmp)
            target = root / "clip" / "comparison"
            (target / "detections").mkdir(parents=True)
            (target / "old_model").mkdir()
            compare_file(Path("clip.mp4"), POINT, out=root, splits=root / "none.csv")
            self.assertFalse((target / "old_model").exists())
            self.assertTrue((target / "detections").exists())
```

Con los imports `from vtia.application.compare_models import Comparison`, `from vtia.domain.model_selection import Disagreement, Verdict` y `compare_file` desde `vtia.compose`.

- [ ] **Step 2: Run; expect ImportError.**

- [ ] **Step 3: Implement** — añadir a `vtia/compose.py`:

```python
@dataclass
class ComparisonRun:
    comparison: Comparison
    out: Path


def compare_file(video: Path, point_path: Path, *, annotations: Path | None = None, manual_csv: str | None = None,
                 models: Sequence[str] = tuple(MODELS), out: Path = Path("outputs/dashboard"),
                 splits: Path = Path("data/splits.csv"),
                 on_model: Callable[[int, int, str], None] | None = None) -> ComparisonRun:
    """Runs every model on one video. Annotations and manual counts are optional; see compare_models."""
    video, splits = Path(video), Path(splits)
    if annotations and splits.is_file() and read_split_manifest(splits).get(video.stem) == "test":
        raise ValueError(f"{video.name} pertenece a la partición de prueba: elegir un modelo con ella invalidaría "
                         "esa evaluación. Usa un clip de validación o compara sin anotaciones.")
    point = load_point(point_path)
    fps, _ = video_info(video)
    truth = map_metric = mot_metric = None
    if annotations:
        from vtia.adapters.coco_map import coco_map as map_metric
        from vtia.adapters.cvat_mot import read_annotations
        from vtia.adapters.trackeval_metrics import mot_metrics as mot_metric
        truth = read_annotations(annotations, video)
    manual = parse_manual_counts(manual_csv, point) if manual_csv else None
    target = Path(out) / video.stem / "comparison"
    clear_results(target)

    def build(model: str):
        return CachedDetector(make_detector(model, 0.01), target / CACHE_DIR), make_tracker("bytetrack", model)

    comparison = compare_models(video, point, fps, models, build, truth=truth, manual=manual,
                                map_metric=map_metric, mot_metric=mot_metric, on_model=on_model)
    write_comparison(target, comparison)
    return ComparisonRun(comparison, target)
```

Imports nuevos: `Sequence` de `collections.abc`; `CachedDetector`; `CACHE_DIR, clear_results, write_comparison`; `parse_manual_counts`; `Comparison, compare_models`; `read_split_manifest`.

- [ ] **Step 4: Run all tests; expect PASS.**

---

### Task 5: Vista en el tablero

**Files:** Create `app/comparison_view.py`, `tests/test_comparison_view.py`. Modify `app/dashboard.py`, `vtia/adapters/labels_es.py`.

**Interfaces:** Consumes `compare_file`, `ComparisonRun`, `confusion_cells`. Produces `render(get_video, video_name, point, frames, root)`, `estimate_seconds(frames)`, `level_message(has_annotations, has_counts)`, `verdict_lines(verdict)`, `matrix_rows(counts)`.

- [ ] **Step 1: Write the failing tests**

```python title="tests/test_comparison_view.py"
"""Text and data of the comparison tab, plus a headless render.

Run: python -m unittest discover -s tests
"""
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import comparison_view as view  # noqa: E402
from vtia.domain.classes import RoadClass as C  # noqa: E402
from vtia.domain.evaluation import MISSED  # noqa: E402
from vtia.domain.model_selection import Verdict  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run; expect ImportError.**

- [ ] **Step 3: Implement**

Añadir a `vtia/adapters/labels_es.py`: `BACKGROUND_LABEL = "Falso positivo"`, `MISSED_LABEL = "No detectado"`.

```python title="app/comparison_view.py"
"""Dashboard tab: run the four detectors on the chosen video and show what the data can support."""
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from vtia.adapters import labels_es as es
from vtia.domain.evaluation import BACKGROUND, MISSED, confusion_cells
from vtia.domain.model_selection import Verdict

BAR, BEST = "#3987e5", "#9ec5f4"          # one hue; the best stands out, as the dashboard's peak interval does
RAMP = ["#16263b", "#2a5f9e", "#9ec5f4"]  # sequential, dark surface: more objects, lighter
EMPTY = "#262730"
GPU_FPS = (65, 37, 23, 13)                # measured on the project's AMD GPU (Docs/Explicacion_Fase3a.md, AVANCE.md)


def level_message(has_annotations: bool, has_counts: bool) -> str:
    if has_annotations:
        return ("Con anotaciones verás la matriz de confusión de cada modelo y sus métricas de detección y seguimiento."
                + ("" if has_counts else " Si añades un conteo manual verás también quién cuenta mejor."))
    if has_counts:
        return ("Con el conteo manual se puede indicar el mejor contando. "
                "Para ver matrices de confusión faltan las anotaciones CVAT.")
    return ("Solo con el video no se puede declarar un modelo más preciso: verás velocidad, cruces y dónde discrepan. "
            "Añade un conteo manual o anotaciones CVAT para medir aciertos.")


def estimate_seconds(frames: int) -> float:
    return frames * sum(1 / fps for fps in GPU_FPS)


def _names(names: tuple[str, ...]) -> str:
    return names[0] if len(names) == 1 else "empate entre " + ", ".join(names[:-1]) + f" y {names[-1]}"


def verdict_lines(v: Verdict) -> list[str]:
    lines = []
    if v.best_by_annotations:
        lines.append(f"**Mejor detectando y clasificando en este video:** {_names(v.best_by_annotations)}.")
    if v.best_by_counts:
        lines.append(f"**Mejor contando en este video:** {_names(v.best_by_counts)}.")
    if v.fastest:
        lines.append(f"**Más rápido:** {v.fastest}.")
    return lines or ["No hay datos para declarar un mejor modelo."]


def _label(value) -> str:
    return es.BACKGROUND_LABEL if value == BACKGROUND else es.MISSED_LABEL if value == MISSED else es.CLASS[value]


def matrix_rows(counts: Counter) -> list[dict]:
    return [{"Real": _label(c.true), "Predicho": _label(c.predicted), "Objetos": c.count, "Proporción": c.share,
             "Texto": "s/d" if c.share is None else str(c.count)} for c in confusion_cells(counts)]


def _matrix_chart(counts: Counter) -> alt.LayerChart:
    df = pd.DataFrame(matrix_rows(counts))
    order_y = [es.CLASS[c] for c in es.CLASS] + [es.BACKGROUND_LABEL]
    order_x = [es.CLASS[c] for c in es.CLASS] + [es.MISSED_LABEL]
    base = alt.Chart(df).encode(
        x=alt.X("Predicho:N", sort=order_x, axis=alt.Axis(labelAngle=-40, title="Predicho")),
        y=alt.Y("Real:N", sort=order_y, title="Real"),
        tooltip=["Real", "Predicho", "Objetos", alt.Tooltip("Proporción:Q", format=".0%")])
    cells = base.mark_rect(stroke="#0e1117", strokeWidth=2).encode(color=alt.condition(
        "isValid(datum['Proporción'])",
        alt.Color("Proporción:Q", scale=alt.Scale(domain=[0, 0.5, 1], range=RAMP),
                  legend=alt.Legend(format=".0%", title="% de la fila")),
        alt.value(EMPTY)))
    text = base.mark_text(fontSize=11).encode(text="Texto:N", color=alt.condition(
        "datum['Proporción'] > 0.6", alt.value("#0e1117"), alt.value("#f0efec")))
    return (cells + text).properties(height=300)


def _bars(rows: list[tuple[str, float]], title: str, best: tuple[str, ...], fmt: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["Modelo", title])
    df["Mejor"] = df["Modelo"].isin(best)
    st.altair_chart(alt.Chart(df).mark_bar(size=18, cornerRadiusEnd=4).encode(
        x=alt.X(f"{title}:Q"), y=alt.Y("Modelo:N", sort=None, title=None),
        color=alt.condition("datum.Mejor", alt.value(BEST), alt.value(BAR)),
        tooltip=["Modelo", alt.Tooltip(f"{title}:Q", format=fmt)]).properties(title=title, height=150), width="stretch")


def _table(models) -> pd.DataFrame:
    def row(m):
        e, c = m.evaluation, m.counting
        data = {"Modelo": m.name, "Estado": f"Error: {m.error}" if m.error else "OK", "Cruces": m.crossings,
                "Objetos seguidos": m.tracks, "FPS inferencia": m.fps}
        if e:
            data |= {"F1 macro": e.classes["macro_f1"], "mAP50": e.detection["mAP50"],
                     "mAP50-95": e.detection["mAP50_95"], "HOTA": e.tracking["HOTA"], "MOTA": e.tracking["MOTA"],
                     "IDF1": e.tracking["IDF1"], "ID switches": e.tracking["IDSW"]}
        if c:
            data |= {"MAE conteo": c["overall"]["mae"], "RMSE conteo": c["overall"]["rmse"]}
        return data
    return pd.DataFrame([row(m) for m in models])


def show(run, root: Path) -> None:
    c = run.comparison
    for line in verdict_lines(c.verdict):
        st.markdown(line)
    st.caption("Resultado de un solo video con umbrales sin calibrar: no permite generalizar a otras cámaras.")
    st.subheader("Tabla comparativa")
    st.dataframe(_table(c.models).round(3), hide_index=True, width="stretch")

    valid = [m for m in c.models if not m.error]
    st.subheader("Comparación por métrica")
    left, right = st.columns(2)
    with left:
        _bars([(m.name, m.fps) for m in valid if m.fps is not None], "FPS inferencia",
              (c.verdict.fastest,) if c.verdict.fastest else (), ".1f")
        _bars([(m.name, m.evaluation.classes["macro_f1"]) for m in valid if m.evaluation], "F1 macro",
              c.verdict.best_by_annotations, ".3f")
    with right:
        _bars([(m.name, m.crossings) for m in valid], "Cruces", (), "d")
        _bars([(m.name, m.counting["overall"]["mae"]) for m in valid if m.counting], "MAE de conteo",
              c.verdict.best_by_counts, ".2f")

    if c.disagreement.by_class:
        st.subheader("Discrepancia entre modelos")
        st.caption("Muestra desacuerdo entre modelos, no error: sin referencia humana no se sabe cuál acierta.")
        st.dataframe(pd.DataFrame([(es.CLASS[k], lo, hi, hi - lo) for k, (lo, hi) in c.disagreement.by_class.items()],
                                  columns=["Clase", "Mínimo", "Máximo", "Diferencia"]), hide_index=True, width="stretch")

    evaluated = [m for m in valid if m.evaluation]
    if evaluated:
        st.subheader("Matrices de confusión")
        st.caption("Color: porcentaje de cada clase real. Número: objetos. Confianza ≥ 0,25, IoU ≥ 0,5. s/d: sin datos.")
        columns = st.columns(2)
        for i, m in enumerate(evaluated):
            with columns[i % 2]:
                st.markdown(f"**{m.name}**")
                st.altair_chart(_matrix_chart(m.evaluation.confusion), width="stretch")

    st.subheader("Descargas")
    for f in sorted(p for p in run.out.rglob("*.csv") if "detections" not in p.parts):
        name = f.relative_to(run.out).as_posix()
        st.download_button(name, f.read_bytes(), file_name=name.replace("/", "_"), key=f"cmp-dl-{name}")


def render(get_video: Callable[[], Path], video_name: str, point: Path | None, frames: int, root: Path) -> None:
    from vtia.compose import compare_file, device_name

    st.subheader("Comparar los 4 modelos en este video")
    annotations = st.file_uploader("Anotaciones CVAT (zip MOT 1.1, opcional)", type=["zip"], key="cmp-cvat")
    counts = st.file_uploader("Conteo manual (CSV con class y manual, opcional)", type=["csv"], key="cmp-csv")
    st.info(level_message(annotations is not None, counts is not None))
    if frames:
        note = "" if device_name() != "CPU" else " Este equipo está usando CPU, así que tardará bastante más."
        st.caption(f"Tiempo estimado de detección con GPU: {es.duration(estimate_seconds(frames))}.{note}")

    if st.button("Comparar", type="primary", disabled=not video_name or point is None, key="cmp-run"):
        bar = st.progress(0.0, text="Preparando...")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                archive = None
                if annotations:
                    archive = Path(tmp) / "annotations.zip"
                    archive.write_bytes(annotations.getbuffer())
                st.session_state.comparison = compare_file(
                    get_video(), point, annotations=archive,
                    manual_csv=counts.getvalue().decode("utf-8-sig") if counts else None,
                    out=root / "outputs" / "dashboard", splits=root / "data" / "splits.csv",
                    on_model=lambda i, n, name: bar.progress((i - 1) / n, text=f"Modelo {i} de {n}: {name}"))
        except Exception as e:  # bad files or a blocked partition: explain it in the page
            bar.empty()
            st.session_state.pop("comparison", None)
            st.error(f"No se pudo comparar: {e}")
            return
        bar.empty()

    run = st.session_state.get("comparison")
    if run:
        show(run, root)
```

En `app/dashboard.py`:
1. Importar `comparison_view` (el directorio `app/` ya está en la ruta con `streamlit run`) y `video_info` desde `vtia.adapters.video_io`.
2. Añadir `save_upload()`, que guarda el video subido en `data/uploads` y devuelve su ruta; usarla en el bloque `if run:`.
3. Tras el panel lateral, crear `analysis_tab, compare_tab = st.tabs(["Análisis", "Comparar modelos"])`, renderizar `comparison_view.render` dentro de `compare_tab` y envolver el resto del script (de `if run:` al final) en `with analysis_tab:`. El orden importa: `st.stop()` del análisis no debe impedir dibujar la pestaña de comparación.

- [ ] **Step 4: Run all tests; expect PASS.**

---

### Task 6: Verificación real

- [ ] **Step 1:** Ejecutar el tablero headless con `streamlit.testing.v1.AppTest` sobre `app/dashboard.py` y comprobar que no hay excepciones y que existen las dos pestañas.
- [ ] **Step 2:** Renderizar `comparison_view.show` con datos sintéticos de los 3 niveles (sin material de personas), abrirlo en el navegador y revisar etiquetas, solapamientos, contraste de celdas y tooltips.
- [ ] **Step 3:** Ejecutar `compare_file` con los 4 modelos reales sobre un clip de Bogotá, fuera del proyecto, y confirmar 4 resultados, caché reutilizada en una segunda ejecución y `summary.csv`.
- [ ] **Step 4:** Actualizar el README con una sección breve sobre la pestaña y el formato del CSV.
