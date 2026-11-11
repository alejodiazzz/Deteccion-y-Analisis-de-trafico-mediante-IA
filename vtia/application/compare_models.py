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
