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
