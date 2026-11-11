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
