"""Reads a camera-point YAML (see configs/points/) into a domain PointConfig.

    lines:          # required, at least one
      - {name, start: [x, y], end: [x, y], to_right, to_left, allowed?}
    zones:          # optional
      - {name, polygon: [[x, y], ...], forbidden?: [class, ...], max_dwell_s?, max_occupancy?}
    restrictions:   # optional
      - {class, hours: ["HH:MM-HH:MM", ...]}
"""
from datetime import datetime
from pathlib import Path

import yaml

from vtia.domain.classes import RoadClass
from vtia.domain.models import Line, PointConfig, Restriction, Zone


def load_point(path: Path) -> PointConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        config = PointConfig(
            lines=tuple(_line(item) for item in data["lines"]),
            zones=tuple(_zone(item) for item in data.get("zones") or []),
            restrictions=tuple(_restriction(item) for item in data.get("restrictions") or []),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{path}: invalid camera-point config ({e!r}); see the format in {__file__}") from e
    for kind, items in (("line", config.lines), ("zone", config.zones)):
        names = [x.name for x in items]
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: duplicate {kind} names {names}")
    if not config.lines:
        raise ValueError(f"{path}: at least one line is required")
    return config


def _line(item: dict) -> Line:
    line = Line(str(item["name"]), _point(item["start"]), _point(item["end"]),
                str(item["to_right"]), str(item["to_left"]), item.get("allowed"))
    if line.allowed is not None and line.allowed not in (line.to_right, line.to_left):
        raise ValueError(f"line {line.name}: allowed must be {line.to_right!r} or {line.to_left!r}")
    return line


def _zone(item: dict) -> Zone:
    polygon = tuple(_point(p) for p in item["polygon"])
    if len(polygon) < 3:
        raise ValueError(f"zone {item['name']}: a polygon needs at least 3 points")
    max_dwell = item.get("max_dwell_s")
    max_occupancy = item.get("max_occupancy")
    return Zone(str(item["name"]), polygon, frozenset(RoadClass(c) for c in item.get("forbidden") or []),
                None if max_dwell is None else float(max_dwell),
                None if max_occupancy is None else int(max_occupancy))


def _restriction(item: dict) -> Restriction:
    windows = []
    for span in item["hours"]:
        a, b = (datetime.strptime(t.strip(), "%H:%M").time() for t in str(span).split("-"))
        windows.append((a, b))
    return Restriction(RoadClass(item["class"]), tuple(windows))


def _point(value) -> tuple[float, float]:
    x, y = value
    return float(x), float(y)
