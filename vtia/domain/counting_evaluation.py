"""Scientific evaluation of manual versus automatic crossing counts."""
from __future__ import annotations

from collections import defaultdict
from math import sqrt
from statistics import mean
from vtia.domain.classes import RoadClass


def count_error(manual: int, automatic: int) -> dict:
    if isinstance(manual, bool) or isinstance(automatic, bool) or not isinstance(manual, int) or not isinstance(automatic, int):
        raise TypeError("manual and automatic counts must be integers")
    if manual < 0 or automatic < 0:
        raise ValueError("counts cannot be negative")
    difference = automatic - manual
    return {
        "manual": manual,
        "automatic": automatic,
        "difference": difference,
        "absolute_difference": abs(difference),
        "error_percent": abs(difference) / manual * 100 if manual else None,
    }


def evaluate_count_rows(rows: list[dict]) -> dict:
    """Rows must contain manual, automatic, class; optional video/point/date/band/direction/line."""
    if not rows:
        return {"status": "pending_data", "reason": "No manual/automatic count records supplied.",
                "records": [], "overall": {"n": 0, "mae": None, "rmse": None}}
    records = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows, 1):
        try:
            manual, automatic = int(row["manual"]), int(row["automatic"])
            klass = str(row["class"]).strip()
            if klass not in {c.value for c in RoadClass}:
                raise ValueError(f"unknown class {klass!r}; expected one of {[c.value for c in RoadClass]}")
            metrics = count_error(manual, automatic)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid count record at row {index}: {exc}") from exc
        record = {**{k: row.get(k) for k in ("video", "point", "date", "time_band", "class", "direction", "line")}, **metrics}
        records.append(record)
        for key in (("class", klass), ("direction", row.get("direction")), ("line", row.get("line")),
                    ("video", row.get("video")), ("point", row.get("point")), ("date", row.get("date")),
                    ("time_band", row.get("time_band"))):
            if key[1] is not None:
                groups[key].append(record)

    errors = [r["difference"] for r in records]
    overall = {"n": len(errors), "mae": mean(abs(x) for x in errors),
               "rmse": sqrt(mean(x * x for x in errors))}
    breakdown = {}
    for (dimension, value), items in groups.items():
        diffs = [r["difference"] for r in items]
        manual_total = sum(r["manual"] for r in items)
        automatic_total = sum(r["automatic"] for r in items)
        breakdown.setdefault(dimension, {})[str(value)] = {
            "n": len(diffs), "manual_total": manual_total,
            "automatic_total": automatic_total,
            "difference_total": automatic_total - manual_total,
            "absolute_error_total": sum(abs(x) for x in diffs),
            "aggregate_error_percent": abs(automatic_total - manual_total) / manual_total * 100 if manual_total else None,
            "mae": mean(abs(x) for x in diffs),
            "rmse": sqrt(mean(x * x for x in diffs)),
        }
    return {"status": "evaluated", "records": records, "overall": overall, "breakdown": breakdown}
