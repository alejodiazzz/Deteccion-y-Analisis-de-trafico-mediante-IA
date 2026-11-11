"""Evidence-aware robustness summaries by manually assigned capture conditions."""
from collections import defaultdict
from statistics import mean

CONDITION_FIELDS = ("lighting", "occlusion", "vehicle_density", "pedestrian_density", "object_size", "distance")


def summarize_robustness(per_sequence: dict, condition_rows: list[dict], min_sequences: int = 3) -> dict:
    """Join per-sequence evaluation metrics with a manually reviewed condition CSV."""
    if min_sequences < 2:
        raise ValueError("min_sequences must be at least 2")
    conditions = {}
    for row in condition_rows:
        name = (row.get("sequence") or "").strip()
        if not name or name in conditions:
            raise ValueError(f"condition metadata has an empty or duplicate sequence: {name!r}")
        conditions[name] = row
    groups = defaultdict(list)
    missing_metrics, missing_conditions = [], []
    for name, metrics in per_sequence.items():
        row = conditions.get(name)
        if row is None:
            missing_conditions.append(name)
            continue
        for field in CONDITION_FIELDS:
            value = (row.get(field) or "").strip()
            if value:
                groups[(field, value)].append(metrics)
    for name in conditions:
        if name not in per_sequence:
            missing_metrics.append(name)
    output = {}
    for (field, value), items in sorted(groups.items()):
        n = len(items)
        if n < min_sequences:
            output.setdefault(field, {})[value] = {"status": "insufficient_evidence", "sequences": n,
                                                    "minimum_sequences": min_sequences}
            continue
        metrics = {}
        for key, section in (("mAP50", "detection"), ("mAP50_95", "detection"),
                             ("HOTA", "tracking"), ("MOTA", "tracking"), ("IDF1", "tracking"),
                             ("macro_f1", "classes")):
            values = [item[section].get(key) for item in items if item[section].get(key) is not None]
            metrics[key] = mean(values) if values else None
        output.setdefault(field, {})[value] = {"status": "supported", "sequences": n, "metrics_mean": metrics}
    return {"minimum_sequences": min_sequences, "conditions": output,
            "sequences_without_condition_metadata": sorted(missing_conditions),
            "condition_rows_without_evaluation": sorted(missing_metrics)}
