"""Tracking metrics with TrackEval: HOTA, CLEAR (MOTA, MOTP, ID switches) and Identity (IDF1).

Class-agnostic: identity is scored over all six classes together.
"""
import numpy as np

# ponytail: TrackEval's last release (2022) uses np.float/np.int, removed in numpy 2 (np.bool came
# back in numpy 2 as numpy's own type, so it must not be overwritten). Restore only what is missing;
# drop this when a maintained release supports numpy 2.
for _name, _alias in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _alias)

from trackeval.metrics import CLEAR, HOTA, Identity  # noqa: E402  (needs the aliases above)

from vtia.domain.models import Detection, GroundTruth  # noqa: E402


def _iou_matrix(a: list, b: list) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    a, b = np.array(a, dtype=float)[:, None, :], np.array(b, dtype=float)[None, :, :]
    w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = w * h
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])  # noqa: E731
    return inter / np.maximum(area(a) + area(b) - inter, 1e-9)


def _sequence_data(truths: list[list[GroundTruth]], tracks: list[list[Detection]]) -> dict:
    """TrackEval's per-sequence input. Ids are remapped to 0..n-1 as TrackEval requires."""
    gt_map, tr_map = {}, {}
    gt_ids, tr_ids, sims = [], [], []
    for frame_truths, frame_tracks in zip(truths, tracks):
        gt_ids.append(np.array([gt_map.setdefault(t.object_id, len(gt_map)) for t in frame_truths], dtype=int))
        tr_ids.append(np.array([tr_map.setdefault(d.track_id, len(tr_map)) for d in frame_tracks], dtype=int))
        sims.append(_iou_matrix([t.box for t in frame_truths], [d.box for d in frame_tracks]))
    return {
        "num_timesteps": len(truths), "num_gt_ids": len(gt_map), "num_tracker_ids": len(tr_map),
        "num_gt_dets": sum(map(len, gt_ids)), "num_tracker_dets": sum(map(len, tr_ids)),
        "gt_ids": gt_ids, "tracker_ids": tr_ids, "similarity_scores": sims,
    }


def mot_metrics(sequences: list[tuple[list[list[GroundTruth]], list[list[Detection]]]]) -> dict:
    config = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    metrics = {"HOTA": HOTA(), "CLEAR": CLEAR(config), "Identity": Identity(config)}
    per_seq = {name: {} for name in metrics}
    gt_objects = 0
    for i, (truths, tracks) in enumerate(sequences):
        data = _sequence_data(truths, tracks)
        gt_objects += data["num_gt_ids"]
        for name, m in metrics.items():
            per_seq[name][i] = m.eval_sequence(data)
    res = {name: m.combine_sequences(per_seq[name]) for name, m in metrics.items()}
    idsw = int(res["CLEAR"]["IDSW"])
    return {
        "HOTA": float(np.mean(res["HOTA"]["HOTA"])),  # averaged over IoU thresholds 0.05..0.95
        "MOTA": float(res["CLEAR"]["MOTA"]),
        "MOTP": float(res["CLEAR"]["MOTP"]),
        "IDF1": float(res["Identity"]["IDF1"]),
        "IDSW": idsw,
        "IDSW_per_100_objects": 100 * idsw / gt_objects if gt_objects else 0.0,
    }
