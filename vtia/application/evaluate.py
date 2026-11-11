"""Use case: score a detector + tracker against an annotated dataset."""
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vtia.application.ports import Detector, Tracker
from vtia.domain.evaluation import class_report, confusion, drop_ignored
from vtia.domain.models import Box, Detection, GroundTruth

Frames = list[list[GroundTruth]]
DetFrames = list[list[Detection]]
# Metric ports as plain functions (implemented in adapters with pycocotools / TrackEval).
MapMetric = Callable[[Frames, DetFrames], dict]              # all frames of all sequences
MotMetric = Callable[[list[tuple[Frames, DetFrames]]], dict]  # (truths, tracks) per sequence


@dataclass(frozen=True)
class AnnotatedSequence:
    name: str
    source: Path         # video file or folder of frames
    truths: Frames       # per frame
    ignore: list[list[Box]]  # per frame: regions excluded from scoring
    fps: float = 30.0    # of the source; folders of frames don't record one


@dataclass
class EvalResult:
    confusion: Counter   # (true class | BACKGROUND, predicted class | MISSED) -> count
    classes: dict        # domain.evaluation.class_report
    detection: dict      # MapMetric output
    tracking: dict       # MotMetric output
    per_sequence: dict   # sequence -> detection/tracking/class metrics


def evaluate(sequences: list[AnnotatedSequence], detector: Detector, tracker: Tracker,
             map_metric: MapMetric, mot_metric: MotMetric, conf: float = 0.25) -> EvalResult:
    """`conf` is the operating threshold for the confusion matrix; mAP uses every detection."""
    counts: Counter = Counter()
    all_truths: Frames = []
    all_dets: DetFrames = []
    tracking_sequences = []
    sequence_results = {}
    for seq in sequences:
        raw = list(detector.detect(seq.source))
        if len(raw) != len(seq.truths):
            raise ValueError(f"{seq.name}: {len(raw)} frames read but {len(seq.truths)} annotated")
        truths = [drop_ignored(f, ig) for f, ig in zip(seq.truths, seq.ignore)]
        dets = [drop_ignored(f, ig) for f, ig in zip(raw, seq.ignore)]
        sequence_counts = Counter()
        for frame_truths, frame in zip(truths, dets):
            frame_counts = confusion(frame_truths, [d for d in frame if d.confidence >= conf])
            counts += frame_counts
            sequence_counts += frame_counts
        # The tracker sees everything, as in production; ignore regions only apply to scoring.
        tracks = [drop_ignored(f, ig) for f, ig in zip(tracker.track(raw, seq.fps), seq.ignore)]
        all_truths += truths
        all_dets += dets
        tracking_sequences.append((truths, tracks))
        sequence_results[seq.name] = {
            "classes": class_report(sequence_counts),
            "detection": map_metric(truths, dets),
            "tracking": mot_metric([(truths, tracks)]),
            "frames": len(truths),
            "ground_truth_instances": sum(len(frame) for frame in truths),
        }
    return EvalResult(counts, class_report(counts), map_metric(all_truths, all_dets),
                      mot_metric(tracking_sequences), sequence_results)
