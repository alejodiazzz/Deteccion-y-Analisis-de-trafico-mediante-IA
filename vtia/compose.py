"""Composition root shared by both entry points (cli.py and app/dashboard.py).

Builds the adapters, runs the use cases and writes every output of a video analysis.
"""
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from vtia.adapters.annotated_video import AnnotatedVideo
from vtia.adapters.anonymized_video import blur_video, contact_sheet
from vtia.adapters.faster_rcnn_detector import FasterRCNNDetector
from vtia.adapters.labels_es import duration
from vtia.adapters.mot_tracker import MotTracker
from vtia.adapters.pdf_report import write_pdf
from vtia.adapters.report_files import write_summary_tables, write_tables
from vtia.adapters.ultralytics_detector import UltralyticsDetector
from vtia.adapters.video_io import video_info
from vtia.adapters.yaml_config import load_point
from vtia.application.anonymize import regions_per_frame
from vtia.application.postprocessed import PostProcessed
from vtia.application.process_video import VideoAnalysis, process_video
from vtia.application.summary import Summary, summarize
from vtia.domain.postprocess import merge_riders, suppress_duplicates
from vtia.domain.timeline import video_start

MODELS = ["yolo26n.pt", "yolo26s.pt", "rtdetr-l.pt", "faster_rcnn"]  # any Ultralytics weights also work
# Tracker threshold per model until calibrated on the project's own data. yolo26n gives low
# confidences: 0.3 counted 9/9 on the sample video and roughly doubled MOTA on VisDrone (2026-10-07).
TRACK_THRESH = {"yolo26n": 0.3}
DEFAULT_TRACK_THRESH = 0.6  # library default


def make_detector(name: str, conf: float):
    return FasterRCNNDetector(conf) if name == "faster_rcnn" else UltralyticsDetector(name, conf)


def default_track_thresh(model: str) -> float:
    return TRACK_THRESH.get(Path(model).stem, DEFAULT_TRACK_THRESH)


def make_tracker(algorithm: str, model: str, track_thresh: float | None = None) -> MotTracker:
    return MotTracker(algorithm, default_track_thresh(model) if track_thresh is None else track_thresh)


def with_postprocess(detector, dedup: float | None = None, riders: float | None = None):
    """Optional clean-up steps (experiments): duplicate removal, then rider merge (E2)."""
    if dedup:
        detector = PostProcessed(detector, lambda f: suppress_duplicates(f, dedup), "dedup")
    if riders:
        detector = PostProcessed(detector, lambda f: merge_riders(f, riders), "riders")
    return detector


def device_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


@dataclass
class Run:
    analysis: VideoAnalysis
    summary: Summary
    out: Path            # folder with every output file
    seconds: float       # processing time


def analyze_file(video: Path, point_path: Path, *, model: str = "yolo26n.pt", tracker: str = "bytetrack",
                 track_thresh: float | None = None, conf: float = 0.1, dedup: float | None = None,
                 riders: float | None = None, start: datetime | None = None, out: Path = Path("outputs"), save_video: bool = False,
                 progress: Callable[[int, int], None] | None = None) -> Run:
    """Counts, zones and alerts for one video; writes CSVs, report.pdf and optionally annotated.mp4."""
    video = Path(video)
    point = load_point(point_path)
    start = start or video_start(video.stem)
    fps, total_frames = video_info(video)
    out = Path(out) / video.stem
    out.mkdir(parents=True, exist_ok=True)
    if not save_video:
        (out / "annotated.mp4").unlink(missing_ok=True)  # an earlier run's video doesn't match these results
    drawer = AnnotatedVideo(video, out / "annotated.mp4", point) if save_video else None

    def on_frame(i, tracked, crossings, alerts):
        if drawer:
            drawer(i, tracked, crossings, alerts)
        if progress:
            progress(i + 1, total_frames)

    began = time.perf_counter()
    mot = make_tracker(tracker, model, track_thresh)
    try:
        detector = with_postprocess(make_detector(model, conf), dedup, riders)
        analysis = process_video(video, detector, mot, point, fps, start, on_frame)
    finally:
        if drawer:
            drawer.close()
    seconds = time.perf_counter() - began

    summary = summarize(analysis)
    write_tables(analysis, out)
    write_summary_tables(summary, out)
    write_pdf(out / "report.pdf", summary, analysis, {
        "Video": video.name,
        "Punto de cámara": Path(point_path).name,
        "Inicio": f"{start:%Y-%m-%d %H:%M:%S}" if start else "desconocido",
        "Duración": duration(summary.duration_s),
        "Modelo": f"{Path(model).stem} (confianza mínima {conf})" + (f", dedup {dedup}" if dedup else "")
                  + (f", jinetes unidos ({riders})" if riders else ""),
        "Tracker": f"{tracker}, umbral {mot.track_thresh}",
        "Procesado en": f"{device_name()}, {analysis.frames / seconds:.1f} cuadros/s",
    })
    return Run(analysis, summary, out, seconds)


ANONYMIZE_MODEL = "yolo26s.pt"  # recall matters more than speed here: a missed person keeps their face
ANONYMIZE_CONF = 0.15           # low on purpose: blurring a false positive costs nothing


def anonymize_file(video: Path, out_dir: Path, *, model: str = ANONYMIZE_MODEL, conf: float = ANONYMIZE_CONF,
                   hold_frames: int = 3, progress: Callable[[int, int], None] | None = None) -> tuple[Path, Path]:
    """Blurs heads and plates; writes <out_dir>/<name>.mp4 and <name>_check.jpg (frames to review by eye)."""
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out, sheet = out_dir / f"{video.stem}.mp4", out_dir / f"{video.stem}_check.jpg"
    if out.resolve() == video.resolve():
        raise ValueError("the anonymised copy would overwrite the original: choose another output folder")
    _, total = video_info(video)
    regions = regions_per_frame(video, make_detector(model, conf), hold_frames)
    blur_video(video, out, regions, (lambda i: progress(i, total)) if progress else None)
    contact_sheet(out, sheet)
    return out, sheet


PRELABEL_MODEL = "yolo26x.pt"  # not one of the compared models: keeps the ground truth from leaning toward them


def prelabel_file(video: Path, out_dir: Path, *, model: str = PRELABEL_MODEL, conf: float = 0.25,
                  track_thresh: float = 0.4, keyframe_every: int = 5,
                  progress: Callable[[int, int], None] | None = None) -> tuple[Path, int]:
    """Pre-annotations for CVAT: tracks from a large model, riders merged into their vehicle as the
    annotation guide asks. Returns the MOT 1.1 zip and the number of tracks."""
    from vtia.adapters.cvat_mot import write_preannotations  # only this command needs it

    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps, total = video_info(video)
    detector = with_postprocess(make_detector(model, conf), riders=0.4)
    tracked = []
    for i, frame in enumerate(make_tracker("bytetrack", model, track_thresh).track(detector.detect(video), fps)):
        tracked.append(frame)
        if progress:
            progress(i + 1, total)
    out = out_dir / f"{video.stem}.zip"
    return out, write_preannotations(tracked, out, keyframe_every)
