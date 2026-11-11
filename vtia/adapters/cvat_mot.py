"""CVAT "MOT 1.1" archives: write pre-annotations to import, read finished annotations back.

Archive: gt/gt.txt rows `frame (from 1), track id, x, y, w, h, not ignored, class id, visibility`
and gt/labels.txt with one label per line (class id = line number, from 1).
"""
import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path

from vtia.adapters.video_io import VIDEO_SUFFIXES, video_info
from vtia.application.evaluate import AnnotatedSequence
from vtia.domain.classes import RoadClass
from vtia.domain.counting import ClassVotes
from vtia.domain.models import Detection, GroundTruth

LABELS = [c.value for c in RoadClass] + ["ignore"]  # same order as the CVAT project labels (README)


def write_preannotations(tracked_frames: list[list[Detection]], out: Path, keyframe_every: int = 5) -> int:
    """Tracks -> MOT 1.1 zip for CVAT's "Upload annotations". One label per track (its majority class).
    Only every `keyframe_every`-th frame plus each track's first and last are written; CVAT
    interpolates between them, which keeps corrections cheap. Returns the number of tracks."""
    votes = ClassVotes()
    frames_of: dict[int, list[Detection]] = defaultdict(list)
    for frame in tracked_frames:
        for d in frame:
            votes.add(d)
            frames_of[d.track_id].append(d)
    rows = []
    for track, dets in frames_of.items():
        class_id = LABELS.index(votes[track].value) + 1
        keep = [d for i, d in enumerate(dets) if i == 0 or i == len(dets) - 1 or d.frame % keyframe_every == 0]
        for d in keep:
            x1, y1, x2, y2 = d.box
            rows.append((d.frame + 1, track, round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1), 1, class_id, 1))
    rows.sort()
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("gt/gt.txt", buf.getvalue())
        z.writestr("gt/labels.txt", "\n".join(LABELS) + "\n")
    return len(frames_of)


def read_annotations(archive: Path, video: Path) -> AnnotatedSequence:
    """Read CVAT MOT 1.1 and reject malformed/out-of-range annotations.

    MOT frame numbers are one-based. Invalid rows are errors, never silently dropped.
    """
    files = _read_files(Path(archive))
    raw_labels = files["gt/labels.txt"].splitlines()
    labels = [label.strip() for label in raw_labels]
    if not labels or any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError(f"{archive}: labels.txt must contain unique, non-empty labels without blank lines")
    valid_labels = {c.value for c in RoadClass} | {"ignore"}
    unknown = sorted(set(labels) - valid_labels)
    if unknown:
        raise ValueError(f"{archive}: unsupported labels in MOT export: {unknown}")
    fps, n = video_info(video)
    if n <= 0:
        raise ValueError(f"{video}: cannot determine a positive video frame count")
    truths = [[] for _ in range(n)]
    ignore = [[] for _ in range(n)]
    seen = set()
    reader = csv.reader(io.StringIO(files["gt/gt.txt"]))
    for line_no, row in enumerate(reader, 1):
        if not row:
            continue
        if len(row) != 9:
            raise ValueError(f"{archive}: gt/gt.txt row {line_no} must have 9 MOT 1.1 columns, got {len(row)}")
        try:
            frame_raw, track_raw = float(row[0]), float(row[1])
            x, y, w, h = (float(v) for v in row[2:6])
            considered_raw, label_raw, visibility = float(row[6]), float(row[7]), float(row[8])
        except ValueError as exc:
            raise ValueError(f"{archive}: non-numeric value at gt.txt row {line_no}") from exc
        values = (frame_raw, track_raw, x, y, w, h, considered_raw, label_raw, visibility)
        import math
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{archive}: non-finite value at gt.txt row {line_no}")
        frame, track = int(frame_raw) - 1, int(track_raw)
        if frame_raw != int(frame_raw) or track_raw != int(track_raw) or track < 0:
            raise ValueError(f"{archive}: frame and track id must be integers (row {line_no})")
        if not 0 <= frame < n:
            raise ValueError(f"{archive}: frame {frame + 1} at row {line_no} is outside video range 1..{n}")
        if w <= 0 or h <= 0:
            raise ValueError(f"{archive}: bounding-box width/height must be positive (row {line_no})")
        if considered_raw not in (0, 1) or label_raw != int(label_raw) or not 1 <= int(label_raw) <= len(labels):
            raise ValueError(f"{archive}: invalid MOT mark/class id at row {line_no}")
        if not 0 <= visibility <= 1:
            raise ValueError(f"{archive}: visibility must be in [0,1] (row {line_no})")
        key = (frame, track)
        if key in seen:
            raise ValueError(f"{archive}: duplicate frame/track pair {key} at row {line_no}")
        seen.add(key)
        label = labels[int(label_raw) - 1]
        box = (x, y, x + w, y + h)
        if label == "ignore" or not considered_raw:
            ignore[frame].append(box)
        else:
            truths[frame].append(GroundTruth(frame, track, RoadClass(label), box))
    return AnnotatedSequence(Path(video).stem, Path(video), truths, ignore, fps)


def _read_files(archive: Path) -> dict[str, str]:
    names = ("gt/gt.txt", "gt/labels.txt")
    if archive.is_dir():
        missing = [n for n in names if not (archive / n).is_file()]
        if missing:
            raise ValueError(f"{archive}: missing required MOT files: {missing}")
        return {n: (archive / n).read_text(encoding="utf-8-sig") for n in names}
    try:
        with zipfile.ZipFile(archive) as z:
            selected = {}
            for required in names:
                matches = [i for i in z.namelist() if i == required or i.endswith("/" + required)]
                if len(matches) != 1:
                    raise ValueError(f"{archive}: expected exactly one {required}, found {len(matches)}")
                selected[required] = z.read(matches[0]).decode("utf-8-sig")
            return selected
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{archive}: invalid ZIP archive") from exc



def load_cvat_dir(annotations: Path, videos: Path) -> list[AnnotatedSequence]:
    """Pair each CVAT archive/folder with exactly one video sharing its stem."""
    annotations, videos = Path(annotations), Path(videos)
    if not annotations.is_dir() or not videos.is_dir():
        raise FileNotFoundError("annotations and videos must both be existing directories")
    archives = sorted(p for p in annotations.iterdir() if p.suffix.lower() == ".zip" or p.is_dir())
    video_files = sorted(p for p in videos.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    by_stem = {}
    for video in video_files:
        if video.stem in by_stem:
            raise ValueError(f"multiple videos share sequence stem {video.stem!r}")
        by_stem[video.stem] = video
    seen = set()
    sequences = []
    for archive in archives:
        if archive.stem in seen:
            raise ValueError(f"multiple annotation exports share sequence stem {archive.stem!r}")
        seen.add(archive.stem)
        video = by_stem.get(archive.stem)
        if video is None:
            raise FileNotFoundError(f"no video named {archive.stem}.* in {videos} for annotations {archive.name}")
        sequences.append(read_annotations(archive, video))
    return sequences


def validate_cvat_dataset(videos: Path, annotations: Path) -> dict:
    """Dataset inventory and annotation quality summary; no inference is performed."""
    videos, annotations = Path(videos), Path(annotations)
    if not videos.is_dir() or not annotations.is_dir():
        raise FileNotFoundError("videos and annotations must be existing directories")
    video_files = sorted(p for p in videos.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    exports = sorted(p for p in annotations.iterdir() if p.suffix.lower() == ".zip" or p.is_dir())
    export_by_stem = {}
    duplicates = []
    for item in exports:
        if item.stem in export_by_stem:
            duplicates.append(item.stem)
        export_by_stem[item.stem] = item
    video_stems = {v.stem for v in video_files}
    annotated = video_stems & set(export_by_stem)
    pending = sorted(video_stems - set(export_by_stem))
    orphan = sorted(set(export_by_stem) - video_stems)
    sequences, problems = [], []
    for stem in sorted(annotated):
        try:
            seq = read_annotations(export_by_stem[stem], next(v for v in video_files if v.stem == stem))
            sequences.append(seq)
        except Exception as exc:
            problems.append({"sequence": stem, "error": str(exc)})
    from collections import Counter
    from vtia.domain.splits import parse_clip
    by_class = Counter(t.road_class.value for seq in sequences for frame in seq.truths for t in frame)
    by_sequence = {seq.name: dict(Counter(t.road_class.value for frame in seq.truths for t in frame)) for seq in sequences}
    by_point, by_time_band = Counter(), Counter()
    for seq in sequences:
        try:
            clip = parse_clip(seq.name)
            by_point[clip.stratum.split("/")[0]] += sum(len(frame) for frame in seq.truths)
            by_time_band[clip.stratum.split("/")[-1]] += sum(len(frame) for frame in seq.truths)
        except ValueError:
            problems.append({"sequence": seq.name, "error": "filename does not follow capture naming protocol"})
    return {
        "videos_available": len(video_files), "videos_annotated": len(annotated),
        "videos_validated": len(sequences), "videos_pending": len(pending), "pending_videos": pending,
        "annotation_exports": len(exports), "orphan_annotations": orphan,
        "duplicate_sequence_names": sorted(set(duplicates)),
        "annotated_frames": sum(sum(bool(truths or ignored) for truths, ignored in zip(seq.truths, seq.ignore)) for seq in sequences),
        "objects_by_class": dict(by_class), "object_instances_by_class": dict(by_class), "objects_by_sequence": by_sequence,
        "objects_by_observation_point": dict(by_point), "objects_by_time_band": dict(by_time_band),
        "ignore_regions": sum(len(frame) for seq in sequences for frame in seq.ignore),
        "invalid_sequences": problems,
        "status": ("pending_data" if not video_files else
                   "issues_found" if problems or orphan or duplicates else
                   "pending_annotations" if pending else "valid"),
    }
