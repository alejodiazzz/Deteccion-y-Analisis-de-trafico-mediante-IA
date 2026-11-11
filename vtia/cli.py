"""Command-line entry point.

    python -m vtia.cli count VIDEO --point configs/points/<point>.yaml [--save-video]
    python -m vtia.cli anonymize data/raw --out data/anon
    python -m vtia.cli prelabel data/anon/P1_20261015_0705_s0312.mp4 --out data/prelabels
    python -m vtia.cli split data/anon --out data/splits.csv
    python -m vtia.cli evaluate --visdrone data/visdrone/VisDrone2019-MOT-val
    python -m vtia.cli evaluate --cvat data/annotations --videos data/anon --splits data/splits.csv --partition test
    python -m vtia.cli calibrate --cvat data/annotations --videos data/anon --splits data/splits.csv --partition fold1 --model yolo26n.pt
    python -m vtia.cli train --visdrone data/visdrone/VisDrone2019-MOT-val --val <seq> <seq>
"""
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from vtia.adapters.cached_detector import CachedDetector
from vtia.adapters.table_files import markdown_table, write_csv
from vtia.adapters.video_io import VIDEO_SUFFIXES
from vtia.adapters.visdrone import load_visdrone
from vtia.application.evaluate import EvalResult, evaluate
MODELS = ["yolo26n.pt", "yolo26s.pt", "rtdetr-l.pt", "faster_rcnn"]
PRELABEL_MODEL = "yolo26x.pt"
ALGORITHMS = ("bytetrack", "botsort")
from vtia.domain.classes import RoadClass
from vtia.domain.evaluation import BACKGROUND, MISSED
from vtia.domain.splits import split, validate_split_assignment
from vtia.domain.dataset_validation import read_split_manifest, select_training_sequences
from vtia.domain.counting_evaluation import evaluate_count_rows

DETECTORS = f"{', '.join(MODELS)} (or any Ultralytics weights, e.g. a fine-tuned best.pt)"


# --- count ---------------------------------------------------------------------------------

def count(args) -> None:
    from vtia.compose import analyze_file, device_name
    if not args.video.is_file():
        raise SystemExit(f"video not found: {args.video}")
    run = analyze_file(args.video, args.point, model=args.model, tracker=args.tracker, track_thresh=args.track_thresh,
                       conf=args.conf, dedup=args.dedup, riders=args.riders, start=args.start, out=args.out,
                       save_video=args.save_video)
    a, s = run.analysis, run.summary
    for line, direction, road_class, n in s.by_line:
        print(f"{line:<14}{direction:<20}{road_class:<12}{n}")
    print(f"{len(a.events)} crossings, {len(a.visits)} zone visits, {len(a.alerts)} alerts | "
          f"{a.frames} frames in {run.seconds:.1f} s ({a.frames / run.seconds:.1f} fps, {device_name()})")
    print(f"start time: {a.start:%Y-%m-%d %H:%M}" if a.start else
          "start time: unknown (name the file P<point>_<YYYYMMDD>_<HHMM> or pass --start)")
    print(f"report: {run.out / 'report.pdf'}" + (f" | video: {run.out / 'annotated.mp4'}" if args.save_video else ""))


# --- data preparation (phase 2) ------------------------------------------------------------

def _videos(inputs: list[Path]) -> list[Path]:
    found = []
    for p in inputs:
        found += sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_SUFFIXES) if p.is_dir() else [p]
    if not found:
        raise SystemExit("no videos found")
    return found


def _progress(name: str):
    def show(i: int, total: int) -> None:
        if i == total or i % 50 == 0:
            print(f"\r{name}: frame {i}/{total}", end="" if i < total else "\n", flush=True)
    return show


def anonymize(args) -> None:
    from vtia.compose import anonymize_file
    for video in _videos(args.inputs):
        out, sheet = anonymize_file(video, args.out, model=args.model, conf=args.conf, hold_frames=args.hold,
                                    progress=_progress(video.name))
        print(f"  -> {out}\n  check by eye: {sheet}")
    print("Review every *_check.jpg (and play the videos) before deleting the raw files: "
          "only anonymised copies may leave this PC (capture protocol §8).")


def prelabel(args) -> None:
    from vtia.compose import prelabel_file
    for video in _videos(args.inputs):
        out, tracks = prelabel_file(video, args.out, model=args.model, keyframe_every=args.keyframe_every,
                                    progress=_progress(video.name))
        print(f"  -> {out} ({tracks} tracks). In CVAT: task > Actions > Upload annotations > MOT 1.1")


def split_cmd(args) -> None:
    if args.folds != 5:
        raise SystemExit("VTIA protocol fixes five validation folds; use --folds 5")
    names = [v.stem for v in _videos(args.inputs)]
    assignment = split(names, args.test_share, args.folds, args.seed)
    problems = validate_split_assignment(names, assignment, args.folds)
    if problems:
        raise SystemExit("invalid split generated:\n- " + "\n- ".join(problems))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, ["clip", "split"], sorted(assignment.items()))
    manifest = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": args.seed, "test_share": args.test_share, "folds": args.folds,
                "assignment_csv": args.out.name, "assignment_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                "clips": len(assignment), "sessions": len({__import__('vtia.domain.splits', fromlist=['parse_clip']).parse_clip(n).session for n in names})}
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for part in sorted(set(assignment.values())):
        print(f"{part:<6} {sum(1 for v in assignment.values() if v == part)} clips")
    print(f"-> {args.out}")


# --- evaluation (phases 3 and 5) -----------------------------------------------------------

def _sequences(args):
    if args.visdrone and args.cvat:
        raise SystemExit("choose one evaluation source: --visdrone or --cvat, not both")
    if args.visdrone:
        sequences = load_visdrone(args.visdrone)
    elif args.cvat and args.videos:
        from vtia.adapters.cvat_mot import load_cvat_dir
        sequences = load_cvat_dir(args.cvat, args.videos)
    else:
        raise SystemExit("give --visdrone ROOT, or --cvat ANNOTATIONS_DIR together with --videos VIDEOS_DIR")
    if args.sequences:
        sequences = [s for s in sequences if s.name in args.sequences]
        missing = set(args.sequences) - {s.name for s in sequences}
        if missing:
            raise SystemExit(f"unknown sequences: {sorted(missing)}")
    if args.cvat and getattr(args, "command_name", "") != "train":
        if not getattr(args, "splits", None) or not getattr(args, "partition", None):
            raise SystemExit("evaluating own CVAT data requires --splits MANIFEST.csv and --partition test (calibration must use a fold)")
        assignment = read_split_manifest(args.splits)
        problems = validate_split_assignment(list(assignment), assignment, folds=5)
        if problems:
            raise SystemExit("invalid split manifest:\n- " + "\n- ".join(problems))
        missing = sorted(s.name for s in sequences if s.name not in assignment)
        if missing:
            raise SystemExit(f"sequences missing from split manifest: {missing}")
        if args.partition == "test" and getattr(args, "command_name", "") == "calibrate":
            raise SystemExit("the fixed test partition cannot be used for calibration")
        if args.sequences:
            wrong = sorted(name for name in args.sequences if assignment.get(name) != args.partition)
            if wrong:
                raise SystemExit(f"explicitly requested sequences do not belong to {args.partition}: {wrong}")
        sequences = [seq for seq in sequences if assignment[seq.name] == args.partition]
        if not sequences:
            raise SystemExit(f"no annotated sequences available in partition {args.partition}")
    if not sequences:
        raise SystemExit("no annotated sequences were loaded")
    return sequences


def _source_name(args) -> str:
    return args.visdrone.name if args.visdrone else Path(args.cvat).name


SUMMARY_HEADER = ["model", "tracker", "track_thresh", "mAP50", "mAP50_95", "AP_small", "macro_F1", "balanced_acc",
                  "micro_recall", "HOTA", "MOTA", "MOTP", "IDF1", "IDSW", "IDSW_per_100",
                  "parameters", "model_size_mb", "fps_inference", "fps_processing", "inference_seconds", "processing_seconds"]


def _model_resources(detector):
    """Best-effort parameter count and checkpoint size; unavailable values remain N/D."""
    model = getattr(detector, "model", None)
    module = getattr(model, "model", model)
    parameters = None
    if module is not None and hasattr(module, "parameters"):
        try:
            parameters = sum(p.numel() for p in module.parameters())
        except Exception:
            pass
    weights = getattr(detector, "weights_path", None)
    if weights and not Path(str(weights)).expanduser().is_file():
        weights = getattr(model, "ckpt_path", None) or weights
    weights = weights or getattr(model, "ckpt_path", None)
    size_mb = None
    if weights:
        path = Path(str(weights)).expanduser()
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
    return parameters, size_mb


def evaluate_cmd(args) -> None:
    from vtia.adapters.coco_map import coco_map
    from vtia.adapters.trackeval_metrics import mot_metrics
    from vtia.compose import device_name, make_detector, make_tracker, with_postprocess
    sequences = _sequences(args)
    out = args.out or Path("outputs/eval") / _source_name(args)
    print(f"{len(sequences)} sequences, {sum(len(s.truths) for s in sequences)} frames | device: {device_name()}")
    summary, sections = [], []
    per_model_sequences = {}
    model_results = {}
    for name in args.models:
        base_detector = make_detector(name, conf=0.01)
        parameters, model_size_mb = _model_resources(base_detector)
        cached = CachedDetector(base_detector, out / "detections")
        detector = with_postprocess(cached, args.dedup, args.riders)
        tracker = make_tracker(args.tracker, name, args.track_thresh)
        processing_start = time.perf_counter()
        r = evaluate(sequences, detector, tracker, coco_map, mot_metrics, conf=args.conf)
        processing_seconds = time.perf_counter() - processing_start
        fps_inference = cached.frames / cached.seconds if cached.seconds else None
        total_frames = sum(len(seq.truths) for seq in sequences)
        fps_processing = total_frames / processing_seconds if processing_seconds else None
        d, t, c = r.detection, r.tracking, r.classes
        summary.append([detector.name, tracker.algorithm, tracker.track_thresh, d["mAP50"], d["mAP50_95"],
                        d["AP_small"], c["macro_f1"], c["balanced_accuracy"], c["micro_recall"], t["HOTA"], t["MOTA"],
                        t["MOTP"], t["IDF1"], t["IDSW"], t["IDSW_per_100_objects"],
                        parameters if parameters is not None else "N/D", model_size_mb if model_size_mb is not None else "N/D",
                        fps_inference if fps_inference is not None else "N/D",
                        fps_processing if fps_processing is not None else "N/D", cached.seconds, processing_seconds])
        sections.append(_write_model_tables(out / detector.name, detector.name, r))
        per_model_sequences[detector.name] = r.per_sequence
        model_results[detector.name] = r
        seq_header = ["sequence", "frames", "ground_truth_instances", "mAP50", "mAP50_95", "macro_F1", "HOTA", "MOTA", "IDF1"]
        seq_rows = [[seq, m["frames"], m["ground_truth_instances"], m["detection"].get("mAP50"),
                     m["detection"].get("mAP50_95"), m["classes"].get("macro_f1"),
                     m["tracking"].get("HOTA"), m["tracking"].get("MOTA"), m["tracking"].get("IDF1")]
                    for seq, m in r.per_sequence.items()]
        write_csv(out / detector.name / "per_sequence.csv", seq_header, seq_rows)
        print(f"done: {detector.name}")

    suffix = ((f"_dedup{args.dedup}" if args.dedup else "") + (f"_riders{args.riders}" if args.riders else "")
              + (f"_{args.tracker}" if args.tracker != "bytetrack" else "")
              + (f"_tt{args.track_thresh}" if args.track_thresh is not None else "")
              + (f"_{'-'.join(args.sequences)}" if args.sequences else ""))
    write_csv(out / f"summary{suffix}.csv", SUMMARY_HEADER, summary)
    report = (f"# Evaluation on {_source_name(args)}\n\nDevice: {device_name()} | confusion threshold: "
              f"conf >= {args.conf}, IoU >= 0.5 | dedup: {args.dedup or 'off'} | riders: {args.riders or 'off'}"
              + (f" | sequences: {', '.join(args.sequences)}" if args.sequences else "") + "\n\n"
              + markdown_table(SUMMARY_HEADER, summary) + "\n" + "\n".join(sections))
    (out / f"summary{suffix}.md").write_text(report, encoding="utf-8")
    robustness = {}
    if getattr(args, "conditions", None):
        from vtia.domain.robustness import summarize_robustness
        with args.conditions.open(newline="", encoding="utf-8-sig") as f:
            condition_rows = list(csv.DictReader(f))
        robustness = {model: summarize_robustness(per_seq, condition_rows) for model, per_seq in per_model_sequences.items()}
        (out / f"robustness{suffix}.json").write_text(json.dumps(robustness, indent=2, ensure_ascii=False), encoding="utf-8")
    payload = {"dataset": _source_name(args), "dataset_type": "VisDrone2019-MOT" if args.visdrone else "own_cvat_mot",
               "partition": getattr(args, "partition", None), "sequences": [s.name for s in sequences],
               "per_sequence": per_model_sequences, "robustness": robustness,
               "device": device_name(), "confusion_threshold": args.conf, "tracker": args.tracker,
               "models": [{"model": row[0], "tracker": row[1], "metrics": dict(zip(SUMMARY_HEADER[2:], row[2:]))}
                          for row in summary]}
    (out / f"summary{suffix}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    from vtia.adapters.scientific_report import write_scientific_pdf
    class_tables = []
    for model_name, result in model_results.items():
        class_headers = ["Clase", "Precision", "Recall", "F1", "Soporte", "AP50", "AP50-95"]
        class_rows = [[str(klass), f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}",
                       f"{metrics['f1']:.4f}", str(metrics["support"]),
                       f"{result.detection['per_class'][klass]['AP50']:.4f}",
                       f"{result.detection['per_class'][klass]['AP50_95']:.4f}"]
                      for klass, metrics in result.classes["per_class"].items()]
        class_tables.append((f"Resultados por clase — {model_name}", class_headers, class_rows))
    write_scientific_pdf(out / f"summary{suffix}.pdf", "Informe de evaluación científica VTIA",
                         {"Objetivo": "Evaluar detección y seguimiento contra anotaciones de referencia.",
                          "Conjunto": payload["dataset"], "Tipo de datos": payload["dataset_type"],
                          "Partición": payload["partition"] or "No aplica", "Secuencias": len(sequences),
                          "Cuadros anotados": sum(len(s.truths) for s in sequences), "Dispositivo": device_name(),
                          "Modelos": ", ".join(args.models), "Tracker": args.tracker,
                          "Umbral de matriz de confusión": args.conf, "IoU de emparejamiento": 0.5},
                         [[str(row[i]) for i in (0, 3, 4, 6, 9, 10, 12, 13, 15, 16, 17, 18)] for row in summary],
                         ["Modelo", "mAP50", "mAP50-95", "F1 macro", "HOTA", "MOTA", "IDF1", "IDSW",
                          "Parámetros", "Tamaño MB", "FPS inferencia", "FPS procesamiento"],
                         ["Metodología: las métricas de detección se calculan con COCO mAP; la matriz de confusión usa el umbral indicado e IoU >= 0.5.",
                          "Seguimiento: HOTA, MOTA, MOTP, IDF1 e ID switches se calculan con TrackEval.",
                          "Rendimiento: FPS de inferencia del detector y FPS de procesamiento integral se informan por separado.",
                          "Los resultados corresponden exclusivamente al conjunto y partición indicados en este informe.",
                          "Los resultados de VisDrone no representan el desempeño sobre cámaras propias.",
                          "El error real de conteo requiere registros manuales independientes; no se infiere de las métricas de detección.",
                          "Limitación: las condiciones de robustez dependen de metadatos manuales; la calidad semántica de las anotaciones requiere revisión humana.",
                          "Conclusión: este informe presenta métricas observadas; no declara un ganador universal ni sustituye el análisis de contexto."],
                         extra_tables=class_tables)
    print(report)


def _write_model_tables(out: Path, name: str, r: EvalResult) -> str:
    out.mkdir(parents=True, exist_ok=True)
    classes = list(RoadClass)
    per_class_header = ["class", "precision", "recall", "f1", "support", "AP50", "AP50_95"]
    per_class = [[c, m["precision"], m["recall"], m["f1"], m["support"],
                  r.detection["per_class"][c]["AP50"], r.detection["per_class"][c]["AP50_95"]]
                 for c, m in r.classes["per_class"].items()]
    confusion_header = ["true \\ predicted", *classes, MISSED]
    confusion = [[row, *(r.confusion[(row, col)] for col in [*classes, MISSED])] for row in [*classes, BACKGROUND]]
    write_csv(out / "per_class.csv", per_class_header, per_class)
    write_csv(out / "confusion.csv", confusion_header, confusion)
    return (f"## {name}\n\n" + markdown_table(per_class_header, per_class) + "\n"
            + markdown_table(confusion_header, confusion))


def calibrate(args) -> None:
    """Tracking metrics for a grid of tracker thresholds, reusing cached detections (no mAP: it doesn't change)."""
    from vtia.adapters.trackeval_metrics import mot_metrics
    from vtia.compose import make_detector, make_tracker, with_postprocess
    args.command_name = "calibrate"
    sequences = _sequences(args)
    out = args.out or Path("outputs/eval") / _source_name(args)
    cached = CachedDetector(make_detector(args.model, conf=0.01), out / "detections")
    detector = with_postprocess(cached, args.dedup, args.riders)
    header = ["track_thresh", "HOTA", "MOTA", "IDF1", "IDSW", "macro_F1"]
    rows = []
    for t in args.grid:
        r = evaluate(sequences, detector, make_tracker(args.tracker, args.model, t), lambda *_: {}, mot_metrics,
                     conf=args.conf)
        rows.append([t, r.tracking["HOTA"], r.tracking["MOTA"], r.tracking["IDF1"], r.tracking["IDSW"], r.classes["macro_f1"]])
        print(f"track_thresh {t}: HOTA {rows[-1][1]:.3f}  MOTA {rows[-1][2]:.3f}  IDF1 {rows[-1][3]:.3f}  IDSW {rows[-1][4]}")
    best = max(rows, key=lambda r: r[header.index(args.metric)])
    name = f"calibration_{detector.name}_{args.tracker}"
    write_csv(out / f"{name}.csv", header, rows)
    (out / f"{name}.md").write_text(
        f"# Tracker threshold calibration: {detector.name} + {args.tracker} on {_source_name(args)}\n\n"
        + markdown_table(header, rows) + f"\nBest by {args.metric}: **{best[0]}**\n", encoding="utf-8")
    print(f"best by {args.metric}: track_thresh {best[0]} -> {out / name}.md")


# --- dataset and count validation (phase 3b) -----------------------------------------------

def validate_dataset_cmd(args) -> None:
    from vtia.adapters.cvat_mot import validate_cvat_dataset
    result = validate_cvat_dataset(args.videos, args.cvat)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nquality report: {args.out}")
    if result["invalid_sequences"] or result["orphan_annotations"] or result["duplicate_sequence_names"]:
        raise SystemExit("dataset validation found structural problems; inspect the JSON report")


def evaluate_counting_cmd(args) -> None:
    with args.csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    result = evaluate_count_rows(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "counting_evaluation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["video", "point", "date", "time_band", "class", "direction", "line", "manual", "automatic", "difference", "absolute_difference", "error_percent"]
    with (args.out / "counting_evaluation.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["records"])
    summary = result["overall"]
    class_rows = result.get("breakdown", {}).get("class", {})
    class_table = "| Clase | Conteo real | Conteo IA | Diferencia | Error % | MAE | RMSE |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for klass, metrics in sorted(class_rows.items()):
        pct = "N/D" if metrics["aggregate_error_percent"] is None else f"{metrics['aggregate_error_percent']:.2f}%"
        class_table += (f"| {klass} | {metrics['manual_total']} | {metrics['automatic_total']} | "
                        f"{metrics['difference_total']} | {pct} | {metrics['mae']:.3f} | {metrics['rmse']:.3f} |\n")
    report = ("# Evaluación del error de conteo\n\n" +
              (f"Registros: {summary['n']} | MAE: {summary['mae']:.4f} | RMSE: {summary['rmse']:.4f}\n\n" if summary["n"] else
               "Estado: pendiente de datos. No se recibieron registros manuales/automáticos.\n\n") +
              class_table + "\nEl error porcentual es N/D cuando el conteo manual es cero.\n")
    (args.out / "counting_evaluation.md").write_text(report, encoding="utf-8")
    from vtia.adapters.scientific_report import write_scientific_pdf
    pdf_rows = [[klass, str(m["manual_total"]), str(m["automatic_total"]), str(m["difference_total"]),
                 "N/D" if m["aggregate_error_percent"] is None else f"{m['aggregate_error_percent']:.2f}%",
                 f"{m['mae']:.3f}", f"{m['rmse']:.3f}"] for klass, m in sorted(class_rows.items())]
    write_scientific_pdf(args.out / "counting_evaluation.pdf", "VTIA — Validación del error de conteo",
                         {"Registros comparados": summary["n"], "MAE global": summary["mae"] if summary["n"] else "N/D",
                          "RMSE global": summary["rmse"] if summary["n"] else "N/D",
                          "Regla de error porcentual": "N/D cuando el conteo manual es cero"},
                         pdf_rows, ["Clase", "Conteo real", "Conteo IA", "Diferencia", "Error %", "MAE", "RMSE"],
                         ["Los conteos reales deben proceder de anotación/revisión manual.",
                          "Cada fila debe representar una unidad de comparación homogénea."])
    print(report)
    print(f"outputs: {args.out}")


# --- fine-tuning (phase 4) -----------------------------------------------------------------

def train_cmd(args) -> None:
    from vtia.compose import device_name
    args.command_name = "train"
    from vtia.adapters.yolo_training import export_dataset, train  # heavy imports only for this command

    sequences = _sequences(args)
    if args.splits:
        if args.val:
            raise SystemExit("use either --splits or --val, not both; own CVAT experiments must use the split manifest")
        part = read_split_manifest(args.splits)
        missing_from_manifest = sorted(s.name for s in sequences if s.name not in part)
        if missing_from_manifest:
            raise SystemExit(f"sequences missing from split manifest: {missing_from_manifest}")
        problems = validate_split_assignment(list(part), part, folds=5)
        if problems:
            raise SystemExit("invalid split manifest:\n- " + "\n- ".join(problems))
        try:
            train_seqs, val_seqs = select_training_sequences(sequences, part, args.fold)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc  # the fixed "test" split is never selected
    else:
        if args.cvat:
            raise SystemExit("training on own CVAT data requires --splits; --val alone cannot guarantee test isolation")
        val_seqs = [s for s in sequences if s.name in set(args.val or [])]
        train_seqs = [s for s in sequences if s not in val_seqs]
    if not train_seqs or not val_seqs:
        raise SystemExit("need at least one training and one validation sequence (--val NAMES or --splits + --fold)")
    print(f"train: {[s.name for s in train_seqs]}\nval:   {[s.name for s in val_seqs]}")
    data = export_dataset(train_seqs, val_seqs, Path("outputs/train") / args.name / "dataset", every=args.every)
    best = train(data, args.weights, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, name=args.name)
    print(f"best weights: {best}\nEvaluate them like any model: evaluate ... --models {best} --sequences <val>")


# --- arguments -----------------------------------------------------------------------------

def parse_start(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M")


def _postprocess_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dedup", type=float, metavar="IOU",
                   help="drop boxes overlapping a more confident one by more than this IoU, any class (e.g. 0.7)")
    p.add_argument("--riders", type=float, metavar="OVERLAP",
                   help="experiment E2: merge a person into the bicycle/motorcycle they ride (e.g. 0.4)")


def _tracker_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tracker", choices=ALGORITHMS, default="bytetrack", help="(default: %(default)s)")


def _source_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--visdrone", type=Path, help="VisDrone2019-MOT folder with sequences/ and annotations/")
    p.add_argument("--cvat", type=Path, help="folder of CVAT MOT 1.1 exports, one <clip>.zip per clip")
    p.add_argument("--videos", type=Path, help="folder with the (anonymised) videos the CVAT exports annotate")
    p.add_argument("--sequences", nargs="+", metavar="NAME", help="use only these sequences / clips")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="vtia")
    sub = p.add_subparsers(required=True)

    c = sub.add_parser("count", help="counts, zone visits, alerts and report for one video")
    c.add_argument("video", type=Path)
    c.add_argument("--point", "--lines", dest="point", type=Path, required=True,
                   help="camera-point YAML: lines, zones, restrictions (see configs/points/)")
    c.add_argument("--start", type=parse_start, metavar='"YYYY-MM-DD HH:MM"',
                   help="wall-clock start of the video, if its name does not follow the capture protocol")
    c.add_argument("--model", default="yolo26n.pt", help=f"{DETECTORS} (default: %(default)s)")
    c.add_argument("--conf", type=float, default=0.1, help="minimum detector confidence (default: %(default)s)")
    c.add_argument("--out", type=Path, default=Path("outputs"), help="output folder (default: %(default)s)")
    c.add_argument("--save-video", action="store_true", help="also write annotated.mp4 with tracks, zones, counts, alerts")
    c.add_argument("--track-thresh", type=float, metavar="T",
                   help="tracker high-confidence threshold; new tracks need T + 0.1 (default: 0.3 for yolo26n, else 0.6)")
    _tracker_options(c)
    _postprocess_options(c)
    c.set_defaults(func=count)

    a = sub.add_parser("anonymize", help="blur heads and plates; writes copies + a contact sheet to check by eye")
    a.add_argument("inputs", type=Path, nargs="+", help="videos or folders of videos")
    a.add_argument("--out", type=Path, required=True, help="folder for the anonymised copies")
    a.add_argument("--model", default="yolo26s.pt", help="(default: %(default)s)")
    a.add_argument("--conf", type=float, default=0.15, help="low on purpose: blurring a false positive is free (default: %(default)s)")
    a.add_argument("--hold", type=int, default=3, help="keep regions blurred this many extra frames (default: %(default)s)")
    a.set_defaults(func=anonymize)

    pl = sub.add_parser("prelabel", help="CVAT pre-annotations (MOT 1.1 zip) so annotators only correct")
    pl.add_argument("inputs", type=Path, nargs="+", help="anonymised videos or folders")
    pl.add_argument("--out", type=Path, required=True)
    pl.add_argument("--model", default=PRELABEL_MODEL, help="use a model outside the comparison (default: %(default)s)")
    pl.add_argument("--keyframe-every", type=int, default=5, help="(default: %(default)s)")
    pl.set_defaults(func=prelabel)

    sp = sub.add_parser("split", help="test set + k folds, whole capture sessions together")
    sp.add_argument("inputs", type=Path, nargs="+", help="clips or folders, named P<point>_<YYYYMMDD>_<HHMM>[_s<SSSS>]")
    sp.add_argument("--out", type=Path, required=True, help="CSV clip,split")
    sp.add_argument("--test-share", type=float, default=0.3, help="(default: %(default)s)")
    sp.add_argument("--folds", type=int, default=5, help="(default: %(default)s)")
    sp.add_argument("--seed", type=int, default=0, help="(default: %(default)s)")
    sp.set_defaults(func=split_cmd)

    vd = sub.add_parser("validate-dataset", help="validate video/CVAT MOT dataset integrity without inference")
    vd.add_argument("--videos", type=Path, required=True)
    vd.add_argument("--cvat", type=Path, required=True, help="directory with CVAT MOT 1.1 exports")
    vd.add_argument("--out", type=Path, default=Path("outputs/dataset_quality.json"))
    vd.set_defaults(func=validate_dataset_cmd)

    ec = sub.add_parser("evaluate-counting", help="compare manual and automatic crossing-count records")
    ec.add_argument("--csv", type=Path, required=True, help="CSV with manual,automatic,class and optional grouping columns")
    ec.add_argument("--out", type=Path, default=Path("outputs/counting_eval"))
    ec.set_defaults(func=evaluate_counting_cmd)

    e = sub.add_parser("evaluate", help="score detectors + tracker on VisDrone or on CVAT annotations")
    _source_options(e)
    e.add_argument("--models", nargs="+", default=MODELS, help=DETECTORS)
    e.add_argument("--conf", type=float, default=0.25, help="operating threshold for the confusion matrix (default: %(default)s)")
    e.add_argument("--out", type=Path, help="(default: outputs/eval/<dataset>)")
    e.add_argument("--splits", type=Path, help="required for own CVAT data: split manifest from `split`")
    e.add_argument("--partition", choices=["test", "fold1", "fold2", "fold3", "fold4", "fold5"],
                   help="CVAT partition to evaluate; use test only for final evaluation")
    e.add_argument("--conditions", type=Path, help="CSV: sequence,lighting,occlusion,vehicle_density,pedestrian_density,object_size,distance")
    e.add_argument("--track-thresh", type=float, metavar="T", help="(default: 0.3 for yolo26n, else 0.6)")
    _tracker_options(e)
    _postprocess_options(e)
    e.set_defaults(func=evaluate_cmd)

    k = sub.add_parser("calibrate", help="pick the tracker threshold of one model on annotated data")
    _source_options(k)
    k.add_argument("--model", required=True, help=DETECTORS)
    k.add_argument("--grid", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7], help="(default: %(default)s)")
    k.add_argument("--metric", choices=["HOTA", "MOTA", "IDF1"], default="HOTA", help="(default: %(default)s)")
    k.add_argument("--conf", type=float, default=0.25)
    k.add_argument("--out", type=Path, help="(default: outputs/eval/<dataset>)")
    k.add_argument("--splits", type=Path, help="required for own CVAT data: split manifest from `split`")
    k.add_argument("--partition", choices=["test", "fold1", "fold2", "fold3", "fold4", "fold5"],
                   help="CVAT validation fold; test is prohibited for calibration")
    _tracker_options(k)
    _postprocess_options(k)
    k.set_defaults(func=calibrate)

    t = sub.add_parser("train", help="fine-tune a YOLO model (phase 4)")
    _source_options(t)
    t.add_argument("--val", nargs="+", metavar="NAME", help="validation sequences (the rest train)")
    t.add_argument("--splits", type=Path, help="CSV from `split`: trains on the folds except --fold, never on test")
    t.add_argument("--fold", type=int, default=1, help="validation fold when using --splits (default: %(default)s)")
    t.add_argument("--weights", default="yolo26n.pt", help="starting weights (default: %(default)s)")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--every", type=int, default=5, help="use 1 of every N frames (default: %(default)s)")
    t.add_argument("--name", default="run", help="run name under outputs/train/ (default: %(default)s)")
    t.set_defaults(func=train_cmd)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
