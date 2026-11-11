"""Files a model comparison leaves on disk."""
import shutil
from pathlib import Path

from vtia.adapters.table_files import markdown_table, write_csv
from vtia.application.compare_models import Comparison, ModelResult
from vtia.application.evaluate import EvalResult
from vtia.domain.classes import RoadClass
from vtia.domain.evaluation import BACKGROUND, MISSED

CACHE_DIR = "detections"
SUMMARY_HEADER = ["model", "error", "crossings", "tracks", "fps_inference", "macro_F1", "mAP50", "mAP50_95",
                  "HOTA", "MOTA", "IDF1", "IDSW", "count_MAE", "count_RMSE"]


def write_model_tables(out: Path, name: str, r: EvalResult) -> str:
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


def clear_results(out: Path) -> None:
    """Removes an earlier comparison so runs never mix; the detection cache stays."""
    if not out.is_dir():
        return
    for item in out.iterdir():
        if item.name == CACHE_DIR:
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()


def write_comparison(out: Path, comparison: Comparison) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "summary.csv", SUMMARY_HEADER, [_summary_row(m) for m in comparison.models])
    for m in comparison.models:
        if m.evaluation:
            write_model_tables(out / m.name, m.name, m.evaluation)


def _summary_row(m: ModelResult) -> list:
    e, c = m.evaluation, m.counting
    return [m.name, m.error or "", m.crossings, m.tracks, m.fps if m.fps is not None else "",
            *([e.classes["macro_f1"], e.detection["mAP50"], e.detection["mAP50_95"], e.tracking["HOTA"],
               e.tracking["MOTA"], e.tracking["IDF1"], e.tracking["IDSW"]] if e else [""] * 7),
            *([c["overall"]["mae"], c["overall"]["rmse"]] if c else ["", ""])]
