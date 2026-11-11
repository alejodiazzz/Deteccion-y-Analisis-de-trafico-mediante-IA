"""Writes tables as CSV (UTF-8, opens in Excel and pandas) and as Markdown."""
import csv
from collections.abc import Iterable, Sequence
from pathlib import Path


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def markdown_table(header: Sequence[str], rows: Iterable[Sequence]) -> str:
    def cell(v) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"
