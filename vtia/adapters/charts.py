"""Static report charts (matplotlib -> PNG bytes) for the PDF.

Single-series magnitude charts: one blue hue from the reference ramp, the busiest
interval emphasised with a darker step of the same hue; text in neutral ink.
"""
import io

import matplotlib

matplotlib.use("Agg")  # no window: render to memory
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

BAR = "#3987e5"    # blue 400, >= 3:1 on white
PEAK = "#184f95"   # blue 600, emphasis
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def composition_png(rows: list[tuple[str, int, float]]) -> bytes:
    """Horizontal bars, largest on top: (label, count, share)."""
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.3 * len(rows) + 0.8), dpi=200)
    ax.barh([r[0] for r in rows], [r[1] for r in rows], height=0.55, color=BAR)  # thin bars, air between
    top = max(r[1] for r in rows)
    for y, (_, n, share) in enumerate(rows):
        ax.text(n + top * 0.01, y, f"{n}  ({share:.0%})", va="center", color=TEXT, fontsize=8)
    ax.set_xlim(0, top * 1.22)
    ax.set_xlabel("Cruces", color=TEXT_2, fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style(ax, grid_axis="x")
    return _png(fig)


def intervals_png(rows: list[tuple[str, int]], peak: str | None) -> bytes:
    """Columns per time interval: (label, count); the peak interval in the darker step."""
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=200)
    x = range(len(rows))
    width = 0.6 if len(rows) >= 8 else 0.6 * len(rows) / 8  # few intervals: keep columns slim
    ax.bar(x, [n for _, n in rows], width=width, color=[PEAK if label == peak else BAR for label, _ in rows])
    step = max(1, len(rows) // 12)  # at most ~12 tick labels
    ax.set_xticks(list(x)[::step], [label for label, _ in rows][::step], fontsize=7, rotation=0)
    if peak:
        i = next(i for i, (label, _) in enumerate(rows) if label == peak)
        ax.annotate(f"pico: {rows[i][1]}", (i, rows[i][1]), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7, color=TEXT)
    ax.set_ylabel("Cruces", color=TEXT_2, fontsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # counts are whole numbers
    ax.margins(y=0.15)
    _style(ax, grid_axis="y")
    return _png(fig)


def _style(ax, grid_axis: str) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=7, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
