"""Dashboard tab: run the four detectors on the chosen video and show what the data can support."""
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from vtia.adapters import labels_es as es
from vtia.domain.evaluation import BACKGROUND, MISSED, confusion_cells
from vtia.domain.model_selection import Verdict

BAR, BEST = "#3987e5", "#9ec5f4"          # one hue; the best stands out, as the dashboard's peak interval does
RAMP = ["#16263b", "#2a5f9e", "#9ec5f4"]  # sequential, dark surface: more objects, lighter
GPU_FPS = (65, 37, 23, 13)                # measured on the project's AMD GPU (Docs/Explicacion_Fase3a.md, AVANCE.md)


def level_message(has_annotations: bool, has_counts: bool) -> str:
    if has_annotations:
        return ("Con anotaciones verás la matriz de confusión de cada modelo y sus métricas de detección y seguimiento."
                + ("" if has_counts else " Si añades un conteo manual verás también quién cuenta mejor."))
    if has_counts:
        return ("Con el conteo manual se puede indicar el mejor contando. "
                "Para ver matrices de confusión faltan las anotaciones CVAT.")
    return ("Solo con el video no se puede declarar un modelo más preciso: verás velocidad, cruces y dónde discrepan. "
            "Añade un conteo manual o anotaciones CVAT para medir aciertos.")


def estimate_seconds(frames: int) -> float:
    return frames * sum(1 / fps for fps in GPU_FPS)


def _names(names: tuple[str, ...]) -> str:
    return names[0] if len(names) == 1 else "empate entre " + ", ".join(names[:-1]) + f" y {names[-1]}"


def verdict_lines(v: Verdict) -> list[str]:
    lines = []
    if v.best_by_annotations:
        lines.append(f"**Mejor detectando y clasificando en este video:** {_names(v.best_by_annotations)}.")
    if v.best_by_counts:
        lines.append(f"**Mejor contando en este video:** {_names(v.best_by_counts)}.")
    if v.fastest:
        lines.append(f"**Más rápido:** {v.fastest}.")
    return lines or ["No hay datos para declarar un mejor modelo."]


def _label(value) -> str:
    return es.BACKGROUND_LABEL if value == BACKGROUND else es.MISSED_LABEL if value == MISSED else es.CLASS[value]


def matrix_rows(counts: Counter) -> list[dict]:
    return [{"Real": _label(c.true), "Predicho": _label(c.predicted), "Objetos": c.count, "Proporción": c.share,
             "Texto": "s/d" if c.share is None else str(c.count)} for c in confusion_cells(counts)]


def _matrix_chart(counts: Counter) -> alt.LayerChart:
    df = pd.DataFrame(matrix_rows(counts))
    order_y = [es.CLASS[c] for c in es.CLASS] + [es.BACKGROUND_LABEL]
    order_x = [es.CLASS[c] for c in es.CLASS] + [es.MISSED_LABEL]
    base = alt.Chart(df).encode(
        x=alt.X("Predicho:N", sort=order_x, axis=alt.Axis(labelAngle=-40, labelOverlap=False, title="Predicho")),
        y=alt.Y("Real:N", sort=order_y, axis=alt.Axis(labelOverlap=False, title="Real")),
        tooltip=["Real", "Predicho", "Objetos", alt.Tooltip("Proporción:Q", format=".0%")])
    # A row without objects has no share: Vega drops its rectangles and only the "s/d" text remains.
    cells = base.mark_rect(stroke="#0e1117", strokeWidth=2).encode(
        color=alt.Color("Proporción:Q", scale=alt.Scale(domain=[0, 0.5, 1], range=RAMP),
                        legend=alt.Legend(format=".0%", title="% de la fila")))
    text = base.mark_text(fontSize=11).encode(text="Texto:N", color=alt.condition(
        "datum['Proporción'] > 0.6", alt.value("#0e1117"), alt.value("#f0efec")))
    return (cells + text).properties(height=340)


def _bars(rows: list[tuple[str, float]], title: str, best: tuple[str, ...], fmt: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["Modelo", title])
    df["Mejor"] = df["Modelo"].isin(best)
    st.altair_chart(alt.Chart(df).mark_bar(size=18, cornerRadiusEnd=4).encode(
        x=alt.X(f"{title}:Q", title=None), y=alt.Y("Modelo:N", sort=None, axis=alt.Axis(labelOverlap=False, title=None)),
        color=alt.condition("datum.Mejor", alt.value(BEST), alt.value(BAR)),
        tooltip=["Modelo", alt.Tooltip(f"{title}:Q", format=fmt)]).properties(title=title, height=230), width="stretch")


def _table(models) -> pd.DataFrame:
    def row(m):
        e, c = m.evaluation, m.counting
        data = {"Modelo": m.name, "Estado": f"Error: {m.error}" if m.error else "OK", "Cruces": m.crossings,
                "Objetos seguidos": m.tracks, "FPS inferencia": m.fps}
        if e:
            data |= {"F1 macro": e.classes["macro_f1"], "mAP50": e.detection["mAP50"],
                     "mAP50-95": e.detection["mAP50_95"], "HOTA": e.tracking["HOTA"], "MOTA": e.tracking["MOTA"],
                     "IDF1": e.tracking["IDF1"], "ID switches": e.tracking["IDSW"]}
        if c:
            data |= {"MAE conteo": c["overall"]["mae"], "RMSE conteo": c["overall"]["rmse"]}
        return data
    return pd.DataFrame([row(m) for m in models])


def show(run) -> None:
    c = run.comparison
    for line in verdict_lines(c.verdict):
        st.markdown(line)
    st.caption("Resultado de un solo video con umbrales sin calibrar: no permite generalizar a otras cámaras.")
    st.subheader("Tabla comparativa")
    st.dataframe(_table(c.models).round(3), hide_index=True, width="stretch")

    valid = [m for m in c.models if not m.error]
    st.subheader("Comparación por métrica")
    left, right = st.columns(2)
    with left:
        _bars([(m.name, m.fps) for m in valid if m.fps is not None], "FPS inferencia",
              (c.verdict.fastest,) if c.verdict.fastest else (), ".1f")
        _bars([(m.name, m.evaluation.classes["macro_f1"]) for m in valid if m.evaluation], "F1 macro",
              c.verdict.best_by_annotations, ".3f")
    with right:
        _bars([(m.name, m.crossings) for m in valid], "Cruces", (), "d")
        _bars([(m.name, m.counting["overall"]["mae"]) for m in valid if m.counting], "MAE de conteo",
              c.verdict.best_by_counts, ".2f")

    if c.disagreement.by_class:
        st.subheader("Discrepancia entre modelos")
        st.caption("Muestra desacuerdo entre modelos, no error: sin referencia humana no se sabe cuál acierta.")
        st.dataframe(pd.DataFrame([(es.CLASS[k], lo, hi, hi - lo) for k, (lo, hi) in c.disagreement.by_class.items()],
                                  columns=["Clase", "Mínimo", "Máximo", "Diferencia"]), hide_index=True, width="stretch")

    evaluated = [m for m in valid if m.evaluation]
    if evaluated:
        st.subheader("Matrices de confusión")
        st.caption("Color: porcentaje de cada clase real. Número: objetos. Confianza ≥ 0,25, IoU ≥ 0,5. s/d: sin datos.")
        columns = st.columns(2)
        for i, m in enumerate(evaluated):
            with columns[i % 2]:
                st.markdown(f"**{m.name}**")
                st.altair_chart(_matrix_chart(m.evaluation.confusion), width="stretch")

    st.subheader("Descargas")
    for f in sorted(p for p in run.out.rglob("*.csv") if "detections" not in p.parts):
        name = f.relative_to(run.out).as_posix()
        st.download_button(name, f.read_bytes(), file_name=name.replace("/", "_"), key=f"cmp-dl-{name}")


def render(get_video: Callable[[], Path], video_name: str, point: Path | None, frames: int, root: Path) -> None:
    from vtia.compose import compare_file, device_name

    key = (video_name, str(point))
    st.subheader("Comparar los 4 modelos en este video")
    annotations = st.file_uploader("Anotaciones CVAT (zip MOT 1.1, opcional)", type=["zip"], key="cmp-cvat")
    counts = st.file_uploader("Conteo manual (CSV con class y manual, opcional)", type=["csv"], key="cmp-csv")
    st.info(level_message(annotations is not None, counts is not None))
    if frames:
        note = "" if device_name() != "CPU" else " Este equipo está usando CPU, así que tardará bastante más."
        st.caption(f"Tiempo estimado de detección con GPU: {es.duration(estimate_seconds(frames))}.{note}")

    if st.button("Comparar", type="primary", disabled=not video_name or point is None, key="cmp-run"):
        bar = st.progress(0.0, text="Preparando...")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                archive = None
                if annotations:
                    archive = Path(tmp) / "annotations.zip"
                    archive.write_bytes(annotations.getbuffer())
                st.session_state.comparison = compare_file(
                    get_video(), point, annotations=archive,
                    manual_csv=counts.getvalue().decode("utf-8-sig") if counts else None,
                    out=root / "outputs" / "dashboard", splits=root / "data" / "splits.csv",
                    on_model=lambda i, n, name: bar.progress((i - 1) / n, text=f"Modelo {i} de {n}: {name}"))
        except Exception as e:  # bad files or a blocked partition: explain it in the page
            bar.empty()
            st.session_state.pop("comparison", None)
            st.error(f"No se pudo comparar: {e}")
            return
        st.session_state.comparison_key = key
        bar.empty()

    run = st.session_state.get("comparison")
    if run and st.session_state.get("comparison_key") == key:  # never show another video's results
        show(run)
