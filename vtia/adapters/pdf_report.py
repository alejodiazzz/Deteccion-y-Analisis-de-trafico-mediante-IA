"""The human-readable mobility report (PDF, ReportLab)."""
import io
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from vtia.adapters import labels_es as es
from vtia.adapters.charts import composition_png, intervals_png
from vtia.application.process_video import VideoAnalysis
from vtia.application.summary import Summary
from vtia.domain.timeline import clock

MAX_ALERT_ROWS = 40
GRID = colors.HexColor("#e4e3df")
HEADER_BG = colors.HexColor("#f0efec")


def write_pdf(path: Path, s: Summary, a: VideoAnalysis, meta: dict[str, str]) -> None:
    """`meta`: ordered (label -> value) facts about the run, shown in the header table."""
    st = getSampleStyleSheet()
    body, h2, small = st["BodyText"], st["Heading2"], st["Italic"]
    story = [Paragraph("Reporte de movilidad", st["Title"]), _table(list(meta.items()), header=False), Spacer(1, 10)]

    def section(title, *content):
        story.append(KeepTogether([Paragraph(title, h2), *content]))

    section("Resumen", _table([
        ["Cruces de línea", "Objetos seguidos", "Alertas", "Duración"],
        [s.total, s.tracks, len(a.alerts), es.duration(s.duration_s)],
    ]))
    section("Conteo por línea y sentido",
            _table([["Línea", "Sentido", "Clase", "Cruces"]] + [[l, d, es.CLASS[c], n] for l, d, c, n in s.by_line])
            if s.by_line else Paragraph("Ningún objeto cruzó las líneas configuradas.", body))
    if s.by_class:
        section("Composición del tráfico",
                _image(composition_png([(es.CLASS[c], n, share) for c, n, share in s.by_class]), 15))

    unit = f"{s.interval_s} s" if s.interval_s < 60 else f"{s.interval_s // 60} min"
    peak = (f" Intervalo con más cruces: el que empieza en <b>{s.peak[0]}</b> ({s.peak[1]} cruces)."
            if s.peak else "")
    clock_note = "" if s.start else " Hora real desconocida: el eje muestra tiempo transcurrido."
    section("Distribución temporal", _image(intervals_png(s.by_interval, s.peak[0] if s.peak else None), 15),
            Paragraph(f"Cruces por intervalos de {unit}.{peak}{clock_note}", body))

    if s.dwell:
        section("Permanencia en zonas", _table(
            [["Zona", "Visitas", "Promedio (s)", "Máximo (s)"]] + [[z, n, f"{m:.1f}", f"{x:.1f}"] for z, n, m, x in s.dwell]))

    if a.alerts:
        section("Alertas", _table([["Tipo", "Cantidad"]] + [[es.ALERT.get(k, k), n] for k, n in s.alerts_by_kind]))
        story.append(Spacer(1, 6))
        rows = [[_when(a, x.frame), es.ALERT.get(x.kind, x.kind), x.where, es.CLASS.get(x.road_class, "-"),
                 "-" if x.track_id is None else x.track_id] for x in a.alerts[:MAX_ALERT_ROWS]]
        story.append(_table([["Momento", "Tipo", "Dónde", "Clase", "ID"]] + rows))  # long: may span pages
        if len(a.alerts) > MAX_ALERT_ROWS:
            story.append(Paragraph(f"... y {len(a.alerts) - MAX_ALERT_ROWS} más (ver alerts.csv).", small))
    else:
        section("Alertas", Paragraph("Sin alertas.", body))

    section("Anexo de calidad", Paragraph(
        f"Objetos seguidos: {s.tracks}. Confianza media de las detecciones seguidas: {s.mean_confidence:.2f}. "
        "Los cambios de identidad (ID switches) solo se pueden medir contra anotaciones manuales; "
        "ver la evaluación del proyecto. Los conteos dependen del modelo y de los umbrales indicados arriba, "
        "y deben validarse con conteos manuales antes de usarse para decisiones.", body))

    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                      bottomMargin=1.8 * cm, title="Reporte de movilidad").build(story)


def _when(a: VideoAnalysis, frame: int) -> str:
    if a.start:
        return f"{clock(a.start, frame, a.fps):%H:%M:%S}"
    seconds = round(frame / a.fps)
    return f"{seconds // 60}:{seconds % 60:02d}"


def _table(rows: list[list], header: bool = True) -> Table:
    t = Table([[str(c) for c in r] for r in rows], hAlign="LEFT", repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.5, GRID), ("FONTSIZE", (0, 0), (-1, -1), 9),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), HEADER_BG), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    else:
        style += [("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def _image(png: bytes, width_cm: float) -> Image:
    img = Image(io.BytesIO(png))
    img.drawHeight = img.drawHeight * width_cm * cm / img.drawWidth
    img.drawWidth = width_cm * cm
    return img
