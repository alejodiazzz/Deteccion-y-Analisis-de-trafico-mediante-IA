"""Compact reproducible scientific evaluation report (PDF, ReportLab)."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def write_scientific_pdf(path: Path, title: str, metadata: dict, rows: list[list],
                         headers: list[str], notes: list[str] | None = None,
                         extra_tables: list[tuple[str, list[str], list[list]]] | None = None) -> None:
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 8)]
    meta_rows = [[str(k), str(v)] for k, v in metadata.items()]
    if meta_rows:
        table = Table(meta_rows, colWidths=[4 * cm, 22 * cm])
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#CBD5E1")),
                                   ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                                   ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.extend([table, Spacer(1, 10)])
    if headers:
        table = Table([headers, *rows], repeatRows=1)
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#CBD5E1")),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                                   ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                   ("FONTSIZE", (0, 0), (-1, -1), 7), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.extend([Paragraph("Resultados", styles["Heading2"]), table, Spacer(1, 8)])
    for section_title, section_headers, section_rows in extra_tables or []:
        story.append(Paragraph(section_title, styles["Heading2"]))
        table = Table([section_headers, *section_rows], repeatRows=1)
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#CBD5E1")),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                                   ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                   ("FONTSIZE", (0, 0), (-1, -1), 7), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.extend([table, Spacer(1, 8)])
    for note in notes or []:
        story.append(Paragraph(note, styles["BodyText"]))
        story.append(Spacer(1, 5))
    SimpleDocTemplate(str(path), pagesize=landscape(A4), leftMargin=1.2*cm, rightMargin=1.2*cm,
                      topMargin=1.2*cm, bottomMargin=1.2*cm, title=title).build(story)
