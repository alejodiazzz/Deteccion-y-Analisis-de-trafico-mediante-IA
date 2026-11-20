"""Dashboard tab: draw a camera point by hand over a frame of the video, or fix a suggested one.

Geometry (lines and zone polygons) is drawn in a small component (static/point_editor.js);
names, directions and zone rules are filled in with Streamlit widgets under the drawing.
Saving writes configs/points/<name>.yaml with the same checks as reading one.
"""
import base64
import hashlib
import re
from pathlib import Path

import cv2
import streamlit as st

from vtia.adapters import labels_es as es
from vtia.adapters.scene_preview import draw_point, read_frame
from vtia.adapters.video_io import video_info
from vtia.adapters.yaml_config import dump_point, load_point, parse_point
from vtia.domain.classes import RoadClass
from vtia.domain.models import PointConfig
from vtia.domain.scene import OPPOSITE

JS = (Path(__file__).parent / "static" / "point_editor.js").read_text(encoding="utf-8")
CSS = """
.pe { font-family: var(--st-font, sans-serif); color: var(--st-text-color, #fafafa); }
.pe-bar { display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; margin-bottom: .4rem; }
.pe-bar button { background: transparent; color: inherit; border: 1px solid rgba(250,250,250,.3);
  border-radius: .5rem; padding: .3rem .8rem; cursor: pointer; font: inherit; font-size: .9rem; }
.pe-bar button:hover { border-color: var(--st-primary-color, #ff4b4b); }
.pe-bar button.on { background: var(--st-primary-color, #ff4b4b); border-color: transparent; color: #fff; }
.pe-sep { flex: 0 0 1rem; }
.pe-hint { font-size: .85rem; opacity: .75; margin-bottom: .4rem; min-height: 1.2em; }
canvas { width: 100%; height: auto; display: block; border-radius: .5rem; cursor: crosshair; outline: none; }
"""
EDITOR = st.components.v2.component("vtia_point_editor", js=JS, css=CSS)

BLANK = "(en blanco)"
DIRECTION = {
    "toward_camera": "Se acerca a la cámara",
    "away_from_camera": "Se aleja de la cámara",
    "left_to_right": "De izquierda a derecha",
    "right_to_left": "De derecha a izquierda",
}
ALLOWED = {None: "Ambos sentidos", "right": "Solo hacia la flecha", "left": "Solo contra la flecha"}
MAX_PREVIEW = 1280  # px wide: the frame sent to the browser; coordinates stay in video pixels


def shapes_from_point(point: PointConfig) -> tuple[list[dict], dict[str, dict]]:
    """Editor shapes and their attributes (keyed by shape id) for an existing camera point."""
    shapes, attrs = [], {}
    for i, ln in enumerate(point.lines):
        sid = f"l{i}"
        shapes.append({"id": sid, "kind": "line", "pts": [list(ln.start), list(ln.end)]})
        attrs[sid] = {"name": ln.name, "to_right": ln.to_right, "to_left": ln.to_left,
                      "auto": arrow_direction([ln.start, ln.end]),
                      "allowed": None if ln.allowed is None else "right" if ln.allowed == ln.to_right else "left"}
    for i, z in enumerate(point.zones):
        sid = f"z{i}"
        shapes.append({"id": sid, "kind": "zone", "pts": [list(p) for p in z.polygon]})
        attrs[sid] = {"name": z.name, "forbidden": sorted(c.value for c in z.forbidden),
                      "max_dwell_s": z.max_dwell_s, "max_occupancy": z.max_occupancy}
    return shapes, attrs


def arrow_direction(pts: list) -> str:
    """Where the line's arrow points (its right-hand side walking start -> end, y down), as a
    direction label: what an object crossing toward the arrow is doing on screen."""
    (x0, y0), (x1, y1) = pts[0], pts[1]
    nx, ny = -(y1 - y0), x1 - x0  # right-hand normal on screen
    if abs(ny) >= abs(nx):
        return "toward_camera" if ny > 0 else "away_from_camera"  # down the image = closer
    return "left_to_right" if nx > 0 else "right_to_left"


def sync_direction(a: dict, pts: list) -> None:
    """Follows the arrow: a new, inverted or re-drawn line gets the direction its arrow shows;
    a direction chosen by hand stays while the arrow keeps pointing the same way."""
    auto = arrow_direction(pts)
    if a.get("auto") != auto:
        if a.get("auto") is not None or not a.get("to_right"):  # loaded lines keep their YAML label
            a["to_right"] = auto
        a["auto"] = auto


def canvas_labels(shapes: list[dict], attrs: dict[str, dict]) -> dict[str, dict]:
    """What the drawing writes next to each shape: its name and, for lines, the arrow's direction."""
    names = default_names(shapes, attrs)
    return {s["id"]: {"name": names[s["id"]],
                      "right": DIRECTION.get((attrs.get(s["id"]) or {}).get("to_right") or "", "")
                      if s["kind"] == "line" else ""}
            for s in shapes}


def default_names(shapes: list[dict], attrs: dict[str, dict]) -> dict[str, str]:
    """A name for every shape: its own if it has one, else linea_N / zona_N not used by another."""
    taken = {a["name"] for a in attrs.values() if a.get("name")}
    names, count = {}, {"line": 0, "zone": 0}
    for s in shapes:
        own = (attrs.get(s["id"]) or {}).get("name")
        if own:
            names[s["id"]] = own
            continue
        prefix = "linea" if s["kind"] == "line" else "zona"
        while True:
            count[s["kind"]] += 1
            name = f"{prefix}_{count[s['kind']]}"
            if name not in taken:
                break
        taken.add(name)
        names[s["id"]] = name
    return names


def point_data(shapes: list[dict], attrs: dict[str, dict], restrictions: list[dict] = ()) -> dict:
    """The camera point in the YAML layout (what parse_point / load_point read)."""
    lines, zones = [], []
    for s in shapes:
        a = attrs.get(s["id"]) or {}
        xy = [[round(x), round(y)] for x, y in s["pts"]]
        if s["kind"] == "line":
            right = a.get("to_right") or "toward_camera"
            left = OPPOSITE.get(right) or a.get("to_left") or f"not_{right}"
            item = {"name": a["name"], "start": xy[0], "end": xy[1], "to_right": right, "to_left": left}
            if a.get("allowed"):
                item["allowed"] = right if a["allowed"] == "right" else left
            lines.append(item)
        else:
            item = {"name": a["name"], "polygon": xy}
            if a.get("forbidden"):
                item["forbidden"] = list(a["forbidden"])
            if a.get("max_dwell_s"):
                item["max_dwell_s"] = float(a["max_dwell_s"])
            if a.get("max_occupancy"):
                item["max_occupancy"] = int(a["max_occupancy"])
            zones.append(item)
    return {"lines": lines, "zones": zones, "restrictions": list(restrictions)}


def problems(data: dict) -> list[str]:
    """What stops the drawing from being saved, in words for the user."""
    found = []
    if not data["lines"]:
        found.append("Dibuja al menos una línea de conteo.")
    names = [x["name"] for x in data["lines"] + data["zones"]]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        found.append(f"Nombres repetidos: {', '.join(repeated)}. Cada línea y zona necesita uno distinto.")
    if any(not n.strip() for n in names):
        found.append("Hay una línea o zona sin nombre.")
    return found


def file_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_")


@st.cache_data(max_entries=8, show_spinner=False)
def frame_image(video: str, frame: int) -> tuple[str, int, int]:
    """The frame as a JPEG data URL (at most MAX_PREVIEW wide) and the video's own size."""
    img = read_frame(Path(video), frame)
    h, w = img.shape[:2]
    if w > MAX_PREVIEW:
        img = cv2.resize(img, (MAX_PREVIEW, round(h * MAX_PREVIEW / w)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode(), w, h


def render(video: Path | None, configs: Path) -> None:
    st.subheader("Dibujar el punto de cámara")
    if not video:
        st.info("Elige o sube un video en el panel izquierdo para dibujar sobre uno de sus cuadros.")
        return
    st.caption("Para cuando la sugerencia automática no acierta o no encuentra flujos: dibuja las líneas de conteo "
               "y las zonas sobre un cuadro del video. Puedes partir de un punto existente (por ejemplo el sugerido) "
               "y corregirlo.")
    fps, total = video_info(video)
    c1, c2 = st.columns([2, 1])
    files = sorted(configs.glob("*.yaml"))
    suggested = configs / f"sugerido_{video.stem}.yaml"
    base = c1.selectbox("Partir de", [BLANK, *files], format_func=lambda p: p if p == BLANK else p.name,
                        index=1 + files.index(suggested) if suggested in files else 0, key="pe-base")
    last = max(0.0, (total - 1) / fps) if fps and total else 0.0
    second = c2.slider("Cuadro del video (s)", 0.0, last, 0.0, 1.0, disabled=last == 0,
                       help="Elige un momento con la vía despejada para ver bien los carriles.")

    point = None
    if base != BLANK:
        try:
            point = load_point(base)
        except ValueError as e:
            st.error(f"No se pudo leer {base.name}: {e}")
            return
    # One editor per video and base; a hash because component keys must not contain "__".
    key = "pe-" + hashlib.sha1(f"{video}|{base if base == BLANK else base.name}".encode()).hexdigest()[:12]
    if f"{key}-attrs" not in st.session_state:  # first time with this base: seed shapes and attributes
        shapes, attrs = shapes_from_point(point) if point else ([], {})
        st.session_state[f"{key}-shapes"], st.session_state[f"{key}-attrs"] = shapes, attrs
    attrs = st.session_state[f"{key}-attrs"]

    image, w, h = frame_image(str(video), min(round(second * fps), max(0, total - 1)) if fps else 0)
    labels = canvas_labels(st.session_state[f"{key}-shapes"], attrs)
    result = EDITOR(key=key, data={"image": image, "w": w, "h": h, "shapes": st.session_state[f"{key}-shapes"],
                                   "labels": labels},
                    default={"shapes": st.session_state[f"{key}-shapes"]}, on_shapes_change=lambda: None)
    shapes = result.shapes or []
    st.session_state[f"{key}-shapes"] = shapes  # a remount starts from here, not from the base
    names = default_names(shapes, attrs)

    lines = [s for s in shapes if s["kind"] == "line"]
    zones = [s for s in shapes if s["kind"] == "zone"]
    if lines:
        st.markdown("**Líneas de conteo**")
        st.caption("«Hacia la flecha» es el sentido de quien cruza la línea hacia donde apunta la flecha verde. "
                   "Si solo se permite un sentido, el otro genera alertas de contraflujo.")
    for s in lines:
        a = attrs.setdefault(s["id"], {})
        sync_direction(a, s["pts"])
        c1, c2, c3 = st.columns([1.2, 1, 1])
        a["name"] = c1.text_input("Nombre", names[s["id"]], key=f"{key}-{s['id']}-name")
        options = list(DIRECTION) + ([a["to_right"]] if a["to_right"] not in DIRECTION else [])
        a["to_right"] = c2.selectbox("Hacia la flecha", options, format_func=lambda d: DIRECTION.get(d, d),
                                     index=options.index(a["to_right"]),
                                     key=f"{key}-{s['id']}-dir-{a['auto']}")  # new widget when the arrow turns
        a["allowed"] = c3.selectbox("Sentido permitido", list(ALLOWED), format_func=ALLOWED.get,
                                    index=list(ALLOWED).index(a.get("allowed")), key=f"{key}-{s['id']}-allowed")
    if zones:
        st.markdown("**Zonas**")
        st.caption("Las zonas cuyo nombre empieza por «peatonal» se dibujan en naranja. Deja en 0 las reglas que no apliquen.")
    for s in zones:
        a = attrs.setdefault(s["id"], {})
        c1, c2, c3, c4 = st.columns([1.2, 1.4, 0.8, 0.8])
        a["name"] = c1.text_input("Nombre", names[s["id"]], key=f"{key}-{s['id']}-name")
        a["forbidden"] = c2.multiselect("Clases prohibidas", [c.value for c in RoadClass],
                                        default=a.get("forbidden") or [],
                                        format_func=lambda c: es.CLASS[RoadClass(c)], key=f"{key}-{s['id']}-forbidden")
        a["max_dwell_s"] = c3.number_input("Permanencia máx. (s)", 0.0, 3600.0, float(a.get("max_dwell_s") or 0),
                                           5.0, key=f"{key}-{s['id']}-dwell")
        a["max_occupancy"] = c4.number_input("Ocupación máx.", 0, 500, int(a.get("max_occupancy") or 0), 1,
                                             key=f"{key}-{s['id']}-occupancy")
    if not shapes:
        st.caption("Aún no hay figuras: elige «Línea» o «Zona» y haz clic sobre la imagen.")

    restrictions = []
    if point and point.restrictions:
        restrictions = [{"class": r.road_class.value, "hours": [f"{x:%H:%M}-{y:%H:%M}" for x, y in r.windows]}
                        for r in point.restrictions]
        st.caption(f"Se conservan las restricciones horarias de {base.name} ({len(restrictions)}); "
                   "se editan en el YAML.")

    st.divider()
    c1, c2 = st.columns([2, 1])
    name = file_name(c1.text_input("Guardar como (configs/points/)", f"manual_{video.stem}", key=f"{key}-file"))
    target = configs / f"{name}.yaml"
    overwrite = True
    if name and target.exists():
        overwrite = c2.checkbox(f"Reemplazar {target.name}", key=f"{key}-overwrite")
    data = point_data(shapes, attrs, restrictions)
    issues = problems(data)
    for issue in issues:
        st.warning(issue)
    if c2.button("Guardar punto", type="primary", width="stretch", disabled=bool(issues) or not name or not overwrite,
                 key=f"{key}-save"):
        try:
            config = parse_point(data, target.name)
        except ValueError as e:
            st.error(f"El punto no es válido: {e}")
            return
        target.write_text(dump_point(config, f"Dibujado a mano sobre {video.name} en el tablero."), encoding="utf-8")
        st.session_state.drawn = (str(target), str(draw_point(video, config, target.with_suffix(".png"),
                                                               round(second * fps) if fps else 0)))
        st.session_state.pick_point = str(target)  # selected in «Punto de cámara» on the rerun
        st.rerun()
    if st.session_state.get("drawn"):
        yaml_file, preview = (Path(p) for p in st.session_state.drawn)
        if yaml_file.exists():
            st.success(f"Guardado como {yaml_file.name} y elegido en «Punto de cámara». Ya puedes analizar.")
            with st.expander("Ver cómo quedó y descargar el YAML"):
                st.image(str(preview))
                st.download_button("Descargar el YAML", yaml_file.read_text(encoding="utf-8"),
                                   file_name=yaml_file.name, key="pe-download")
    if canvas_labels(shapes, attrs) != labels:  # names/directions on the drawing lag one step: redraw now
        st.rerun()
