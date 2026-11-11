"""Local dashboard: pick a video and a camera point, run the analysis, explore and download the results.

    python -m streamlit run app/dashboard.py
"""
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `streamlit run` only puts app/ on the path

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from vtia.adapters import labels_es as es  # noqa: E402
from vtia.adapters.mot_tracker import ALGORITHMS  # noqa: E402
from vtia.compose import MODELS, analyze_file, default_track_thresh, device_name  # noqa: E402
from vtia.domain.timeline import video_start  # noqa: E402

DATA, CONFIGS, OUT = ROOT / "data", ROOT / "configs" / "points", ROOT / "outputs" / "dashboard"
BAR, PEAK = "#3987e5", "#9ec5f4"  # blue steps for the dark theme (peak = lighter, stands out on dark)
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]
UPSTREAM_JS = (ROOT / "app" / "static" / "upstream.global.js").read_text(encoding="utf-8")


def animated_background() -> None:
    """upstream-fx WebGL streaks (MIT, vendored in app/static) behind the whole page.

    components.html renders in an iframe: CSS stretches it over the viewport behind everything
    (z-index -1), the app's own backgrounds go transparent and the content sits on a translucent
    panel to stay readable. The iframe never gets the pointer, so the parent's is forwarded.
    """
    with st.container(key="upstream-bg"):
        components.html(
            '<style>html,body{margin:0;height:100%}</style>'
            '<upstream-fx style="position:fixed;inset:0"></upstream-fx>'
            "<script>" + UPSTREAM_JS + "</script>"
            "<script>if (window.parent !== window) {"  # guard: without an iframe this would loop forever
            ' for (const t of ["pointermove", "pointerdown", "pointerup", "pointerleave"])'
            "  window.parent.addEventListener(t, e => window.dispatchEvent(new PointerEvent(t, e)), {passive: true});"
            "}</script>",
            height=0,
        )
    st.html("""<style>
    body { background: #000 !important; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] {
        background: transparent !important;
    }
    .st-key-upstream-bg { position: fixed !important; inset: 0; z-index: -1; pointer-events: none; }
    .st-key-upstream-bg iframe { position: fixed; inset: 0; width: 100vw !important; height: 100vh !important; border: 0; }
    [data-testid="stMainBlockContainer"] {
        margin-top: 2.5rem; margin-bottom: 2rem; padding: 2rem 2.5rem !important;
        background: rgba(14, 17, 23, 0.82); border-radius: 16px;  /* no backdrop-filter: it would trap the fixed background inside the panel */
    }
    [data-testid="stSidebar"] { background: rgba(14, 17, 23, 0.88) !important; backdrop-filter: blur(6px); }
    </style>""")


st.set_page_config(page_title="VTIA · Movilidad", layout="wide")
animated_background()
st.title("Conteo, zonas y alertas en video")

with st.sidebar:
    st.header("Entrada")
    uploaded = st.file_uploader("Subir un video", type=VIDEO_TYPES)
    existing = sorted(p for p in DATA.rglob("*") if p.suffix.lower().lstrip(".") in VIDEO_TYPES)
    picked = st.selectbox("o elegir uno de data/", existing, format_func=lambda p: str(p.relative_to(DATA)),
                          disabled=uploaded is not None)
    points = sorted(CONFIGS.glob("*.yaml"))
    point = st.selectbox("Punto de cámara", points, format_func=lambda p: p.name)
    st.caption("Las líneas, zonas y reglas de cada punto se editan en configs/points/.")

    st.header("Modelo")
    model = st.selectbox("Detector", MODELS)
    tracker = st.selectbox("Tracker", ALGORITHMS)
    thresh = st.slider("Umbral del tracker", 0.1, 0.9, default_track_thresh(model), 0.05, key=f"thresh-{model}",
                       help="Confianza para crear un track nuevo (+0.1). Más bajo: encuentra más objetos, con más falsos.")

    st.header("Hora de inicio")
    name = uploaded.name if uploaded else (picked.name if picked else "")
    from_name = video_start(Path(name).stem) if name else None
    st.caption(f"Del nombre del archivo: {from_name:%Y-%m-%d %H:%M}" if from_name
               else "El nombre no sigue P<punto>_<AAAAMMDD>_<HHMM>.")
    manual = st.checkbox("Indicarla a mano", value=False)
    start = None
    if manual:
        c1, c2 = st.columns(2)
        day, hour = c1.date_input("Fecha"), c2.time_input("Hora", step=60)
        start = datetime.combine(day, hour)

    save_video = st.checkbox("Generar video anotado", value=True)
    run = st.button("Analizar", type="primary", width="stretch",
                    disabled=not (uploaded or picked) or not point)

if run:
    if uploaded:
        target = DATA / "uploads" / uploaded.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(uploaded.getbuffer())
    else:
        target = picked
    bar = st.progress(0.0, text="Cargando el modelo...")
    try:
        st.session_state.run = analyze_file(
            target, point, model=model, tracker=tracker, track_thresh=thresh, start=start, out=OUT,
            save_video=save_video,
            progress=lambda i, n: bar.progress(min(i / max(n, 1), 1.0), text=f"Cuadro {i} de {n} ({device_name()})"))
    except Exception as e:  # show config/video problems in the page instead of a stack trace
        bar.empty()
        st.error(f"No se pudo analizar el video: {e}")
        st.stop()
    bar.empty()

result = st.session_state.get("run")
if not result:
    st.info("Elige un video y un punto de cámara en el panel izquierdo y presiona **Analizar**.")
    st.stop()

a, s = result.analysis, result.summary
st.caption(f"{result.out.name} · {a.frames} cuadros en {result.seconds:.1f} s · "
           f"inicio {a.start:%Y-%m-%d %H:%M}" if a.start else
           f"{result.out.name} · {a.frames} cuadros en {result.seconds:.1f} s · hora de inicio desconocida")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Cruces de línea", s.total)
k2.metric("Objetos seguidos", s.tracks)
k3.metric("Alertas", len(a.alerts))
k4.metric("Duración", es.duration(s.duration_s))

summary_tab, video_tab, zones_tab, files_tab = st.tabs(["Resumen", "Video", "Zonas y alertas", "Descargas"])

with summary_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Composición del tráfico")
        if s.by_class:
            df = pd.DataFrame([(es.CLASS[c], n, share) for c, n, share in s.by_class], columns=["Clase", "Cruces", "Proporción"])
            st.altair_chart(alt.Chart(df).mark_bar(color=BAR, size=18, cornerRadiusEnd=4).encode(
                x=alt.X("Cruces:Q", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("Clase:N", sort="-x", title=None),
                tooltip=["Clase", "Cruces", alt.Tooltip("Proporción:Q", format=".0%")]), width="stretch")
        else:
            st.write("Ningún objeto cruzó las líneas.")
        st.subheader("Por línea y sentido")
        st.dataframe(pd.DataFrame([(l, d, es.CLASS[c], n) for l, d, c, n in s.by_line],
                                  columns=["Línea", "Sentido", "Clase", "Cruces"]), hide_index=True, width="stretch")
    with right:
        unit = f"{s.interval_s} s" if s.interval_s < 60 else f"{s.interval_s // 60} min"
        st.subheader(f"Cruces por intervalo de {unit}")
        df = pd.DataFrame(s.by_interval, columns=["Inicio", "Cruces"])
        df["Pico"] = df["Inicio"] == (s.peak[0] if s.peak else None)
        st.altair_chart(alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Inicio:N", sort=None, title="Inicio del intervalo"),
            y=alt.Y("Cruces:Q", axis=alt.Axis(tickMinStep=1)),
            color=alt.condition("datum.Pico", alt.value(PEAK), alt.value(BAR)),
            tooltip=["Inicio", "Cruces"]), width="stretch")
        if s.peak:
            st.caption(f"Pico: el intervalo que empieza en **{s.peak[0]}**, con {s.peak[1]} cruces."
                       + ("" if s.start else " Sin hora real: tiempo transcurrido desde el inicio."))

with video_tab:
    video = result.out / "annotated.mp4"
    if video.exists():
        st.video(str(video))
        st.caption("Cajas por clase con su ID; la línea se ilumina en cada cruce; en rojo, objetos con alerta.")
    else:
        st.info("Este análisis se hizo sin video anotado.")

with zones_tab:
    st.subheader("Permanencia en zonas")
    if s.dwell:
        st.dataframe(pd.DataFrame(s.dwell, columns=["Zona", "Visitas", "Promedio (s)", "Máximo (s)"]).round(1),
                     hide_index=True, width="stretch")
    else:
        st.write("Este punto no tiene zonas configuradas, o nadie entró en ellas.")
    st.subheader("Alertas")
    if a.alerts:
        st.dataframe(pd.DataFrame([(round(x.frame / a.fps, 1), es.ALERT.get(x.kind, x.kind), x.where,
                                    es.CLASS.get(x.road_class, "-"), x.track_id, x.detail) for x in a.alerts],
                                  columns=["Segundo", "Tipo", "Dónde", "Clase", "ID", "Detalle"]),
                     hide_index=True, width="stretch")
    else:
        st.write("Sin alertas.")

with files_tab:
    st.write(f"Archivos en `{result.out.relative_to(ROOT)}`:")
    for f in sorted(result.out.iterdir()):
        st.download_button(f.name, f.read_bytes(), file_name=f.name, key=f"dl-{f.name}")
