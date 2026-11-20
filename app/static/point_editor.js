// Camera-point editor (st.components.v2): draw counting lines and zone polygons over a frame.
// Coordinates are kept in the video's own pixels (data.w x data.h), the ones the YAML uses.
// Geometry only: names, directions and rules are edited in Python, which sends them back as
// data.labels to be drawn. Python is told about a change only when a shape is finished, moved
// or deleted (each change reruns the app), not on every click.
export default function (component) {
  const { data, parentElement } = component;
  let ed = parentElement.__pointEditor;
  if (!ed) {
    ed = parentElement.__pointEditor = build(parentElement, data);
  }
  ed.api = component;
  ed.labels = data.labels || {};
  if (ed.imageSrc !== data.image) {
    ed.imageSrc = data.image;
    ed.img.src = data.image;
  }
  ed.draw();
}

const LINE = "#22c55e", ROAD = "#3b82f6", WALK = "#f59e0b", SEL = "#facc15", DRAFT = "#f472b6";
const HIT = 12; // screen pixels to grab a vertex or close a zone

function build(root, data) {
  const ed = {
    w: data.w, h: data.h, labels: {}, mode: "line", draft: [], hover: null,
    shapes: (data.shapes || []).map((s) => ({ ...s, pts: s.pts.map((p) => [...p]) })),
    selected: null, drag: null, img: new Image(),
  };
  const wrap = document.createElement("div");
  wrap.className = "pe";
  wrap.innerHTML = `
    <div class="pe-bar">
      <button data-mode="line">Línea</button>
      <button data-mode="zone">Zona</button>
      <button data-mode="edit">Mover / elegir</button>
      <span class="pe-sep"></span>
      <button data-act="close">Cerrar zona</button>
      <button data-act="undo">Quitar último punto</button>
      <button data-act="flip">Invertir línea</button>
      <button data-act="delete">Borrar elegida</button>
    </div>
    <div class="pe-hint"></div>
    <canvas tabindex="0"></canvas>`;
  root.appendChild(wrap);
  const canvas = wrap.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const hint = wrap.querySelector(".pe-hint");
  canvas.width = ed.w;
  canvas.height = ed.h;
  ed.img.onload = () => ed.draw();

  const k = () => ed.w / Math.max(1, canvas.clientWidth); // video pixels per screen pixel
  const at = (e) => {
    const r = canvas.getBoundingClientRect();
    return [Math.round(Math.min(ed.w, Math.max(0, (e.clientX - r.left) * k()))),
            Math.round(Math.min(ed.h, Math.max(0, (e.clientY - r.top) * k())))];
  };
  const near = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]) <= HIT * k();
  const newId = () => "s" + Date.now().toString(36) + Math.random().toString(36).slice(2, 5);

  const send = () => ed.api.setStateValue("shapes", ed.shapes.map((s) => ({ id: s.id, kind: s.kind, pts: s.pts })));
  const finish = (kind, pts) => {
    const s = { id: newId(), kind, pts };
    ed.shapes.push(s);
    ed.selected = s.id;
    ed.draft = [];
    send();
  };
  const closeZone = () => {
    if (ed.mode === "zone" && ed.draft.length >= 3) finish("zone", ed.draft);
    ed.draw();
  };
  const setMode = (m) => {
    ed.mode = m;
    ed.draft = [];
    ed.draw();
  };

  wrap.querySelectorAll("[data-mode]").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  wrap.querySelector('[data-act="close"]').onclick = closeZone;
  wrap.querySelector('[data-act="undo"]').onclick = () => {
    ed.draft.pop();
    ed.draw();
  };
  wrap.querySelector('[data-act="flip"]').onclick = () => {
    const s = ed.shapes.find((s) => s.id === ed.selected && s.kind === "line");
    if (!s) return;
    s.pts.reverse(); // start <-> end: the arrow points to the other side
    send();
    ed.draw();
  };
  wrap.querySelector('[data-act="delete"]').onclick = () => {
    if (!ed.selected) return;
    ed.shapes = ed.shapes.filter((s) => s.id !== ed.selected);
    ed.selected = null;
    send();
    ed.draw();
  };

  canvas.addEventListener("mousedown", (e) => {
    const p = at(e);
    canvas.focus();
    if (ed.mode === "line") {
      if (ed.draft.length === 0) ed.draft = [p];
      else if (!near(ed.draft[0], p)) finish("line", [ed.draft[0], p]);
    } else if (ed.mode === "zone") {
      if (ed.draft.length >= 3 && near(ed.draft[0], p)) return closeZone();
      ed.draft.push(p);
    } else {
      ed.drag = null;
      for (const s of [...ed.shapes].reverse()) {
        const i = s.pts.findIndex((q) => near(q, p));
        if (i >= 0) { ed.drag = { s, i }; ed.selected = s.id; break; }
      }
      if (!ed.drag) {
        const s = [...ed.shapes].reverse().find((s) => hits(s, p, HIT * k()));
        ed.selected = s ? s.id : null;
        if (s) ed.drag = { s, i: -1, from: p, orig: s.pts.map((q) => [...q]) }; // drag the whole shape
      }
    }
    ed.draw();
  });
  canvas.addEventListener("mousemove", (e) => {
    const p = at(e);
    ed.hover = p;
    if (ed.drag) {
      const { s, i } = ed.drag;
      if (i >= 0) s.pts[i] = p;
      else s.pts = ed.drag.orig.map((q) => [q[0] + p[0] - ed.drag.from[0], q[1] + p[1] - ed.drag.from[1]]);
      ed.drag.moved = true;
    }
    ed.draw();
  });
  const drop = () => {
    if (ed.drag && ed.drag.moved) send();
    ed.drag = null;
  };
  canvas.addEventListener("mouseup", drop);
  canvas.addEventListener("mouseleave", () => { drop(); ed.hover = null; ed.draw(); });
  canvas.addEventListener("dblclick", (e) => { e.preventDefault(); closeZone(); });
  canvas.addEventListener("keydown", (e) => {
    if (e.key === "Enter") closeZone();
    else if (e.key === "Escape") setMode(ed.mode);
    else if ((e.key === "Delete" || e.key === "Backspace") && ed.selected && !ed.draft.length) {
      wrap.querySelector('[data-act="delete"]').click();
    } else return;
    e.preventDefault();
  });

  ed.draw = () => {
    const u = k(); // keep strokes and text the same size on screen whatever the video size
    ctx.clearRect(0, 0, ed.w, ed.h);
    if (ed.img.complete && ed.img.naturalWidth) ctx.drawImage(ed.img, 0, 0, ed.w, ed.h);
    for (const s of ed.shapes) {
      const label = ed.labels[s.id] || {};
      const sel = s.id === ed.selected;
      if (s.kind === "zone") {
        const color = (label.name || "").startsWith("peatonal") ? WALK : ROAD;
        path(ctx, s.pts, true);
        ctx.fillStyle = color + "40";
        ctx.fill();
        ctx.lineWidth = 2 * u;
        ctx.strokeStyle = sel ? SEL : color;
        ctx.stroke();
        tag(ctx, label.name, [Math.min(...s.pts.map((p) => p[0])), Math.min(...s.pts.map((p) => p[1]))], u);
      } else {
        const [a, b] = s.pts;
        path(ctx, s.pts, false);
        ctx.lineWidth = 4 * u;
        ctx.strokeStyle = sel ? SEL : LINE;
        ctx.stroke();
        // Arrow to the right-hand side walking start -> end: crossings that way are "to_right".
        const dx = b[0] - a[0], dy = b[1] - a[1], n = Math.max(1, Math.hypot(dx, dy));
        const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], reach = Math.min(0.3 * n, 40 * u);
        const tip = [m[0] - (dy / n) * reach, m[1] + (dx / n) * reach];
        arrow(ctx, m, tip, sel ? SEL : LINE, u);
        if (label.right) tag(ctx, "→ " + label.right, tip, u, true);
        tag(ctx, label.name, b, u);
        dot(ctx, a, u, sel ? SEL : LINE); // the start: the arrow is to its right
      }
      if (sel) s.pts.forEach((p) => dot(ctx, p, u, SEL));
    }
    if (ed.draft.length) {
      const pts = ed.hover ? [...ed.draft, ed.hover] : ed.draft;
      path(ctx, pts, false);
      ctx.lineWidth = 2 * u;
      ctx.setLineDash([8 * u, 6 * u]);
      ctx.strokeStyle = DRAFT;
      ctx.stroke();
      ctx.setLineDash([]);
      ed.draft.forEach((p) => dot(ctx, p, u, DRAFT));
    }
    wrap.querySelectorAll("[data-mode]").forEach((b) => b.classList.toggle("on", b.dataset.mode === ed.mode));
    hint.textContent = {
      line: ed.draft.length ? "Clic en el final de la línea (Esc cancela)."
        : "Línea de conteo: clic en el inicio y clic en el final. La flecha marca el lado «hacia la derecha».",
      zone: ed.draft.length >= 3 ? "Sigue marcando vértices; cierra con clic en el primero, doble clic o Enter."
        : "Zona: clic en cada vértice del polígono (mínimo 3).",
      edit: "Arrastra un vértice o la figura entera; clic para elegirla, «Invertir línea» para cambiar la flecha "
        + "de lado y Supr para borrarla.",
    }[ed.mode];
  };
  return ed;
}

function path(ctx, pts, closed) {
  ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
  if (closed) ctx.closePath();
}

function dot(ctx, p, u, color) {
  ctx.beginPath();
  ctx.arc(p[0], p[1], 5 * u, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = 1.5 * u;
  ctx.strokeStyle = "#000";
  ctx.stroke();
}

function arrow(ctx, from, to, color, u) {
  const ang = Math.atan2(to[1] - from[1], to[0] - from[0]), head = 10 * u;
  ctx.beginPath();
  ctx.moveTo(from[0], from[1]);
  ctx.lineTo(to[0], to[1]);
  ctx.lineTo(to[0] - head * Math.cos(ang - 0.5), to[1] - head * Math.sin(ang - 0.5));
  ctx.moveTo(to[0], to[1]);
  ctx.lineTo(to[0] - head * Math.cos(ang + 0.5), to[1] - head * Math.sin(ang + 0.5));
  ctx.lineWidth = 3 * u;
  ctx.strokeStyle = color;
  ctx.stroke();
}

function tag(ctx, text, at, u, small) {
  if (!text) return;
  ctx.font = `${(small ? 12 : 14) * u}px sans-serif`;
  const w = ctx.measureText(text).width + 8 * u, h = (small ? 17 : 20) * u;
  const x = Math.min(Math.max(0, at[0]), ctx.canvas.width - w), y = Math.min(Math.max(0, at[1] - h), ctx.canvas.height - h);
  ctx.fillStyle = "rgba(0,0,0,.75)";
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = "#fff";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + 4 * u, y + h / 2);
}

function hits(s, p, tol) {
  if (s.kind === "zone") {
    let inside = false;
    for (let i = 0, j = s.pts.length - 1; i < s.pts.length; j = i++) {
      const [xi, yi] = s.pts[i], [xj, yj] = s.pts[j];
      if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }
  const [a, b] = s.pts, dx = b[0] - a[0], dy = b[1] - a[1];
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / Math.max(1, dx * dx + dy * dy)));
  return Math.hypot(a[0] + t * dx - p[0], a[1] + t * dy - p[1]) <= tol;
}
