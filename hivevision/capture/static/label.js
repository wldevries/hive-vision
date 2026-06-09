// Per-photo manual labeling: mark each tile's icon center and assign its class.
// Points are stored in image-pixel coordinates (the normalized frame the server
// serves and saves). A pan/zoom view transform maps image <-> screen.
//
// Input (touch-first, for the Surface): tap empty = add point; tap a dot = select;
// drag a dot = move it; one-finger drag on empty = pan; two-finger pinch = zoom+pan.
// Touchpad: pinch (ctrl+wheel) = zoom, two-finger scroll = pan. Plus on-screen
// zoom -/+ and Remove buttons so no keyboard is needed.

const params = new URLSearchParams(location.search);
const SRC = params.get("src");

const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const dpr = window.devicePixelRatio || 1;

let classes = [];
let current = null; // current class code
const points = []; // {label, x, y} in image pixels
let selected = -1;

const img = new Image();
let view = { scale: 1, ox: 0, oy: 0 };

// pointer interaction state
const pointers = new Map(); // pointerId -> {x, y} (CSS px, canvas-relative)
let mode = null; // 'point' | 'pan' | 'gesture' | 'maybe'
let dragIndex = -1;
let start = { x: 0, y: 0 };
let viewStart = { ox: 0, oy: 0 };
let moved = false;
let gesture = null; // {dist, mx, my} two-finger baseline

const HIT_PX = 14;
const TAP_PX = 4; // movement under this counts as a tap, not a drag

// ---- coordinate transforms (CSS pixels) ---------------------------------- //
const toScreen = (p) => ({ x: p.x * view.scale + view.ox, y: p.y * view.scale + view.oy });
const toImage = (sx, sy) => ({ x: (sx - view.ox) / view.scale, y: (sy - view.oy) / view.scale });

function eventPos(e) {
  const r = cv.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function twoPointer() {
  const [a, b] = [...pointers.values()];
  return { dist: Math.hypot(a.x - b.x, a.y - b.y) || 1, mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 };
}

// ---- palette ------------------------------------------------------------- //
async function buildPalette() {
  classes = await fetch("/api/classes").then((r) => r.json());
  const pal = document.getElementById("palette");
  pal.innerHTML = "";
  for (const color of ["w", "b"]) {
    const head = document.createElement("div");
    head.className = "group";
    head.textContent = color === "w" ? "White" : "Black";
    pal.appendChild(head);
    for (const c of classes.filter((c) => c.code[0] === color)) {
      const b = document.createElement("button");
      b.className = "swatch";
      b.dataset.code = c.code;
      b.title = c.name;
      const fill = color === "w" ? "#ededed" : "#1c1c1c";
      b.innerHTML = `<span class="dot" style="background:${fill}"></span>` +
        `<span>${c.code} <span class="muted">${c.name.split(" ").slice(1).join(" ") || c.name}</span></span>`;
      b.onclick = () => setCurrent(c.code);
      pal.appendChild(b);
    }
  }
  setCurrent(classes[0].code);
}

function setCurrent(code) {
  current = code;
  for (const b of document.querySelectorAll(".swatch"))
    b.classList.toggle("active", b.dataset.code === code);
}

// ---- view ---------------------------------------------------------------- //
function resizeCanvas() {
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  draw();
}

function resetView() {
  const w = cv.clientWidth, h = cv.clientHeight;
  const s = Math.min(w / img.naturalWidth, h / img.naturalHeight);
  view.scale = s;
  view.ox = (w - img.naturalWidth * s) / 2;
  view.oy = (h - img.naturalHeight * s) / 2;
  draw();
}

function zoomAround(sx, sy, factor) {
  const before = toImage(sx, sy);
  view.scale *= factor;
  view.ox = sx - before.x * view.scale;
  view.oy = sy - before.y * view.scale;
  draw();
}

function zoomBy(factor) {
  zoomAround(cv.clientWidth / 2, cv.clientHeight / 2, factor);
}

function draw() {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cv.clientWidth, cv.clientHeight);
  if (img.complete && img.naturalWidth) {
    ctx.drawImage(
      img, view.ox, view.oy, img.naturalWidth * view.scale, img.naturalHeight * view.scale,
    );
  }
  points.forEach((p, i) => {
    const s = toScreen(p);
    const isW = p.label[0] === "w";
    ctx.beginPath();
    ctx.arc(s.x, s.y, 13, 0, Math.PI * 2);
    ctx.fillStyle = isW ? "#ededed" : "#1c1c1c";
    ctx.fill();
    ctx.lineWidth = i === selected ? 3.5 : 2;
    ctx.strokeStyle = i === selected ? "#ff3b30" : "#2b8cff";
    ctx.stroke();
    ctx.fillStyle = isW ? "#111" : "#ededed";
    ctx.font = "bold 12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(p.label.slice(1), s.x, s.y);
  });
  document.getElementById("count").textContent = `${points.length} pieces`;
}

function hitTest(sx, sy) {
  for (let i = points.length - 1; i >= 0; i--) {
    const s = toScreen(points[i]);
    if (Math.hypot(s.x - sx, s.y - sy) <= HIT_PX) return i;
  }
  return -1;
}

// ---- pointer gestures ---------------------------------------------------- //
cv.addEventListener("pointerdown", (e) => {
  cv.setPointerCapture(e.pointerId);
  const p = eventPos(e);
  pointers.set(e.pointerId, p);

  if (pointers.size === 2) {
    mode = "gesture";
    gesture = twoPointer();
    return;
  }
  if (pointers.size > 2) return;

  // single pointer
  start = p;
  moved = false;
  viewStart = { ox: view.ox, oy: view.oy };
  const hit = hitTest(p.x, p.y);
  if (hit >= 0) {
    mode = "point";
    dragIndex = hit;
    selected = hit;
    draw();
  } else {
    mode = "maybe"; // becomes a tap-add (no move) or a pan (on move)
  }
});

cv.addEventListener("pointermove", (e) => {
  if (!pointers.has(e.pointerId)) return;
  const p = eventPos(e);
  pointers.set(e.pointerId, p);

  if (mode === "gesture" && pointers.size >= 2) {
    const g = twoPointer();
    // Keep the image point under the old pinch midpoint fixed under the new one,
    // while scaling by the change in finger distance -> simultaneous zoom + pan.
    const anchor = toImage(gesture.mx, gesture.my);
    view.scale *= g.dist / gesture.dist;
    view.ox = g.mx - anchor.x * view.scale;
    view.oy = g.my - anchor.y * view.scale;
    gesture = g;
    draw();
    return;
  }

  if (Math.hypot(p.x - start.x, p.y - start.y) > TAP_PX) moved = true;
  if (mode === "point" && moved) {
    const ip = toImage(p.x, p.y);
    points[dragIndex].x = ip.x;
    points[dragIndex].y = ip.y;
    draw();
  } else if ((mode === "maybe" || mode === "pan") && moved) {
    mode = "pan";
    view.ox = viewStart.ox + (p.x - start.x);
    view.oy = viewStart.oy + (p.y - start.y);
    draw();
  }
});

function endPointer(e) {
  const p = eventPos(e);
  const wasGesture = mode === "gesture";
  pointers.delete(e.pointerId);

  if (wasGesture) {
    // A finger lifted mid-pinch. If one remains, continue as a pan with a fresh
    // baseline (no jump, no accidental tap-add); otherwise the gesture is over.
    if (pointers.size === 1) {
      start = [...pointers.values()][0];
      viewStart = { ox: view.ox, oy: view.oy };
      moved = true;
      mode = "pan";
    } else {
      mode = null;
    }
    return;
  }

  if (mode === "maybe" && !moved && current) {
    const ip = toImage(p.x, p.y);
    points.push({ label: current, x: ip.x, y: ip.y });
    selected = points.length - 1;
    draw();
    scheduleRecover();
  } else if (mode === "point" && moved) {
    scheduleRecover(); // a tile was dragged to a new position
  }
  if (pointers.size === 0) {
    mode = null;
    dragIndex = -1;
  }
}
cv.addEventListener("pointerup", endPointer);
cv.addEventListener("pointercancel", (e) => {
  pointers.delete(e.pointerId);
  if (pointers.size === 0) { mode = null; dragIndex = -1; }
});

// Touchpad / mouse wheel: pinch (ctrl+wheel) zooms; plain scroll pans.
cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  const p = eventPos(e);
  if (e.ctrlKey) {
    zoomAround(p.x, p.y, Math.exp(-e.deltaY * 0.01));
  } else {
    view.ox -= e.deltaX;
    view.oy -= e.deltaY;
    draw();
  }
}, { passive: false });

function removeSelected() {
  if (selected >= 0) {
    points.splice(selected, 1);
    selected = -1;
    draw();
    scheduleRecover();
  }
}

window.addEventListener("keydown", (e) => {
  if ((e.key === "Delete" || e.key === "Backspace") && selected >= 0) {
    e.preventDefault();
    removeSelected();
  }
});

// ---- save ---------------------------------------------------------------- //
// The next inbox photo (in listing order, wrapping) that still has no label.
async function nextUnlabeled() {
  const rows = await fetch("/api/inbox").then((r) => r.json()).catch(() => []);
  if (!rows.length) return null;
  const here = Math.max(0, rows.findIndex((r) => r.src === SRC));
  for (let k = 1; k <= rows.length; k++) {
    const cand = rows[(here + k) % rows.length];
    if (!cand.labeled) return cand.src; // current row now reads as labeled, so it's skipped
  }
  return null;
}

async function save() {
  const status = document.getElementById("status");
  status.textContent = "saving…";
  const resp = await fetch("/api/label", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ src: SRC, points }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    status.textContent = "error: " + (err.detail || resp.status);
    return;
  }
  status.textContent = "saved ✓";
  const next = await nextUnlabeled();
  if (next && next !== SRC) {
    location.href = "/label?src=" + encodeURIComponent(next);
  } else {
    status.textContent = "saved ✓ · all photos labeled";
    setTimeout(() => (status.textContent = ""), 2000);
  }
}

window.save = save;
window.resetView = resetView;
window.zoomBy = zoomBy;
window.removeSelected = removeSelected;

// ---- recovered-board preview ---------------------------------------------- //
const board = document.getElementById("board");
const bctx = board.getContext("2d");
let recoverTimer = null;

function scheduleRecover() {
  clearTimeout(recoverTimer);
  recoverTimer = setTimeout(runRecover, 350);
}

async function runRecover() {
  const stat = document.getElementById("board-stat");
  if (points.length < 3) {
    drawBoard([]);
    stat.textContent = "place ≥3 tiles";
    return;
  }
  let data;
  try {
    data = await fetch("/api/recover", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points }),
    }).then((r) => r.json());
  } catch {
    return;
  }
  if (!data.ok) {
    drawBoard([]);
    stat.textContent = data.reason || "recovery failed";
    return;
  }
  drawBoard(data.placements, data.orient_deg || 0);
  const pct = data.residual_frac * 100;
  const cls = pct < 7 ? "good" : pct < 12 ? "warn" : "bad";
  stat.innerHTML =
    `<b>${data.n}</b> tiles · fit <span class="conf ${cls}">${pct.toFixed(1)}%</span>` +
    (cls === "good" ? "" : " — check the board");
}

// Fixed logical size set explicitly in JS so the canvas can't feed back on its own
// clientHeight (a missing/stale CSS rule would otherwise grow it by dpr each redraw).
const BOARD_W = 150, BOARD_H = 120;

function drawBoard(placements, orientDeg = 0) {
  board.style.width = BOARD_W + "px";
  board.style.height = BOARD_H + "px";
  board.width = Math.round(BOARD_W * dpr);
  board.height = Math.round(BOARD_H * dpr);
  const w = BOARD_W, h = BOARD_H;
  bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  bctx.clearRect(0, 0, w, h);
  if (!placements.length) return;

  // dx,dy are already oriented to match the photo (server applies the homography's
  // local rotation); fall back to raw axial layout if absent.
  const S3 = Math.sqrt(3);
  const cells = placements.map((p) => ({
    x: p.dx ?? 1.5 * p.q, y: p.dy ?? S3 * (p.r + p.q / 2), label: p.label,
  }));
  const xs = cells.map((c) => c.x), ys = cells.map((c) => c.y);
  const minx = Math.min(...xs) - 1, maxx = Math.max(...xs) + 1;
  const miny = Math.min(...ys) - 1, maxy = Math.max(...ys) + 1;
  const scale = Math.min(w / (maxx - minx), h / (maxy - miny));
  const ox = (w - (maxx - minx) * scale) / 2 - minx * scale;
  const oy = (h - (maxy - miny) * scale) / 2 - miny * scale;
  const R = scale; // circumradius 1 in plane units -> tiles touch
  const off = (Math.PI / 180) * orientDeg; // hex orientation matches the lattice

  for (const c of cells) {
    const cx = c.x * scale + ox, cy = c.y * scale + oy;
    const isW = c.label[0] === "w";
    bctx.beginPath();
    for (let k = 0; k < 6; k++) {
      const a = off + (Math.PI / 180) * (60 * k);
      const vx = cx + R * Math.cos(a), vy = cy + R * Math.sin(a);
      k ? bctx.lineTo(vx, vy) : bctx.moveTo(vx, vy);
    }
    bctx.closePath();
    bctx.fillStyle = isW ? "#ededed" : "#1c1c1c";
    bctx.fill();
    bctx.lineWidth = 1;
    bctx.strokeStyle = "rgba(90,150,230,0.55)"; // soft blue so dark tiles read on the dark panel
    bctx.stroke();
    bctx.fillStyle = isW ? "#111" : "#ededed";
    bctx.font = `bold ${Math.max(8, R * 0.7)}px system-ui, sans-serif`;
    bctx.textAlign = "center";
    bctx.textBaseline = "middle";
    bctx.fillText(c.label.slice(1), cx, cy);
  }
}

// ---- boot ---------------------------------------------------------------- //
async function boot() {
  if (!SRC) { document.getElementById("filename").textContent = "no ?src= given"; return; }
  document.getElementById("filename").textContent = SRC;
  document.getElementById("filename").classList.remove("muted");
  await buildPalette();

  const existing = await fetch("/api/label?src=" + encodeURIComponent(SRC)).then((r) => r.json());
  if (existing && existing.points) {
    for (const p of existing.points) points.push({ label: p.label, x: p.x, y: p.y });
  }

  img.onload = () => { resizeCanvas(); resetView(); };
  img.src = "/api/image?src=" + encodeURIComponent(SRC);
  window.addEventListener("resize", resizeCanvas);
  runRecover();
}

boot();
