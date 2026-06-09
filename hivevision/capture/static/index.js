async function load() {
  const grid = document.getElementById("grid");
  const summary = document.getElementById("summary");
  const rows = await fetch("/api/inbox").then((r) => r.json());

  if (!rows.length) {
    summary.textContent = "";
    grid.className = "";
    grid.innerHTML =
      '<div class="empty">No photos yet. Drop images into ' +
      "<code>data/store/inbox/</code> (subfolders OK), then press Refresh.</div>";
    return;
  }

  grid.className = "grid";
  const done = rows.filter((r) => r.labeled).length;
  summary.textContent = `${rows.length} photos · ${done} labeled`;

  grid.innerHTML = "";
  for (const r of rows) {
    const card = document.createElement("div");
    card.className = "card";
    card.onclick = () => (location.href = "/label?src=" + encodeURIComponent(r.src));
    const badge = r.labeled
      ? `<span class="badge done">labeled · ${r.n_points}</span>`
      : '<span class="badge todo">to do</span>';
    card.innerHTML =
      `<img class="thumb" loading="lazy" src="/api/thumb?src=${encodeURIComponent(r.src)}&w=480" />` +
      `<div class="meta"><span class="name" title="${r.src}">${r.src}</span>${badge}</div>`;
    grid.appendChild(card);
  }
}

load();
