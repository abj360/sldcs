/* SLDCS UI controller.
   Drives the four screens, talks to the FastAPI backend, and plays the
   scan-sweep once per result. Honest by design: pipeline stage labels are the
   real stages the backend runs; no fabricated per-tile progress counters are
   shown, and a zero count is rendered exactly like any other value. */

(() => {
  "use strict";

  const MAX_FILE_SIZE = 52428800; // 50 MB, mirrors Settings.MAX_FILE_SIZE.
  const ALLOWED_EXT = ["jpg", "jpeg", "png", "bmp"];
  const SLOW_NOTICE_MS = 8000;
  const SWEEP_MS = 900;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const el = (id) => document.getElementById(id);
  const screens = {
    tray: el("screen-tray"),
    processing: el("screen-processing"),
    report: el("screen-report"),
    info: el("screen-info"),
  };

  const session = { processed: 0, confSum: 0, confCount: 0, latencySum: 0, latencyCount: 0 };
  let lastResults = [];
  let activeIndex = 0;

  // --- Header progress hairline (activity light, not a spinner) ---
  const headerProgress = document.createElement("div");
  headerProgress.id = "header-progress";
  document.body.appendChild(headerProgress);

  function showScreen(name) {
    Object.entries(screens).forEach(([key, node]) => {
      node.hidden = key !== name;
    });
  }

  // ---------------- Startup state ----------------
  async function initInstrumentState() {
    setStatus("loading");
    try {
      const health = await (await fetch("/health")).json();
      if (!health.model_loaded) {
        setOffline();
        return;
      }
      setStatus("ready");
      const cfg = await (await fetch("/config")).json();
      el("tray-params").textContent =
        `${cfg.tile_size}×${cfg.tile_size} tiles · ${cfg.tile_overlap}px overlap · conf ≥ ${cfg.conf_threshold.toFixed(2)}`;
      el("model-tag-conf").textContent = `conf ≥ ${cfg.conf_threshold.toFixed(2)}`;
      const info = await (await fetch("/model/info")).json();
      renderModelTag(info);
    } catch (err) {
      setOffline();
    }
  }

  function setStatus(state) {
    const dot = el("status-dot");
    dot.className = "status-dot " + state;
  }

  function setOffline() {
    setStatus("offline");
    const tray = el("dropzone");
    tray.classList.add("disabled");
    el("tray-instruction").textContent = "Instrument offline — detection model failed to load.";
  }

  function renderModelTag(info) {
    const label = info.trained_on_project_data
      ? `TRAINED · ${info.version}`
      : `PRETRAINED · yolov5s (COCO)`;
    el("model-tag-name").textContent = label;
  }

  // ---------------- Upload handling ----------------
  const dropzone = el("dropzone");
  const fileInput = el("file-input");

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => handleFiles([...fileInput.files]));

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      if (evt === "dragleave" && dropzone.contains(e.relatedTarget)) return;
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const files = [...(e.dataTransfer?.files || [])];
    handleFiles(files);
  });

  function trayError(message) {
    const node = el("tray-error");
    node.textContent = message;
    node.hidden = false;
    dropzone.classList.add("error-flash");
    setTimeout(() => dropzone.classList.remove("error-flash"), 600);
  }

  function validateFiles(files) {
    for (const file of files) {
      const ext = file.name.split(".").pop().toLowerCase();
      if (!ALLOWED_EXT.includes(ext)) {
        trayError("Unsupported file — use JPG, PNG, or BMP.");
        return false;
      }
      if (file.size > MAX_FILE_SIZE) {
        trayError("Image exceeds 50MB — resize and try again.");
        return false;
      }
    }
    return true;
  }

  async function handleFiles(files) {
    if (!files.length || dropzone.classList.contains("disabled")) return;
    el("tray-error").hidden = true;
    if (!validateFiles(files)) return;
    await runDetection(files);
  }

  // ---------------- Processing ----------------
  const STAGES = ["TILING…", "RUNNING INFERENCE…", "STITCHING…", "REMOVING DUPLICATES…"];

  async function runDetection(files) {
    showScreen("processing");
    const thumbUrl = URL.createObjectURL(files[0]);
    el("processing-thumb").src = thumbUrl;
    el("processing-slow").hidden = true;
    headerProgress.style.width = "35%";

    let stageIdx = 0;
    el("processing-status").textContent = STAGES[0];
    const stageTimer = reducedMotion
      ? null
      : setInterval(() => {
          stageIdx = Math.min(stageIdx + 1, STAGES.length - 1);
          el("processing-status").textContent = STAGES[stageIdx];
        }, 550);
    const slowTimer = setTimeout(() => (el("processing-slow").hidden = false), SLOW_NOTICE_MS);

    const form = new FormData();
    files.forEach((f) => form.append("files", f));

    try {
      const resp = await fetch("/detect", { method: "POST", body: form });
      headerProgress.style.width = "100%";
      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({ detail: "Detection failed." }));
        throw new Error(detail.detail || "Detection failed.");
      }
      const results = await resp.json();
      recordSession(results);
      lastResults = results;
      activeIndex = 0;
      renderReport(results);
      showScreen("report");
    } catch (err) {
      showScreen("tray");
      trayError(String(err.message || err));
    } finally {
      if (stageTimer) clearInterval(stageTimer);
      clearTimeout(slowTimer);
      setTimeout(() => (headerProgress.style.width = "0"), 300);
      URL.revokeObjectURL(thumbUrl);
    }
  }

  function recordSession(results) {
    results.forEach((r) => {
      session.processed += 1;
      session.confCount += 1;
      session.confSum += r.average_confidence;
      session.latencyCount += 1;
      session.latencySum += r.processing_time_ms;
    });
    el("stat-processed").textContent = String(session.processed);
    el("stat-avgconf").textContent =
      session.confCount ? (session.confSum / session.confCount).toFixed(3) : "—";
    el("stat-latency").textContent =
      session.latencyCount ? `${Math.round(session.latencySum / session.latencyCount)} ms` : "—";
  }

  // ---------------- Report rendering ----------------
  function renderReport(results) {
    const isBatch = results.length > 1;
    el("filmstrip").hidden = !isBatch;
    el("aggregate-bar").hidden = !isBatch;
    if (isBatch) renderFilmstrip(results);
    renderSingle(results[activeIndex]);
  }

  function renderFilmstrip(results) {
    const strip = el("filmstrip");
    strip.innerHTML = "";
    const total = results.reduce((s, r) => s + r.larvae_count, 0);
    const avg = (total / results.length).toFixed(2);
    el("aggregate-bar").textContent =
      `${results.length} IMAGES · ${total} TOTAL LARVAE · AVG ${avg} / IMAGE`;
    results.forEach((r, i) => {
      const img = document.createElement("img");
      img.src = `data:image/png;base64,${r.annotated_image}`;
      img.alt = r.filename || `image ${i + 1}`;
      if (i === activeIndex) img.classList.add("active");
      img.addEventListener("click", () => {
        activeIndex = i;
        [...strip.children].forEach((c, ci) => c.classList.toggle("active", ci === i));
        renderSingle(results[i]);
      });
      strip.appendChild(img);
    });
  }

  function animateCount(node, target) {
    // Tally the hero count up from zero, like a settling instrument readout.
    if (reducedMotion) { node.textContent = String(target); return; }
    const duration = 650;
    const start = performance.now();
    node.classList.remove("count-pop");
    void node.offsetWidth;
    node.classList.add("count-pop");
    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      node.textContent = String(Math.round(target * eased));
      if (t < 1) requestAnimationFrame(tick);
      else node.textContent = String(target);
    }
    requestAnimationFrame(tick);
  }

  function renderSingle(result) {
    animateCount(el("count-value"), result.larvae_count);
    el("r-avgconf").textContent = result.average_confidence.toFixed(3);
    el("r-time").textContent = `${Math.round(result.processing_time_ms)} ms`;
    el("r-tiles").textContent = String(result.tiles_scanned);
    el("r-dups").textContent = String(result.duplicates_merged);

    renderConfBars(result.confidence_distribution);
    renderDetectionList(result.detections);

    const img = el("result-image");
    img.onload = () => renderBoxes(result);
    img.src = `data:image/png;base64,${result.annotated_image}`;
    resetZoom();
  }

  function renderConfBars(distribution) {
    const wrap = el("conf-bars");
    wrap.innerHTML = "";
    const max = Math.max(1, ...distribution);
    distribution.forEach((count) => {
      const bar = document.createElement("div");
      bar.className = "conf-bar" + (count > 0 ? " filled" : "");
      bar.style.height = `${Math.max(1, (count / max) * 100)}%`;
      bar.title = String(count);
      wrap.appendChild(bar);
    });
  }

  function renderDetectionList(detections) {
    const list = el("detection-list");
    list.innerHTML = "";
    detections.forEach((d) => {
      const li = document.createElement("li");
      const confClass = d.confidence < 0.5 ? "det-conf-low" : "";
      li.innerHTML =
        `<span>#${d.id}</span>` +
        `<span class="${confClass}">${d.confidence.toFixed(2)}</span>`;
      const locate = document.createElement("button");
      locate.className = "det-locate";
      locate.type = "button";
      locate.textContent = "locate";
      locate.addEventListener("click", () => highlightBox(d.id));
      li.appendChild(locate);
      list.appendChild(li);
    });
  }

  function renderBoxes(result) {
    const layer = el("box-layer");
    layer.innerHTML = "";
    const w = result.image_width;
    const h = result.image_height;
    const boxNodes = [];
    result.detections.forEach((d) => {
      const box = document.createElement("div");
      box.className = "det-box";
      box.dataset.id = String(d.id);
      box.style.left = `${(d.x1 / w) * 100}%`;
      box.style.top = `${(d.y1 / h) * 100}%`;
      box.style.width = `${((d.x2 - d.x1) / w) * 100}%`;
      box.style.height = `${((d.y2 - d.y1) / h) * 100}%`;
      box.style.visibility = "hidden";
      const tag = document.createElement("span");
      tag.className = "det-tag mono";
      tag.textContent = d.confidence.toFixed(2);
      box.appendChild(tag);
      layer.appendChild(box);
      boxNodes.push({ node: box, cy: (d.y1 + d.y2) / 2 / h });
    });
    playScanSweep(boxNodes);
  }

  function playScanSweep(boxNodes) {
    const scan = el("scan-line");
    if (reducedMotion) {
      boxNodes.forEach((b) => (b.node.style.visibility = "visible"));
      return;
    }
    scan.classList.remove("sweeping");
    void scan.offsetWidth; // restart animation
    scan.classList.add("sweeping");
    boxNodes.forEach((b) => {
      setTimeout(() => {
        b.node.style.visibility = "visible";
        b.node.classList.add("materialize");
      }, Math.max(0, b.cy * SWEEP_MS));
    });
  }

  function highlightBox(id) {
    const box = document.querySelector(`.det-box[data-id="${id}"]`);
    if (!box) return;
    document.querySelectorAll(".det-box.active").forEach((b) => b.classList.remove("active"));
    box.style.visibility = "visible";
    box.classList.add("active");
    box.scrollIntoView({ block: "nearest", behavior: reducedMotion ? "auto" : "smooth" });
  }

  // ---------------- Image toolbar ----------------
  let zoom = 1;
  function applyZoom() {
    const t = `scale(${zoom})`;
    el("result-image").style.transform = t;
    el("box-layer").style.transform = t;
    el("box-layer").style.transformOrigin = "top left";
    el("result-image").style.transformOrigin = "top left";
  }
  function resetZoom() { zoom = 1; applyZoom(); }
  el("btn-zoom-in").addEventListener("click", () => { zoom = Math.min(zoom + 0.25, 4); applyZoom(); });
  el("btn-zoom-out").addEventListener("click", () => { zoom = Math.max(zoom - 0.25, 1); applyZoom(); });
  el("btn-toggle-boxes").addEventListener("click", (e) => {
    const layer = el("box-layer");
    const hidden = layer.classList.toggle("hide-boxes");
    e.currentTarget.setAttribute("aria-pressed", String(!hidden));
  });
  el("btn-toggle-labels").addEventListener("click", (e) => {
    const layer = el("box-layer");
    const hidden = layer.classList.toggle("hide-labels");
    e.currentTarget.setAttribute("aria-pressed", String(!hidden));
  });
  el("btn-download").addEventListener("click", () => {
    const result = lastResults[activeIndex];
    if (!result) return;
    const a = document.createElement("a");
    a.href = `data:image/png;base64,${result.annotated_image}`;
    a.download = (result.filename || "specimen") + "_annotated.png";
    a.click();
  });

  // ---------------- Report actions ----------------
  el("btn-export").addEventListener("click", () => {
    const payload = lastResults.map((r) => ({ ...r, annotated_image: undefined }));
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "sldcs_report.json";
    a.click();
    URL.revokeObjectURL(a.href);
  });
  el("btn-again").addEventListener("click", () => {
    fileInput.value = "";
    showScreen("tray");
  });

  // ---------------- Instrument info ----------------
  el("model-tag").addEventListener("click", showInfo);
  el("btn-info-back").addEventListener("click", () => showScreen("tray"));

  async function showInfo() {
    try {
      const info = await (await fetch("/model/info")).json();
      const body = el("info-table-body");
      const rows = [
        ["Version", info.version],
        ["Source", info.source],
        ["Trained on project data", info.trained_on_project_data ? "Yes" : "No"],
        ["Status", info.status],
        ["Device", info.device],
        ["Detectable classes", String(info.class_count)],
      ];
      body.innerHTML = rows
        .map(([k, v]) => `<tr><th>${k}</th><td>${escapeHtml(String(v))}</td></tr>`)
        .join("");
      el("info-note").textContent = info.note;
      showScreen("info");
    } catch (err) {
      /* If info is unavailable the instrument is offline; stay on tray. */
    }
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  }

  initInstrumentState();
})();
