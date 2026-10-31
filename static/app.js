"use strict";
// Model Lab front end: plain JS, no build step, no CDN scripts (works on an offline Jetson network).

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const isNum = v => typeof v === "number" && Number.isFinite(v);
const fmt = (v, d = 1) => (isNum(v) ? v.toFixed(d) : "–");
const trunc = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
const ic = name => `<svg class="ic"><use href="#i-${name}"/></svg>`;
const COLORS = ["#2f5bff", "#f0643c", "#16a864", "#e0a100", "#8b5cf6", "#0ea5b7", "#e04891", "#64748b"];
const dig = (o, path) => path.split(".").reduce((x, k) => (x == null ? undefined : x[k]), o);
const VLABEL = { pt_fp32: "PyTorch FP32", onnx_fp32: "ONNX FP32", trt_fp32: "TensorRT FP32", trt_fp16: "TensorRT FP16", trt_int8: "TensorRT INT8" };
const precTag = p => `<span class="tag ${esc((p || "").toLowerCase())}">${esc(p)}</span>`;

const state = { models: [], videos: [], env: null, jobs: [], disk: null };

async function api(url, opts = {}) {
  const r = await fetch(url, opts);
  let d = null;
  try { d = await r.json(); } catch { /* not json */ }
  if (!r.ok) throw new Error((d && d.error) || `${r.status} ${r.statusText}`);
  return d;
}

function toast(text, kind = "") {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.textContent = text;
  $("#toasts").append(t);
  setTimeout(() => t.remove(), 4000);
}

// POST a FormData with upload progress (models and videos can be large).
function send(url, data, onProgress) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("POST", url);
    x.upload.onprogress = e => e.lengthComputable && onProgress && onProgress(Math.round((100 * e.loaded) / e.total));
    x.onload = () => {
      let d = {};
      try { d = JSON.parse(x.responseText); } catch { /* ignore */ }
      x.status < 400 ? resolve(d) : reject(new Error(d.error || `${x.status} ${x.statusText}`));
    };
    x.onerror = () => reject(new Error("Network error: is the server still running?"));
    x.send(data);
  });
}

async function submitForm(form, url, check) {
  const msg = $(".msg", form), btn = $("button[type=submit]", form);
  msg.textContent = "";
  msg.style.color = "";
  const data = new FormData(form);
  const problem = check && check(data);
  if (problem) { msg.textContent = problem; return null; }
  btn.disabled = true;
  try {
    const hasFile = [...data.values()].some(v => v instanceof File && v.size);
    return await send(url, data, hasFile ? p => { msg.style.color = "var(--muted)"; msg.textContent = `Uploading ${p}%`; } : null);
  } catch (e) {
    msg.style.color = "";
    msg.textContent = e.message;
    return null;
  } finally {
    btn.disabled = false;
    if (msg.textContent.startsWith("Uploading")) msg.textContent = "";
  }
}

// ---------------------------------------------------------------------------------- layout
const PAGES = ["overview", "convert", "infer", "compare", "models", "videos"];
function showTab(name, sub) {
  if (!PAGES.includes(name)) name = "overview";
  $$(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
  $$(".panel").forEach(p => p.classList.toggle("active", p.id === "tab-" + name));
  $("#pageTitle").textContent = $(`.tab[data-tab="${name}"] span`).textContent;
  history.replaceState(null, "", "#" + name + (sub ? "/" + sub : ""));
  if (!sub) window.scrollTo({ top: 0 });
}
$$(".tab").forEach(t => t.addEventListener("click", () => showTab(t.dataset.tab)));
document.addEventListener("click", ev => {
  const g = ev.target.closest("[data-goto]");
  if (g) showTab(g.dataset.goto);
});
showTab(...location.hash.slice(1).split("/"));  // #compare/<run-id> links straight to a result

function bindShow(form) {
  const update = () => $$("[data-show]", form).forEach(el => {
    const [n, v] = el.dataset.show.split("=");
    const c = form.querySelector(`[name="${n}"]:checked`);
    el.hidden = !(c && c.value === v);
  });
  form.addEventListener("change", update);
  form.addEventListener("reset", () => setTimeout(update));
  update();
}

// drag & drop file fields: the transparent <input type=file> covers the whole box
function bindDrop(drop) {
  const inp = $("input", drop), f = $(".drop-f", drop);
  const update = () => {
    const file = inp.files[0];
    drop.classList.toggle("has-file", !!file);
    f.innerHTML = file ? `${esc(file.name)}<small>${(file.size / 1e6).toFixed(1)} MB · click to change</small>` : "";
  };
  inp.addEventListener("change", update);
  inp.addEventListener("dragenter", () => drop.classList.add("over"));
  ["dragleave", "drop"].forEach(e => inp.addEventListener(e, () => drop.classList.remove("over")));
  inp.form && inp.form.addEventListener("reset", () => setTimeout(update));
  drop._update = update;
}

$$(".sourceBox").forEach(b => b.append($("#sourceTpl").content.cloneNode(true)));
$$(".settingsBox").forEach(b => b.append($("#settingsTpl").content.cloneNode(true)));
$$("form").forEach(bindShow);
$$(".drop").forEach(bindDrop);

// theme: system → light → dark
function paintThemeBtn() {
  const dark = document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  $("#themeBtn").innerHTML = `${ic(dark ? "sun" : "moon")}<span>${dark ? "Light" : "Dark"} theme</span>`;
  return dark;
}
$("#themeBtn").onclick = () => {
  const next = paintThemeBtn() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch { /* private mode */ }
  paintThemeBtn();
};
paintThemeBtn();

// ------------------------------------------------------------------------------ environment
async function loadEnv() {
  let d;
  try { d = await api("/api/env"); } catch {
    $("#env").innerHTML = `<span class="pill bad">Server not reachable</span>`;
    $("#sideDevice").innerHTML = `<span class="dot bad"></span><span>Offline</span>`;
    return;
  }
  const e = d.env;
  state.env = e;
  state.disk = d.disk_free_gb;
  state.storage = d.storage;
  const pills = [];
  if (e) {
    const ortGpu = (e.onnxruntime_providers || []).some(p => /CUDA|Tensorrt/.test(p));
    pills.push(e.cuda ? `<span class="pill ok"><span class="dot ok"></span>CUDA ${esc(e.cuda_version || "")}</span>` : `<span class="pill bad">No CUDA</span>`);
    pills.push(e.tensorrt ? `<span class="pill ok">TensorRT ${esc(e.tensorrt)}</span>` : `<span class="pill bad" title="Engines can only be built on the Jetson">No TensorRT</span>`);
    pills.push(e.onnxruntime ? `<span class="pill ${ortGpu ? "ok" : "warn"}" title="${esc((e.onnxruntime_providers || []).join(", "))}">ORT ${ortGpu ? "GPU" : "CPU"}</span>` : `<span class="pill warn">No ONNX Runtime</span>`);
    if (e.power_mode) pills.push(`<span class="pill" title="sudo nvpmodel -m 0 && sudo jetson_clocks for stable numbers">${ic("zap")}${esc(e.power_mode)}</span>`);
    if (!e.ffmpeg_h264) pills.push(`<span class="pill warn" title="sudo apt install ffmpeg">No ffmpeg</span>`);
    $("#sideDevice").innerHTML = `<span class="dot ${e.cuda && e.tensorrt ? "ok" : "warn"}"></span><span title="${esc(e.device)}">${esc(e.device)}</span>`;
  } else {
    pills.push(`<span class="pill">Reading device…</span>`);
  }
  pills.push(`<span class="pill">${ic("disk")}${d.disk_free_gb} GB free</span>`);
  $("#env").innerHTML = pills.join("");
  renderOverview();
  const q = d.queue, qel = $("#queue");
  qel.classList.toggle("busy", !!q.running);
  qel.innerHTML = q.running ? `<span class="spin"></span>${esc(trunc(q.running.title, 42))}${q.queued ? ` · +${q.queued}` : ""}`
    : (q.queued ? `${q.queued} queued` : `<span class="dot ok" style="box-shadow:none"></span>Idle`);
  renderDevice();
}
$("#envRefresh").onclick = async () => {
  await api("/api/env/refresh", { method: "POST" });
  toast("Reading device info again (takes a few seconds)");
  setTimeout(loadEnv, 8000);
};

function renderDevice() {
  const e = state.env;
  if (!e) return;
  const ok = v => `<span class="dot ${v ? "ok" : "bad"}"></span>`;
  const ortGpu = (e.onnxruntime_providers || []).some(p => /CUDA|Tensorrt/.test(p));
  const rows = [
    ["Device", esc(e.device)], ["L4T / JetPack", esc([e.l4t, e.jetpack && "JetPack " + e.jetpack.split("+")[0]].filter(Boolean).join(" · ") || "–")],
    ["Power mode", esc(e.power_mode || "–")], ["GPU", `${ok(e.cuda)}${esc(e.gpu || "not available")}`],
    ["CUDA / TensorRT", `${ok(e.cuda && e.tensorrt)}${esc(e.cuda_version || "–")} / ${esc(e.tensorrt || "not installed")}`],
    ["PyTorch / Ultralytics", `${esc(e.torch || "–")} / ${esc(e.ultralytics || "–")}`],
    ["ONNX Runtime", `<span class="dot ${ortGpu ? "ok" : "warn"}"></span>${esc(e.onnxruntime || "not installed")} ${ortGpu ? "(GPU)" : "(CPU only)"}`],
    ["Video encoding", `${ok(e.ffmpeg_h264)}${e.ffmpeg_h264 ? "H.264 (plays in browser)" : "mp4v only: install ffmpeg"}`],
    ["Power readings", `${ok(e.tegrastats)}${e.tegrastats ? "tegrastats" : "not available (CPU/RAM only)"}`],
    ["Storage", `${state.disk} GB free${state.storage ? ` · models ${state.storage.models_mb} MB · videos ${state.storage.videos_mb} MB · runs ${state.storage.jobs_mb} MB` : ""}`],
  ];
  $("#ovDevice").innerHTML = `<dl class="kv">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
}

// ---------------------------------------------------------------------------------- overview
function renderOverview() {
  const variants = state.models.flatMap(g => g.variants);
  const engines = variants.filter(v => v.variant.startsWith("trt")).length;
  const runs = state.jobs.filter(j => j.kind !== "convert");
  const kpi = (icon, label, value, sub) => `<div class="kpi"><div class="kic">${ic(icon)}</div><div><span>${label}</span><b>${value}</b>${sub ? `<small>${sub}</small>` : ""}</div></div>`;
  $("#ovKpis").innerHTML = [
    kpi("layers", "Models", state.models.length, `${variants.length} files`),
    kpi("cpu", "TensorRT engines", engines, "built on this device"),
    kpi("film", "Test videos", state.videos.length, `${state.videos.reduce((a, v) => a + (v.frames || 0), 0).toLocaleString()} frames`),
    kpi("chart", "Saved runs", runs.length, `${runs.filter(j => j.kind === "compare").length} comparisons`),
    kpi("disk", "Free space", state.disk != null ? `${state.disk} GB` : "–", ""),
  ].join("");
  const step = (tab, icon, title, text) => `<button class="wstep" data-goto="${tab}"><span class="n">${ic(icon)}</span><span><b>${title}</b><small>${text}</small></span>${ic("arrow")}</button>`;
  $("#ovSteps").innerHTML = [
    step("convert", "convert", "1 · Convert", `Upload a .pt and build ONNX and TensorRT FP32 / FP16. ${state.models.length} model${state.models.length === 1 ? "" : "s"} in the library.`),
    step("infer", "play", "2 · Inference", "Run one model on a video, image or camera and check the output."),
    step("compare", "chart", "3 · Compare", "Benchmark several models on the same video and export the results."),
  ].join("");
  const recent = state.jobs.slice(0, 8);
  $("#ovRuns").innerHTML = recent.length ? `<div class="tablewrap"><table><tr><th>Run</th><th>Type</th><th>Source</th><th>Created</th><th>Status</th><th></th></tr>
    ${recent.map(j => `<tr><td><b>${esc(trunc(j.title, 60))}</b></td><td>${{ convert: "Conversion", infer: "Inference", compare: "Comparison" }[j.kind]}</td>
      <td>${esc(j.source || "–")}</td><td><small>${esc(j.created)}</small></td><td><span class="state ${j.state}">${j.state}</span></td>
      <td class="rowact">${j.kind === "convert" ? `<button class="btn small" data-goto="models">Library</button>` : `<button class="btn small" data-openrun="${j.kind}/${j.id}">Open</button>`}</td></tr>`).join("")}</table></div>`
    : `<p class="empty">No runs yet. Start with <b>Convert</b>.</p>`;
}
$("#ovRuns").addEventListener("click", ev => {
  const b = ev.target.closest("[data-openrun]");
  if (!b) return;
  const [kind, id] = b.dataset.openrun.split("/");
  showTab(kind);
  openResult(id, kind);
});

async function loadJobs() {
  state.jobs = await api("/api/jobs");
  renderOverview();
}

// ------------------------------------------------------------------------------- models
async function loadModels() {
  state.models = await api("/api/models");
  renderLibrary();
  fillModelInputs();
  renderOverview();
}

function renderLibrary() {
  const n = state.models.length;
  $("#navModels").textContent = n || "";
  $("#miniLibrary").innerHTML = n ? `<div class="minilist">${state.models.map(g => `<div><b>${esc(g.name)}</b><span class="tags">${g.variants.map(v => precTag(v.variant.startsWith("pt") ? "PT" : v.variant.startsWith("onnx") ? "ONNX" : v.precision || v.variant.split("_")[1].toUpperCase())).join("")}</span></div>`).join("")}</div>`
    : `<p class="empty">Converted models will appear here.</p>`;
  const el = $("#library");
  if (!n) {
    el.innerHTML = `<div class="card"><p class="empty">No models yet. <button class="btn small primary" data-goto="convert">${ic("plus")}Convert your first model</button></p></div>`;
    return;
  }
  el.innerHTML = state.models.map(g => {
    const i = g.info || {};
    const names = Object.values(i.names || {});
    const facts = [i.arch && `<span class="tag">${esc(i.arch)}</span>`, i.task && `<span class="tag">${esc(i.task)}</span>`,
      `<span class="tag">${names.length} classes</span>`, i.params_m && `<span class="tag">${i.params_m} M params</span>`,
      i.gflops && `<span class="tag">${i.gflops} GFLOPs</span>`, i.train_imgsz && `<span class="tag">trained @ ${i.train_imgsz}</span>`,
      i.imported && `<span class="tag">imported</span>`, g.busy && `<span class="state running">busy</span>`].filter(Boolean).join("");
    const rows = g.variants.map(v => `<tr>
      <td><b>${esc(v.format || v.label.split(" ")[0])}</b></td><td>${precTag(v.precision || v.label.split(" ")[1])}</td>
      <td class="num">${fmt(v.size_mb, 1)} MB</td><td class="num">${esc(v.imgsz)}</td>
      <td>${esc(v.built_on || "–")}${v.tensorrt ? ` <small>· TRT ${esc(v.tensorrt)}</small>` : ""}</td>
      <td class="num">${v.build_s ? fmt(v.build_s, 0) + " s" : "–"}</td><td><small>${esc(v.created || "")}</small></td>
      <td class="rowact"><button class="btn small" data-run="${esc(v.key)}">${ic("play")}Run</button>
        <a class="btn small icon" title="Download" href="/files/models/${encodeURIComponent(g.name)}/${encodeURIComponent(v.file)}">${ic("download")}</a>
        <button class="btn small icon danger" title="Delete this file" data-del="${esc(g.name)}/${esc(v.variant)}" ${g.busy ? "disabled" : ""}>${ic("trash")}</button></td></tr>`).join("");
    return `<div class="group"><header>
      <div class="ghead"><div class="gavatar">${esc(g.name.slice(0, 2))}</div><div><b>${esc(g.name)}</b><div class="facts">${facts}</div></div></div>
      <div class="actions"><button class="btn small" data-add="${esc(g.name)}" ${g.variants.some(v => v.variant === "pt_fp32") ? "" : "disabled"}>${ic("plus")}Add formats</button>
      <button class="btn small danger" data-delg="${esc(g.name)}" ${g.busy ? "disabled" : ""}>${ic("trash")}Delete model</button></div></header>
      <div class="tablewrap"><table><tr><th>Format</th><th>Precision</th><th class="num">Size</th><th class="num">Input</th><th>Built on</th><th class="num">Build time</th><th>Created</th><th></th></tr>${rows}</table></div>
      ${names.length ? `<div class="classes">${names.slice(0, 80).map(c => `<span class="tag">${esc(c)}</span>`).join("")}</div>` : ""}</div>`;
  }).join("");
}

$("#library").addEventListener("click", async ev => {
  const b = ev.target.closest("button");
  if (!b) return;
  if (b.dataset.run) {
    $("#inferForm [name=model]").value = b.dataset.run;
    syncImgsz($("#inferForm"));
    showTab("infer");
  } else if (b.dataset.add) {
    const f = $("#convertForm");
    f.querySelector("[name=source][value=existing]").checked = true;
    f.group.value = b.dataset.add;
    f.dispatchEvent(new Event("change"));
    showTab("convert");
  } else if (b.dataset.del || b.dataset.delg) {
    const what = b.dataset.del || b.dataset.delg;
    if (!confirm(`Delete ${b.dataset.del ? "this file" : "the whole model and all its formats"}: ${what}?`)) return;
    try { await api(`/api/models/${what}`, { method: "DELETE" }); toast(`Deleted ${what}`); } catch (e) { toast(e.message, "bad"); }
    loadModels();
  }
});
$("#refreshModels").onclick = loadModels;

function fillModelInputs() {
  const groups = state.models;
  $$(".groupSelect").forEach(s => {
    const cur = s.value;
    s.innerHTML = groups.filter(g => g.variants.some(v => v.variant === "pt_fp32")).map(g => `<option>${esc(g.name)}</option>`).join("") || `<option value="">No models with a .pt yet</option>`;
    if (cur) s.value = cur;
  });
  $("#groupNames").innerHTML = groups.map(g => `<option value="${esc(g.name)}">`).join("");
  $$(".modelSelect").forEach(s => {
    const cur = s.value;
    s.innerHTML = groups.map(g => `<optgroup label="${esc(g.name)}">${g.variants.map(v => `<option value="${esc(v.key)}">${esc(g.name)} · ${esc(v.label)} (${v.imgsz})</option>`).join("")}</optgroup>`).join("")
      || `<option value="">No models yet — convert one first</option>`;
    if (cur && [...s.options].some(o => o.value === cur)) s.value = cur;
  });
  syncImgsz($("#inferForm"));
  const box = $("#compareModels");
  const checked = new Set($$("input:checked", box).map(i => i.value));
  box.innerHTML = groups.map(g => `<div class="grp"><div class="grpname"><span>${esc(g.name)}</span><button type="button" class="btn ghost small" data-all="${esc(g.name)}">Select all</button></div>
    <div>${g.variants.map(v => `<label><input type="checkbox" name="models" value="${esc(v.key)}" ${checked.has(v.key) ? "checked" : ""}> ${esc(v.label)} <small>${v.imgsz}</small></label>`).join("")}</div></div>`).join("")
    || `<p class="empty">No models yet.</p>`;
  syncBaseline();
}

const findVariant = key => state.models.flatMap(g => g.variants).find(v => v.key === key);

// exported models have a fixed input size; only .pt accepts another one
function syncImgsz(form) {
  const v = findVariant(form.model && form.model.value), inp = form.imgsz;
  if (!v) return;
  inp.disabled = v.variant !== "pt_fp32";
  inp.placeholder = inp.disabled ? `fixed: ${v.imgsz}` : `auto (${v.imgsz})`;
  if (inp.disabled) inp.value = "";
}
$("#inferForm").model.addEventListener("change", e => syncImgsz(e.target.form));

function syncBaseline() {
  const sel = $("#baseline"), cur = sel.value;
  const keys = $$("#compareModels input:checked").map(i => i.value);
  sel.innerHTML = keys.map(k => `<option value="${esc(k)}">${esc(k.replace("/", " · "))}</option>`).join("") || `<option value="">Tick models first</option>`;
  if (keys.includes(cur)) sel.value = cur;
  else sel.value = keys.find(k => k.endsWith("/pt_fp32")) || keys[0] || "";
}
$("#compareModels").addEventListener("change", syncBaseline);
$("#compareModels").addEventListener("click", ev => {
  const b = ev.target.closest("[data-all]");
  if (!b) return;
  const boxes = $$("input", b.closest(".grp"));
  const on = !boxes.every(i => i.checked);
  boxes.forEach(i => (i.checked = on));
  syncBaseline();
});

// ------------------------------------------------------------------------------- videos
async function loadVideos() {
  state.videos = await api("/api/videos");
  $("#navVideos").textContent = state.videos.length || "";
  $("#videoList").innerHTML = state.videos.length ? `<div class="tablewrap"><table><tr><th>Video</th><th class="num">Resolution</th><th class="num">FPS</th><th class="num">Frames</th><th class="num">Duration</th><th class="num">Size</th><th></th></tr>
    ${state.videos.map(v => `<tr><td><b>${esc(v.name)}</b></td><td class="num">${v.width ? `${v.width}×${v.height}` : "–"}</td><td class="num">${v.fps || "–"}</td>
      <td class="num">${v.frames ? v.frames.toLocaleString() : "–"}</td><td class="num">${v.duration_s ? `${v.duration_s} s` : "–"}</td><td class="num">${v.size_mb} MB</td>
      <td class="rowact"><a class="btn small icon" title="Preview / download" href="/files/videos/${encodeURIComponent(v.name)}" target="_blank">${ic("play")}</a>
      <button class="btn small icon danger" title="Delete" data-delv="${esc(v.name)}">${ic("trash")}</button></td></tr>`).join("")}</table></div>`
    : `<p class="empty">No videos yet. Upload one above.</p>`;
  $$(".videoSelect").forEach(s => {
    const cur = s.value;
    s.innerHTML = state.videos.map(v => `<option value="${esc(v.name)}">${esc(v.name)}${v.frames ? ` · ${v.frames} frames` : ""}${v.width ? ` · ${v.width}×${v.height}` : ""}</option>`).join("") || `<option value="">Library is empty — upload a video</option>`;
    if (cur) s.value = cur;
  });
  renderOverview();
}
$("#videoList").addEventListener("click", async ev => {
  const b = ev.target.closest("[data-delv]");
  if (!b || !confirm(`Delete video ${b.dataset.delv}?`)) return;
  try { await api(`/api/videos/${encodeURIComponent(b.dataset.delv)}`, { method: "DELETE" }); toast("Video deleted"); } catch (e) { toast(e.message, "bad"); }
  loadVideos();
});
$("#refreshVideos").onclick = loadVideos;
$("#videoForm").addEventListener("submit", async ev => {
  ev.preventDefault();
  const r = await submitForm(ev.target, "/api/videos", d => (d.get("file").size ? null : "Choose a video"));
  if (r) { ev.target.reset(); toast(`Uploaded ${r.name}`); loadVideos(); }
});

// --------------------------------------------------------------------------------- jobs
const ICON = { done: "✓", failed: "✕", running: "", pending: "", cancelled: "–", queued: "" };

function renderJob(el, j) {
  const logOpen = $("details.log", el)?.open;
  const steps = j.steps.map((s, k) => {
    const st = s.state || "pending";
    let bar = "";
    const p = s.progress;
    if (st === "running" && p) {
      if (p.total && p.frame != null) bar = `<div class="bar"><i style="width:${Math.min(100, (100 * p.frame) / p.total)}%"></i></div><small>${p.frame.toLocaleString()} / ${p.total.toLocaleString()} frames</small>`;
      else if (p.frame != null) bar = `<small>${p.frame.toLocaleString()} frames</small>`;
      else if (p.current) bar = `<div class="bar"><i style="width:${(100 * (p.step - 1)) / Math.max(p.total, 1)}%"></i></div><small>Step ${p.step}/${p.total}: ${esc(VLABEL[p.current] || p.current)}</small>`;
    }
    const mark = st === "running" ? `<span class="spin" style="width:10px;height:10px"></span>` : (ICON[st] || k + 1);
    return `<li><span class="ic-st ${st}">${mark}</span><div>${esc(s.label)}${bar}</div></li>`;
  }).join("");
  const active = ["queued", "running"].includes(j.state);
  el.innerHTML = `<div class="job"><div class="jh"><b>${esc(j.title)}</b><span class="actions"><span class="state ${j.state}">${j.state}</span>
    ${active ? `<button class="btn small danger" data-cancel="${j.id}">${ic("x")}Cancel</button>` : ""}</span></div>
    <div class="jbody">
    ${j.state === "queued" ? `<div class="note info" style="margin:0 0 10px">${ic("info")}Waiting for the running job to finish (one job at a time).</div>` : ""}
    <ul class="steps">${steps || `<li><span class="ic-st"><span class="spin" style="width:10px;height:10px"></span></span><div>Starting…</div></li>`}</ul>
    ${(j.warnings || []).map(w => `<div class="note warn">${ic("info")}<span>${esc(w)}</span></div>`).join("")}
    ${j.error ? `<div class="note bad">${ic("x")}<span>${esc(j.error)}</span></div>` : ""}
    ${j.kind === "convert" && j.state === "done" ? `<div class="note ok">${ic("info")}<span>Done. The new files are in the <a href="#models" data-goto="models">model library</a>.</span></div>` : ""}
    <details class="log" ${logOpen ? "open" : ""}><summary>Console log</summary><pre>${esc(j.log || "(empty)")}</pre></details></div></div>`;
  const pre = $("pre", el);
  pre.scrollTop = pre.scrollHeight;
  const c = $("[data-cancel]", el);
  if (c) c.onclick = async () => { if (confirm("Cancel this job?")) await api(`/api/jobs/${j.id}/cancel`, { method: "POST" }); };
}

function watchJob(id, el, onDone) {
  clearTimeout(el._timer);
  el._job = id;
  const tick = async () => {
    if (el._job !== id) return;
    let j;
    try { j = await api(`/api/jobs/${id}`); } catch (e) { el.innerHTML = `<div class="note bad">${esc(e.message)}</div>`; return; }
    renderJob(el, j);
    if (["done", "failed", "cancelled"].includes(j.state)) {
      loadEnv();
      loadJobs();
      toast(`${j.title}: ${j.state}`, j.state === "done" ? "" : "bad");
      onDone && onDone(j);
      return;
    }
    el._timer = setTimeout(tick, 1500);
  };
  tick();
}

const UI = kind => ({ result: $(`#${kind}Result`), history: $(`#${kind}History`), job: $(`#${kind}Job`) });

async function loadHistory(kind) {
  const { history: el, result } = UI(kind);
  const list = await api(`/api/jobs?kind=${kind}`);
  el.innerHTML = list.map(j => `<div class="hitem ${result._shown === j.id ? "active" : ""}" data-open="${j.id}" data-state="${j.state}">
    <div class="t"><b>${esc(j.title)}</b><small>${esc(j.created)} · ${j.models} model${j.models === 1 ? "" : "s"}</small></div>
    <div class="r"><span class="state ${j.state}">${j.state}</span>${["queued", "running"].includes(j.state) ? "" : `<button class="btn small icon ghost" data-deljob="${j.id}" title="Delete run">${ic("trash")}</button>`}</div></div>`).join("")
    || `<p class="empty">Nothing yet.</p>`;
  el.onclick = async ev => {
    const del = ev.target.closest("[data-deljob]");
    if (del) {
      ev.stopPropagation();
      if (!confirm("Delete this run and all its files (videos, CSVs)?")) return;
      try { await api(`/api/jobs/${del.dataset.deljob}`, { method: "DELETE" }); toast("Run deleted"); } catch (e) { toast(e.message, "bad"); }
      if (result._shown === del.dataset.deljob) { result.innerHTML = ""; result._shown = null; }
      loadJobs();
      return loadHistory(kind);
    }
    const it = ev.target.closest("[data-open]");
    if (!it) return;
    if (["queued", "running"].includes(it.dataset.state)) watchJob(it.dataset.open, UI(kind).job, () => openResult(it.dataset.open, kind));
    else openResult(it.dataset.open, kind);
  };
}

async function openResult(id, kind) {
  const { result: el, job } = UI(kind);
  el._shown = id;
  try {
    const res = await api(`/api/jobs/${id}/results`);
    el.innerHTML = kind === "infer" ? renderSingle(res) : renderCompare(res);
    bindVideos(el);
    showTab(kind, id);
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch {
    el.innerHTML = `<div class="card"><div class="note bad">${ic("x")}<span>No results for this run: it failed before any model finished. See the console log above.</span></div></div>`;
    watchJob(id, job);
  }
  loadHistory(kind);
}

// -------------------------------------------------------------------------------- forms
const convertForm = $("#convertForm");
const nameField = form => form.elements.namedItem("name");
convertForm.weights.addEventListener("change", e => {
  const f = e.target.files[0];
  if (f && !nameField(convertForm).value) nameField(convertForm).value = f.name.replace(/\.pt$/i, "").replace(/[^A-Za-z0-9_.-]+/g, "_");
});
convertForm.addEventListener("submit", async ev => {
  ev.preventDefault();
  const r = await submitForm(convertForm, "/api/convert", d => {
    if (d.get("source") === "upload" && !d.get("weights").size) return "Choose a .pt file";
    if (d.get("source") === "existing" && !d.get("group")) return "Choose a model";
    return null;
  });
  if (!r) return;
  convertForm.reset();
  loadModels();
  loadJobs();
  watchJob(r.id, $("#convertJob"), loadModels);
});

const importForm = $("#importForm");
importForm.file.addEventListener("change", e => {
  const f = e.target.files[0];
  if (f && !nameField(importForm).value) nameField(importForm).value = f.name.replace(/\.(onnx|engine)$/i, "").replace(/[^A-Za-z0-9_.-]+/g, "_");
});
importForm.addEventListener("submit", async ev => {
  ev.preventDefault();
  const r = await submitForm(importForm, "/api/models/import", d => (d.get("file").size ? null : "Choose a file"));
  if (!r) return;
  importForm.reset();
  toast(`Imported as ${r.key.replace("/", " · ")}`);
  loadModels();
});

function checkSource(d) {
  const t = d.get("source_type");
  if (t === "library" && !d.get("video_name")) return "Choose a video (or upload one)";
  if (t === "upload" && !d.get("source_file").size) return "Choose a file to upload";
  if (t === "path" && !d.get("source_path").trim()) return "Enter a path, camera index or stream URL";
  return null;
}

function benchForm(kind, url, check) {
  const form = $(`#${kind}Form`);
  form.addEventListener("submit", async ev => {
    ev.preventDefault();
    const r = await submitForm(form, url, check);
    if (!r) return;
    if (form.source_type.value === "upload") { form.source_file.value = ""; $(".drop", form)._update(); loadVideos(); }
    const ui = UI(kind);
    ui.result.innerHTML = "";
    loadHistory(kind);
    loadJobs();
    watchJob(r.id, ui.job, () => openResult(r.id, kind));
  });
}
benchForm("infer", "/api/infer", d => (!d.get("model") ? "Choose a model" : checkSource(d)));
benchForm("compare", "/api/compare", d => (!d.getAll("models").length ? "Tick at least one model" : checkSource(d)));
$("#refreshHistory").onclick = () => loadHistory("compare");

// ------------------------------------------------------------------------------- charts
function niceTicks(min, max, n = 5) {
  if (max === min) max = min + 1;
  const raw = (max - min) / n, mag = 10 ** Math.floor(Math.log10(raw)), r = raw / mag;
  const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
  const ticks = [];
  for (let v = Math.floor(min / step) * step; v <= max + step * 0.5; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

function barChart(title, items, { unit = "", digits = 1, better } = {}) {
  items = items.filter(i => isNum(i.value));
  if (!items.length) return "";
  const W = 560, rowH = 30, labelW = 210, padR = 74, H = items.length * rowH + 4;
  const max = Math.max(...items.map(i => i.value)) || 1, span = W - labelW - padR;
  const bars = items.map((it, k) => {
    const y = k * rowH + 2, w = Math.max(2, (span * it.value) / max);
    return `<text x="${labelW - 10}" y="${y + 18}" text-anchor="end">${esc(trunc(it.label, 32))}</text>
      <rect class="track" x="${labelW}" y="${y + 7}" width="${span}" height="${rowH - 14}" rx="4"/>
      <rect x="${labelW}" y="${y + 7}" width="${w}" height="${rowH - 14}" rx="4" fill="${it.color}"/>
      <text x="${labelW + span + 8}" y="${y + 18}" style="fill:var(--text);font-weight:600">${fmt(it.value, digits)}${unit}</text>`;
  }).join("");
  return `<div class="chart"><h4>${esc(title)}${better ? `<small>${better === "high" ? "↑ higher" : "↓ lower"} is better</small>` : ""}</h4>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">${bars}</svg></div>`;
}

function lineChart(title, series, { xLabel = "frame", unit = "", zero = true } = {}) {
  series = series.filter(s => s.x && s.x.length);
  if (!series.length) return "";
  const W = 560, H = 230, L = 46, R = 10, T = 8, B = 30;
  const xs = series.flatMap(s => s.x), ys = series.flatMap(s => s.y).filter(isNum);
  if (!ys.length) return "";
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = zero ? 0 : Math.min(...ys), y1 = Math.max(...ys);
  if (x1 === x0) x1 = x0 + 1;
  const yt = niceTicks(y0, y1), xt = niceTicks(x0, x1, 6);
  y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]);
  const sx = x => L + ((x - x0) / (x1 - x0)) * (W - L - R), sy = y => T + (1 - (y - y0) / (y1 - y0 || 1)) * (H - T - B);
  const grid = yt.map(v => `<line class="grid" x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}"/><text x="${L - 6}" y="${sy(v) + 4}" text-anchor="end">${+v.toFixed(2)}</text>`).join("")
    + xt.filter(v => v >= x0 && v <= x1).map(v => `<text x="${sx(v)}" y="${H - 12}" text-anchor="middle">${+v.toFixed(1)}</text>`).join("");
  const paths = series.map(s => {
    let d = "", pen = false;
    s.x.forEach((x, i) => {
      const y = s.y[i];
      if (!isNum(y)) { pen = false; return; }
      d += `${pen ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`;
      pen = true;
    });
    return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="1.7" stroke-linejoin="round" opacity=".92"/>`;
  }).join("");
  const legend = series.length > 1 ? `<div class="legend">${series.map(s => `<span><i class="sw" style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")}</div>` : "";
  return `<div class="chart"><h4>${esc(title)}${unit ? `<small>${unit}</small>` : ""}</h4>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">${grid}${paths}
    <text x="${(L + W - R) / 2}" y="${H - 1}" text-anchor="middle">${esc(xLabel)}</text></svg>${legend}</div>`;
}

// ------------------------------------------------------------------------------ results
const S = (r, path) => dig(r.summary, path);
const fileUrl = (id, path, dl) => `/files/jobs/${id}/${path}${dl ? "?dl=1" : ""}`;
const sectionT = t => `<div class="section-t">${t}</div>`;

function downloads(res) {
  const id = res.job.id;
  return `<div class="downloads">
    <a class="btn small" href="${fileUrl(id, "summary.csv", 1)}">${ic("download")}summary.csv</a>
    <a class="btn small" href="${fileUrl(id, "comparison.json", 1)}">${ic("download")}JSON</a>
    <a class="btn small" href="/api/jobs/${id}/export">${ic("download")}All data .zip</a>
    <a class="btn small primary" href="/api/jobs/${id}/export?videos=1">${ic("download")}Data + videos .zip</a></div>`;
}

function resultHead(res, sub) {
  return `<div class="rhead"><div><h2>${esc(res.job.title)}</h2><div class="rmeta">${sub}</div>
    ${res.job.params.notes ? `<p class="muted" style="margin-top:8px">${esc(res.job.params.notes)}</p>` : ""}</div>${downloads(res)}</div>`;
}

function envBlock(res, r) {
  const e = res.env || {}, s = r && r.summary;
  const items = [
    ["Device", e.device], ["L4T / JetPack", [e.l4t, e.jetpack].filter(Boolean).join(" / ")], ["Power mode", e.power_mode],
    ["GPU", e.gpu], ["CUDA", e.cuda_version], ["TensorRT", e.tensorrt], ["PyTorch", e.torch], ["Ultralytics", e.ultralytics],
    ["ONNX Runtime", e.onnxruntime && `${e.onnxruntime} (${(e.onnxruntime_providers || []).join(", ")})`],
    ["Source", s && `${s.source.path}${s.source.width ? ` · ${s.source.width}×${s.source.height}` : ""}${s.source.fps ? ` · ${s.source.fps} fps` : ""}`],
    ["Settings", s && `conf ${s.settings.conf} · IoU ${s.settings.iou} · warm-up ${s.settings.warmup} · max frames ${s.settings.max_frames || "all"}`],
    ["Notes", res.job.params.notes],
  ].filter(([, v]) => v);
  return `${sectionT("Test setup")}<dl class="kv">${items.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`;
}

function mediaFor(res, r, i) {
  const id = res.job.id, o = S(r, "outputs") || {};
  if (o.image) return `<div class="media"><img src="${fileUrl(id, `results/${i}/${o.image}`)}" alt="annotated output"></div>`;
  if (!o.video) return "";
  const url = fileUrl(id, `results/${i}/${o.video}`);
  return o.video_codec === "h264"
    ? `<video src="${url}" controls muted preload="metadata" playsinline></video>`
    : `<div class="note warn" style="margin:12px">${ic("info")}<span>Saved as mp4v, which browsers cannot play. <a href="${url}?dl=1">Download the video</a>, or install ffmpeg for in-browser playback.</span></div>`;
}

function stat(label, value, unit = "", hl) {
  return `<div class="stat ${hl ? "hl" : ""}"><span>${esc(label)}</span><b>${value}</b>${unit ? `<small>${unit}</small>` : ""}</div>`;
}

function perClassTable(rows) {
  const classes = [...new Set(rows.flatMap(r => Object.keys(S(r, "detections.per_class") || {})))].sort();
  if (!classes.length) return "";
  return `${sectionT("Detections per class")}<div class="tbl-card tablewrap"><table><tr><th>Model</th>${classes.map(c => `<th class="num">${esc(c)}</th>`).join("")}</tr>
    ${rows.map(r => `<tr><td><i class="sw" style="background:${r.color}"></i>${esc(r.label)}</td>${classes.map(c => `<td class="num">${(S(r, "detections.per_class")?.[c] ?? 0).toLocaleString()}</td>`).join("")}</tr>`).join("")}</table></div>`;
}

function renderSingle(res) {
  const r = res.models[0];
  if (!r || !r.ok) return `<div class="card"><div class="note bad">${ic("x")}<span>The run failed. See the console log.</span></div></div>`;
  r.color = COLORS[0];
  const img = S(r, "source.type") === "image";
  const sys = S(r, "system") || {};
  const boxes = r.summary.boxes || [];
  const meta = [`<span class="pill">${esc(r.label)}</span>`, `<span class="pill">${S(r, "runtime.device") === "cpu" ? "CPU" : "GPU"}</span>`,
    `<span class="pill">imgsz ${S(r, "settings.imgsz")}</span>`, `<span class="pill">${esc(res.job.params.source_name || "")}</span>`,
    `<span class="pill">${esc(res.job.finished || "")}</span>`].join("");
  return `<div class="result"><div class="card">${resultHead(res, meta)}
    <div class="stats">
      ${stat("Model FPS", fmt(S(r, "fps_model"), 1), "", true)}
      ${stat("Latency p50", fmt(S(r, "latency_ms.p50"), 2), "ms")}
      ${stat("Inference p50", fmt(S(r, "inference_ms.p50"), 2), "ms")}
      ${img ? stat("Detections", boxes.length) : stat("Pipeline FPS", fmt(S(r, "fps_pipeline"), 1))}
      ${img ? "" : stat("Detections / frame", fmt(S(r, "detections.per_frame_mean"), 2))}
      ${isNum(sys.power_w_mean) ? stat("Power", fmt(sys.power_w_mean, 1), "W") : ""}
      ${isNum(sys.gpu_temp_max) ? stat("GPU temp max", fmt(sys.gpu_temp_max, 1), "°C") : ""}
      ${isNum(sys.gpu_util_mean) ? stat("GPU load", fmt(sys.gpu_util_mean, 0), "%") : ""}
    </div>
    ${mediaFor(res, r, 0)}
    ${img && boxes.length ? `${sectionT("Boxes")}<div class="tbl-card tablewrap"><table><tr><th>Class</th><th class="num">Confidence</th><th>Box (x1, y1, x2, y2)</th></tr>
      ${boxes.map(b => `<tr><td>${esc(b.class)}</td><td class="num">${fmt(b.conf, 3)}</td><td>${b.xyxy.map(v => v.toFixed(0)).join(", ")}</td></tr>`).join("")}</table></div>` : ""}
    ${img ? "" : `${sectionT("Over time")}<div class="charts">
      ${lineChart("Latency per frame", [{ name: r.label, color: COLORS[0], x: r.series.latency_ms?.[0], y: r.series.latency_ms?.[1] }], { unit: "ms" })}
      ${lineChart("Detections per frame", [{ name: r.label, color: COLORS[1], x: r.series.detections?.[0], y: r.series.detections?.[1] }])}
      ${lineChart("Power", [{ name: r.label, color: COLORS[3], x: r.series.power_w?.[0], y: r.series.power_w?.[1] }], { xLabel: "seconds", unit: "W" })}
      ${lineChart("GPU temperature", [{ name: r.label, color: COLORS[1], x: r.series.gpu_temp_c?.[0], y: r.series.gpu_temp_c?.[1] }], { xLabel: "seconds", unit: "°C", zero: false })}
    </div>`}
    ${perClassTable([r])}
    ${envBlock(res, r)}</div></div>`;
}

const COLS = [
  { h: "Size MB", v: r => S(r, "model.size_mb"), d: 1, better: "low" },
  { h: "Infer p50 ms", v: r => S(r, "inference_ms.p50"), d: 2, better: "low" },
  { h: "Infer p95 ms", v: r => S(r, "inference_ms.p95"), d: 2, better: "low" },
  { h: "Latency ms", v: r => S(r, "latency_ms.mean"), d: 2, better: "low", extra: r => ` <small>± ${fmt(S(r, "latency_ms.std"), 2)}</small>` },
  { h: "Model FPS", v: r => S(r, "fps_model"), d: 1, better: "high" },
  { h: "Pipeline FPS", v: r => S(r, "fps_pipeline"), d: 1, better: "high" },
  { h: "Speed-up", v: r => r.speedup, d: 2, better: "high", unit: "×" },
  { h: "Det / frame", v: r => S(r, "detections.per_frame_mean"), d: 2 },
  { h: "Agreement F1", v: r => r.agreement?.f1, d: 3, better: "high", skipBase: true },
  { h: "Mean IoU", v: r => r.agreement?.mean_iou, d: 3, better: "high", skipBase: true },
  { h: "Power W", v: r => S(r, "system.power_w_mean"), d: 1, better: "low" },
  { h: "FPS / W", v: r => S(r, "fps_per_watt"), d: 2, better: "high" },
  { h: "GPU %", v: r => S(r, "system.gpu_util_mean"), d: 0 },
  { h: "GPU °C max", v: r => S(r, "system.gpu_temp_max"), d: 1 },
  { h: "mAP50", v: r => r.val?.map50, d: 3, better: "high" },
  { h: "mAP50-95", v: r => r.val?.map50_95, d: 3, better: "high" },
];

function renderCompare(res) {
  const rows = res.models.map((r, i) => ({ ...r, color: COLORS[i % COLORS.length] }));
  const ok = rows.filter(r => r.ok), base = rows[res.baseline];
  const cols = COLS.filter(c => ok.some(r => isNum(c.v(r))));
  const best = cols.map(c => {
    const vals = ok.filter(r => !(c.skipBase && r === base)).map(c.v).filter(isNum);
    return c.better && vals.length > 1 ? (c.better === "high" ? Math.max(...vals) : Math.min(...vals)) : null;
  });
  const table = `<div class="tbl-card tablewrap"><table><tr><th>Model</th><th>Device</th>${cols.map(c => `<th class="num">${c.h}</th>`).join("")}</tr>
    ${rows.map(r => `<tr><td><i class="sw" style="background:${r.color}"></i><b>${esc(r.label)}</b>${r === base ? `<span class="base">BASELINE</span>` : ""}</td>
      ${r.ok ? `<td>${esc(S(r, "runtime.device") === "cpu" ? "CPU" : "GPU")}${(S(r, "runtime.onnx_providers") || [])[0] === "CPUExecutionProvider" ? ` <span class="tag int8">ORT CPU</span>` : ""}</td>`
        + cols.map((c, k) => { const v = c.v(r); return `<td class="num ${isNum(v) && v === best[k] ? "best" : ""}">${c.skipBase && r === base ? "<small>ref</small>" : fmt(v, c.d) + (isNum(v) && c.unit ? c.unit : "") + (c.extra && isNum(v) ? c.extra(r) : "")}</td>`; }).join("")
        : `<td colspan="${cols.length + 1}"><span class="state failed">failed</span> <small>see console log</small></td>`}</tr>`).join("")}</table></div>`;

  const bar = (title, fn, o) => barChart(title, ok.map(r => ({ label: r.label, value: fn(r), color: r.color })), o);
  const line = (title, key, o) => lineChart(title, ok.map(r => ({ name: r.label, color: r.color, x: r.series[key]?.[0], y: r.series[key]?.[1] })), o);
  const nonBase = ok.filter(r => r !== base);
  const charts = [
    bar("Model FPS", r => S(r, "fps_model"), { better: "high" }),
    bar("Inference time p50", r => S(r, "inference_ms.p50"), { unit: " ms", digits: 2, better: "low" }),
    bar("Speed-up vs baseline", r => r.speedup, { unit: "×", digits: 2, better: "high" }),
    barChart("Agreement with baseline (F1)", nonBase.map(r => ({ label: r.label, value: r.agreement?.f1, color: r.color })), { digits: 3, better: "high" }),
    bar("Average power", r => S(r, "system.power_w_mean"), { unit: " W", better: "low" }),
    bar("FPS per watt", r => S(r, "fps_per_watt"), { digits: 2, better: "high" }),
    bar("mAP50-95", r => r.val?.map50_95, { digits: 3, better: "high" }),
    line("Latency per frame", "latency_ms", { unit: "ms" }),
    line("Detections per frame", "detections", {}),
    line("Power over time", "power_w", { xLabel: "seconds", unit: "W" }),
    line("GPU temperature", "gpu_temp_c", { xLabel: "seconds", unit: "°C", zero: false }),
  ].join("");

  // headline tiles: fastest model and its gain over the baseline
  const fastest = ok.filter(r => isNum(S(r, "fps_model"))).sort((a, b) => S(b, "fps_model") - S(a, "fps_model"))[0];
  const agreeMin = nonBase.map(r => r.agreement?.f1).filter(isNum);
  const s0 = ok[0] && ok[0].summary;
  const tiles = fastest ? `<div class="stats">
    <div class="stat hl" title="${esc(fastest.label)}"><span>Fastest</span><b class="name">${esc(fastest.label.split(" · ")[0])}</b><small>${esc(fastest.label.split(" · ").slice(1).join(" · "))}</small></div>
    ${stat("Top model FPS", fmt(S(fastest, "fps_model"), 1))}
    ${stat("Speed-up vs baseline", fmt(fastest.speedup, 2), "×")}
    ${agreeMin.length ? stat("Lowest agreement F1", fmt(Math.min(...agreeMin), 3)) : ""}
    ${stat("Measured frames", s0 ? s0.measured_frames.toLocaleString() : "–")}
  </div>` : "";

  const vids = ok.map(r => ({ r, m: mediaFor(res, r, r.index) })).filter(x => x.m);
  const meta = [`<span class="pill">${ic("film")}${esc(res.job.params.source_name || "")}</span>`, `<span class="pill">${rows.length} models</span>`,
    `<span class="pill">${esc(res.job.finished || res.job.state)}</span>`].join("");
  return `<div class="result"><div class="card">
    ${resultHead(res, meta)}
    ${tiles}
    ${sectionT("Results")}${table}
    <p class="explain"><b>Model FPS</b> = 1000 ÷ (pre-process + inference + post-process ms): the number to compare between formats.
      <b>Pipeline FPS</b> also includes reading frames, drawing boxes and writing the video.
      <b>Agreement</b> compares each model's boxes with the baseline's (same class, IoU ≥ 0.5) on every frame; 1.000 means identical output.
      It shows how much a conversion changed the predictions, not accuracy against ground truth (use the optional mAP for that).
      Green cells mark the best value in each column.</p>
    ${sectionT("Charts")}<div class="charts">${charts}</div>
    ${perClassTable(ok)}
    ${vids.length ? `${sectionT("Output videos")}<div class="actions" style="margin-bottom:12px"><button class="btn small" data-v="play">${ic("play")}Play all</button><button class="btn small" data-v="pause">Pause all</button><button class="btn small" data-v="restart">${ic("refresh")}Restart all</button><small class="muted">Starts every video from the same moment.</small></div>
      <div class="videos">${vids.map(({ r, m }) => `<figure><figcaption><i class="sw" style="background:${r.color}"></i>${esc(r.label)}</figcaption>${m}</figure>`).join("")}</div>` : ""}
    ${envBlock(res, ok[0])}
  </div></div>`;
}

function bindVideos(el) {
  el.querySelectorAll("[data-v]").forEach(b => (b.onclick = () => {
    const vs = $$("video", el);
    if (!vs.length) return;
    if (b.dataset.v === "pause") return vs.forEach(v => v.pause());
    const t = b.dataset.v === "restart" ? 0 : vs[0].currentTime;
    vs.forEach(v => { v.currentTime = t; v.play().catch(() => {}); });
  }));
}

// --------------------------------------------------------------------------------- start
(async function init() {
  loadEnv();
  setInterval(loadEnv, 5000);
  await Promise.all([loadModels(), loadVideos(), loadJobs()]);
  loadHistory("infer");
  loadHistory("compare");
  const [tab, runId] = location.hash.slice(1).split("/");
  if (runId && (tab === "infer" || tab === "compare")) openResult(runId, tab);
  // re-attach to jobs that are still running after a page reload
  for (const j of state.jobs.filter(j => ["queued", "running"].includes(j.state)).reverse()) {
    if (j.kind === "convert") watchJob(j.id, $("#convertJob"), loadModels);
    else watchJob(j.id, UI(j.kind).job, () => openResult(j.id, j.kind));
  }
})();
