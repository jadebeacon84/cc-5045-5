"use strict";

const $ = (id) => document.getElementById(id);
let lastSchedule = null;
let lastMeta = null;
let currentJobId = null;
let pollTimer = null;

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
  return body;
}

function showError(msg) {
  const el = $("error-banner");
  if (!msg) {
    el.hidden = true;
    el.textContent = "";
  } else {
    el.hidden = false;
    el.textContent = msg;
  }
}

function readForm() {
  const indices = (raw) =>
    raw.split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  return {
    fps: Number($("fps").value),
    frame_index: Number($("frame_index").value),
    t_open_offset: Number($("t_open_offset").value),
    shutter_angle: Number($("shutter_angle").value),
    samples: Number($("samples").value),
    reference_samples: Number($("reference_samples").value),
    curve: $("curve").value,
    workers: Number($("workers").value),
    width: Number($("width").value),
    height: Number($("height").value),
    max_retries: Number($("max_retries").value),
    fail_indices: indices($("fail_indices").value),
    duplicate_indices: indices($("duplicate_indices").value),
  };
}

async function refreshSchedule() {
  const f = readForm();
  const qs = new URLSearchParams({
    fps: f.fps,
    angle: f.shutter_angle,
    samples: f.samples,
    curve: f.curve,
    frame_index: f.frame_index,
    t_open_offset: f.t_open_offset,
  });
  try {
    const s = await fetchJSON(`/api/schedule?${qs}`);
    lastSchedule = s;
    showError("");
    const boundaryMs = ((s.frame_index + 1) * s.frame_period) * 1000;
    const crosses = s.t_close > (s.frame_index + 1) * s.frame_period + 1e-12;
    const boundary = crosses
      ? `<span class="bad">快门关闭越过本帧边界 ${boundaryMs.toFixed(3)} ms（应已被服务端拒绝）</span>`
      : `<span class="ok">快门区间闭合于本帧内（边界 ${boundaryMs.toFixed(3)} ms）</span>`;
    $("range-info").innerHTML =
      `帧号 <b>#${s.frame_index}</b>，帧周期 <b>${(s.frame_period * 1000).toFixed(3)} ms</b>，` +
      `帧起点 <b>${(s.frame_start * 1000).toFixed(3)} ms</b><br>` +
      `快门绝对区间 <b>${(s.t_open * 1000).toFixed(3)} → ${(s.t_close * 1000).toFixed(3)} ms</b>` +
      `（帧内偏移 ${(s.t_open_offset * 1000).toFixed(3)} ms，曝光 ${(s.exposure * 1000).toFixed(3)} ms），共 ${s.times.length} 个采样时刻<br>` +
      boundary;
    drawWeights(s, null);
  } catch (err) {
    showError(`快门采样参数无效：${err.message}`);
    $("range-info").textContent = "采样范围无法计算，请修正参数。";
    $("weight-chart").innerHTML = "";
  }
}

function drawWeights(schedule, meta) {
  const chart = $("weight-chart");
  chart.innerHTML = "";
  const maxW = Math.max(...schedule.weights);
  const missing = new Set(meta ? meta.merge.missing : []);
  const dups = new Set(meta ? meta.merge.duplicates_rejected : []);
  const attempts = meta ? meta.merge.attempts : {};
  const framePeriod = schedule.frame_period;
  const frameIndex = schedule.frame_index || 0;
  const frameStart = schedule.frame_start !== undefined
    ? schedule.frame_start : frameIndex * framePeriod;
  const frameEnd = frameStart + framePeriod;

  schedule.weights.forEach((w, i) => {
    const bar = document.createElement("div");
    bar.className = "weight-bar";
    if (missing.has(i)) bar.classList.add("missing");
    else if (dups.includes(i)) bar.classList.add("duplicate");
    bar.style.height = `${Math.max(2, (w / maxW) * 100)}%`;
    const t = schedule.times[i];
    // 标出靠近帧边界的采样时刻（时间区间偏差重点回归区）。
    const edgeGap = Math.min(t - frameStart, frameEnd - t);
    const nearEdge = edgeGap >= 0 && edgeGap < 0.5 * framePeriod / schedule.weights.length;
    if (nearEdge && !missing.has(i) && !dups.includes(i)) bar.classList.add("edge");
    const attempt = attempts[i] ? `，实际尝试 ${attempts[i]} 次` : "";
    const flag = missing.has(i)
      ? "（遗漏）"
      : dups.includes(i)
        ? "（重复已拒绝）"
        : nearEdge
          ? "（贴近帧边界）"
          : "";
    bar.dataset.tip =
      `#${i}  t=${(t * 1000).toFixed(4)}ms（绝对） w=${w.toExponential(3)}${attempt}${flag}`;
    chart.appendChild(bar);
  });
}

async function startRender() {
  showError("");
  $("render-btn").disabled = true;
  $("job-status").textContent = "任务提交中…";
  let job;
  try {
    job = await fetchJSON("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readForm()),
    });
  } catch (err) {
    showError(err.message);
    $("render-btn").disabled = false;
    $("job-status").textContent = "";
    return;
  }

  $("job-status").textContent = `任务 ${job.job_id} 渲染中（多进程时间采样）…`;
  pollTimer = setInterval(() => pollJob(job.job_id), 700);
}

async function pollJob(jobId) {
  let status;
  try {
    status = await fetchJSON(`/api/jobs/${jobId}/status`);
  } catch (err) {
    clearInterval(pollTimer);
    showError(`状态查询失败：${err.message}`);
    $("render-btn").disabled = false;
    return;
  }
  if (status.status === "running") return;

  clearInterval(pollTimer);
  $("render-btn").disabled = false;
  if (status.status === "error") {
    $("job-status").textContent = "任务失败";
    showError(`渲染异常（由快门采样界面提示）：\n${status.error}`);
    return;
  }

  $("job-status").textContent = `任务 ${jobId} 完成`;
  currentJobId = jobId;
  const meta = await fetchJSON(`/api/jobs/${jobId}/meta`);
  lastMeta = meta;
  showResults(jobId, meta);
}

function showResults(jobId, meta) {
  $("result-panel").hidden = false;
  const base = `/jobs/${jobId}`;
  const bust = `?v=${Date.now()}`;
  $("img-integral").src = `${base}/integral.png${bust}`;
  $("img-reference").src = `${base}/reference.png${bust}`;
  $("img-error").src = `${base}/error.png${bust}`;

  const m = meta.merge;
  const e = meta.error;
  $("metrics").innerHTML = [
    ["接受时刻样本", `${m.accepted.length} / ${meta.config.samples}`],
    ["重复并拒绝", m.duplicates_rejected.length ? m.duplicates_rejected.join(", ") : "无"],
    ["遗漏时刻", m.missing.length ? m.missing.join(", ") : "无"],
    ["接受权重和", m.accepted_weight_sum.toFixed(6)],
    ["平均绝对误差", e.mean_abs.toExponential(4)],
    ["最大像素误差", e.max_abs.toExponential(4)],
    ["参考样本密度", `${meta.reference.samples} 时刻/曝光`],
  ].map(([k, v]) => `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");

  const missingBad = m.missing.length
    ? `<span class="bad">检测到遗漏时刻：${m.missing.join(", ")}</span>`
    : `<span class="ok">无遗漏</span>`;
  const dupBad = m.duplicates_rejected.length
    ? `<span class="bad">重复时刻已拒绝，未二次累加：${m.duplicates_rejected.join(", ")}</span>`
    : `<span class="ok">无重复投递</span>`;
  $("merge-report").innerHTML =
    `合并记录（实际接受的时间样本）：${missingBad} ｜ ${dupBad}<br>` +
    `重试情况：${Object.entries(m.attempts).filter(([, a]) => a > 1)
      .map(([i, a]) => `#${i} 尝试 ${a} 次`).join("，") || "全部一次成功"}`;

  const sel = $("state-select");
  sel.innerHTML = "";
  m.accepted.forEach((idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    const s = lastSchedule && lastSchedule.times[idx] !== undefined
      ? lastSchedule.times[idx] : meta.schedule.times[idx];
    opt.textContent = `样本 #${idx} · t=${(s * 1000).toFixed(4)} ms · w=${meta.schedule.weights[idx].toExponential(3)}`;
    sel.appendChild(opt);
  });
  selectState(jobId);
  drawWeights(
    {
      times: meta.schedule.times,
      weights: meta.schedule.weights,
      positions: meta.schedule.positions,
      frame_period: meta.schedule.frame_period,
      frame_index: meta.schedule.frame_index,
      frame_start: meta.schedule.frame_start,
    },
    meta
  );
}

function selectState(jobId) {
  const idx = Number($("state-select").value);
  const t = lastMeta.schedule.times[idx];
  const ft = lastMeta.schedule.frame_times[idx];
  const w = lastMeta.schedule.weights[idx];
  $("state-meta").innerHTML =
    `绝对时刻 t=<b>${(t * 1000).toFixed(4)} ms</b>` +
    `（帧 #${lastMeta.schedule.frame_index} 内 ${(ft * 1000).toFixed(4)} ms），` +
    `快门位置 u=${lastMeta.schedule.positions[idx].toFixed(4)}，权重 w=${w.toExponential(4)}`;
  $("img-state").src = `/jobs/${jobId}/samples/sample_${String(idx).padStart(4, "0")}.png?v=${Date.now()}`;
}

function bindTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".tab-body").forEach((body) => {
        body.hidden = body.dataset.body !== btn.dataset.tab;
      });
    });
  });
}

async function init() {
  const { curves } = await fetchJSON("/api/curves");
  const sel = $("curve");
  Object.entries(curves).forEach(([k, label]) => {
    const opt = document.createElement("option");
    opt.value = k;
    opt.textContent = label;
    if (k === "trapezoid") opt.selected = true;
    sel.appendChild(opt);
  });

  ["fps", "frame_index", "t_open_offset", "shutter_angle", "samples", "curve"].forEach((id) =>
    $(id).addEventListener("change", refreshSchedule)
  );
  $("render-btn").addEventListener("click", startRender);
  $("state-select").addEventListener("change", () => {
    if (currentJobId) selectState(currentJobId);
  });
  bindTabs();
  refreshSchedule();
}

init();
