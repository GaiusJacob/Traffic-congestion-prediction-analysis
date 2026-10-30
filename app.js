/* app.js - draws the simulation that runs in Python (engine.py).
   The browser asks the server to advance time a little (/api/advance), receives the new state
   (cars, road loads, Random Forest forecast, accidents) and draws it on the canvas. */

const $ = s => document.querySelector(s);
const canvas = $('#map'), ctx = canvas.getContext('2d');
const SECONDS_PER_HOUR = 4;                 // at speed 1x one simulated hour takes 4 real seconds
const CAR_COLORS = ['#b7d3ee', '#54d6a0', '#ff5f65', '#ffb23f'];   // normal, rerouted, stuck, slow

let NET = null, nodeById = {}, geo = [], roadGeo = [];
let S = null, since = 0;                    // latest server state, number of log events already shown
let running = false, speed = 4, layer = 0, selected = null;
let inflight = false, lastAdv = 0, polledAt = performance.now();
let accSig = '', fcSig = '';

// ------------------------------------------------------------------ helpers
async function api(url, method = 'GET', body = null) {
  try {
    const r = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : null });
    return await r.json();
  } catch (e) { return { error: 'Server not reachable: ' + e.message }; }
}

function ramp(r) {                           // load ratio -> colour (green > yellow > orange > red)
  r = Math.max(0, Math.min(1, r));
  const st = [[0, [47, 143, 107]], [0.35, [215, 194, 74]], [0.65, [232, 128, 58]], [1, [255, 75, 85]]];
  for (let i = 1; i < st.length; i++) {
    if (r <= st[i][0]) {
      const [r0, c0] = st[i - 1], [r1, c1] = st[i], t = (r - r0) / (r1 - r0);
      return `rgb(${c0.map((v, k) => Math.round(v + (c1[k] - v) * t)).join(',')})`;
    }
  }
  return 'rgb(255,75,85)';
}

function addLog(ev) {
  const el = document.createElement('div');
  el.className = 'log-entry ' + (ev.kind || '');
  el.textContent = `${ev.clock}: ${ev.text}`;
  $('#log').prepend(el);
  while ($('#log').children.length > 80) $('#log').lastChild.remove();
}

// ----------------------------------------------------------------- geometry
function buildGeometry() {
  nodeById = Object.fromEntries(NET.nodes.map(n => [n.id, n]));
  geo = NET.edges.map(e => {                 // one lane per direction, drawn slightly to the side
    const a = nodeById[e.frm], b = nodeById[e.to];
    const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy), ux = dx / len, uy = dy / len;
    const nx = -uy * 5, ny = ux * 5;
    return { x1: a.x + nx, y1: a.y + ny, x2: b.x + nx, y2: b.y + ny, ang: Math.atan2(dy, dx) };
  });
  roadGeo = NET.roads.map(r => ({ a: nodeById[r.a], b: nodeById[r.b] }));
}

// ------------------------------------------------------------------ drawing
function layerValues() {
  if (layer > 0 && S.pred) return S.pred.map(p => p[layer - 1]);
  return S.loads;
}

function halo(text, x, y) {
  ctx.lineWidth = 3; ctx.strokeStyle = '#091522'; ctx.strokeText(text, x, y); ctx.fillText(text, x, y);
}

function draw(now) {
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#091522'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#102337'; ctx.lineWidth = 1;
  for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
  for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
  if (!NET || !S) { requestAnimationFrame(draw); return; }

  const dh = running ? Math.min(0.3, (now - polledAt) / 1000 * speed / SECONDS_PER_HOUR) : 0;
  const pulse = 3 * Math.sin(now / 180);
  ctx.lineCap = 'round';

  // roads: dark asphalt, then one coloured lane per direction
  for (const r of roadGeo) {
    ctx.lineWidth = 19; ctx.strokeStyle = '#162a3e';
    ctx.beginPath(); ctx.moveTo(r.a.x, r.a.y); ctx.lineTo(r.b.x, r.b.y); ctx.stroke();
  }
  const vals = layerValues();
  geo.forEach((g, i) => {
    ctx.lineWidth = 6; ctx.strokeStyle = ramp(vals[i] ?? 0); ctx.globalAlpha = 0.9;
    ctx.beginPath(); ctx.moveTo(g.x1, g.y1); ctx.lineTo(g.x2, g.y2); ctx.stroke();
  });
  ctx.globalAlpha = 1;

  // selected road highlight
  if (selected && selected.kind === 'road') {
    const r = roadGeo[selected.id];
    ctx.lineWidth = 24; ctx.strokeStyle = '#58d7e855';
    ctx.beginPath(); ctx.moveTo(r.a.x, r.a.y); ctx.lineTo(r.b.x, r.b.y); ctx.stroke();
  }

  // junctions and gateways
  for (const n of NET.nodes) {
    if (n.kind === 'gate') {
      ctx.fillStyle = '#1b3550'; ctx.strokeStyle = '#58d7e8'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.roundRect(n.x - 13, n.y - 13, 26, 26, 6); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#9fdcec'; ctx.font = 'bold 12px Arial'; ctx.textAlign = 'center';
      ctx.fillText('⇄', n.x, n.y + 4);
      ctx.font = 'bold 11px Arial'; ctx.fillStyle = '#7fb4c8'; halo(n.id + ' IN/OUT', n.x, n.y - 20);
    } else {
      const sel = selected && selected.kind === 'node' && selected.id === n.id;
      ctx.beginPath(); ctx.arc(n.x, n.y, 15, 0, 7); ctx.fillStyle = '#123a3a'; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = '#55d6a2'; ctx.stroke();
      if (sel) { ctx.beginPath(); ctx.arc(n.x, n.y, 24, 0, 7); ctx.lineWidth = 3; ctx.strokeStyle = '#58d7e8'; ctx.stroke(); }
      ctx.textAlign = 'left'; ctx.fillStyle = '#e2edf8'; ctx.font = 'bold 12px Arial'; halo(n.id, n.x + 15, n.y - 12);
      ctx.fillStyle = '#7d97ae'; ctx.font = '10px Arial'; halo(n.name, n.x + 15, n.y + 22);
    }
  }

  // accidents
  for (const a of S.accidents) {
    if (a.status === 'cleared' || a.status === 'cancelled') continue;
    let x, y;
    if (a.kind === 'node') { const n = nodeById[a.target]; x = n.x; y = n.y; }
    else { const r = roadGeo[a.target]; x = (r.a.x + r.b.x) / 2; y = (r.a.y + r.b.y) / 2; }
    const col = a.status === 'active' ? '#ff5964' : a.status === 'warning' ? '#ffc34f' : '#8aa0b5';
    ctx.beginPath(); ctx.arc(x, y, (a.kind === 'node' ? 30 : 22) + (a.status === 'scheduled' ? 0 : pulse), 0, 7);
    ctx.lineWidth = 3; ctx.strokeStyle = col; ctx.setLineDash(a.status === 'scheduled' ? [4, 5] : []); ctx.stroke(); ctx.setLineDash([]);
    if (a.status === 'active') { ctx.fillStyle = '#ff596433'; ctx.fill(); }
    ctx.fillStyle = col; ctx.font = 'bold 15px Arial'; ctx.textAlign = 'center'; ctx.fillText('⚠', x, y + 5);
    ctx.font = 'bold 11px Arial';
    const txt = a.status === 'active' ? `#${a.id} ACTIVE` : `#${a.id} in ${Math.max(0, a.start - S.t).toFixed(1)} h`;
    halo(txt, x, y - 36);
  }

  // cars (position extrapolated between server updates so motion looks smooth)
  for (const c of S.cars) {
    const g = geo[c[1]], f = Math.min(1, c[2] + c[3] * dh);
    ctx.save();
    ctx.translate(g.x1 + (g.x2 - g.x1) * f, g.y1 + (g.y2 - g.y1) * f);
    ctx.rotate(g.ang);
    ctx.fillStyle = CAR_COLORS[c[4]];
    ctx.strokeStyle = '#06101b'; ctx.lineWidth = 1.2;
    ctx.fillRect(-5.5, -3, 11, 6);
    ctx.strokeRect(-5.5, -3, 11, 6);
    ctx.restore();
  }

  ctx.textAlign = 'left'; ctx.fillStyle = '#5e7d97'; ctx.font = '11px Arial';
  const act = S.accidents.filter(a => a.status === 'active').length, warn = S.accidents.filter(a => a.status === 'warning').length;
  ctx.fillText(act ? `NETWORK STATUS: ${act} ACCIDENT(S) ACTIVE` : warn ? `NETWORK STATUS: ${warn} WARNING(S) ACTIVE` : 'NETWORK STATUS: NORMAL', 16, 628);
  requestAnimationFrame(draw);
}

// -------------------------------------------------------------- UI updates
function apply(st) {
  if (st.error) { addLog({ clock: S ? S.clock : '', text: st.error, kind: 'warn' }); return; }
  S = st; polledAt = performance.now();
  st.events.forEach(addLog); since = st.n_events;
  const s = st.stats;
  $('#sClock').textContent = st.clock;
  $('#sCars').textContent = s.in_network;
  $('#sFlow').textContent = `${s.entered} / ${s.exited}`;
  $('#sWait').textContent = s.waiting;
  $('#sStuck').textContent = s.stuck;
  $('#sRe').textContent = s.rerouted;
  $('#sDelay').textContent = `${s.avg_delay_min} min`;
  $('#predClock').textContent = st.pred_clock ? `Forecast made at ${st.pred_clock}` : (st.settings.model ? 'forecast starts at the next full hour' : 'model not loaded');

  const live = st.accidents.filter(a => a.status === 'active').length;
  const warn = st.accidents.filter(a => a.status === 'warning').length;
  const sched = st.accidents.filter(a => a.status === 'scheduled').length;
  const b = $('#badge');
  if (live) { b.textContent = `⛔ ${live} active` + (warn ? ` · ⚠ ${warn} warning` : ''); b.className = 'badge active'; }
  else if (warn) { b.textContent = `⚠ ${warn} warning` + (sched ? ` · ${sched} scheduled` : ''); b.className = 'badge warning'; }
  else if (sched) { b.textContent = `${sched} scheduled`; b.className = 'badge neutral'; }
  else { b.textContent = 'No accidents scheduled'; b.className = 'badge neutral'; }

  $('#demand').value = st.settings.demand; $('#mode').value = st.settings.mode; $('#useRf').checked = st.settings.use_rf;
  updateAccidents(); updateForecast();
}

const SEV = v => v >= 0.999 ? 'full closure' : `${Math.round(v * 100)}% capacity lost`;

function updateAccidents() {
  const sig = JSON.stringify(S.accidents) + Math.floor(S.t * 2);
  if (sig === accSig) return;
  accSig = sig;
  const list = S.accidents.filter(a => a.status !== 'cancelled').slice().reverse();
  $('#accList').innerHTML = list.length ? list.map(a => {
    const label = { scheduled: 'scheduled', warning: `warning · starts in ${Math.max(0, a.start - S.t).toFixed(1)} h`, active: 'ACTIVE', cleared: 'cleared' }[a.status];
    const impact = (a.status === 'active' || a.status === 'cleared')
      ? `<div>Stuck at start: <b>${a.n_stuck}</b>${a.status === 'active' ? ` (still waiting: ${a.stuck_now})` : ''} · rerouted: <b>${a.n_rerouted}</b></div>` : '';
    const cancel = (a.status === 'cleared') ? '' : `<button class="ghost small" data-cancel="${a.id}">Cancel</button>`;
    return `<div class="acc ${a.status}"><div class="top"><b>#${a.id} ${a.label}</b><span class="st">${label}</span></div>
      <div>${a.start_clock} → ${a.end_clock} · ${SEV(a.sev)}</div>${impact}<div style="text-align:right">${cancel}</div></div>`;
  }).join('') : '<p class="empty">No accidents yet. Pick a place on the map and press “Add accident”.</p>';
}

function updateForecast() {
  const key = (S.pred_clock || '') + '|' + JSON.stringify(selected) + '|' + Math.floor(S.t * 4);
  if (key === fcSig) return;
  fcSig = key;
  let idx, title;
  if (selected && selected.kind === 'road') {
    idx = NET.edges.filter(e => e.road === selected.id).map(e => e.idx);
    const r = NET.roads[selected.id]; title = `Selected road ${r.a}–${r.b} (both directions)`;
  } else if (selected) {
    idx = NET.edges.filter(e => e.to === selected.id).map(e => e.idx); title = `Roads leading into junction ${selected.id}`;
  } else {
    const score = i => S.pred ? S.pred[i][3] : S.loads[i];
    idx = NET.edges.map(e => e.idx).sort((a, b) => score(b) - score(a)).slice(0, 8);
    title = S.pred ? 'Most congested roads expected in 4 hours' : 'Most loaded roads right now';
  }
  $('#fcTitle').textContent = title + ' · load = cars ÷ capacity';
  if (!S.pred) { $('#fcBody').innerHTML = `<p class="empty">${S.settings.model ? 'The first forecast is made at the next full hour.' : 'Random Forest model not loaded – see the terminal.'}</p>`; return; }
  const cell = v => `<td class="c" style="background:${ramp(v)}">${Math.round(v * 100)}%</td>`;
  $('#fcBody').innerHTML = `<table class="ft"><tr><th>Road</th><th>Now</th><th>+1h</th><th>+2h</th><th>+3h</th><th>+4h</th></tr>` +
    idx.map(i => `<tr><td>${NET.edges[i].frm}→${NET.edges[i].to}</td>${cell(S.loads[i])}${S.pred[i].map(cell).join('')}</tr>`).join('') + '</table>';
}

async function loadModelInfo() {
  const m = await api('/api/model');
  const box = $('#modelBody');
  if (!m.metrics) { box.innerHTML = `<p>${m.message || 'No model.'}</p>`; return; }
  const k = m.metrics, imp = (a, b) => `${Math.round((1 - a / b) * 100)}%`;
  box.innerHTML = `<p><b>RandomForestRegressor</b> · ${k.n_trees} trees · trained on ${k.n_train.toLocaleString()} simulated road-hours (${k.episodes} episodes × ${k.hours} h). Predicts every road's load 1–4 hours ahead.</p>
   <table class="ft metrics"><tr><th>Error on unseen data (MAE)</th>${k.horizons.map(h => `<th>${h}</th>`).join('')}</tr>
   <tr><td>Random Forest</td>${k.mae_rf.map(v => `<td>${v.toFixed(3)}</td>`).join('')}</tr>
   <tr><td>“Same as now” baseline</td>${k.mae_baseline.map(v => `<td>${v.toFixed(3)}</td>`).join('')}</tr>
   <tr><td><b>Improvement</b></td>${k.mae_rf.map((v, i) => `<td><b>${imp(v, k.mae_baseline[i])}</b></td>`).join('')}</tr></table>
   <p style="margin-top:12px">Most important inputs: ${Object.keys(k.feature_importance).slice(0, 6).map(f => `<span class="pill">${f}</span>`).join('')}</p>
   <p class="footnote">Trained ${k.trained_at}. Lower error = better. Re-train any time with <code>python train_model.py</code>.</p>`;
}

// ------------------------------------------------------------- simulation loop
setInterval(async () => {
  if (!running || inflight || !NET) return;
  const now = performance.now();
  const hours = Math.min(0.5, (now - lastAdv) / 1000 * speed / SECONDS_PER_HOUR);
  lastAdv = now; inflight = true;
  apply(await api('/api/advance', 'POST', { hours, since }));
  inflight = false;
}, 110);

function setRunning(on) {
  running = on; lastAdv = performance.now();
  $('#playBtn').textContent = on ? 'Ⅱ Pause' : '▶ Start';
  $('#liveText').textContent = on ? 'SIMULATION LIVE' : 'SIMULATION PAUSED';
}

// ------------------------------------------------------------------- controls
$('#playBtn').onclick = () => setRunning(!running);
$('#resetBtn').onclick = async () => { $('#log').innerHTML = ''; since = 0; accSig = fcSig = ''; apply(await api('/api/reset', 'POST', { demand: +$('#demand').value })); };
$('#speed').onchange = e => { speed = +e.target.value; };
$('#demand').onchange = async e => apply(await api('/api/settings', 'POST', { demand: +e.target.value, since }));
$('#mode').onchange = async e => apply(await api('/api/settings', 'POST', { mode: e.target.value, since }));
$('#useRf').onchange = async e => apply(await api('/api/settings', 'POST', { use_rf: e.target.checked, since }));
document.querySelectorAll('.layers button').forEach(b => b.onclick = () => {
  layer = +b.dataset.layer;
  document.querySelectorAll('.layers button').forEach(x => x.classList.toggle('on', x === b));
});

$('#addAcc').onclick = async () => {
  if (!selected) return;
  apply(await api('/api/accidents', 'POST', {
    kind: selected.kind, target: selected.id, start_in: +$('#fStart').value,
    duration: +$('#fDur').value, severity: +$('#fSev').value, since }));
};
$('#randAcc').onclick = async () => apply(await api('/api/accidents/random', 'POST', { since }));
$('#accList').addEventListener('mousedown', async e => {       // mousedown: the list is redrawn often
  const id = e.target.dataset && e.target.dataset.cancel;
  if (id) apply(await api(`/api/accidents/${id}?since=${since}`, 'DELETE'));
});

function distToSegment(px, py, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y, t = Math.max(0, Math.min(1, ((px - a.x) * dx + (py - a.y) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(px - (a.x + t * dx), py - (a.y + t * dy));
}
canvas.addEventListener('click', ev => {
  if (!NET) return;
  const r = canvas.getBoundingClientRect();
  const x = (ev.clientX - r.left) * canvas.width / r.width, y = (ev.clientY - r.top) * canvas.height / r.height;
  let pick = null;
  const node = NET.nodes.find(n => Math.hypot(n.x - x, n.y - y) < (n.kind === 'gate' ? 18 : 24));
  if (node && node.kind === 'gate') { $('#picked').innerHTML = 'Gateways (map entrances) cannot have accidents – pick a junction or a road.'; return; }
  if (node) pick = { kind: 'node', id: node.id };
  else {
    let best = 12;
    roadGeo.forEach((g, i) => { const d = distToSegment(x, y, g.a, g.b); if (d < best) { best = d; pick = { kind: 'road', id: i }; } });
  }
  selected = pick; fcSig = '';
  if (!pick) { $('#picked').textContent = 'Nothing selected – click the map.'; $('#addAcc').disabled = true; }
  else {
    $('#picked').innerHTML = pick.kind === 'node'
      ? `Selected junction <b>${pick.id}</b> – ${nodeById[pick.id].name}`
      : `Selected road <b>${NET.roads[pick.id].a}–${NET.roads[pick.id].b}</b>`;
    $('#addAcc').disabled = false;
  }
  if (S) updateForecast();
});

// ----------------------------------------------------------------------- start
(async function init() {
  NET = await api('/api/network');
  buildGeometry();
  apply(await api('/api/state'));
  loadModelInfo();
  requestAnimationFrame(draw);
})();
