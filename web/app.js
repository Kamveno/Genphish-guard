/* genphish-guard console — frontend.
   Thin client over the FastAPI backend. No framework, no build step. */

const API = '';
const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const fx = (v, n = 3) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(n);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const state = { meta: null, samples: [], drift: null, lastDetect: null };

async function api(path, opts = {}) {
  const r = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' }, ...opts,
  });
  if (!r.ok) {
    let d; try { d = (await r.json()).detail; } catch { d = r.statusText; }
    throw new Error(d || `HTTP ${r.status}`);
  }
  return r.json();
}

/* ══════════════════════════ boot ══════════════════════════ */
async function boot() {
  wireTabs();
  wireDetect();
  wireDrift();
  wireSoar();
  wireApi();

  try {
    const h = await api('/api/health');
    $('#dot').className = 'dot ' + (h.ready ? 'live' : 'dead');
    $('#pill-text').textContent = h.ready ? 'API live' : 'degraded';
    $('#mode').textContent = h.data_mode;
    if (h.data_mode !== 'REAL') $('#pill-mode').style.color = 'var(--defuse)';
  } catch (e) {
    $('#dot').className = 'dot dead';
    $('#pill-text').textContent = 'API unreachable';
    return;
  }

  const [stats, samples] = await Promise.all([
    api('/api/stats'), api('/api/samples'),
  ]);
  state.meta = stats;
  state.samples = samples.samples;

  fillSelect('#sel-src', [...stats.generators, 'pooled'], 'pooled');
  fillSelect('#d-src', stats.generators, stats.generators[0]);
  fillSelect('#d-tgt', stats.generators, stats.generators[1]);
  renderChips();
  renderApiTab();

  loadTransfer();
  loadCalibration();
  refreshSoar();
}

function fillSelect(sel, items, def) {
  const el = $(sel);
  el.innerHTML = items.map(i => `<option value="${i}"${i === def ? ' selected' : ''}>${i}</option>`).join('');
}

function wireTabs() {
  $$('#tabs button').forEach(b => b.onclick = () => showTab(b.dataset.v));
}
function showTab(v) {
  $$('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
  $$('.view').forEach(s => s.classList.toggle('on', s.id === 'v-' + v));
  if (v === 'soar') refreshSoar();
}

/* ══════════════════════════ detect ══════════════════════════ */
function renderChips() {
  $('#chips').innerHTML = state.samples.map((s, i) =>
    `<span class="chip" data-i="${i}">${esc(s.name)}<b>${esc(s.expect)}</b></span>`).join('');
  $$('#chips .chip').forEach(c => c.onclick = () => {
    const s = state.samples[+c.dataset.i];
    $('#email').value = s.text;
    const n = $('#sample-note');
    if (s.note) { n.innerHTML = `<div><b>${esc(s.name)}.</b> ${esc(s.note)}</div>`; n.hidden = false; }
    else { n.hidden = true; }
  });
}

function wireDetect() {
  $('#btn-detect').onclick = runDetect;
  $('#btn-clear').onclick = () => {
    $('#email').value = '';
    resetVerdict();
  };
  $('#btn-goto-soar').onclick = () => showTab('soar');
  $('#email').addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') runDetect();
  });
}

function resetVerdict() {
  $('#tier').className = 'tier-badge idle';
  $('#tier').textContent = 'AWAITING INPUT';
  $('#vtop').className = 'verdict-top';
  $('#vsub').textContent = 'Submit an email to get a zero-trust decision.';
  $('#score').textContent = '—';
  $('#needle').style.left = '0%';
  $('#fill').style.width = '0%';
  $('#contrib').innerHTML = '<div class="empty">No analysis yet.</div>';
  $('#gpills').innerHTML = '';
  $('#thrline').textContent = 'operating threshold —';
  $('#inc-card').hidden = true;
}

async function runDetect() {
  const text = $('#email').value.trim();
  if (!text) { $('#email').focus(); return; }
  const btn = $('#btn-detect');
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span> analysing';

  try {
    const r = await api('/api/detect', {
      method: 'POST',
      body: JSON.stringify({
        text,
        detector: $('#sel-kind').value,
        trained_on: $('#sel-src').value,
        calibration: $('#sel-cal').value,
        raise_incident: $('#chk-soar').checked,
      }),
    });
    state.lastDetect = r;
    paintVerdict(r);
  } catch (e) {
    $('#vsub').textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false; btn.textContent = 'Analyse email';
  }
}

function paintVerdict(r) {
  const pct = v => (Math.max(0, Math.min(1, v)) * 100).toFixed(1) + '%';

  $('#tier').className = 'tier-badge ' + r.tier;
  $('#tier').textContent = r.tier;
  $('#vtop').className = 'verdict-top ' + r.tier;
  $('#score').textContent = fx(r.score, 4);

  const sub = {
    BLOCK:   'Confident phishing. Quarantine, block sender at the MTA, raise a P1.',
    DEFUSE:  'Uncertain. Deliver a stripped copy — links removed, attachments sandboxed.',
    DELIVER: 'Below the detection threshold. Deliver normally, keep the audit record.',
  }[r.tier];
  $('#vsub').textContent = sub;

  $('#z-del').style.width = pct(r.t_lo);
  $('#z-def').style.left  = pct(r.t_lo);
  $('#z-def').style.width = pct(r.t_hi - r.t_lo);
  $('#z-blk').style.width = pct(1 - r.t_hi);
  $('#fill').style.width  = pct(r.score);
  $('#needle').style.left = pct(r.score);
  $('#thrmark').style.left = pct(r.threshold);

  $('#thrline').innerHTML =
    `operating threshold <b style="color:var(--violet)">${fx(r.threshold, 4)}</b> ` +
    `· calibration <b>${esc(r.calibration)}</b> · detector <b>${esc(r.detector.family)}/${esc(r.detector.trained_on)}</b>` +
    `<br>tiers anchored to it — DELIVER &lt; ${fx(r.t_lo, 4)} ≤ DEFUSE &lt; ${fx(r.t_hi, 4)} ≤ BLOCK. ` +
    `Change the calibration method and the whole policy moves with the threshold.`;

  const top = r.contributions.slice(0, 8);
  const max = Math.max(...top.map(c => Math.abs(c.contribution)), 1e-9);
  $('#contrib').innerHTML = top.map(c => {
    const w = (Math.abs(c.contribution) / max) * 50;
    const pos = c.contribution >= 0;
    return `<div class="contrib-row">
      <span class="cf"><b>${esc(c.feature)}</b> = ${fx(c.value, 2)}</span>
      <span class="cbar"><i class="${pos ? 'pos' : 'neg'}" style="width:${w}%"></i></span>
      <span class="cv ${pos ? 'bad' : 'good'}">${c.contribution >= 0 ? '+' : ''}${fx(c.contribution, 2)}</span>
    </div>`;
  }).join('');

  $('#gpills').innerHTML = r.group_contributions.map(g =>
    `<span class="gpill ${g.contribution > 0.01 ? 'hot' : g.contribution < -0.01 ? 'cold' : ''}">
      ${esc(g.group)} ${g.contribution >= 0 ? '+' : ''}${fx(g.contribution, 2)}</span>`).join('');

  if (r.incident) {
    $('#inc-card').hidden = false;
    const i = r.incident;
    $('#inc-preview').innerHTML = `
      <div class="row small mono" style="margin-bottom:9px">
        <span class="inc-id">#${esc(i.id)}</span>
        <span class="inc-t ${esc(i.tier)}">${esc(i.tier)}</span>
        <span class="inc-m">${esc(i.severity)}</span>
        <span class="inc-m">conf ${fx(i.confidence, 4)}</span>
      </div>
      ${(i.playbook || []).map(s => `<div class="pb-step">
        <span class="${s.status === 'executed' ? 'pb-ok' : 'pb-skip'}">${s.status === 'executed' ? '✓' : '·'}</span>
        <span><span class="pb-a">${esc(s.action)}</span><br><span class="pb-d">${esc(s.detail)}</span></span>
      </div>`).join('')}`;
  } else {
    $('#inc-card').hidden = true;
  }
}

/* ══════════════════════════ transfer ══════════════════════════ */
async function loadTransfer() {
  let t;
  try { t = await api('/api/transfer'); } catch { return; }

  const ramp = (v, lo, hi) => {
    const x = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
    // red (bad) → amber → green (good)
    const stops = [[244, 82, 77], [240, 169, 44], [53, 208, 127]];
    const i = x < .5 ? 0 : 1, f = x < .5 ? x * 2 : (x - .5) * 2;
    const c = stops[i].map((a, k) => Math.round(a + (stops[i + 1][k] - a) * f));
    return `rgba(${c.join(',')},0.22)`;
  };
  const txt = (v, lo, hi) => {
    const x = (v - lo) / (hi - lo);
    return x < .4 ? 'var(--block)' : x < .72 ? 'var(--defuse)' : 'var(--deliver)';
  };

  const build = (tbl, mat, lo, hi) => {
    let h = '<thead><tr><th></th>' + t.targets.map(c => `<th>${esc(c)}</th>`).join('') + '</tr></thead><tbody>';
    mat.forEach((row, i) => {
      h += `<tr><th class="rowh">${esc(t.sources[i])}</th>` + row.map((v, j) =>
        `<td class="${i === j ? 'diag' : ''}" style="background:${ramp(v, lo, hi)};color:${txt(v, lo, hi)}">
          ${fx(v, 3)}<small>${i === j ? 'in-domain' : 'transfer'}</small></td>`).join('') + '</tr>';
    });
    $(tbl).innerHTML = h + '</tbody></table>';
  };

  build('#heat-auc', t.auc, 0.5, 1.0);
  build('#heat-f1', t.f1, 0.3, 1.0);

  const off = (m) => {
    const v = [];
    m.forEach((r, i) => r.forEach((x, j) => { if (i !== j) v.push(x); }));
    return v.reduce((a, b) => a + b, 0) / v.length;
  };
  const dia = (m) => m.map((r, i) => r[i]).reduce((a, b) => a + b, 0) / m.length;

  const aucOff = off(t.auc), f1Off = off(t.f1), f1Dia = dia(t.f1);
  $('#transfer-tiles').innerHTML = `
    <div class="tile"><div class="k">In-domain F1</div><div class="v ok">${fx(f1Dia, 3)}</div><div class="n">detector on its own generator</div></div>
    <div class="tile"><div class="k">Cross-generator F1</div><div class="v bad">${fx(f1Off, 3)}</div><div class="n">same detector, stale threshold</div></div>
    <div class="tile"><div class="k">Transfer gap</div><div class="v warn">${fx((f1Dia - f1Off) * 100, 2)} pp</div><div class="n">what the threshold costs</div></div>
    <div class="tile"><div class="k">Cross-generator AUC</div><div class="v acc">${fx(aucOff, 3)}</div><div class="n">ranking survives the shift</div></div>`;
}

/* ══════════════════════════ calibration ══════════════════════════ */
async function loadCalibration() {
  let c;
  try { c = await api('/api/calibration'); } catch { return; }
  const s = c.summary;
  if (!s || !s.n_pairs) return;

  $('#calib-tiles').innerHTML = `
    <div class="tile"><div class="k">Stale threshold F1</div><div class="v bad">${fx(s.f1_stale, 3)}</div><div class="n">no correction applied</div></div>
    <div class="tile"><div class="k">Distribution matching</div><div class="v ok">${fx(s.f1_dm, 3)}</div><div class="n">zero target labels</div></div>
    <div class="tile"><div class="k">Labelled oracle</div><div class="v acc">${fx(s.f1_oracle, 3)}</div><div class="n">upper bound, uses labels</div></div>
    <div class="tile"><div class="k">Gap recovered</div><div class="v ok">${fx(s.gap_recovered_dm * 100, 1)}%</div><div class="n">label-free, over ${s.n_pairs} pairs</div></div>`;

  const bars = [
    ['stale',    s.f1_stale,     'var(--block)'],
    ['EM',       s.f1_em,        'var(--defuse)'],
    ['self-train', s.f1_selftrain, 'var(--defuse)'],
    ['DM',       s.f1_dm,        'var(--deliver)'],
    ['oracle',   s.f1_oracle,    'var(--accent)'],
  ];
  const max = Math.max(...bars.map(b => b[1]));
  $('#calib-bars').innerHTML = bars.map(([n, v, col]) => `
    <div style="display:grid;grid-template-columns:86px 1fr 52px;gap:10px;align-items:center;padding:7px 0">
      <span class="mono small dim">${n}</span>
      <span style="height:14px;background:var(--bg-grid);border-radius:4px;position:relative">
        <i style="position:absolute;inset:0 auto 0 0;width:${(v / max) * 100}%;background:${col};border-radius:4px;opacity:.75;display:block"></i>
      </span>
      <span class="mono small" style="text-align:right;color:${col}">${fx(v, 3)}</span>
    </div>`).join('');

  $('#pi-hat').textContent = fx(s.pi_hat_mean, 3);
  $('#pi-true').textContent = fx(s.true_prior, 3);

  $('#calib-table').innerHTML = `
    <thead><tr>
      <th>source → target</th>
      <th class="num">thr stale</th><th class="num">thr DM</th><th class="num">thr oracle</th>
      <th class="num">F1 stale</th><th class="num">F1 EM</th><th class="num">F1 self</th>
      <th class="num">F1 DM</th><th class="num">F1 oracle</th><th class="num">recovered</th>
    </tr></thead><tbody>` + c.rows.map(r => {
      const rec = (r.f1_oracle - r.f1_stale) > 1e-9
        ? (r.f1_dm - r.f1_stale) / (r.f1_oracle - r.f1_stale) : 0;
      return `<tr>
        <td>${esc(r.source)} → ${esc(r.target)}</td>
        <td class="num">${fx(r.thr_stale)}</td><td class="num">${fx(r.thr_dm)}</td><td class="num">${fx(r.thr_oracle)}</td>
        <td class="num bad">${fx(r.f1_stale)}</td><td class="num">${fx(r.f1_em)}</td><td class="num">${fx(r.f1_selftrain)}</td>
        <td class="num good">${fx(r.f1_dm)}</td><td class="num">${fx(r.f1_oracle)}</td>
        <td class="num ${rec > .8 ? 'good' : 'warnc'}">${fx(rec * 100, 1)}%</td>
      </tr>`;
    }).join('') + '</tbody>';
}

/* ══════════════════════════ drift ══════════════════════════ */
function wireDrift() { $('#btn-drift').onclick = runDrift; }

async function runDrift() {
  const btn = $('#btn-drift');
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span> streaming';
  $('#drift-status').textContent = '';
  try {
    const q = new URLSearchParams({
      n: $('#d-n').value, shift_at: $('#d-at').value,
      source: $('#d-src').value, target: $('#d-tgt').value,
    });
    const d = await api('/api/drift/simulate?' + q);
    state.drift = d;
    paintDrift(d);
  } catch (e) {
    $('#drift-status').textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false; btn.textContent = 'Run stream';
  }
}

function paintDrift(d) {
  const fired = d.alarm_index !== null && d.alarm_index !== undefined;
  $('#drift-tiles').innerHTML = `
    <div class="tile"><div class="k">Alarm</div>
      <div class="v ${fired ? 'bad' : 'ok'}">${fired ? 'FIRED' : 'silent'}</div>
      <div class="n">${fired ? 'wealth crossed 1/δ' : 'no drift detected'}</div></div>
    <div class="tile"><div class="k">Alarm at email</div>
      <div class="v">${fired ? '#' + d.alarm_index : '—'}</div>
      <div class="n">shift injected at #${d.shift_at}</div></div>
    <div class="tile"><div class="k">Time to detection</div>
      <div class="v warn">${d.ttd !== null && d.ttd !== undefined ? d.ttd : '—'}</div>
      <div class="n">emails after the switch</div></div>
    <div class="tile"><div class="k">Alarm level δ</div>
      <div class="v acc">${d.delta}</div>
      <div class="n">ceiling 1/δ = ${d.ceiling}</div></div>`;

  $('#drift-chart').innerHTML = wealthChart(d);
  $('#score-chart').innerHTML = scoreChart(d);
}

function wealthChart(d) {
  const W = 1000, H = 260, P = { t: 14, r: 16, b: 26, l: 48 };
  const iw = W - P.l - P.r, ih = H - P.t - P.b;
  const w = d.wealth.map(v => Math.max(v, 1e-3));
  const lo = Math.log10(Math.min(...w, 0.1));
  const hi = Math.log10(Math.max(...w, d.ceiling * 1.6));
  const X = i => P.l + (i / (w.length - 1)) * iw;
  const Y = v => P.t + ih - ((Math.log10(v) - lo) / (hi - lo)) * ih;

  const path = w.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join('');
  const ticks = [];
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) {
    ticks.push(`<line class="grid-l" x1="${P.l}" x2="${W - P.r}" y1="${Y(10 ** e)}" y2="${Y(10 ** e)}"/>
      <text x="${P.l - 7}" y="${Y(10 ** e) + 3}" text-anchor="end">1e${e}</text>`);
  }
  const shiftX = X(d.shift_at);
  const alarmX = (d.alarm_index !== null && d.alarm_index !== undefined) ? X(d.alarm_index) : null;

  return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
    ${ticks.join('')}
    <line class="axis" x1="${P.l}" x2="${W - P.r}" y1="${P.t + ih}" y2="${P.t + ih}"/>
    <line class="axis" x1="${P.l}" x2="${P.l}" y1="${P.t}" y2="${P.t + ih}"/>
    <line x1="${P.l}" x2="${W - P.r}" y1="${Y(d.ceiling)}" y2="${Y(d.ceiling)}"
          stroke="var(--block)" stroke-width="1.5" stroke-dasharray="6 4"/>
    <text x="${W - P.r}" y="${Y(d.ceiling) - 6}" text-anchor="end" fill="var(--block)">Ville ceiling 1/δ = ${d.ceiling}</text>
    <line x1="${shiftX}" x2="${shiftX}" y1="${P.t}" y2="${P.t + ih}"
          stroke="var(--violet)" stroke-width="1.5" stroke-dasharray="4 4"/>
    <text x="${shiftX + 5}" y="${P.t + 11}" fill="var(--violet)">generator shift → ${esc(d.target)}</text>
    ${alarmX !== null ? `<line x1="${alarmX}" x2="${alarmX}" y1="${P.t}" y2="${P.t + ih}" stroke="var(--block)" stroke-width="1.5"/>
      <circle cx="${alarmX}" cy="${Y(d.ceiling)}" r="4.5" fill="var(--block)"/>
      <text x="${alarmX + 6}" y="${P.t + 26}" fill="var(--block)">ALARM</text>` : ''}
    <path d="${path}" fill="none" stroke="var(--accent)" stroke-width="1.8"/>
    <text x="${P.l}" y="${H - 7}">email #0</text>
    <text x="${W - P.r}" y="${H - 7}" text-anchor="end">#${w.length}</text>
  </svg>`;
}

function scoreChart(d) {
  const W = 1000, H = 190, P = { t: 12, r: 16, b: 34, l: 48 };
  const iw = W - P.l - P.r, ih = H - P.t - P.b;
  const X = i => P.l + (i / (d.scores.length - 1)) * iw;
  const Y = v => P.t + ih - v * ih;
  const step = Math.max(1, Math.floor(d.scores.length / 480));

  // The band the monitor actually watches, derived from calibration quantiles.
  const [bLo, bHi] = d.band || [0.35, 0.65];

  const dots = d.scores.map((s, i) => (i % step) ? '' :
    `<circle cx="${X(i).toFixed(1)}" cy="${Y(s).toFixed(1)}" r="1.5"
      fill="${d.labels[i] ? 'var(--block)' : 'var(--deliver)'}" opacity=".55"/>`).join('');

  const band = `<rect x="${P.l}" y="${Y(bHi)}" width="${iw}" height="${Y(bLo) - Y(bHi)}"
      fill="var(--defuse)" opacity=".10"/>`;

  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const lift = d.band_rate_pre > 1e-9
    ? (d.band_rate_post / d.band_rate_pre).toFixed(1) + '×' : '—';

  return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
    ${band}
    ${ticks.map(v => `<line class="grid-l" x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}"/>
      <text x="${P.l - 7}" y="${Y(v) + 3}" text-anchor="end">${v.toFixed(2)}</text>`).join('')}
    ${[bLo, bHi].map(v => `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}"
      stroke="var(--defuse)" stroke-width="1" stroke-dasharray="3 3" opacity=".75"/>`).join('')}
    <line class="axis" x1="${P.l}" x2="${P.l}" y1="${P.t}" y2="${P.t + ih}"/>
    ${dots}
    <line x1="${X(d.shift_at)}" x2="${X(d.shift_at)}" y1="${P.t}" y2="${P.t + ih}"
          stroke="var(--violet)" stroke-width="1.5" stroke-dasharray="4 4"/>
    <text x="${P.l}" y="${H - 6}" fill="var(--deliver)">● legitimate</text>
    <text x="${P.l + 92}" y="${H - 6}" fill="var(--block)">● phishing</text>
  </svg>
  <div class="tiles" style="grid-template-columns:repeat(4,1fr);margin:13px 0 0">
    <div class="tile"><div class="k">Watched band</div>
      <div class="v" style="font-size:15px">${fx(bLo, 3)} – ${fx(bHi, 3)}</div>
      <div class="n">gap between saturated ham and phishing</div></div>
    <div class="tile"><div class="k">Band rate before</div>
      <div class="v ok" style="font-size:17px">${fx(d.band_rate_pre, 4)}</div>
      <div class="n">null rate r₀ = ${fx(d.r0, 4)}</div></div>
    <div class="tile"><div class="k">Band rate after</div>
      <div class="v bad" style="font-size:17px">${fx(d.band_rate_post, 4)}</div>
      <div class="n">${lift} the pre-shift rate</div></div>
    <div class="tile"><div class="k">Labels used</div>
      <div class="v acc" style="font-size:17px">0</div>
      <div class="n">the rate change is all it sees</div></div>
  </div>`;
}

/* ══════════════════════════ soar ══════════════════════════ */
function wireSoar() {
  $('#btn-refresh-soar').onclick = refreshSoar;
  $('#btn-clear-soar').onclick = async () => {
    await api('/api/soar/incidents', { method: 'DELETE' });
    refreshSoar();
  };
  $$('[data-pb]').forEach(b => b.onclick = async () => {
    const p = await api('/api/soar/playbook/' + b.dataset.pb);
    $('#pb-view').innerHTML = p.steps.map(s => `<div class="pb-step">
      <span class="${s.status === 'executed' ? 'pb-ok' : 'pb-skip'}">${s.status === 'executed' ? '✓' : '·'}</span>
      <span><span class="pb-a">${esc(s.action)}</span><br><span class="pb-d">${esc(s.detail)}</span></span>
    </div>`).join('');
  });
}

async function refreshSoar() {
  let d;
  try { d = await api('/api/soar/incidents?limit=100'); } catch { return; }

  const n = { BLOCK: 0, DEFUSE: 0 };
  d.incidents.forEach(i => { n[i.tier] = (n[i.tier] || 0) + 1; });

  $('#soar-tiles').innerHTML = `
    <div class="tile"><div class="k">Total incidents</div><div class="v">${d.count}</div><div class="n">this session</div></div>
    <div class="tile"><div class="k">P1 — BLOCK</div><div class="v bad">${n.BLOCK || 0}</div><div class="n">quarantined</div></div>
    <div class="tile"><div class="k">P2 — DEFUSE</div><div class="v warn">${n.DEFUSE || 0}</div><div class="n">stripped and delivered</div></div>
    <div class="tile"><div class="k">Connector</div><div class="v acc" style="font-size:15px">${$('#soar-endpoint').value ? 'WEBHOOK' : 'SIMULATION'}</div><div class="n">${$('#soar-endpoint').value ? 'live POST' : 'in-memory queue'}</div></div>`;

  if (!d.incidents.length) {
    $('#soar-list').innerHTML = '<div class="empty">No incidents yet. Analyse an email on the Detect tab.</div>';
    return;
  }

  $('#soar-list').innerHTML = d.incidents.map(i => `
    <div class="inc ${esc(i.tier)}" data-id="${esc(i.id)}">
      <div class="inc-h">
        <span class="inc-id">#${esc(i.id)}</span>
        <span class="inc-t ${esc(i.tier)}">${esc(i.tier)}</span>
        <span class="inc-m">${esc(i.severity)}</span>
        <span class="inc-m">conf ${fx(i.confidence, 4)} / thr ${fx(i.threshold, 4)}</span>
        <span class="inc-m">${esc(i.source_det)}</span>
        <span class="inc-m">${esc(i.cal_method)}</span>
        <span class="spacer"></span>
        <span class="inc-m">${new Date(i.timestamp * 1000).toLocaleTimeString()}</span>
      </div>
      <div class="inc-body">
        <div class="card-t" style="margin:11px 0 7px">Playbook</div>
        ${(i.playbook || []).map(s => `<div class="pb-step">
          <span class="${s.status === 'executed' ? 'pb-ok' : 'pb-skip'}">${s.status === 'executed' ? '✓' : '·'}</span>
          <span><span class="pb-a">${esc(s.action)}</span><br><span class="pb-d">${esc(s.detail)}</span></span>
        </div>`).join('')}
        <div class="row" style="margin-top:11px">
          <button class="btn ghost sm" data-dispatch="${esc(i.id)}">Dispatch to connector</button>
          <span class="small faint" data-res="${esc(i.id)}"></span>
        </div>
        <pre class="json">${esc(JSON.stringify(i, null, 2))}</pre>
      </div>
    </div>`).join('');

  $$('#soar-list .inc-h').forEach(h => h.onclick = () => h.parentElement.classList.toggle('open'));
  $$('[data-dispatch]').forEach(b => b.onclick = async e => {
    e.stopPropagation();
    const id = b.dataset.dispatch;
    const out = $(`[data-res="${id}"]`);
    out.textContent = 'dispatching…';
    try {
      const r = await api('/api/soar/dispatch', {
        method: 'POST',
        body: JSON.stringify({ incident_id: id, endpoint: $('#soar-endpoint').value || null }),
      });
      out.textContent = `${r.mode}: ${JSON.stringify(r.result)}`;
    } catch (err) { out.textContent = 'error: ' + err.message; }
  });
}

/* ══════════════════════════ api tab ══════════════════════════ */
const ENDPOINTS = [
  ['GET', '/api/health', 'Liveness, data mode, which detectors are loaded.'],
  ['GET', '/api/stats', 'Corpus composition, feature groups, run_all and extended summaries.'],
  ['GET', '/api/samples', 'Built-in demo emails covering every zero-trust tier.'],
  ['POST', '/api/detect', 'Score one email → confidence, tier, per-feature contributions, SOAR incident.'],
  ['POST', '/api/detect/batch', 'Score up to 500 emails in one call; optionally raise incidents for each.'],
  ['GET', '/api/features', 'Extract the 17 stylometric features without scoring.'],
  ['GET', '/api/transfer', 'N×N cross-generator AUC and F1 matrices.'],
  ['GET', '/api/calibration', 'Per-pair thresholds and F1 for stale / DM / EM / self-training / oracle.'],
  ['GET', '/api/drift/simulate', 'Score a stream that switches generator mid-way; returns martingale wealth and alarm index.'],
  ['POST', '/api/drift/session', 'Open an online martingale session for one-score-at-a-time streaming.'],
  ['POST', '/api/drift/step', 'Feed one score into an open session; returns wealth and alarm flag.'],
  ['GET', '/api/soar/incidents', 'Incident log with the playbook resolved per incident.'],
  ['DELETE', '/api/soar/incidents', 'Clear the log.'],
  ['POST', '/api/soar/dispatch', 'Send an incident to a webhook, or to the simulation connector.'],
  ['GET', '/api/soar/playbook/{tier}', 'The response steps mapped to BLOCK / DEFUSE / DELIVER.'],
];

function renderApiTab() {
  $('#eplist').innerHTML = ENDPOINTS.map(([v, p, d]) => `
    <div class="ep"><div class="ep-h"><span class="verb ${v}">${v}</span><span class="ep-p">${esc(p)}</span></div>
    <div class="ep-d">${esc(d)}</div></div>`).join('');

  const origin = location.origin || 'http://localhost:8000';
  $('#curl-block').textContent =
`# score one email
curl -s -X POST ${origin}/api/detect \\
  -H 'Content-Type: application/json' \\
  -d '{"text":"URGENT verify your account at https://secure-verify.com","detector":"xgboost","trained_on":"pooled"}'

# transfer matrix
curl -s ${origin}/api/transfer

# drift stream, shift injected at email 300
curl -s '${origin}/api/drift/simulate?n=600&shift_at=300'

# incident log
curl -s ${origin}/api/soar/incidents`;
}

function wireApi() {
  $('#btn-api-send').onclick = async () => {
    const out = $('#api-out'), st = $('#api-status');
    st.innerHTML = '<span class="spin"></span>';
    try {
      const body = JSON.parse($('#api-body').value);
      const t0 = performance.now();
      const r = await api('/api/detect', { method: 'POST', body: JSON.stringify(body) });
      st.textContent = `200 OK · ${(performance.now() - t0).toFixed(0)} ms`;
      out.textContent = JSON.stringify(r, null, 2);
    } catch (e) {
      st.textContent = 'error';
      out.textContent = e.message;
    }
  };
}

boot();
