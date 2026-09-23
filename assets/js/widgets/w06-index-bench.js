// w06 Flat vs HNSW 속도 비교
// One concept: an approximate index (HNSW) trades exactness for speed. The widget
// builds an exact Flat index and an HNSW graph over the same seeded random vectors,
// runs the same queries on both, and reports query time, speedup and Recall@10.
// Failure demos: tiny M / efSearch → HNSW misses true neighbors; small N → no speedup.

import { registerOutput, unregisterOutput } from '../site/result.js';

const K = 10; // top-k compared
const Q = 100; // queries per measurement
const SEED = 42;
const SLICE_MS = 30; // work per slice before yielding to the UI
const MIN_TIMED_MS = 25; // repeat a query batch until this much time was measured (coarse browser timers)
const EF_LIST = [1, 2, 4, 8, 10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 400];
const EFC_LIST = [8, 16, 24, 32, 48, 64, 100, 128, 200, 300, 400];
const CURVE_EF = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const DIMS = [32, 64, 128, 384];

const BASE = { n: 3000, dim: 128, dist: 'topic', M: 12, efC: 48, ef: 32 };
const PRESETS = {
  base: { label: '기본: 3천 개', ...BASE },
  ef1: { label: '실패: efSearch 1', ...BASE, ef: 1 },
  m2: { label: '실패: M 2', ...BASE, M: 2 },
  small: { label: '실패: N 1천 · 32차원', ...BASE, n: 1000, dim: 32, ef: 64 },
  random: { label: '실패: 무작위 분포', ...BASE, dist: 'random' },
  big: { label: '큰 N: 2만 개 (수 초)', ...BASE, n: 20000, ef: 64 },
};

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget" data-w06>
    <h3 class="widget__title">Flat vs HNSW 속도 비교</h3>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">벡터 수 N <output data-out="n"></output></span>
        <input type="range" data-in="n" min="1000" max="50000" step="1000">
      </label>
      <label class="field">
        <span class="field__label">차원</span>
        <select data-in="dim">${DIMS.map((d) => `<option value="${d}">${d}${d === 384 ? ' (MiniLM·e5-small 크기, 느림)' : ''}</option>`).join('')}</select>
      </label>
      <label class="field">
        <span class="field__label">데이터 분포</span>
        <select data-in="dist">
          <option value="topic">주제 군집 20개 (실제 임베딩과 비슷)</option>
          <option value="random">완전 무작위 (구조 없음)</option>
        </select>
      </label>
      <label class="field">
        <span class="field__label">M (노드당 이웃 수) <output data-out="M"></output></span>
        <input type="range" data-in="M" min="2" max="48" step="1">
      </label>
      <label class="field">
        <span class="field__label">efConstruction (구축 탐색 폭) <output data-out="efC"></output></span>
        <input type="range" data-in="efC" min="0" max="${EFC_LIST.length - 1}" step="1">
      </label>
      <label class="field">
        <span class="field__label">efSearch (검색 탐색 폭) <output data-out="ef"></output></span>
        <input type="range" data-in="ef" min="0" max="${EF_LIST.length - 1}" step="1">
      </label>
    </div>
    <div class="btn-row" data-slot="presets"></div>
    <div class="w06-progress" data-slot="progress" role="status" aria-live="polite">
      <div class="w06-progress__row">
        <span class="spinner" data-slot="spin" aria-hidden="true"></span>
        <span data-slot="phase">준비 중…</span>
        <span class="w06-spacer"></span>
        <button type="button" class="btn small" data-act="cancel">■ 취소</button>
        <button type="button" class="btn small" data-act="rerun" hidden>⟳ 다시 측정</button>
      </div>
      <div class="bar-progress"><div class="bar-progress__fill" data-slot="bar"></div></div>
    </div>
    <div data-slot="error"></div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w06-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>
    <h4>질의 1회 평균 시간</h4>
    <div class="w06-bars" data-slot="bars"></div>
    <h4>질의 #1의 정답 이웃 10개 중 HNSW가 찾은 것</h4>
    <div class="w06-hits" data-slot="hits"></div>
    <h4>efSearch에 따른 Recall@10</h4>
    <div data-slot="curve"></div>
    <p class="w06-note" data-slot="layers"></p>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ n?: number, dim?: number, M?: number, efC?: number, ef?: number, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w06:${++seq}`;
  const st = { ctrl, outputId, job: null, timer: 0 };
  state.set(el, st);

  const s = { ...BASE };
  for (const key of Object.keys(BASE)) if (options[key] !== undefined) s[key] = options[key];
  // caches: data (N, dim, dist) → flat ground truth; graph (M, efC); search (ef)
  let dataC = null;
  let graphC = null;
  let searchC = null;

  $('[data-slot=presets]').innerHTML = Object.entries(PRESETS)
    .map(([k, p]) => `<button type="button" class="btn ghost" data-preset="${k}">${p.label}</button>`)
    .join('');

  const nearest = (list, v) => list.reduce((b, x, i) => (Math.abs(x - v) < Math.abs(list[b] - v) ? i : b), 0);

  function syncInputs() {
    $('[data-in=n]').value = s.n;
    $('[data-in=dim]').value = s.dim;
    $('[data-in=dist]').value = s.dist;
    $('[data-in=M]').value = s.M;
    $('[data-in=efC]').value = nearest(EFC_LIST, s.efC);
    $('[data-in=ef]').value = nearest(EF_LIST, s.ef);
    $('[data-out=n]').textContent = `${s.n.toLocaleString()}개`;
    $('[data-out=M]').textContent = s.M;
    $('[data-out=efC]').textContent = s.efC;
    $('[data-out=ef]').textContent = s.ef < K ? `${s.ef} (< k)` : s.ef;
  }

  // ---------- progress ----------
  function progress(phase, frac) {
    $('[data-slot=progress]').classList.remove('is-idle');
    $('[data-slot=spin]').hidden = false;
    $('[data-slot=phase]').textContent = phase;
    $('[data-slot=bar]').style.width = `${Math.round(frac * 100)}%`;
    $('[data-act=cancel]').hidden = false;
    $('[data-act=rerun]').hidden = true;
  }
  function idle(text) {
    $('[data-slot=progress]').classList.add('is-idle');
    $('[data-slot=spin]').hidden = true;
    $('[data-slot=phase]').textContent = text;
    $('[data-slot=bar]').style.width = '100%';
    $('[data-act=cancel]').hidden = true;
    $('[data-act=rerun]').hidden = false;
  }

  /** Run fn(i) for i in [from, to) in time slices; returns busy milliseconds (yields excluded). */
  async function sliced(job, from, to, fn, onSlice) {
    let busy = 0;
    let i = from;
    while (i < to) {
      const t0 = performance.now();
      do fn(i++);
      while (i < to && performance.now() - t0 < SLICE_MS);
      busy += performance.now() - t0;
      onSlice?.(i, busy);
      await new Promise((r) => setTimeout(r, 0));
      if (job.cancelled) throw new DOMException('cancelled', 'AbortError');
    }
    return busy;
  }

  /** Time fn(q, first) over the Q queries; extra timing-only passes if the batch was too fast to measure. */
  async function perQuery(job, fn, onSlice, repeat = true) {
    let busy = await sliced(job, 0, Q, (q) => fn(q, true), onSlice);
    let calls = Q;
    while (repeat && busy < MIN_TIMED_MS && calls < Q * 50) {
      busy += await sliced(job, 0, Q, (q) => fn(q, false));
      calls += Q;
    }
    return busy / calls;
  }

  // ---------- pipeline ----------
  async function run(from) {
    if (st.job) Object.assign(st.job, { cancelled: true, superseded: true });
    const job = { cancelled: false, superseded: false };
    st.job = job;
    const cfg = { ...s };
    const dataKey = `${cfg.n}|${cfg.dim}|${cfg.dist}`;
    const graphKey = `${dataKey}|${cfg.M}|${cfg.efC}`;
    if (from === 'data' || dataC?.key !== dataKey) dataC = null;
    if (!dataC || from === 'graph' || graphC?.key !== graphKey) graphC = null;
    searchC = null;
    $('[data-slot=error]').replaceChildren();

    try {
      if (!dataC) {
        const { n, dim } = cfg;
        const gen = new VectorGen(dim, { seed: SEED, clusters: cfg.dist === 'topic' ? 20 : 0 });
        const queries = new Float32Array(Q * dim);
        gen.fill(queries, 0, Q);
        const data = new Float32Array(n * dim);
        const ROWS = 250;
        await sliced(job, 0, Math.ceil(n / ROWS), (b) => gen.fill(data, b * ROWS, Math.min(n, (b + 1) * ROWS)), (b) =>
          progress(`벡터 ${n.toLocaleString()}개 생성 중…`, (b * ROWS) / n),
        );
        // warm up the JIT so the first timed queries are not penalized
        for (let q = 0; q < 3; q++) flatSearch(data, dim, queries, q * dim, K);
        const gt = [];
        const flatMs = await perQuery(
          job,
          (q, first) => {
            const r = flatSearch(data, dim, queries, q * dim, K);
            if (first) gt.push(r);
          },
          (q) => progress(`Flat: 질의 ${Q}개를 벡터 전부와 비교해 검색 중…`, q / Q),
        );
        dataC = { key: dataKey, n, dim, data, queries, gt, flatMs };
      }

      if (!graphC) {
        const { data, dim, n } = dataC;
        const index = new HNSW(data, dim, { M: cfg.M, efConstruction: cfg.efC });
        const buildMs = await sliced(job, 0, n, (i) => index.insert(i), (i, busy) => {
          const left = (busy / i) * (n - i);
          progress(`HNSW 그래프 구축 ${i.toLocaleString()} / ${n.toLocaleString()} · 남은 시간 약 ${fmtSec(left)}`, i / n);
        });
        graphC = { key: graphKey, index, buildMs };
      }

      const { index } = graphC;
      const { dim, queries, gt } = dataC;
      const measure = async (ef, label, repeat) => {
        for (let q = 0; q < 3; q++) index.search(queries, q * dim, K, ef);
        const res = [];
        let dist = 0;
        const ms = await perQuery(
          job,
          (q, first) => {
            const before = index.dcount;
            const r = index.search(queries, q * dim, K, ef);
            if (first) {
              res.push(r);
              dist += index.dcount - before;
            }
          },
          (q) => progress(label, q / Q),
          repeat,
        );
        const recall = res.reduce((a, r, q) => a + recallOverlap(r, gt[q], K), 0) / Q;
        return { ef, res, recall, ms, dist: dist / Q };
      };
      const main = await measure(cfg.ef, `HNSW: 같은 질의 ${Q}개 검색 중 (efSearch ${cfg.ef})…`, true);
      searchC = { ...main, curve: [] };
      render(cfg);
      for (const ef of CURVE_EF) {
        const r = ef === cfg.ef ? main : await measure(ef, `efSearch ${ef}로 Recall 곡선 측정 중…`, false);
        searchC.curve.push({ ef, recall: r.recall, ms: r.ms });
      }
      render(cfg);
      idle(`완료 · N ${cfg.n.toLocaleString()} · ${cfg.dim}차원 · 질의 ${Q}개 · 같은 설정은 항상 같은 벡터(시드 ${SEED})`);
    } catch (err) {
      if (err.name === 'AbortError') {
        if (!job.superseded && !ctrl.signal.aborted) idle('취소했다. 설정을 바꾸거나 “다시 측정”을 누른다.');
        return;
      }
      idle('오류로 멈췄다.');
      $('[data-slot=error]').innerHTML = `<div class="widget__error" role="alert">측정 중 오류가 났다 (${escapeHtml(
        err.message,
      )}). N이나 차원을 줄여 다시 시도한다. 메모리가 부족하면 탭이 느려질 수 있다.</div>`;
    } finally {
      if (st.job === job) st.job = null;
    }
  }

  // ---------- rendering ----------
  function render(cfg) {
    if (!dataC || !graphC || !searchC) return;
    const { flatMs, n, gt } = dataC;
    const { buildMs, index } = graphC;
    const { recall, ms, dist, res, curve } = searchC;
    const speed = flatMs / ms;
    const saved = flatMs - ms;
    const breakEven = saved > 0 ? Math.ceil(buildMs / saved) : Infinity;

    $('[data-slot=verdict]').innerHTML = verdictHtml(cfg, recall, speed, flatMs, buildMs);
    $('[data-slot=stats]').innerHTML = [
      ['Flat 질의', `${fmtMs(flatMs)}`],
      ['HNSW 질의', `${fmtMs(ms)}`],
      ['속도 향상', `×${speed.toFixed(1)}`],
      ['Recall@10', recall.toFixed(2)],
      ['HNSW 구축', fmtSec(buildMs)],
      ['질의당 거리 계산', `${n.toLocaleString()} → ${Math.round(dist).toLocaleString()}`],
      ['본전 질의 수', Number.isFinite(breakEven) ? `${breakEven.toLocaleString()}회` : '없음'],
    ]
      .map(([l, v]) => `<div class="stat"><span class="stat__label">${l}</span><span class="stat__value">${v}</span></div>`)
      .join('');

    const max = Math.max(flatMs, ms);
    $('[data-slot=bars]').innerHTML = [
      ['Flat (정확)', flatMs, 'flat'],
      ['HNSW (근사)', ms, 'hnsw'],
    ]
      .map(
        ([l, v, c]) => `<div class="w06-bar"><span class="w06-bar__label">${l}</span>
        <span class="w06-bar__track"><i class="w06-bar__fill w06-bar__fill--${c}" style="width:${Math.max(1, (v / max) * 100)}%"></i></span>
        <span class="w06-bar__val">${fmtMs(v)}</span></div>`,
      )
      .join('');

    const found = new Set(res[0]);
    const hitN = gt[0].filter((id) => found.has(id)).length;
    $('[data-slot=hits]').innerHTML = `<ol class="w06-hitlist" aria-label="정답 이웃 10개의 발견 여부">${gt[0]
      .map(
        (id, r) => `<li class="${found.has(id) ? 'is-hit' : 'is-miss'}" title="정답 ${r + 1}위 · 벡터 #${id}">
        <b>${r + 1}</b><span>${found.has(id) ? '찾음' : '놓침'}</span></li>`,
      )
      .join('')}</ol><p class="w06-note">정답(Flat) 1~10위 가운데 <b>${hitN}개</b>를 HNSW도 돌려주었다.</p>`;

    $('[data-slot=curve]').innerHTML = curveSvg(curve, cfg.ef, recall);
    const sizes = index.layerSizes();
    $('[data-slot=layers]').innerHTML = `그래프 층별 노드 수 (위층 → 아래층): ${sizes
      .map((c, l) => `L${l} ${c.toLocaleString()}`)
      .reverse()
      .join(' · ')}. 맨 아래층 L0에는 모든 벡터가 있고, 위로 갈수록 드물어진다.`;
  }

  // ---------- events ----------
  const schedule = (from, delay) => {
    clearTimeout(st.timer);
    if (st.job) Object.assign(st.job, { cancelled: true, superseded: true });
    progress('바뀐 설정으로 다시 측정한다…', 0);
    st.timer = setTimeout(() => run(from), delay);
  };
  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  const bindRange = (key, from, map = Number) =>
    on(`[data-in=${key}]`, 'input', (e) => {
      s[key] = map(e.target.value);
      syncInputs();
      schedule(from, from === 'search' ? 80 : 350);
    });
  bindRange('n', 'data');
  bindRange('M', 'graph');
  bindRange('efC', 'graph', (v) => EFC_LIST[Number(v)]);
  bindRange('ef', 'search', (v) => EF_LIST[Number(v)]);
  on('[data-in=dim]', 'change', (e) => {
    s.dim = Number(e.target.value);
    syncInputs();
    schedule('data', 0);
  });
  on('[data-in=dist]', 'change', (e) => {
    s.dist = e.target.value;
    syncInputs();
    schedule('data', 0);
  });
  on('[data-slot=presets]', 'click', (e) => {
    const p = PRESETS[e.target.closest('[data-preset]')?.dataset.preset];
    if (!p) return;
    const { label, ...vals } = p;
    Object.assign(s, vals);
    syncInputs();
    schedule('search', 0); // caches decide which stages really rerun
  });
  on('[data-act=cancel]', 'click', () => {
    clearTimeout(st.timer);
    if (st.job) st.job.cancelled = true;
  });
  on('[data-act=rerun]', 'click', () => schedule('data', 0));

  syncInputs();
  registerOutput(outputId, {
    title: options.outputTitle ?? 'Flat vs HNSW 측정 결과',
    node: out,
    inlineHost: $('[data-slot=out-inline]'),
  });
  $('[data-slot=verdict]').innerHTML =
    '<div class="widget__status"><span class="spinner" aria-hidden="true"></span> 첫 측정을 실행하는 중…</div>';
  run('data');
}

export function unmount(el) {
  const st = state.get(el);
  if (!st) return;
  clearTimeout(st.timer);
  if (st.job) st.job.cancelled = true;
  st.ctrl.abort();
  unregisterOutput(st.outputId);
  state.delete(el);
  el.replaceChildren();
}

// ---------- rendering helpers ----------

function verdictHtml(cfg, recall, speed, flatMs, buildMs) {
  const r = recall.toFixed(2);
  if (recall < 0.9) {
    let why;
    if (cfg.M <= 4) why = `M 값(${cfg.M})이 너무 작아 노드마다 이웃이 몇 개 없다. 그래프가 끊기듯 성겨서 탐색이 진짜 이웃 근처까지 가지 못하고 멈춘다.`;
    else if (cfg.ef < K) why = `efSearch(${cfg.ef})가 k(10)보다 작다. 후보를 ${cfg.ef}개만 들고 탐욕적으로 내려가다 첫 “막다른 곳”에서 멈춘다.`;
    else if (cfg.dist === 'random') why = '구조가 없는 무작위 벡터에서는 “가까운 이웃의 이웃도 가깝다”는 그래프의 전제가 약하다. 같은 ef로도 훨씬 많이 놓친다.';
    else why = `N이 커질수록 같은 efSearch로는 더 많이 놓친다. efSearch를 올려 본다 (속도는 그만큼 준다).`;
    return `<div class="callout callout--danger"><span class="callout__title">HNSW가 진짜 최근접 이웃을 놓친다 — Recall@10 = ${r}</span>${why}
      Flat은 항상 1.00이다. 이 상태로 RAG를 만들면 정답 청크가 top-10에서 빠진다.</div>`;
  }
  if (speed < 1.5) {
    return `<div class="callout"><span class="callout__title">HNSW가 빠르지 않다 — ×${speed.toFixed(1)}</span>
      N=${cfg.n.toLocaleString()}이면 Flat도 질의 한 번에 ${fmtMs(flatMs)}다. 그래프를 따라가는 부가 작업(힙, 방문 표시) 때문에 이득이 없고,
      구축에 ${fmtSec(buildMs)}를 따로 썼다. 문서가 적으면 Flat이 정답이다.</div>`;
  }
  return `<div class="callout callout--ok"><span class="callout__title">HNSW가 ×${speed.toFixed(1)} 빠르고 Recall@10 = ${r}</span>
    정답 이웃을 거의 다 찾으면서 비교 횟수를 크게 줄였다. efSearch를 낮추면 더 빨라지지만 어디서부터 놓치기 시작하는지 확인한다.</div>`;
}

/** Recall vs efSearch (log2 x-axis), current setting highlighted. */
function curveSvg(curve, curEf, curRecall) {
  if (!curve.length) return '<p class="w06-note">측정 중…</p>';
  const W = 360;
  const H = 170;
  const L = 36;
  const R = 12;
  const T = 12;
  const B = 34;
  const xs = (ef) => L + (Math.log2(ef) / Math.log2(400)) * (W - L - R);
  const ys = (r) => T + (1 - r) * (H - T - B);
  const pts = curve.map((c) => `${xs(c.ef).toFixed(1)},${ys(c.recall).toFixed(1)}`).join(' ');
  const yTicks = [0, 0.5, 0.9, 1]
    .map(
      (v) => `<line class="w06-grid${v === 0.9 ? ' w06-grid--target' : ''}" x1="${L}" x2="${W - R}" y1="${ys(v)}" y2="${ys(v)}"/>
      <text class="w06-ax" x="${L - 5}" y="${ys(v) + 4}" text-anchor="end">${v}</text>`,
    )
    .join('');
  const xTicks = [1, 4, 16, 64, 256]
    .map((v) => `<text class="w06-ax" x="${xs(v)}" y="${H - B + 16}" text-anchor="middle">${v}</text>`)
    .join('');
  const dots = curve.map((c) => `<circle class="w06-dot" cx="${xs(c.ef)}" cy="${ys(c.recall)}" r="3"/>`).join('');
  return `<svg class="w06-curve" viewBox="0 0 ${W} ${H}" role="img" aria-label="efSearch가 커질수록 Recall@10이 올라가는 곡선. 현재 efSearch ${curEf}에서 ${curRecall.toFixed(2)}">
    ${yTicks}${xTicks}
    <text class="w06-ax" x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">efSearch (로그 눈금) · 점선 = Recall 0.9</text>
    <polyline class="w06-line" points="${pts}"/>${dots}
    <circle class="w06-cur" cx="${xs(curEf)}" cy="${ys(curRecall)}" r="6"/>
  </svg>`;
}

function fmtMs(ms) {
  if (ms < 0.1) return `${(ms * 1000).toFixed(0)}µs`;
  if (ms < 10) return `${ms.toFixed(2)}ms`;
  return `${ms.toFixed(0)}ms`;
}

function fmtSec(ms) {
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

// ---------- pure algorithm code (DOM-free, tested in Node) ----------

/** Seeded PRNG (mulberry32) so every run with the same settings is identical. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand) {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

/**
 * Streams unit vectors of size dim. With clusters > 0 every vector is a random
 * "topic" center plus the same amount of noise, like real embeddings of documents
 * about a handful of topics; clusters = 0 gives structureless random directions.
 * fill() can be called in slices so the browser can yield between them.
 */
export class VectorGen {
  constructor(dim, { seed = 42, clusters = 20 } = {}) {
    this.dim = dim;
    this.rand = mulberry32(seed);
    this.centers = Array.from({ length: clusters }, () => {
      const v = new Float32Array(dim);
      for (let j = 0; j < dim; j++) v[j] = gaussian(this.rand);
      return v;
    });
  }
  /** write rows [from, to) of out */
  fill(out, from, to) {
    const { dim, rand, centers } = this;
    for (let i = from; i < to; i++) {
      const o = i * dim;
      const c = centers.length ? centers[Math.floor(rand() * centers.length)] : null;
      let s = 0;
      for (let j = 0; j < dim; j++) {
        const x = (c ? c[j] : 0) + gaussian(rand);
        out[o + j] = x;
        s += x * x;
      }
      const inv = 1 / Math.sqrt(s);
      for (let j = 0; j < dim; j++) out[o + j] *= inv;
    }
  }
}

/** n unit vectors in one row-major Float32Array. */
export function makeVectors(n, dim, opts = {}) {
  const out = new Float32Array(n * dim);
  new VectorGen(dim, opts).fill(out, 0, n);
  return out;
}

/** Inner product of row a (offset ao) of A and row b (offset bo) of B, 4-way unrolled. */
export function dotAt(A, ao, B, bo, dim) {
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  let j = 0;
  for (; j + 3 < dim; j += 4) {
    s0 += A[ao + j] * B[bo + j];
    s1 += A[ao + j + 1] * B[bo + j + 1];
    s2 += A[ao + j + 2] * B[bo + j + 2];
    s3 += A[ao + j + 3] * B[bo + j + 3];
  }
  for (; j < dim; j++) s0 += A[ao + j] * B[bo + j];
  return s0 + s1 + s2 + s3;
}

/** Exact top-k by inner product (= cosine for unit vectors). Returns ids, best first. */
export function flatSearch(data, dim, q, qo, k) {
  const n = data.length / dim;
  const ids = new Int32Array(k).fill(-1);
  const sc = new Float32Array(k).fill(-Infinity);
  for (let i = 0; i < n; i++) {
    const s = dotAt(data, i * dim, q, qo, dim);
    if (s <= sc[k - 1]) continue;
    let p = k - 1;
    while (p > 0 && sc[p - 1] < s) {
      sc[p] = sc[p - 1];
      ids[p] = ids[p - 1];
      p--;
    }
    sc[p] = s;
    ids[p] = i;
  }
  return Array.from(ids);
}

/** Binary heap over (id, dist). max=true keeps the largest dist on top. */
class Heap {
  constructor(max) {
    this.sign = max ? -1 : 1; // stored key = sign * d, smallest key on top
    this.ids = [];
    this.ks = [];
  }
  get size() {
    return this.ids.length;
  }
  topId() {
    return this.ids[0];
  }
  topD() {
    return this.ks[0] * this.sign;
  }
  push(id, d) {
    const { ids, ks } = this;
    const k = d * this.sign;
    let i = ids.length;
    ids.push(id);
    ks.push(k);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (ks[p] <= k) break;
      ids[i] = ids[p];
      ks[i] = ks[p];
      i = p;
    }
    ids[i] = id;
    ks[i] = k;
  }
  pop() {
    const { ids, ks } = this;
    const lastId = ids.pop();
    const lastK = ks.pop();
    const n = ids.length;
    if (n === 0) return;
    let i = 0;
    for (;;) {
      let c = 2 * i + 1;
      if (c >= n) break;
      if (c + 1 < n && ks[c + 1] < ks[c]) c++;
      if (ks[c] >= lastK) break;
      ids[i] = ids[c];
      ks[i] = ks[c];
      i = c;
    }
    ids[i] = lastId;
    ks[i] = lastK;
  }
  /** entries as [{id, d}] sorted closest first */
  sorted() {
    const s = this.sign;
    return this.ids.map((id, i) => ({ id, d: this.ks[i] * s })).sort((a, b) => a.d - b.d);
  }
}

/**
 * HNSW (Hierarchical Navigable Small World), plain JS.
 * Distance = 1 − dot (vectors are unit length). Layer 0 keeps up to 2M links,
 * upper layers up to M. Neighbors are chosen with the "heuristic" of the paper
 * (Malkov & Yashunin 2018, Alg. 4), the same rule hnswlib/FAISS use.
 */
export class HNSW {
  constructor(data, dim, { M = 16, efConstruction = 100, seed = 7 } = {}) {
    this.data = data;
    this.dim = dim;
    this.n = data.length / dim;
    this.M = Math.max(2, M);
    this.M0 = this.M * 2;
    this.efC = Math.max(efConstruction, this.M);
    this.mL = 1 / Math.log(this.M);
    this.rand = mulberry32(seed);
    this.level = new Int8Array(this.n);
    this.nbr0 = new Int32Array(this.n * this.M0);
    this.cnt0 = new Int32Array(this.n);
    this.upper = new Array(this.n); // upper[i][l - 1] = neighbor id array for layer l ≥ 1
    this.entry = -1;
    this.maxLevel = -1;
    this.count = 0;
    this.visited = new Uint32Array(this.n);
    this.tag = 0;
    this.dcount = 0; // distance computations (for "work per query")
  }

  /** distance between stored node a and vector q at offset qo */
  _d(a, q, qo) {
    this.dcount++;
    return 1 - dotAt(this.data, a * this.dim, q, qo, this.dim);
  }

  _links(i, l) {
    if (l === 0) return this.nbr0.subarray(i * this.M0, i * this.M0 + this.cnt0[i]);
    return this.upper[i][l - 1];
  }

  _setLinks(i, l, ids) {
    if (l === 0) {
      this.nbr0.set(ids, i * this.M0);
      this.cnt0[i] = ids.length;
    } else this.upper[i][l - 1] = ids.slice();
  }

  _nextTag() {
    if (++this.tag === 0xffffffff) {
      this.visited.fill(0);
      this.tag = 1;
    }
    return this.tag;
  }

  /** Greedy walk with beam 1 on an upper layer. */
  _greedy(q, qo, ep, epD, l) {
    for (let changed = true; changed; ) {
      changed = false;
      for (const e of this._links(ep, l)) {
        const d = this._d(e, q, qo);
        if (d < epD) {
          epD = d;
          ep = e;
          changed = true;
        }
      }
    }
    return [ep, epD];
  }

  /**
   * Beam search on layer l with beam width ef. Returns [{id, d}] sorted closest first.
   * If keep > ef, the best `keep` of every node evaluated are returned (so ef < k
   * still yields k answers — just worse ones).
   */
  _searchLayer(q, qo, ep, epD, ef, l, keep = ef) {
    const tag = this._nextTag();
    const vis = this.visited;
    vis[ep] = tag;
    const cand = new Heap(false);
    const res = new Heap(true);
    const pool = keep > ef ? new Heap(true) : null;
    cand.push(ep, epD);
    res.push(ep, epD);
    pool?.push(ep, epD);
    while (cand.size) {
      const c = cand.topId();
      const cd = cand.topD();
      // stop when the closest unexpanded candidate is worse than the worst kept result
      // (with a pool, also keep walking until k nodes have been evaluated)
      if (cd > res.topD() && res.size >= ef && !(pool && pool.size < keep)) break;
      cand.pop();
      for (const e of this._links(c, l)) {
        if (vis[e] === tag) continue;
        vis[e] = tag;
        const d = this._d(e, q, qo);
        const poolShort = pool && pool.size < keep;
        if (pool && (poolShort || d < pool.topD())) {
          pool.push(e, d);
          if (pool.size > keep) pool.pop();
        }
        if (res.size < ef || d < res.topD()) {
          cand.push(e, d);
          res.push(e, d);
          if (res.size > ef) res.pop();
        } else if (poolShort) cand.push(e, d);
      }
    }
    return (pool ?? res).sorted();
  }

  /** Paper Alg. 4: keep a candidate only if it is closer to the base than to any kept one. */
  _select(cands, M) {
    if (cands.length <= M) return cands.map((c) => c.id);
    const out = [];
    const dim = this.dim;
    for (const c of cands) {
      if (out.length >= M) break;
      let good = true;
      for (const r of out) {
        if (this._d(c.id, this.data, r * dim) < c.d) {
          good = false;
          break;
        }
      }
      if (good) out.push(c.id);
    }
    return out;
  }

  /** Insert node i (0..n−1). Call for i = 0,1,2,… in order. */
  insert(i) {
    const { data, dim } = this;
    const qo = i * dim;
    const lvl = Math.min(16, Math.floor(-Math.log(1 - this.rand()) * this.mL));
    this.level[i] = lvl;
    if (lvl > 0) this.upper[i] = Array.from({ length: lvl }, () => []);
    this.count++;
    if (this.entry < 0) {
      this.entry = i;
      this.maxLevel = lvl;
      return;
    }
    let ep = this.entry;
    let epD = this._d(ep, data, qo);
    for (let l = this.maxLevel; l > lvl; l--) [ep, epD] = this._greedy(data, qo, ep, epD, l);
    for (let l = Math.min(lvl, this.maxLevel); l >= 0; l--) {
      const W = this._searchLayer(data, qo, ep, epD, this.efC, l);
      const nb = this._select(W, this.M);
      this._setLinks(i, l, nb);
      const Mmax = l === 0 ? this.M0 : this.M;
      for (const e of nb) this._addLink(e, i, l, Mmax);
      ep = W[0].id;
      epD = W[0].d;
    }
    if (lvl > this.maxLevel) {
      this.maxLevel = lvl;
      this.entry = i;
    }
  }

  _addLink(e, i, l, Mmax) {
    const cur = this._links(e, l);
    if (cur.length < Mmax) {
      if (l === 0) {
        this.nbr0[e * this.M0 + this.cnt0[e]] = i;
        this.cnt0[e]++;
      } else cur.push(i);
      return;
    }
    // full: re-select among old links + new node, seen from e
    const eo = e * this.dim;
    const cands = [...cur, i].map((id) => ({ id, d: this._d(id, this.data, eo) }));
    cands.sort((a, b) => a.d - b.d);
    this._setLinks(e, l, this._select(cands, Mmax));
  }

  /** Top-k ids for query vector q (offset qo), beam width ef on layer 0. */
  search(q, qo, k, ef) {
    if (this.entry < 0) return [];
    let ep = this.entry;
    let epD = this._d(ep, q, qo);
    for (let l = this.maxLevel; l > 0; l--) [ep, epD] = this._greedy(q, qo, ep, epD, l);
    return this._searchLayer(q, qo, ep, epD, Math.max(1, ef), 0, k)
      .slice(0, k)
      .map((r) => r.id);
  }

  /** number of nodes per layer, top layer last */
  layerSizes() {
    const sizes = new Array(this.maxLevel + 1).fill(0);
    for (let i = 0; i < this.count; i++) for (let l = 0; l <= this.level[i]; l++) sizes[l]++;
    return sizes;
  }
}

/** |A ∩ B| / k where B is the exact answer. */
export function recallOverlap(approx, exact, k) {
  const truth = new Set(exact.slice(0, k));
  let hit = 0;
  for (const id of approx.slice(0, k)) if (truth.has(id)) hit++;
  return hit / k;
}
