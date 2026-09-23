// w10 코드 비교 뷰어
// One concept: the same minimal RAG pipeline written three ways (our core
// modules, LangChain, LlamaIndex) — which lines do each step, what the
// framework does for you, and which hidden defaults silently change results.
// The "숨은 기본값" experiment re-runs retrieval with BM25 (no model download)
// so the effect of a framework default on hit@k is visible instantly.

import { chunk } from '../core/chunker.js';
import { BM25 } from '../core/bm25.js';
import { registerOutput, unregisterOutput } from '../site/result.js';

const DATA = new URL('../../data/', import.meta.url);

// Separator lists: ours = chunker.js default, lc = LangChain RecursiveCharacterTextSplitter default.
export const SEPARATORS = {
  ko: ['\n\n', '\n', '. ', '다. ', ' ', ''],
  lc: ['\n\n', '\n', ' ', ''],
};

export const SIM_OPTIONS = {
  size: [
    [100, '100자'],
    [150, '150자 (우리 설정)'],
    [300, '300자'],
    [1024, '1024 (LlamaIndex 기본, 토큰 → 문자로 근사)'],
    [4000, '4000자 (LangChain 기본)'],
  ],
  overlap: [
    [0, '0'],
    [20, '20 (우리 설정)'],
    [200, '200 (두 프레임워크 기본)'],
  ],
  seps: [
    ['ko', '문장 경계 포함 (우리 chunker.js)'],
    ['lc', '"\\n\\n", "\\n", " ", "" (LangChain 기본)'],
  ],
  k: [
    [1, '1'],
    [2, '2 (LlamaIndex 기본)'],
    [3, '3 (우리 설정)'],
    [4, '4 (LangChain 기본)'],
    [5, '5'],
  ],
};

export const SIM_PRESETS = {
  ours: { label: '우리 설정', size: 150, overlap: 20, seps: 'ko', k: 3 },
  langchain: { label: 'LangChain 기본값', size: 4000, overlap: 200, seps: 'lc', k: 4 },
  llamaindex: { label: 'LlamaIndex 기본값', size: 1024, overlap: 200, seps: 'ko', k: 2 },
  fail: { label: '실패 재현: 구분자만 기본값', size: 150, overlap: 20, seps: 'lc', k: 3 },
};

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w10" data-w10>
    <h3 class="widget__title">코드 비교 뷰어</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> 코드와 문서셋을 불러오는 중…
    </div>
    <div data-slot="body" hidden>
      <div class="w10-steps" role="radiogroup" aria-label="파이프라인 단계" data-slot="steps"></div>
      <div class="w10-layout-row">
        <button type="button" class="btn small ghost w10-layout" data-act="layout" aria-pressed="false">↕ 위아래로 보기</button>
      </div>
      <div class="w10-grid" data-slot="panes"></div>
      <p class="w10-note" data-slot="note"></p>

      <h4 class="w10-h">숨은 기본값 실험</h4>
      <p class="w10-help">프레임워크에 인자를 안 주면 아래 기본값이 쓰인다. 골든셋 12문항을 BM25로 다시 검색해
        “근거 문장이 든 청크가 top-k 안에 있는가(hit@k)”를 바로 계산한다. 기준은 <b>우리 설정</b>이다.</p>
      <div class="btn-row w10-presets" data-slot="presets"></div>
      <div class="widget__controls">
        <label class="field"><span class="field__label">청크 크기 <code>chunk_size</code></span><select data-in="size"></select></label>
        <label class="field"><span class="field__label">오버랩 <code>chunk_overlap</code></span><select data-in="overlap"></select></label>
        <label class="field"><span class="field__label">구분자 <code>separators</code></span><select data-in="seps"></select></label>
        <label class="field"><span class="field__label">검색 개수 <code>k</code> / <code>top_k</code></span><select data-in="k"></select></label>
      </div>
      <div data-slot="out-inline"></div>
    </div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w10-out">
    <h4 data-slot="step-title"></h4>
    <div class="w10-explain" data-slot="explain"></div>
    <div class="callout callout--more" data-slot="hidden"></div>
    <h4>줄 수 (주석·빈 줄 제외 / 전체)</h4>
    <div class="stat-row" data-slot="lines"></div>
    <h4>숨은 기본값 실험 결과</h4>
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="stat-row" data-slot="sim"></div>
    <div data-slot="changed"></div>
    <h4>숨은 기본값 목록</h4>
    <div class="w10-table-wrap"><table class="w10-defaults" data-slot="defaults"></table></div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ step?: string, preset?: keyof SIM_PRESETS, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w10:${++seq}`;
  state.set(el, { ctrl, outputId });

  let data;
  let corpus;
  let golden;
  try {
    [data, corpus, golden] = await Promise.all([
      fetchJson(new URL('w10/snippets.json', DATA), ctrl.signal),
      fetchJson(new URL('corpus/corpus.json', DATA), ctrl.signal),
      fetchJson(new URL('golden/golden.json', DATA), ctrl.signal),
    ]);
  } catch (err) {
    if (err.name === 'AbortError') return;
    $('[data-slot=status]').innerHTML = `<div class="widget__error" role="alert">
      코드 비교 데이터를 불러오지 못했다 (${escapeHtml(err.message)}).<br>
      파일을 더블클릭해 <code>file://</code>로 열었다면 fetch가 동작하지 않는다.
      저장소 폴더에서 <code>python -m http.server</code>를 실행하고 <code>http://localhost:8000</code>으로 연다.</div>`;
    return;
  }

  const s = {
    step: data.steps.some((st) => st.id === options.step) ? options.step : 'chunk',
    sim: { ...SIM_PRESETS[options.preset in SIM_PRESETS ? options.preset : 'langchain'] },
  };

  // ---------- code panes ----------
  const panes = data.versions.map((v) => {
    const lines = v.code.split('\n');
    const pane = document.createElement('section');
    pane.className = 'w10-pane';
    pane.setAttribute('aria-label', `${v.label} 코드`);
    pane.innerHTML = `
      <header class="w10-pane__head">
        <span class="w10-tag">${escapeHtml(v.lang)}</span>
        <b>${escapeHtml(v.label)}</b><span class="w10-sub">${escapeHtml(v.sub)}</span>
        <span class="w10-count" data-count></span>
      </header>
      <div class="w10-file">${escapeHtml(v.file)}</div>
      <pre class="w10-code" tabindex="0" aria-label="${escapeHtml(v.label)} 코드 (스크롤 가능)"><code>${lines
        .map((l, i) => `<span class="w10-line" data-ln="${i + 1}"><span class="w10-ln" aria-hidden="true">${i + 1}</span><span class="w10-src">${highlightLine(l, v.lang)}</span></span>`)
        .join('')}</code></pre>`;
    $('[data-slot=panes]').append(pane);
    return { v, pane, pre: pane.querySelector('pre'), lineEls: [...pane.querySelectorAll('.w10-line')], stats: countLines(v.code, v.lang) };
  });

  $('[data-slot=note]').textContent = `⚠ ${data.note}`;

  // ---------- step selector ----------
  const stepHost = $('[data-slot=steps]');
  stepHost.innerHTML = data.steps
    .map((st, i) => `<button type="button" class="btn small w10-step" role="radio" data-step="${st.id}">${i + 1}. ${escapeHtml(st.label)}</button>`)
    .join('');
  const stepBtns = [...stepHost.querySelectorAll('[data-step]')];

  function renderStep(scroll = true) {
    const st = data.steps.find((x) => x.id === s.step);
    stepBtns.forEach((b) => {
      const on = b.dataset.step === s.step;
      b.setAttribute('aria-checked', String(on));
      b.tabIndex = on ? 0 : -1;
      b.classList.toggle('primary', on);
    });

    for (const p of panes) {
      const ranges = p.v.ranges[s.step] ?? [];
      const inStep = (n) => ranges.some(([a, b]) => n >= a && n <= b);
      let first = null;
      p.lineEls.forEach((ln, i) => {
        const hl = inStep(i + 1);
        ln.classList.toggle('is-hl', hl);
        if (hl && !first) first = ln;
      });
      const stepLines = ranges.reduce((acc, [a, b]) => acc + countLines(p.v.code.split('\n').slice(a - 1, b).join('\n'), p.v.lang).code, 0);
      p.pane.querySelector('[data-count]').textContent = `이 단계 ${stepLines}줄 · 전체 ${p.stats.code}줄`;
      if (scroll && first) p.pre.scrollTo({ top: Math.max(0, first.offsetTop - 8), behavior: 'smooth' });
    }

    $('[data-slot=step-title]').textContent = `${data.steps.indexOf(st) + 1}단계 · ${st.label}: 누가 무엇을 하나`;
    $('[data-slot=explain]').innerHTML = [
      ['우리 코드', st.ours],
      ['LangChain', st.langchain],
      ['LlamaIndex', st.llamaindex],
    ]
      .map(([t, html]) => `<div class="w10-card"><b>${t}</b><p>${html}</p></div>`)
      .join('');
    $('[data-slot=hidden]').innerHTML = `<span class="callout__title">이 단계의 숨은 기본값 · 함정</span>
      <ul>${st.hidden.map((h) => `<li>${escapeHtml(h)}</li>`).join('')}</ul>`;
    $('[data-slot=lines]').innerHTML = panes
      .map((p) => {
        const ranges = p.v.ranges[s.step] ?? [];
        const stepCode = ranges.reduce((acc, [a, b]) => acc + countLines(p.v.code.split('\n').slice(a - 1, b).join('\n'), p.v.lang).code, 0);
        return `<div class="stat"><span class="stat__label">${escapeHtml(p.v.label)}</span>
          <span class="stat__value">${p.stats.code} / ${p.stats.total}</span>
          <span class="w10-stat-sub">${escapeHtml(st.label)} ${stepCode}줄</span></div>`;
      })
      .join('');
  }

  // ---------- hidden-default experiment ----------
  const docs = corpus.documents;
  const cache = new Map();
  const simulate = (cfg) => {
    const key = JSON.stringify(cfg);
    if (!cache.has(key)) cache.set(key, runRetrieval(docs, golden.items, cfg));
    return cache.get(key);
  };
  const base = simulate(SIM_PRESETS.ours);

  for (const [name, opts] of Object.entries(SIM_OPTIONS)) {
    $(`[data-in=${name}]`).innerHTML = opts
      .map(([v, label]) => `<option value="${v}">${escapeHtml(label)}</option>`)
      .join('');
  }
  $('[data-slot=presets]').innerHTML = Object.entries(SIM_PRESETS)
    .map(([k, p]) => `<button type="button" class="btn small" data-preset="${k}">${escapeHtml(p.label)}</button>`)
    .join('');

  function syncInputs() {
    for (const name of Object.keys(SIM_OPTIONS)) $(`[data-in=${name}]`).value = String(s.sim[name]);
    const active = Object.entries(SIM_PRESETS).find(([, p]) => ['size', 'overlap', 'seps', 'k'].every((f) => p[f] === s.sim[f]))?.[0];
    root.querySelectorAll('[data-preset]').forEach((b) => {
      b.classList.toggle('primary', b.dataset.preset === active);
      b.setAttribute('aria-pressed', String(b.dataset.preset === active));
    });
  }

  function renderSim() {
    const r = simulate(s.sim);
    const k = s.sim.k;
    // mode: 'higher' = higher is better, 'lower' = lower is better, 'neutral' = just different
    const d = (v, bv, digits, mode = 'higher') => {
      const diff = v - bv;
      if (Math.abs(diff) < 1e-9) return '<span class="w10-delta">±0 (기준과 같음)</span>';
      const cls = mode === 'neutral' ? '' : (mode === 'lower' ? diff < 0 : diff > 0) ? 'up' : 'down';
      const txt = `${diff > 0 ? '+' : '−'}${Math.abs(diff).toFixed(digits)}`;
      return `<span class="w10-delta ${cls}">${txt}</span>`;
    };
    $('[data-slot=sim]').innerHTML = [
      ['청크 수', r.n, d(r.n, base.n, 0, 'neutral')],
      [`근거 hit@${k}`, r.hit.toFixed(3), d(r.hit, base.hit, 3)],
      ['MRR', r.mrr.toFixed(3), d(r.mrr, base.mrr, 3)],
      ['컨텍스트 (자/질문)', r.ctx, d(r.ctx, base.ctx, 0, 'lower')],
      ['근거 비율', `${(r.density * 100).toFixed(1)}%`, d(r.density * 100, base.density * 100, 1)],
    ]
      .map(([l, v, delta]) => `<div class="stat"><span class="stat__label">${l}</span><span class="stat__value">${v}</span>${delta}</div>`)
      .join('');

    $('[data-slot=verdict]').innerHTML = verdictHtml(r, base, s.sim, docs.length);

    const changed = r.perQ.filter((q, i) => q.rank !== base.perQ[i].rank);
    $('[data-slot=changed]').innerHTML = changed.length
      ? `<details class="w10-changed"><summary>순위가 달라진 질문 ${changed.length}개</summary><ul>${changed
        .map((q) => {
          const b = base.perQ.find((x) => x.id === q.id);
          return `<li><code>${q.id}</code> ${escapeHtml(q.question)} — 근거 청크 순위 ${fmtRank(b.rank)} → <b>${fmtRank(q.rank)}</b></li>`;
        })
        .join('')}</ul></details>`
      : '';
  }

  $('[data-slot=defaults]').innerHTML = `<thead><tr><th>프레임워크</th><th>API · 인자</th><th>기본값</th><th>우리 설정</th><th>결과에 미치는 영향</th></tr></thead><tbody>${data.defaults
    .map((x) => `<tr><td>${escapeHtml(x.fw)}</td><td><code>${escapeHtml(x.api)}</code><br><code>${escapeHtml(x.param)}</code></td><td>${escapeHtml(x.value)}</td><td>${escapeHtml(x.ours)}</td><td>${escapeHtml(x.effect)}</td></tr>`)
    .join('')}</tbody>`;

  // ---------- events ----------
  const on = (target, type, fn) => target.addEventListener(type, fn, { signal: ctrl.signal });
  on(stepHost, 'click', (e) => {
    const b = e.target.closest('[data-step]');
    if (!b) return;
    s.step = b.dataset.step;
    renderStep();
  });
  on(stepHost, 'keydown', (e) => {
    const dir = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (!dir && e.key !== 'Home' && e.key !== 'End') return;
    e.preventDefault();
    const i = stepBtns.findIndex((b) => b.dataset.step === s.step);
    const n = stepBtns.length;
    const next = e.key === 'Home' ? 0 : e.key === 'End' ? n - 1 : (i + dir + n) % n;
    s.step = stepBtns[next].dataset.step;
    renderStep();
    stepBtns[next].focus();
  });
  on($('[data-act=layout]'), 'click', (e) => {
    const stacked = $('[data-slot=panes]').classList.toggle('is-stacked');
    e.currentTarget.setAttribute('aria-pressed', String(stacked));
    e.currentTarget.textContent = stacked ? '↔ 나란히 보기' : '↕ 위아래로 보기';
    renderStep();
  });
  for (const name of Object.keys(SIM_OPTIONS)) {
    on($(`[data-in=${name}]`), 'change', (e) => {
      const v = e.target.value;
      s.sim[name] = name === 'seps' ? v : Number(v);
      syncInputs();
      renderSim();
    });
  }
  on($('[data-slot=presets]'), 'click', (e) => {
    const p = SIM_PRESETS[e.target.closest('[data-preset]')?.dataset.preset];
    if (!p) return;
    s.sim = { ...p };
    syncInputs();
    renderSim();
  });

  $('[data-slot=status]').hidden = true;
  $('[data-slot=body]').hidden = false;
  syncInputs();
  renderStep(false);
  renderSim();
  registerOutput(outputId, {
    title: options.outputTitle ?? '코드 비교 · 숨은 기본값',
    node: out,
    inlineHost: $('[data-slot=out-inline]'),
  });
  // scroll panes once layout exists
  requestAnimationFrame(() => {
    if (!ctrl.signal.aborted) renderStep(true);
  });
}

export function unmount(el) {
  const st = state.get(el);
  st?.ctrl.abort();
  if (st) unregisterOutput(st.outputId);
  state.delete(el);
  el.replaceChildren();
}

// ---------- pure helpers (exported for tests) ----------

/**
 * Chunk every document with cfg, index chunks with BM25 and score the golden set.
 * hit = a chunk containing the evidence quote is in the top k.
 */
export function runRetrieval(docs, items, cfg) {
  const overlap = Math.min(cfg.overlap, cfg.size - 1);
  const chunks = docs.flatMap((d) =>
    chunk(d.text, { strategy: 'recursive', size: cfg.size, overlap, separators: SEPARATORS[cfg.seps] })
      .map((c) => ({ id: `${d.id}#${c.index}`, text: c.text })));
  const bm25 = new BM25().add(chunks);
  let hit = 0;
  let mrr = 0;
  let ctx = 0;
  let density = 0;
  const perQ = items.map((it) => {
    const res = bm25.search(it.question, cfg.k);
    const i = res.findIndex((r) => it.evidence.some((e) => r.text.includes(e.quote)));
    const len = res.reduce((a, r) => a + r.text.length, 0);
    if (i >= 0) {
      hit++;
      mrr += 1 / (i + 1);
    }
    ctx += len;
    density += len ? it.evidence[0].quote.length / len : 0;
    return { id: it.id, question: it.question, rank: i >= 0 ? i + 1 : null };
  });
  const n = items.length || 1;
  return {
    n: chunks.length,
    hit: hit / n,
    mrr: mrr / n,
    ctx: Math.round(ctx / n),
    density: density / n,
    clamped: cfg.overlap >= cfg.size,
    perQ,
  };
}

/** Count code lines (non-empty, not comment-only) and total lines. */
export function countLines(code, lang) {
  const lines = code.split('\n');
  const marker = lang === 'PY' ? '#' : '//';
  const codeLines = lines.filter((l) => l.trim() && !l.trim().startsWith(marker));
  return { total: lines.length, code: codeLines.length };
}

function verdictHtml(r, base, cfg, docCount) {
  const parts = [];
  if (r.clamped) {
    parts.push(`<div class="callout"><span class="callout__title">오버랩 ${cfg.overlap} ≥ 크기 ${cfg.size}</span>
      LangChain이라면 여기서 ValueError가 난다. 우리 <code>chunker.js</code>는 오류 없이 오버랩을 ${cfg.size - 1}로 줄였다. 이것도 숨은 기본값이다.</div>`);
  }
  const k = cfg.k;
  if (r.n <= docCount) {
    parts.push(`<div class="callout callout--danger"><span class="callout__title">청킹이 사라졌다 — 청크 ${r.n}개 = 문서 ${docCount}개</span>
      hit@${k}는 ${r.hit.toFixed(3)}로 좋아 보이지만 질문당 컨텍스트가 ${r.ctx}자로 우리 설정(${base.ctx}자)의 ${(r.ctx / base.ctx).toFixed(1)}배다.
      근거 비율이 ${(base.density * 100).toFixed(1)}% → ${(r.density * 100).toFixed(1)}%로 떨어져 LLM 비용과 잡음이 늘고, 인용이 문서 단위로 뭉개진다.</div>`);
  } else if (r.hit < base.hit - 1e-9) {
    parts.push(`<div class="callout callout--danger"><span class="callout__title">지표 하락 — hit ${base.hit.toFixed(3)} → ${r.hit.toFixed(3)}</span>
      오류는 하나도 나지 않았다. 설정 하나가 바뀌었을 뿐인데 근거 청크를 top-${k} 안에서 놓친 질문이 생겼다. 아래 “순위가 달라진 질문”을 연다.</div>`);
  } else if (r.hit > base.hit + 1e-9) {
    parts.push(`<div class="callout callout--more"><span class="callout__title">지표 상승 — 이유를 설명할 수 있나?</span>
      hit@${k} ${r.hit.toFixed(3)}. k가 커졌거나 청크가 커졌다면 컨텍스트(${r.ctx}자)도 함께 늘었다. 좋아진 것인지 측정 조건이 바뀐 것인지 구분한다.</div>`);
  } else if (['size', 'overlap', 'seps', 'k'].every((f) => SIM_PRESETS.ours[f] === cfg[f])) {
    parts.push(`<div class="callout callout--ok"><span class="callout__title">기준: 우리 설정</span>
      recursive 150자 · 오버랩 20 · 문장 경계 구분자 · k=3. 리팩터링 후에도 이 숫자가 그대로 나와야 한다.</div>`);
  } else {
    parts.push(`<div class="callout callout--ok"><span class="callout__title">hit 동일 — ${r.hit.toFixed(3)}</span>
      지표는 같지만 청크 수(${r.n})와 컨텍스트(${r.ctx}자)가 달라졌다. hit 하나만 보면 이 차이를 놓친다.</div>`);
  }
  return parts.join('');
}

function highlightLine(line, lang) {
  const marker = lang === 'PY' ? '#' : '//';
  // comment = whole-line comment or a marker preceded by whitespace (avoids '#' inside template ids)
  const re = lang === 'PY' ? /(^|\s)#\s/ : /(^|\s)\/\/\s?/;
  const m = line.match(re);
  if (!m) return escapeHtml(line);
  const at = m.index + m[1].length;
  if (!line.slice(at).startsWith(marker)) return escapeHtml(line);
  return `${escapeHtml(line.slice(0, at))}<span class="w10-cm">${escapeHtml(line.slice(at))}</span>`;
}

const fmtRank = (r) => (r ? `${r}위` : '없음');

async function fetchJson(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${res.status} ${url.pathname}`);
  return res.json();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
