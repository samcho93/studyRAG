// w04 청킹 플레이그라운드
// One concept: how chunk strategy / size / overlap decide whether the evidence
// sentence for a question survives intact inside a single chunk.

import { chunk, chunkStats, STRATEGIES } from '../core/chunker.js';
import { registerOutput, unregisterOutput } from '../site/result.js';

const DATA = new URL('../../data/', import.meta.url);
const COLORS = 6;

const PRESETS = {
  fail: { label: '실패 재현: 고정 100자, 오버랩 0', strategy: 'fixed', size: 100, overlap: 0 },
  overlap: { label: '오버랩 추가: 고정 100자, 오버랩 40', strategy: 'fixed', size: 100, overlap: 40 },
  recursive: { label: '재귀 분할 200자', strategy: 'recursive', size: 200, overlap: 0 },
};

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget" data-w04>
    <h3 class="widget__title">청킹 플레이그라운드</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> 문서셋을 불러오는 중…
    </div>
    <div data-slot="body" hidden>
      <div class="widget__controls">
        <label class="field">
          <span class="field__label">문서</span>
          <select data-in="doc"></select>
        </label>
        <label class="field">
          <span class="field__label">확인할 질문 (근거 문장에 물결 밑줄)</span>
          <select data-in="fact"></select>
        </label>
        <label class="field">
          <span class="field__label">전략</span>
          <select data-in="strategy"></select>
        </label>
        <label class="field">
          <span class="field__label">청크 크기 (문자) <output data-out="size"></output></span>
          <input type="range" data-in="size" min="20" max="600" step="10">
        </label>
        <label class="field">
          <span class="field__label">오버랩 (문자) <output data-out="overlap"></output></span>
          <input type="range" data-in="overlap" min="0" max="200" step="5">
        </label>
      </div>
      <div class="btn-row" data-slot="presets" style="margin-bottom: var(--space-4)"></div>
      <details>
        <summary>원문 직접 수정하기</summary>
        <label class="field" style="margin-bottom: var(--space-4)">
          <span class="visually-hidden">원문</span>
          <textarea data-in="text" spellcheck="false"></textarea>
        </label>
      </details>

      <div data-slot="out-inline"></div>
    </div>
  </div>`;

// Output (verdict, stats, highlighted text, chunk list) goes to the 실습 결과 panel
// on wide screens and stays under the controls on narrow ones.
const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w04-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>
    <div class="legend" aria-hidden="true">
      <span><i style="background: var(--chunk-1)"></i><i style="background: var(--chunk-2)"></i><i style="background: var(--chunk-3)"></i>청크(색으로 구분)</span>
      <span><i style="background: var(--chunk-overlap)"></i>오버랩(중복 저장)</span>
      <span>┃ 청크 경계</span>
    </div>
    <div class="chunk-view" data-slot="view" tabindex="0" aria-label="청크 경계가 표시된 원문"></div>
    <h4>청크 목록</h4>
    <ol class="chunk-list" data-slot="list"></ol>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ doc?: string, fact?: string, strategy?: string, size?: number, overlap?: number, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w04:${++seq}`;
  state.set(el, { ctrl, outputId });

  let corpus;
  let golden;
  try {
    [corpus, golden] = await Promise.all([
      fetchJson(new URL('corpus/corpus.json', DATA), ctrl.signal),
      fetchJson(new URL('golden/golden.json', DATA), ctrl.signal),
    ]);
  } catch (err) {
    if (err.name === 'AbortError') return;
    $('[data-slot=status]').innerHTML = `<div class="widget__error" role="alert">
      문서셋을 불러오지 못했다 (${escapeHtml(err.message)}).<br>
      파일을 더블클릭해 <code>file://</code>로 열었다면 ES Module과 fetch가 동작하지 않는다.
      저장소 폴더에서 <code>python -m http.server</code>를 실행하고 <code>http://localhost:8000</code>으로 연다.</div>`;
    return;
  }

  const docs = corpus.documents;
  const s = {
    doc: options.doc ?? 'rag-chunking',
    fact: options.fact ?? 'q05',
    strategy: options.strategy ?? 'fixed',
    size: options.size ?? 100,
    overlap: options.overlap ?? 0,
    text: '',
  };

  const docSel = $('[data-in=doc]');
  docSel.innerHTML = docs
    .map((d) => `<option value="${d.id}">${escapeHtml(d.title)}${d.type === 'broken' ? ' ⚠' : ''}</option>`)
    .join('');
  $('[data-in=strategy]').innerHTML = Object.entries(STRATEGIES)
    .map(([k, v]) => `<option value="${k}">${v}</option>`)
    .join('');
  $('[data-slot=presets]').innerHTML = Object.entries(PRESETS)
    .map(([k, p]) => `<button type="button" class="btn small" data-preset="${k}">${p.label}</button>`)
    .join('');

  const factsFor = (docId) => golden.items.filter((it) => it.evidence.some((e) => e.doc === docId));

  function loadDoc(docId, factId) {
    s.doc = docId;
    s.text = docs.find((d) => d.id === docId)?.text ?? '';
    const facts = factsFor(docId);
    s.fact = facts.some((f) => f.id === factId) ? factId : facts[0]?.id ?? '';
    $('[data-in=fact]').innerHTML = facts.length
      ? facts.map((f) => `<option value="${f.id}">${escapeHtml(f.question)}</option>`).join('')
      : '<option value="">(이 문서에 연결된 질문 없음)</option>';
    $('[data-in=text]').value = s.text;
  }

  function syncInputs() {
    docSel.value = s.doc;
    $('[data-in=fact]').value = s.fact;
    $('[data-in=strategy]').value = s.strategy;
    $('[data-in=size]').value = s.size;
    $('[data-in=overlap]').value = s.overlap;
  }

  function render() {
    // overlap must stay below size, otherwise the window never advances
    s.overlap = Math.min(s.overlap, s.size - 5);
    $('[data-in=overlap]').max = String(Math.min(200, s.size - 5));
    $('[data-in=overlap]').value = s.overlap;
    $('[data-out=size]').textContent = s.size;
    $('[data-out=overlap]').textContent = `${s.overlap} (${Math.round((s.overlap / s.size) * 100)}%)`;

    const chunks = chunk(s.text, { strategy: s.strategy, size: s.size, overlap: s.overlap });
    const stats = chunkStats(chunks, s.text.length);
    const factItem = golden.items.find((it) => it.id === s.fact);
    const quote = factItem?.evidence.find((e) => e.doc === s.doc)?.quote;
    const fStart = quote ? s.text.indexOf(quote) : -1;
    const fact = fStart >= 0 ? { start: fStart, end: fStart + quote.length } : null;

    const holders = fact ? chunks.filter((c) => c.start <= fact.start && c.end >= fact.end) : [];
    const touching = fact ? chunks.filter((c) => c.start < fact.end && c.end > fact.start) : [];

    $('[data-slot=stats]').innerHTML = [
      ['청크 수', stats.count],
      ['평균 길이', stats.avg],
      ['최소 / 최대', `${stats.min} / ${stats.max}`],
      ['중복 저장', `${Math.round(stats.duplicated * 100)}%`],
    ]
      .map(([l, v]) => `<div class="stat"><span class="stat__label">${l}</span><span class="stat__value">${v}</span></div>`)
      .join('');

    $('[data-slot=verdict]').innerHTML = verdictHtml(factItem, quote, fact, holders, touching);
    $('[data-slot=view]').innerHTML = highlight(s.text, chunks, fact);
    $('[data-slot=list]').innerHTML = chunks
      .map((c) => {
        const has = holders.includes(c);
        return `<li class="${has ? 'has-fact' : ''}" style="border-left-color: var(--chunk-${(c.index % COLORS) + 1})">
          <div class="chunk-meta"><span>#${c.index + 1}</span><span>${c.start}–${c.end}</span><span>${c.end - c.start}자</span>
          ${has ? '<span class="badge-ok">근거 온전히 포함</span>' : ''}</div>
          ${escapeHtml(c.text)}</li>`;
      })
      .join('');
  }

  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  on('[data-in=doc]', 'change', (e) => {
    loadDoc(e.target.value);
    syncInputs();
    render();
  });
  on('[data-in=fact]', 'change', (e) => {
    s.fact = e.target.value;
    render();
  });
  on('[data-in=strategy]', 'change', (e) => {
    s.strategy = e.target.value;
    render();
  });
  on('[data-in=size]', 'input', (e) => {
    s.size = Number(e.target.value);
    render();
  });
  on('[data-in=overlap]', 'input', (e) => {
    s.overlap = Number(e.target.value);
    render();
  });
  on('[data-in=text]', 'input', (e) => {
    s.text = e.target.value;
    render();
  });
  on('[data-slot=presets]', 'click', (e) => {
    const p = PRESETS[e.target.closest('[data-preset]')?.dataset.preset];
    if (!p) return;
    Object.assign(s, { strategy: p.strategy, size: p.size, overlap: p.overlap });
    syncInputs();
    render();
  });

  loadDoc(s.doc, s.fact);
  syncInputs();
  render();
  $('[data-slot=status]').hidden = true;
  $('[data-slot=body]').hidden = false;
  registerOutput(outputId, {
    title: options.outputTitle ?? '청킹 결과',
    node: out,
    inlineHost: $('[data-slot=out-inline]'),
  });
}

export function unmount(el) {
  const st = state.get(el);
  st?.ctrl.abort();
  if (st) unregisterOutput(st.outputId);
  state.delete(el);
  el.replaceChildren();
}

// ---------- rendering helpers ----------

function verdictHtml(item, quote, fact, holders, touching) {
  if (!item) return '';
  if (!fact) {
    return `<div class="callout"><span class="callout__title">근거 문장을 찾을 수 없다</span>
      원문을 수정해서 근거 문장 「${escapeHtml(quote ?? '')}」가 사라졌다.</div>`;
  }
  if (holders.length > 0) {
    const ids = holders.map((c) => `#${c.index + 1}`).join(', ');
    return `<div class="callout callout--ok"><span class="callout__title">근거 보존됨 — 청크 ${ids}</span>
      「${escapeHtml(item.question)}」의 근거가 한 청크 안에 온전히 들어 있다. 이 청크가 검색되면 답할 수 있다.</div>`;
  }
  const ids = touching.map((c) => `#${c.index + 1}`).join('와 ');
  return `<div class="callout callout--danger"><span class="callout__title">근거가 잘렸다 — 청크 ${ids}로 쪼개짐</span>
    「${escapeHtml(item.question)}」에 답하려면 두 조각이 모두 필요하지만, 각 조각만으로는 의미가 불완전해 둘 다 검색 순위가 낮아질 수 있다.
    오버랩을 늘리거나 전략을 바꿔 본다.</div>`;
}

/** Split text at every chunk/fact boundary and color each piece by how many chunks cover it. */
function highlight(text, chunks, fact) {
  const cuts = new Set([0, text.length]);
  for (const c of chunks) {
    cuts.add(c.start);
    cuts.add(c.end);
  }
  if (fact) {
    cuts.add(fact.start);
    cuts.add(fact.end);
  }
  const points = [...cuts].sort((a, b) => a - b);
  const starts = new Set(chunks.slice(1).map((c) => c.start));
  let html = '';
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];
    if (a === b) continue;
    if (starts.has(a)) html += '<span class="boundary" aria-hidden="true"></span>';
    const cover = chunks.filter((c) => c.start <= a && c.end >= b);
    const cls = ['seg'];
    if (cover.length === 1) cls.push(`c${cover[0].index % COLORS}`);
    else if (cover.length > 1) cls.push('ov');
    if (fact && a >= fact.start && b <= fact.end) cls.push('fact');
    html += `<span class="${cls.join(' ')}">${escapeHtml(text.slice(a, b))}</span>`;
  }
  return html;
}

async function fetchJson(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${res.status} ${url.pathname}`);
  return res.json();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
