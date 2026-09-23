// w05 유사도 히트맵 + 2D 투영
// One concept: an embedding model turns sentences into vectors, and cosine
// similarity between those vectors decides what "similar" means for retrieval.
// Shows an N×N cosine heatmap, a 2D PCA projection and a query ranking, and
// lets students switch to an English-only model to watch Korean topics collapse.

import { MODELS, DEFAULT_MODEL, loadModelWithUI, embedBatch, modelInfo } from '../core/embed.js';
import { registerOutput, unregisterOutput } from '../site/result.js';

const FAIL_MODEL = 'Xenova/all-MiniLM-L6-v2';
const MAX_SENTENCES = 16;

// Drawn from the shared corpus (rag-chunking, facility-rules, broken-scan).
// S10 paraphrases S4 with different words; S11 negates S7 with the same words.
export const DEFAULT_TEXT = [
  '[청킹] 고정 길이 청킹은 문장 중간에서 잘릴 수 있다.',
  '[청킹] 오버랩은 인접한 청크가 일부 내용을 겹쳐 갖게 한다.',
  '[청킹] 긴 문서를 검색 단위로 나누는 과정을 청킹이라 한다.',
  '[실습실] 실습실 안에서는 모든 음식물 섭취가 금지된다.',
  '[실습실] GPU 서버는 한 사람이 최대 12시간까지 연속으로 쓸 수 있다.',
  '[실습실] 실습실은 평일 오후 9시까지 개방된다.',
  '[장학금] 성적 우수 장학금은 평점 4.0 이상인 학생에게 지급한다.',
  '[장학금] 가계 곤란 장학금은 소득 분위 3분위 이하 학생이 대상이다.',
  '[장학금] 근로 장학생은 주 10시간 이내로 행정실 업무를 보조한다.',
  '[실습실] 랩실에서는 먹거나 마시면 안 된다.',
  '[장학금] 성적 우수 장학금은 평점 4.0 이상인 학생에게 지급하지 않는다.',
  '[무관] 오늘 점심 메뉴는 김치찌개다.',
].join('\n');

const DEFAULT_QUERY = '장학금을 받으려면 성적이 얼마나 좋아야 하나?';

// Trap pairs are found by text so they survive reordering; hidden if edited away.
const TRAPS = [
  {
    key: 'para',
    label: '함정 1: 다른 단어 · 같은 뜻',
    a: '실습실 안에서는 모든 음식물 섭취가 금지된다.',
    b: '랩실에서는 먹거나 마시면 안 된다.',
    expect: '뜻이 같으므로 높아야 한다',
  },
  {
    key: 'neg',
    label: '함정 2: 같은 단어 · 반대 뜻',
    a: '성적 우수 장학금은 평점 4.0 이상인 학생에게 지급한다.',
    b: '성적 우수 장학금은 평점 4.0 이상인 학생에게 지급하지 않는다.',
    expect: '뜻이 반대이므로 낮아야 하지만…',
  },
];

const TOPIC_COLORS = ['var(--color-accent)', 'var(--color-accent2)', 'var(--color-success)', 'var(--color-more)', 'var(--color-danger)'];
const MUTED_TOPICS = new Set(['무관', '기타']);

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w05" data-w05>
    <h3 class="widget__title">유사도 히트맵 + 2D 투영</h3>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">임베딩 모델</span>
        <select data-in="model"></select>
      </label>
      <label class="field">
        <span class="field__label">히트맵 색 범위</span>
        <select data-in="scale">
          <option value="auto">자동 (최솟값~최댓값으로 늘림)</option>
          <option value="fixed">고정 (0~1)</option>
        </select>
      </label>
    </div>
    <div class="btn-row" data-slot="presets" style="margin-bottom: var(--space-3)">
      <button type="button" class="btn small" data-act="fail">실패 재현: 영어 전용 모델</button>
      <button type="button" class="btn small" data-act="multi">다국어 모델로 되돌리기</button>
      <button type="button" class="btn small" data-act="trap-para">함정 1: 다른 단어 · 같은 뜻</button>
      <button type="button" class="btn small" data-act="trap-neg">함정 2: 같은 단어 · 반대 뜻</button>
    </div>
    <div data-slot="model-status" class="w05-status"></div>
    <label class="field w05-query">
      <span class="field__label">질의 (문장들을 이 질의와 가까운 순으로 정렬)</span>
      <input type="text" data-in="query" autocomplete="off">
    </label>
    <div class="w05-sents-head">
      <b>문장 목록</b><span class="w05-legend" data-slot="topics" aria-hidden="true"></span>
    </div>
    <ol class="w05-sents" data-slot="sents"></ol>
    <details>
      <summary>문장 직접 수정하기</summary>
      <p class="w05-hint">한 줄에 한 문장. 줄 앞의 <code>[주제]</code>는 색 구분용 표시이며 임베딩에는 들어가지 않는다. 최대 ${MAX_SENTENCES}문장.</p>
      <label class="field">
        <span class="visually-hidden">문장 목록 (한 줄에 한 문장)</span>
        <textarea data-in="text" spellcheck="false" rows="8"></textarea>
      </label>
      <div class="btn-row" style="margin-top: var(--space-2)">
        <button type="button" class="btn small ghost" data-act="reset">기본 문장으로 되돌리기</button>
      </div>
    </details>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w05-out">
    <div data-slot="calc" class="widget__status" role="status" aria-live="polite"></div>
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>

    <h4>코사인 유사도 히트맵</h4>
    <div class="w05-heat" data-slot="heat" role="grid" aria-label="문장 쌍별 코사인 유사도. 화살표 키로 이동한다."></div>
    <div class="w05-scale" aria-hidden="true">
      <span data-slot="scale-min"></span><i class="w05-scale__bar"></i><span data-slot="scale-max"></span>
    </div>
    <div class="w05-readout" data-slot="readout" aria-live="polite"></div>
    <details class="w05-tablewrap">
      <summary>표로 보기 (전체 값)</summary>
      <div class="w05-tablescroll" data-slot="table"></div>
    </details>

    <h4>함정 쌍 점검</h4>
    <div data-slot="traps"></div>

    <h4>2D 투영 (PCA)</h4>
    <div class="w05-plotwrap">
      <svg class="w05-plot" data-slot="plot" viewBox="0 0 360 250" role="group" aria-label="문장 임베딩의 2차원 주성분 투영"></svg>
    </div>
    <div class="w05-readout w05-readout--plot" data-slot="plot-readout" aria-live="polite"></div>
    <p class="w05-caption" data-slot="plot-caption"></p>

    <h4>질의 순위</h4>
    <ol class="w05-rank" data-slot="rank"></ol>
  </div>`;

const state = new WeakMap();
let seq = 0;
// Embedding cache shared across mounts: key = modelId + '\\u0000' + prefixed text
const cache = new Map();

/**
 * @param {HTMLElement} el
 * @param {{ model?: string, text?: string, query?: string, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w05:${++seq}`;
  const timers = new Set();
  const st = { ctrl, outputId, timers, alive: true };
  state.set(el, st);

  const s = {
    model: MODELS.some((m) => m.id === options.model) ? options.model : DEFAULT_MODEL,
    scale: 'auto',
    text: options.text ?? DEFAULT_TEXT,
    query: options.query ?? DEFAULT_QUERY,
    sentences: [],
    topics: [],
    sel: null, // { i, j }
    sym: null, // vectors for sentence↔sentence (symmetric role)
    pass: null, // vectors for query↔sentence (passage role)
    qVec: null,
    pca: null,
    sim: null,
    error: null,
  };

  $('[data-in=model]').innerHTML = MODELS.map((m) => `<option value="${m.id}">${escapeHtml(m.label)}</option>`).join('');
  $('[data-in=model]').value = s.model;
  $('[data-in=text]').value = s.text;
  $('[data-in=query]').value = s.query;

  // ---------- serialized async work (model load + embedding) ----------
  let chain = Promise.resolve();
  let token = 0;
  let qToken = 0;
  const enqueue = (fn, my) => {
    chain = chain.then(fn).catch((err) => {
      // errors from superseded jobs (model/text changed since) are ignored
      if (!st.alive || my !== token) return;
      s.error = err;
      renderAll();
    });
    return chain;
  };

  const later = (fn, ms) => {
    const t = setTimeout(() => {
      timers.delete(t);
      fn();
    }, ms);
    timers.add(t);
    return t;
  };
  let textTimer = null;
  let queryTimer = null;

  function parse() {
    const lines = s.text
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean)
      .slice(0, MAX_SENTENCES);
    s.sentences = lines.map((line) => {
      const m = /^\[([^\]]{1,20})\]\s*(.+)$/.exec(line);
      return m ? { topic: m[1].trim(), text: m[2].trim() } : { topic: '기타', text: line };
    });
    const topics = [];
    for (const x of s.sentences) if (!topics.includes(x.topic)) topics.push(x.topic);
    s.topics = topics;
    const n = s.sentences.length;
    if (!s.sel || s.sel.i >= n || s.sel.j >= n) s.sel = defaultSel();
  }

  function defaultSel() {
    const idx = trapIndex(TRAPS[0]);
    if (idx) return idx;
    return { i: 0, j: Math.min(1, s.sentences.length - 1) };
  }

  function trapIndex(trap) {
    const i = s.sentences.findIndex((x) => x.text === trap.a);
    const j = s.sentences.findIndex((x) => x.text === trap.b);
    return i >= 0 && j >= 0 ? { i, j } : null;
  }

  function topicColor(topic) {
    if (MUTED_TOPICS.has(topic)) return 'var(--color-text-muted)';
    const colored = s.topics.filter((t) => !MUTED_TOPICS.has(t));
    return TOPIC_COLORS[colored.indexOf(topic) % TOPIC_COLORS.length];
  }

  function cached(modelId, role, texts) {
    const info = modelInfo(modelId);
    const prefix = role === 'query' ? info.queryPrefix : info.passagePrefix;
    return texts.map((t) => cache.get(`${modelId}\u0000${prefix}${t}`) ?? null);
  }

  async function embedCached(modelId, role, texts) {
    const info = modelInfo(modelId);
    const prefix = role === 'query' ? info.queryPrefix : info.passagePrefix;
    const keys = texts.map((t) => `${modelId}\u0000${prefix}${t}`);
    const missing = [...new Set(texts.filter((t, i) => !cache.has(keys[i])))];
    if (missing.length) {
      const vecs = await embedBatch(missing, { role });
      missing.forEach((t, i) => cache.set(`${modelId}\u0000${prefix}${t}`, vecs[i]));
    }
    return keys.map((k) => cache.get(k));
  }

  /** Recompute sentence vectors (and the query) for the current model + text. */
  function refresh() {
    const my = ++token;
    s.error = null;
    const texts = s.sentences.map((x) => x.text);
    const modelId = s.model;
    // Instant render if everything is cached for this model (e.g. switching back)
    const symC = cached(modelId, 'query', texts);
    const passC = cached(modelId, 'passage', texts);
    if (symC.every(Boolean) && passC.every(Boolean)) setVectors(symC, passC);
    else clearVectors();
    renderAll();

    enqueue(async () => {
      if (!st.alive || my !== token) return;
      await loadModelWithUI($('[data-slot=model-status]'), modelId);
      if (!st.alive || my !== token) return;
      if (!s.sym) {
        setCalc('문장 임베딩 계산 중…');
        // e5 guidance: symmetric similarity uses "query: " on both sides;
        // retrieval uses "query: " for the question and "passage: " for documents.
        const sym = await embedCached(modelId, 'query', texts);
        const pass = await embedCached(modelId, 'passage', texts);
        if (!st.alive || my !== token) return;
        setVectors(sym, pass);
      }
      s.qVec = null;
      await embedQuery(my);
      renderAll();
    }, my);
  }

  async function embedQuery(parentToken) {
    const q = s.query.trim();
    if (!q) {
      s.qVec = null;
      return;
    }
    const my = ++qToken;
    const [v] = await embedCached(s.model, 'query', [q]);
    if (!st.alive || my !== qToken || parentToken !== token) return;
    s.qVec = v;
  }

  function refreshQuery() {
    const my = token;
    const q = s.query.trim();
    if (!q) {
      s.qVec = null;
      renderAll();
      return;
    }
    const [c] = cached(s.model, 'query', [q]);
    if (c && s.sym) {
      s.qVec = c;
      renderAll();
      return;
    }
    if (!s.sym) return; // the running refresh() embeds the query when done
    enqueue(async () => {
      if (!st.alive || my !== token) return;
      await embedQuery(my);
      renderAll();
    }, my);
  }

  function setVectors(sym, pass) {
    s.sym = sym;
    s.pass = pass;
    s.sim = similarityMatrix(sym);
    s.pca = sym.length >= 2 ? pca2(sym) : null;
  }

  function clearVectors() {
    s.sym = s.pass = s.sim = s.pca = s.qVec = null;
  }

  function setCalc(msg) {
    $('[data-slot=calc]').innerHTML = msg ? `<span class="spinner" aria-hidden="true"></span> ${escapeHtml(msg)}` : '';
  }

  // ---------- rendering ----------
  function renderAll() {
    if (!st.alive) return;
    renderSentences();
    setCalc('');
    if (s.error) {
      $('[data-slot=calc]').innerHTML = `<div class="widget__error" role="alert">계산하지 못했다: ${escapeHtml(s.error.message ?? s.error)}.
        문장 목록은 그대로 볼 수 있다. 네트워크를 확인하고 모델을 다시 선택한다.</div>`;
    } else if (!s.sym) {
      setCalc('모델을 불러오면 히트맵과 투영이 채워진다 (최초 1회 다운로드, 이후 브라우저 캐시 사용)');
    }
    renderStats();
    renderHeat();
    renderReadout();
    renderTable();
    renderTraps();
    renderPlot();
    renderRank();
  }

  function renderSentences() {
    $('[data-slot=topics]').innerHTML = s.topics
      .map((t) => `<span><i style="background: ${topicColor(t)}"></i>${escapeHtml(t)}</span>`)
      .join('');
    $('[data-slot=sents]').innerHTML = s.sentences
      .map(
        (x, i) => `<li><span class="w05-id" style="--topic: ${topicColor(x.topic)}">S${i + 1}</span>
          <span class="w05-sent-text">${escapeHtml(x.text)}</span></li>`,
      )
      .join('');
  }

  function offDiag() {
    const vals = [];
    const n = s.sentences.length;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) vals.push(s.sim[i][j]);
    return vals;
  }

  function renderStats() {
    const dim = s.sym?.[0]?.length;
    const m = s.sim ? topicMetrics(s.sim, s.sentences.map((x) => x.topic), MUTED_TOPICS) : null;
    const vals = s.sim ? offDiag() : [];
    const cells = [
      ['차원', dim ?? '—'],
      ['최근접 이웃 주제 일치', m?.nnAgree == null ? '—' : `${m.nnHits}/${m.nnTotal}`],
      ['같은 주제 평균', m?.same == null ? '—' : m.same.toFixed(3)],
      ['다른 주제 평균', m?.diff == null ? '—' : m.diff.toFixed(3)],
      ['값 범위', vals.length ? `${Math.min(...vals).toFixed(2)}~${Math.max(...vals).toFixed(2)}` : '—'],
    ];
    $('[data-slot=stats]').innerHTML = cells
      .map(([l, v]) => `<div class="stat"><span class="stat__label">${l}</span><span class="stat__value">${v}</span></div>`)
      .join('');

    const vEl = $('[data-slot=verdict]');
    if (!m || m.nnAgree == null) {
      vEl.innerHTML = '';
      return;
    }
    const gap = m.same - m.diff;
    const pct = Math.round(m.nnAgree * 100);
    const english = s.model === FAIL_MODEL;
    let cls;
    let title;
    if (m.nnAgree >= 0.75 && gap >= 0.05) {
      cls = 'callout--ok';
      title = `주제가 뭉친다 — 최근접 이웃 ${pct}%가 같은 주제`;
    } else if (m.nnAgree >= 0.5) {
      cls = '';
      title = `주제 구분이 약하다 — 최근접 이웃 ${pct}%가 같은 주제`;
    } else {
      cls = 'callout--danger';
      title = `주제가 섞였다 — 최근접 이웃 ${pct}%만 같은 주제`;
    }
    const body = english
      ? '영어 전용 모델은 한글을 제대로 된 단어 조각으로 자르지 못한다. 값 범위와 “같은 주제 − 다른 주제” 차이를 다국어 모델과 비교한다. 색 범위를 <b>고정 (0~1)</b>으로 바꾸면 전체가 얼마나 비슷한 색인지 보인다.'
      : `같은 주제 평균이 다른 주제 평균보다 ${gap.toFixed(3)} 높다. 절댓값보다 <b>순위</b>가 중요하다: 검색은 “가장 가까운 것”을 고른다.`;
    vEl.innerHTML = `<div class="callout ${cls}"><span class="callout__title">${title}</span>${body}</div>`;
  }

  function scaleOf() {
    if (!s.sim) return { lo: 0, hi: 1 };
    if (s.scale === 'fixed') return { lo: 0, hi: 1 };
    const vals = offDiag();
    if (!vals.length) return { lo: 0, hi: 1 };
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    return hi - lo < 1e-6 ? { lo: lo - 0.01, hi: hi + 0.01 } : { lo, hi };
  }

  function renderHeat() {
    const n = s.sentences.length;
    const { lo, hi } = scaleOf();
    const heat = $('[data-slot=heat]');
    heat.style.setProperty('--n', n);
    let html = `<div role="row" class="w05-heat__row"><span role="columnheader" class="w05-heat__corner"></span>`;
    for (let j = 0; j < n; j++) {
      html += `<span role="columnheader" class="w05-heat__lab" style="--topic: ${topicColor(s.sentences[j].topic)}">${j + 1}</span>`;
    }
    html += '</div>';
    for (let i = 0; i < n; i++) {
      html += `<div role="row" class="w05-heat__row"><span role="rowheader" class="w05-heat__lab" style="--topic: ${topicColor(s.sentences[i].topic)}">S${i + 1}</span>`;
      for (let j = 0; j < n; j++) {
        const selected = s.sel && ((s.sel.i === i && s.sel.j === j) || (s.sel.i === j && s.sel.j === i));
        const cls = ['w05-cell'];
        if (i === j) cls.push('is-diag');
        if (selected) cls.push('is-sel');
        if (s.sel && (s.sel.i === i || s.sel.j === j)) cls.push('is-cross');
        let style = '';
        let label = `S${i + 1}와 S${j + 1}`;
        if (s.sim) {
          const v = s.sim[i][j];
          const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
          style = ` style="--t: ${t.toFixed(3)}"`;
          label += ` 코사인 ${v.toFixed(3)}`;
        } else {
          cls.push('is-empty');
          label += ' (계산 전)';
        }
        const focus = s.sel && s.sel.i === i && s.sel.j === j ? 0 : -1;
        html += `<span role="gridcell" class="${cls.join(' ')}" data-i="${i}" data-j="${j}" tabindex="${focus}"
          aria-selected="${selected ? 'true' : 'false'}" aria-label="${label}"${style}></span>`;
      }
      html += '</div>';
    }
    const hadFocus = heat.contains(document.activeElement);
    heat.innerHTML = html;
    if (hadFocus && s.sel) heat.querySelector(`[data-i="${s.sel.i}"][data-j="${s.sel.j}"]`)?.focus();
    $('[data-slot=scale-min]').textContent = s.sim ? lo.toFixed(2) : '';
    $('[data-slot=scale-max]').textContent = s.sim ? hi.toFixed(2) : '';
  }

  function pairHtml(i, j) {
    const a = s.sentences[i];
    const b = s.sentences[j];
    if (!a || !b) return '';
    const lex = charBigramJaccard(a.text, b.text);
    const cos = s.sim ? s.sim[i][j] : null;
    let rankNote = '';
    if (s.sim && i !== j) {
      const r = neighborRank(s.sim, i, j);
      rankNote = ` · S${i + 1}의 이웃 중 <b>${r}위</b>`;
    }
    return `<div class="w05-pair"><b>S${i + 1} ↔ S${j + 1}</b>
      <span class="chip accent">코사인 ${cos == null ? '계산 전' : cos.toFixed(3)}</span>
      <span class="chip">글자 겹침 ${lex.toFixed(2)}</span>${rankNote}</div>
      <div class="w05-pair__texts"><span style="--topic: ${topicColor(a.topic)}">${escapeHtml(a.text)}</span>
      <span style="--topic: ${topicColor(b.topic)}">${escapeHtml(b.text)}</span></div>`;
  }

  let hover = null;
  function renderReadout() {
    const p = hover ?? s.sel;
    $('[data-slot=readout]').innerHTML = p ? pairHtml(p.i, p.j) : '';
  }

  function renderTable() {
    const n = s.sentences.length;
    if (!s.sim) {
      $('[data-slot=table]').innerHTML = '<p class="w05-hint">모델을 불러온 뒤 표시한다.</p>';
      return;
    }
    let html = '<table class="w05-table"><caption class="visually-hidden">문장 쌍별 코사인 유사도</caption><thead><tr><th scope="col"></th>';
    for (let j = 0; j < n; j++) html += `<th scope="col">S${j + 1}</th>`;
    html += '</tr></thead><tbody>';
    for (let i = 0; i < n; i++) {
      html += `<tr><th scope="row" title="${escapeHtml(s.sentences[i].text)}">S${i + 1}</th>`;
      for (let j = 0; j < n; j++) html += `<td>${s.sim[i][j].toFixed(2)}</td>`;
      html += '</tr>';
    }
    $('[data-slot=table]').innerHTML = `${html}</tbody></table>`;
  }

  function renderTraps() {
    const rows = TRAPS.map((t) => ({ t, idx: trapIndex(t) })).filter((r) => r.idx);
    if (!rows.length) {
      $('[data-slot=traps]').innerHTML = '<p class="w05-hint">기본 함정 문장이 목록에서 지워졌다. “기본 문장으로 되돌리기”로 복구한다.</p>';
      return;
    }
    $('[data-slot=traps]').innerHTML = `<table class="w05-traps"><thead><tr><th>쌍</th><th>글자 겹침</th><th>코사인</th><th>이웃 순위</th></tr></thead><tbody>${rows
      .map(({ t, idx }) => {
        const lex = charBigramJaccard(s.sentences[idx.i].text, s.sentences[idx.j].text);
        const cos = s.sim ? s.sim[idx.i][idx.j].toFixed(3) : '—';
        const rank = s.sim ? `${neighborRank(s.sim, idx.i, idx.j)}위 / ${s.sentences.length - 1}` : '—';
        return `<tr><td><button type="button" class="btn small ghost" data-trap="${t.key}">S${idx.i + 1}↔S${idx.j + 1}</button>
          <small>${escapeHtml(t.label)} · ${escapeHtml(t.expect)}</small></td>
          <td>${lex.toFixed(2)}</td><td>${cos}</td><td>${rank}</td></tr>`;
      })
      .join('')}</tbody></table>`;
  }

  function renderPlot() {
    const svg = $('[data-slot=plot]');
    const W = 360;
    const H = 250;
    const pad = 26;
    if (!s.pca) {
      svg.innerHTML = `<rect x="1" y="1" width="${W - 2}" height="${H - 2}" rx="8" class="w05-plot__frame"/>
        <text x="${W / 2}" y="${H / 2}" text-anchor="middle" class="w05-plot__note">모델을 불러오면 점이 찍힌다</text>`;
      $('[data-slot=plot-caption]').textContent = '';
      return;
    }
    const pts = s.pca.coords.map((c) => ({ x: c[0], y: c[1] }));
    let q = null;
    if (s.qVec) {
      const [x, y] = s.pca.project(s.qVec);
      q = { x, y };
    }
    const all = q ? [...pts, q] : pts;
    const xs = all.map((p) => p.x);
    const ys = all.map((p) => p.y);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) || 1;
    // one scale for both axes so distances are not distorted
    const k = Math.min(W - 2 * pad, H - 2 * pad) / span;
    const sx = (x) => W / 2 + (x - cx) * k;
    const sy = (y) => H / 2 - (y - cy) * k;

    let html = `<rect x="1" y="1" width="${W - 2}" height="${H - 2}" rx="8" class="w05-plot__frame"/>
      <line x1="${pad / 2}" y1="${H / 2}" x2="${W - pad / 2}" y2="${H / 2}" class="w05-plot__axis"/>
      <line x1="${W / 2}" y1="${pad / 2}" x2="${W / 2}" y2="${H - pad / 2}" class="w05-plot__axis"/>
      <text x="${W - 8}" y="${H / 2 - 5}" text-anchor="end" class="w05-plot__note">PC1</text>
      <text x="${W / 2 + 5}" y="16" class="w05-plot__note">PC2</text>`;
    pts.forEach((p, i) => {
      const x = sx(p.x);
      const y = sy(p.y);
      const sent = s.sentences[i];
      const anchor = x > W - 40 ? 'end' : 'start';
      const dx = anchor === 'end' ? -9 : 9;
      html += `<g class="w05-pt" tabindex="0" role="img" data-pt="${i}" aria-label="S${i + 1} ${escapeHtml(sent.topic)}: ${escapeHtml(sent.text)}">
        <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6.5" style="fill: ${topicColor(sent.topic)}"/>
        <text x="${(x + dx).toFixed(1)}" y="${(y + 4).toFixed(1)}" text-anchor="${anchor}">S${i + 1}</text></g>`;
    });
    if (q) {
      const x = sx(q.x);
      const y = sy(q.y);
      html += `<g class="w05-pt w05-pt--q" tabindex="0" role="img" data-pt="q" aria-label="질의: ${escapeHtml(s.query)}">
        <path d="M ${x} ${y - 9} L ${x + 8} ${y} L ${x} ${y + 9} L ${x - 8} ${y} Z"/>
        <text x="${(x + 11).toFixed(1)}" y="${(y + 4).toFixed(1)}">Q</text></g>`;
    }
    svg.innerHTML = html;
    const [e1, e2] = s.pca.explained;
    $('[data-slot=plot-caption]').textContent =
      `${s.sym[0].length}차원 벡터를 분산이 가장 큰 두 방향(PC1, PC2)으로 눌러 그렸다. 두 축이 전체 분산의 ${Math.round((e1 + e2) * 100)}%(PC1 ${Math.round(e1 * 100)}%, PC2 ${Math.round(e2 * 100)}%)만 설명하므로, 2D에서 가까워 보여도 실제 코사인은 다를 수 있다. 판단은 히트맵 값으로 한다.${s.qVec ? ' ◆Q는 질의 위치다.' : ''}`;
  }

  function showPoint(key) {
    const el = $('[data-slot=plot-readout]');
    if (key == null) {
      el.innerHTML = '';
      return;
    }
    if (key === 'q') {
      el.innerHTML = `<div class="w05-pair"><b>◆ Q</b> ${escapeHtml(s.query)}</div>`;
      return;
    }
    const i = Number(key);
    const x = s.sentences[i];
    if (!x) return;
    el.innerHTML = `<div class="w05-pair"><b>S${i + 1}</b><span class="chip" style="color: ${topicColor(x.topic)}">${escapeHtml(x.topic)}</span> ${escapeHtml(x.text)}</div>`;
  }

  function renderRank() {
    const list = $('[data-slot=rank]');
    if (!s.query.trim()) {
      list.innerHTML = '<li class="w05-hint">질의를 입력하면 순위가 나온다.</li>';
      return;
    }
    if (!s.pass || !s.qVec) {
      list.innerHTML = s.sentences
        .map((x, i) => `<li class="is-pending"><span class="w05-id" style="--topic: ${topicColor(x.topic)}">S${i + 1}</span><span class="w05-rank__text">${escapeHtml(x.text)}</span><span class="w05-rank__score">…</span></li>`)
        .join('');
      return;
    }
    const ranked = rankBy(s.qVec, s.pass);
    const top = ranked[0]?.score ?? 1;
    const bottom = ranked[ranked.length - 1]?.score ?? 0;
    list.innerHTML = ranked
      .map(({ index, score }, r) => {
        const x = s.sentences[index];
        const w = top - bottom < 1e-9 ? 100 : 8 + ((score - bottom) / (top - bottom)) * 92;
        return `<li class="${r === 0 ? 'is-top' : ''}">
          <span class="w05-id" style="--topic: ${topicColor(x.topic)}">S${index + 1}</span>
          <span class="w05-rank__text">${escapeHtml(x.text)}<i class="w05-rank__bar" style="width: ${w.toFixed(1)}%; background: ${topicColor(x.topic)}"></i></span>
          <span class="w05-rank__score">${score.toFixed(3)}</span></li>`;
      })
      .join('');
  }

  // ---------- events ----------
  const on = (target, type, fn) => target.addEventListener(type, fn, { signal: ctrl.signal });

  on($('[data-in=model]'), 'change', (e) => {
    s.model = e.target.value;
    refresh();
  });
  on($('[data-in=scale]'), 'change', (e) => {
    s.scale = e.target.value;
    renderHeat();
  });
  on($('[data-in=text]'), 'input', (e) => {
    s.text = e.target.value;
    clearTimeout(textTimer);
    timers.delete(textTimer);
    parse();
    clearVectors();
    renderAll();
    textTimer = later(refresh, 600);
  });
  on($('[data-in=query]'), 'input', (e) => {
    s.query = e.target.value;
    s.qVec = null;
    renderPlot();
    renderRank();
    clearTimeout(queryTimer);
    timers.delete(queryTimer);
    queryTimer = later(refreshQuery, 350);
  });

  const selectTrap = (key) => {
    const idx = trapIndex(TRAPS.find((t) => t.key === key));
    if (!idx) return;
    s.sel = idx;
    hover = null;
    renderHeat();
    renderReadout();
  };

  on(root, 'click', (e) => {
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (!act) return;
    if (act === 'fail' || act === 'multi') {
      s.model = act === 'fail' ? FAIL_MODEL : DEFAULT_MODEL;
      $('[data-in=model]').value = s.model;
      refresh();
    } else if (act === 'trap-para') selectTrap('para');
    else if (act === 'trap-neg') selectTrap('neg');
    else if (act === 'reset') {
      s.text = DEFAULT_TEXT;
      $('[data-in=text]').value = s.text;
      s.sel = null;
      parse();
      refresh();
    }
  });

  on(out, 'click', (e) => {
    const trap = e.target.closest('[data-trap]')?.dataset.trap;
    if (trap) {
      selectTrap(trap);
      return;
    }
    const cell = e.target.closest('.w05-cell');
    if (cell) {
      s.sel = { i: Number(cell.dataset.i), j: Number(cell.dataset.j) };
      hover = null;
      renderHeat();
      renderReadout();
      out.querySelector(`.w05-cell[data-i="${s.sel.i}"][data-j="${s.sel.j}"]`)?.focus();
    }
  });

  const heat = $('[data-slot=heat]');
  on(heat, 'mouseover', (e) => {
    const cell = e.target.closest('.w05-cell');
    if (!cell) return;
    hover = { i: Number(cell.dataset.i), j: Number(cell.dataset.j) };
    renderReadout();
  });
  on(heat, 'mouseleave', () => {
    hover = null;
    renderReadout();
  });
  on(heat, 'keydown', (e) => {
    if (!s.sel) return;
    const n = s.sentences.length;
    const moves = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    let { i, j } = s.sel;
    if (moves[e.key]) {
      i = Math.max(0, Math.min(n - 1, i + moves[e.key][0]));
      j = Math.max(0, Math.min(n - 1, j + moves[e.key][1]));
    } else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = n - 1;
    else return;
    e.preventDefault();
    s.sel = { i, j };
    hover = null;
    renderHeat();
    renderReadout();
    heat.querySelector(`[data-i="${i}"][data-j="${j}"]`)?.focus();
  });

  const plot = $('[data-slot=plot]');
  const ptKey = (e) => e.target.closest?.('[data-pt]')?.dataset.pt;
  on(plot, 'mouseover', (e) => showPoint(ptKey(e)));
  on(plot, 'focusin', (e) => showPoint(ptKey(e)));
  on(plot, 'mouseleave', () => showPoint(null));
  on(plot, 'focusout', () => showPoint(null));

  // ---------- start ----------
  parse();
  registerOutput(outputId, {
    title: options.outputTitle ?? '유사도 히트맵 · 2D 투영',
    node: out,
    inlineHost: $('[data-slot=out-inline]'),
  });
  refresh();
}

export function unmount(el) {
  const st = state.get(el);
  if (!st) return;
  st.alive = false;
  st.ctrl.abort();
  st.timers.forEach((t) => clearTimeout(t));
  st.timers.clear();
  unregisterOutput(st.outputId);
  state.delete(el);
  el.replaceChildren();
}

// ---------- pure helpers (exported for testing) ----------

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function cosine(a, b) {
  const d = Math.sqrt(dot(a, a)) * Math.sqrt(dot(b, b));
  return d === 0 ? 0 : dot(a, b) / d;
}

export function similarityMatrix(vectors) {
  const n = vectors.length;
  const m = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    m[i][i] = 1;
    for (let j = i + 1; j < n; j++) m[i][j] = m[j][i] = cosine(vectors[i], vectors[j]);
  }
  return m;
}

/** Rank of j among i's neighbors (1 = nearest), self excluded. */
export function neighborRank(sim, i, j) {
  const v = sim[i][j];
  let r = 1;
  for (let k = 0; k < sim.length; k++) if (k !== i && k !== j && sim[i][k] > v) r++;
  return r;
}

/** Documents sorted by cosine to the query. */
export function rankBy(q, docs) {
  return docs.map((d, index) => ({ index, score: cosine(q, d) })).sort((a, b) => b.score - a.score);
}

/**
 * Topic separation: nearest-neighbor topic agreement (scale-free) plus mean
 * same-topic vs cross-topic similarity. Singleton topics and `ignore` topics
 * are excluded from the NN check (they cannot have a same-topic neighbor).
 */
export function topicMetrics(sim, topics, ignore = new Set()) {
  const n = sim.length;
  const count = {};
  topics.forEach((t) => (count[t] = (count[t] ?? 0) + 1));
  let nnHits = 0;
  let nnTotal = 0;
  let sameSum = 0;
  let sameN = 0;
  let diffSum = 0;
  let diffN = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (topics[i] === topics[j]) {
        sameSum += sim[i][j];
        sameN++;
      } else {
        diffSum += sim[i][j];
        diffN++;
      }
    }
    if (count[topics[i]] < 2 || ignore.has(topics[i])) continue;
    let best = -1;
    let bestV = -Infinity;
    for (let j = 0; j < n; j++) {
      if (j !== i && sim[i][j] > bestV) {
        bestV = sim[i][j];
        best = j;
      }
    }
    nnTotal++;
    if (best >= 0 && topics[best] === topics[i]) nnHits++;
  }
  return {
    nnHits,
    nnTotal,
    nnAgree: nnTotal ? nnHits / nnTotal : null,
    same: sameN ? sameSum / sameN : null,
    diff: diffN ? diffSum / diffN : null,
  };
}

/** Lexical overlap: Jaccard of character bigrams, whitespace and punctuation removed. */
export function charBigramJaccard(a, b) {
  const grams = (s) => {
    const t = s.replace(/[\s.,!?·"'()]/g, '');
    const set = new Set();
    for (let i = 0; i < t.length - 1; i++) set.add(t.slice(i, i + 2));
    return set;
  };
  const A = grams(a);
  const B = grams(b);
  if (!A.size && !B.size) return 1;
  let inter = 0;
  for (const g of A) if (B.has(g)) inter++;
  return inter / (A.size + B.size - inter);
}

/**
 * 2-component PCA by power iteration on the (implicit) covariance X^T X.
 * The second component is found in the orthogonal complement of the first.
 * @param {ArrayLike<number>[]} vectors
 * @returns {{ coords: number[][], components: Float64Array[], mean: Float64Array, explained: number[], project: (v: ArrayLike<number>) => number[] }}
 */
export function pca2(vectors, { iterations = 300, tol = 1e-10 } = {}) {
  const n = vectors.length;
  const d = vectors[0].length;
  const mean = new Float64Array(d);
  for (const v of vectors) for (let k = 0; k < d; k++) mean[k] += v[k] / n;
  const X = vectors.map((v) => {
    const r = new Float64Array(d);
    for (let k = 0; k < d; k++) r[k] = v[k] - mean[k];
    return r;
  });
  let total = 0;
  for (const r of X) total += dot(r, r);

  const covMul = (v) => {
    const out = new Float64Array(d);
    for (const r of X) {
      const p = dot(r, v);
      for (let k = 0; k < d; k++) out[k] += p * r[k];
    }
    return out;
  };
  const orth = (v, comps) => {
    for (const c of comps) {
      const p = dot(v, c);
      for (let k = 0; k < d; k++) v[k] -= p * c[k];
    }
    return v;
  };
  const unit = (v) => {
    const len = Math.sqrt(dot(v, v));
    if (len < 1e-15) return null;
    for (let k = 0; k < d; k++) v[k] /= len;
    return v;
  };

  const components = [];
  const lambdas = [];
  for (let c = 0; c < 2; c++) {
    // deterministic start so the picture does not jump between renders
    let v = new Float64Array(d);
    for (let k = 0; k < d; k++) v[k] = 1 + ((k * 7 + c * 3) % 11) / 10;
    v = unit(orth(v, components));
    if (!v) break;
    for (let it = 0; it < iterations; it++) {
      const w = unit(orth(covMul(v), components));
      if (!w) break;
      let diff = 0;
      for (let k = 0; k < d; k++) diff += (w[k] - v[k]) ** 2;
      v = w;
      if (diff < tol) break;
    }
    // sign convention: largest-magnitude coordinate is positive
    let big = 0;
    for (let k = 1; k < d; k++) if (Math.abs(v[k]) > Math.abs(v[big])) big = k;
    if (v[big] < 0) for (let k = 0; k < d; k++) v[k] = -v[k];
    components.push(v);
    lambdas.push(dot(v, covMul(v)));
  }
  while (components.length < 2) {
    components.push(new Float64Array(d));
    lambdas.push(0);
  }

  const project = (v) => {
    const r = new Float64Array(d);
    for (let k = 0; k < d; k++) r[k] = v[k] - mean[k];
    return components.map((c) => dot(r, c));
  };
  return {
    coords: vectors.map(project),
    components,
    mean,
    explained: lambdas.map((l) => (total > 0 ? l / total : 0)),
    project,
  };
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
