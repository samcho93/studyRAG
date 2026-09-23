// w08 BM25 vs 벡터 vs 하이브리드
// One concept: keyword search and vector search fail on different queries,
// and rank fusion (RRF or weighted min-max) recovers both. BM25 is ready
// instantly; the vector column fills in once the embedding model is loaded.

import { chunk } from '../core/chunker.js';
import { BM25, tokenize } from '../core/bm25.js';
import { VectorStore, dot } from '../core/vectorstore.js';
import { rrf, weightedFusion, minMax, rerank } from '../core/rerank.js';
import { MODELS, DEFAULT_MODEL, loadModelWithUI, embed, embedBatch } from '../core/embed.js';
import { registerOutput, unregisterOutput } from '../site/result.js';

const DATA = new URL('../../data/', import.meta.url);
const CHUNKING = { strategy: 'recursive', size: 200 };
const DEPTH = 10; // candidates taken from each retriever
const SHOW = 5; // rows shown per column
const HIT_K = 3; // "found" means evidence in the top 3

const SHORT = {
  'dept-overview': '학과 소개',
  'course-rag-syllabus': 'RAG 교과목',
  'facility-rules': '실습실 규정',
  'rag-intro': 'RAG란',
  'rag-chunking': '청킹',
  'rag-embedding': '임베딩',
  'rag-vectordb': '벡터DB',
  'rag-hybrid': '하이브리드',
  'rag-evaluation': '평가',
  'broken-scan': '장학금(스캔)',
  'broken-table': '시간표(깨짐)',
  'broken-mixed': 'FAQ(깨짐)',
};

const PRESETS = [
  { id: 'x05', label: '① 바꿔 말하기 (BM25 약점)' },
  { id: 'x01', label: '② 코드 토큰 2l4 (벡터 약점)' },
  { id: 'x07', label: '③ 양쪽 모두 중간 순위' },
  { id: 'q10', label: '④ 표 깨짐 (모두 실패)' },
];

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w08" data-w08>
    <h3 class="widget__title">BM25 vs 벡터 vs 하이브리드</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> 문서셋을 불러오는 중…
    </div>
    <div data-slot="body" hidden>
      <div class="btn-row w08-presets" data-slot="presets" role="group" aria-label="예시 질문"></div>
      <label class="field w08-query">
        <span class="field__label">질문 (직접 입력하거나 아래에서 고른다)</span>
        <input type="text" data-in="query" autocomplete="off" spellcheck="false">
      </label>
      <div class="widget__controls">
        <label class="field">
          <span class="field__label">골든셋 · 예시 질문</span>
          <select data-in="item"></select>
        </label>
        <label class="field">
          <span class="field__label">융합 방식</span>
          <select data-in="mode">
            <option value="rrf">RRF (순위만 사용)</option>
            <option value="weighted">가중합 (min-max 정규화 점수)</option>
          </select>
        </label>
        <label class="field" data-show="rrf">
          <span class="field__label">RRF k <output data-out="k"></output></span>
          <input type="range" data-in="k" min="1" max="100" step="1">
        </label>
        <label class="field" data-show="weighted" hidden>
          <span class="field__label">α (벡터 비중) <output data-out="alpha"></output></span>
          <input type="range" data-in="alpha" min="0" max="1" step="0.05">
        </label>
      </div>
      <label class="w08-check">
        <input type="checkbox" data-in="rerank">
        장난감 리랭커 적용 (하이브리드 상위 ${DEPTH}개를 다시 정렬 · 크로스 인코더 아님)
      </label>
      <details class="w08-model">
        <summary>임베딩 모델 바꾸기</summary>
        <label class="field">
          <span class="visually-hidden">임베딩 모델</span>
          <select data-in="model"></select>
        </label>
      </details>
      <div class="w08-status" data-slot="model-status"></div>
      <div data-slot="out-inline"></div>
    </div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w08-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>
    <div class="w08-cols" data-slot="cols"></div>
    <p class="w08-formula" data-slot="formula"></p>
    <div class="legend" aria-hidden="true">
      <span><b class="w08-gold-mark">✓</b> 정답 근거가 든 청크</span>
      <span><b class="w08-up">▲</b><b class="w08-down">▼</b> 왼쪽 방법 대비 순위 이동</span>
      <span>B·V = BM25·벡터에서의 순위</span>
    </div>
    <div class="w08-detail" data-slot="detail"></div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ preset?: string, model?: string, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w08:${++seq}`;
  const st = { ctrl, outputId, timer: 0, alive: true };
  state.set(el, st);

  let corpus;
  let golden;
  let extra;
  try {
    [corpus, golden, extra] = await Promise.all([
      fetchJson(new URL('corpus/corpus.json', DATA), ctrl.signal),
      fetchJson(new URL('golden/golden.json', DATA), ctrl.signal),
      fetchJson(new URL('w08/hybrid.json', DATA), ctrl.signal),
    ]);
  } catch (err) {
    if (err.name === 'AbortError') return;
    $('[data-slot=status]').innerHTML = `<div class="widget__error" role="alert">
      문서셋을 불러오지 못했다 (${escapeHtml(err.message)}).<br>
      파일을 더블클릭해 <code>file://</code>로 열었다면 ES Module과 fetch가 동작하지 않는다.
      저장소 폴더에서 <code>python -m http.server</code>를 실행하고 <code>http://localhost:8000</code>으로 연다.</div>`;
    return;
  }

  // ---------- index ----------
  const docTitle = Object.fromEntries(corpus.documents.map((d) => [d.id, d.title]));
  const chunks = corpus.documents.flatMap((d) =>
    chunk(d.text, CHUNKING).map((c) => ({
      id: `${d.id}#${c.index}`,
      doc: d.id,
      index: c.index,
      text: c.text.trim(),
      label: `${SHORT[d.id] ?? d.title} #${c.index}`,
    })));
  const byId = new Map(chunks.map((c) => [c.id, c]));
  const bm25 = new BM25().add(chunks.map(({ id, text }) => ({ id, text })));
  const items = [
    ...golden.items.map((it) => ({ ...it, group: '골든셋' })),
    ...extra.extra.map((it) => ({ ...it, group: it.kind === 'exact' ? '예시 · 정확한 토큰' : '예시 · 바꿔 말하기' })),
  ];
  const relevantIds = (item) =>
    item ? chunks.filter((c) => item.evidence.some((e) => e.doc === c.doc && c.text.includes(e.quote))).map((c) => c.id) : [];

  // ---------- vector side (filled in after the model loads) ----------
  const vec = {
    model: options.model ?? DEFAULT_MODEL,
    ready: false,
    error: '',
    stores: new Map(), // modelId → VectorStore of chunk vectors
    titled: new Map(), // modelId → Map(id → vector of "title: chunk") for the toy reranker
    qcache: new Map(), // `${model}|${query}` → query vector
  };

  const s = {
    query: '',
    itemId: '',
    mode: 'rrf',
    k: 60,
    alpha: 0.5,
    rerank: false,
    sel: '',
  };

  // ---------- controls ----------
  $('[data-slot=presets]').innerHTML = PRESETS
    .map((p) => `<button type="button" class="btn small" data-preset="${p.id}">${escapeHtml(p.label)}</button>`)
    .join('');
  const groups = [...new Set(items.map((it) => it.group))];
  $('[data-in=item]').innerHTML = `<option value="">(직접 입력 · 근거 미지정)</option>${groups
    .map((g) => `<optgroup label="${g}">${items
      .filter((it) => it.group === g)
      .map((it) => `<option value="${it.id}">${it.id} · ${escapeHtml(it.question)}</option>`)
      .join('')}</optgroup>`)
    .join('')}`;
  $('[data-in=model]').innerHTML = MODELS
    .map((m) => `<option value="${m.id}">${escapeHtml(m.label)}</option>`)
    .join('');

  function selectItem(id) {
    const item = items.find((it) => it.id === id);
    s.itemId = item ? item.id : '';
    if (item) s.query = item.question;
    syncInputs();
  }

  function syncInputs() {
    $('[data-in=query]').value = s.query;
    $('[data-in=item]').value = s.itemId;
    $('[data-in=mode]').value = s.mode;
    $('[data-in=k]').value = s.k;
    $('[data-in=alpha]').value = s.alpha;
    $('[data-in=rerank]').checked = s.rerank;
    $('[data-in=model]').value = vec.model;
    $('[data-out=k]').textContent = s.k;
    $('[data-out=alpha]').textContent = s.alpha.toFixed(2);
    root.querySelector('[data-show=rrf]').hidden = s.mode !== 'rrf';
    root.querySelector('[data-show=weighted]').hidden = s.mode !== 'weighted';
    root.querySelectorAll('[data-preset]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.preset === s.itemId)));
  }

  // ---------- model loading ----------
  async function prepareModel(modelId) {
    vec.model = modelId;
    vec.ready = false;
    vec.error = '';
    update();
    const statusEl = $('[data-slot=model-status]');
    try {
      await loadModelWithUI(statusEl, modelId);
      if (!st.alive || vec.model !== modelId) return;
      if (!vec.stores.has(modelId)) {
        const note = document.createElement('div');
        note.className = 'widget__status';
        note.innerHTML = '<span class="spinner" aria-hidden="true"></span><span>청크 임베딩 중…</span>';
        statusEl.append(note);
        const vectors = await embedBatch(chunks.map((c) => c.text), {
          role: 'passage',
          onProgress: (done, total) => {
            note.lastElementChild.textContent = `청크 ${done} / ${total}개 임베딩 중…`;
          },
        });
        note.remove();
        const store = new VectorStore();
        chunks.forEach((c, i) => store.add(c.id, vectors[i]));
        vec.stores.set(modelId, store);
      }
      if (!st.alive || vec.model !== modelId) return;
      vec.ready = true;
      update();
    } catch (err) {
      if (!st.alive || vec.model !== modelId) return;
      vec.error = err?.message ?? String(err);
      update();
    }
  }

  async function queryVector(q) {
    const key = `${vec.model}|${q}`;
    if (!vec.qcache.has(key)) vec.qcache.set(key, await embed(q, { role: 'query' }));
    return vec.qcache.get(key);
  }

  // Toy reranker needs "title: chunk" vectors — a stronger text window than the bare chunk.
  async function titledVectors() {
    if (!vec.titled.has(vec.model)) {
      const vs = await embedBatch(chunks.map((c) => `${docTitle[c.doc]}: ${c.text}`), { role: 'passage' });
      vec.titled.set(vec.model, new Map(chunks.map((c, i) => [c.id, vs[i]])));
    }
    return vec.titled.get(vec.model);
  }

  // ---------- update ----------
  let tick = 0;
  async function update() {
    const my = ++tick;
    const q = s.query.trim();
    const item = items.find((it) => it.id === s.itemId) ?? null;
    const rel = relevantIds(item);
    const view = { q, item, rel, bm: q ? bm25.search(q, DEPTH) : [], vec: null, fused: null, final: null, pending: '' };

    if (!q) {
      draw(view);
      return;
    }
    if (vec.error) {
      view.pending = 'error';
      draw(view);
      return;
    }
    if (!vec.ready) {
      view.pending = 'model';
      draw(view);
      return;
    }
    const cachedKey = `${vec.model}|${q}`;
    if (!vec.qcache.has(cachedKey)) {
      view.pending = 'query';
      draw(view);
    }
    try {
      const qv = await queryVector(q);
      if (my !== tick || !st.alive) return;
      view.vec = vec.stores.get(vec.model).search(qv, DEPTH);
      view.fused = s.mode === 'rrf'
        ? rrf([view.bm, view.vec], { k: s.k, topK: DEPTH })
        : weightedFusion(view.vec, view.bm, { alpha: s.alpha, topK: DEPTH });
      view.final = view.fused;
      view.pending = '';
      if (s.rerank) {
        if (!vec.titled.has(vec.model)) {
          view.pending = 'rerank';
          draw(view);
        }
        const tv = await titledVectors();
        if (my !== tick || !st.alive) return;
        view.final = await toyRerank(q, qv, view.fused, tv);
        view.pending = '';
      }
      draw(view);
    } catch (err) {
      if (my !== tick || !st.alive) return;
      vec.error = err?.message ?? String(err);
      view.pending = 'error';
      draw(view);
    }
  }

  // 0.5 × keyword coverage + 0.5 × min-max(cosine to "title: chunk"), over the fused candidates only.
  async function toyRerank(q, qv, candidates, titled) {
    const cos = minMax(candidates.map((c) => ({ id: c.id, score: dot(qv, titled.get(c.id)) })));
    const cosById = new Map(cos.map((c) => [c.id, c.score]));
    const ranked = await rerank(
      q,
      candidates,
      (query, c) => 0.5 * coverage(query, byId.get(c.id).text) + 0.5 * cosById.get(c.id),
      { topK: DEPTH },
    );
    return ranked.map((r) => ({ ...r, score: r.rerankScore }));
  }

  // ---------- drawing ----------
  function draw(v) {
    const rankIn = (list) => new Map((list ?? []).map((r, i) => [r.id, i + 1]));
    const bRank = rankIn(v.bm);
    const vRank = rankIn(v.vec);
    const fRank = rankIn(v.fused);
    const evRank = (list) => {
      if (!list || !v.rel.length) return null;
      const i = list.findIndex((r) => v.rel.includes(r.id));
      return i < 0 ? Infinity : i + 1;
    };
    const ranks = { bm: evRank(v.bm), vec: evRank(v.vec), hy: evRank(v.final) };

    // verdict + stats
    $('[data-slot=verdict]').innerHTML = verdictHtml(v, ranks);
    const fmtRank = (r, list) => (r === null ? (list ? '—' : '…') : r === Infinity ? `${DEPTH}위 밖` : `${r}위`);
    const stat = (label, value, hit) =>
      `<div class="stat ${hit === true ? 'w08-hit' : hit === false ? 'w08-miss' : ''}"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;
    const isHit = (r) => (r === null ? null : r <= HIT_K);
    const scale = (list, d) => (list?.length ? `${list[list.length - 1].score.toFixed(d)}–${list[0].score.toFixed(d)}` : '—');
    $('[data-slot=stats]').innerHTML = [
      stat('근거 순위 · BM25', fmtRank(ranks.bm, v.bm), isHit(ranks.bm)),
      stat('근거 순위 · 벡터', fmtRank(ranks.vec, v.vec), v.vec ? isHit(ranks.vec) : null),
      stat(`근거 순위 · 하이브리드${s.rerank ? '+리랭커' : ''}`, fmtRank(ranks.hy, v.final), v.final ? isHit(ranks.hy) : null),
      stat('점수 범위 · BM25', scale(v.bm, 1), null),
      stat('점수 범위 · 코사인', scale(v.vec, 3), null),
    ].join('');

    // columns
    const cols = [
      {
        key: 'bm',
        name: 'BM25',
        sub: '키워드 · 바로 계산',
        list: v.bm,
        empty: v.q ? '질문의 어떤 토큰도 문서에 없다 → 결과 0개' : '질문을 입력한다',
        fmt: (x) => x.toFixed(2),
        extra: (r) => `<span class="w08-src">V${vRank.get(r.id) ?? '–'}</span>`,
      },
      {
        key: 'vec',
        name: '벡터',
        sub: '코사인 유사도',
        list: v.vec,
        fmt: (x) => x.toFixed(3),
        extra: (r, i) => moveHtml(bRank.get(r.id), i + 1, v.bm ? 'BM25' : ''),
      },
      {
        key: 'hy',
        name: s.rerank ? '하이브리드 + 리랭커' : '하이브리드',
        sub: s.rerank
          ? '장난감 리랭커 점수'
          : s.mode === 'rrf' ? `RRF · k=${s.k}` : `가중합 · α=${s.alpha.toFixed(2)}`,
        list: v.final,
        fmt: (x) => (s.rerank ? x.toFixed(3) : s.mode === 'rrf' ? x.toFixed(4) : x.toFixed(3)),
        extra: (r, i) =>
          `<span class="w08-src">B${bRank.get(r.id) ?? '–'}·V${vRank.get(r.id) ?? '–'}</span>${
            s.rerank ? moveHtml(fRank.get(r.id), i + 1, '융합') : ''}`,
      },
    ];
    $('[data-slot=cols]').innerHTML = cols.map((c) => columnHtml(c, v)).join('');

    // formula line for the current top hybrid item
    $('[data-slot=formula]').innerHTML = formulaHtml(v, bRank, vRank);

    // detail of the selected chunk
    const selId = s.sel && byId.has(s.sel) ? s.sel : v.final?.[0]?.id ?? v.bm[0]?.id ?? '';
    const sel = byId.get(selId);
    $('[data-slot=detail]').innerHTML = sel
      ? `<div class="w08-detail__head"><b>${escapeHtml(sel.label)}</b>
          <span class="w08-src">${escapeHtml(sel.id)}</span>
          ${v.rel.includes(sel.id) ? '<span class="badge-ok">정답 근거 포함</span>' : ''}
          <span class="w08-src">B${bRank.get(sel.id) ?? '–'} · V${v.vec ? vRank.get(sel.id) ?? '–' : '…'} · H${
            v.final ? rankIn(v.final).get(sel.id) ?? '–' : '…'}</span></div>
         <p class="w08-detail__text">${markTokens(sel.text, v.q)}</p>
         <p class="w08-detail__note">노란 표시 = 질문 토큰(바이그램 포함)이 청크에 그대로 있는 부분. BM25는 이것만 본다.</p>`
      : '';
  }

  function columnHtml(c, v) {
    let body;
    if (c.list === null) {
      if (v.pending === 'error') {
        body = `<div class="widget__error" role="alert">벡터 검색을 쓸 수 없다 (${escapeHtml(vec.error)}). BM25 결과만 본다.</div>`;
      } else {
        const msg = { model: '임베딩 모델 준비 중…', query: '질문 임베딩 중…', rerank: '리랭커 준비 중…' }[v.pending] ?? '대기 중…';
        body = v.q
          ? `<div class="widget__status"><span class="spinner" aria-hidden="true"></span>${msg}</div>`
          : `<p class="w08-empty">질문을 입력한다</p>`;
      }
    } else if (c.list.length === 0) {
      body = `<p class="w08-empty">${escapeHtml(c.empty ?? '결과 없음')}</p>`;
    } else {
      body = `<ol class="w08-list">${c.list
        .slice(0, SHOW)
        .map((r, i) => {
          const ch = byId.get(r.id);
          const gold = v.rel.includes(r.id);
          return `<li><button type="button" class="w08-item${gold ? ' is-gold' : ''}${s.sel === r.id ? ' is-sel' : ''}"
            data-id="${r.id}" aria-pressed="${s.sel === r.id}" title="${escapeHtml(ch.text.slice(0, 120))}">
            <span class="w08-row"><span class="w08-rank">${i + 1}</span><span class="w08-label">${escapeHtml(ch.label)}</span>${
              gold ? '<b class="w08-gold-mark" aria-label="정답 근거">✓</b>' : ''}</span>
            <span class="w08-row w08-meta"><span class="w08-score">${c.fmt(r.score)}</span>${c.extra(r, i)}</span>
          </button></li>`;
        })
        .join('')}</ol>`;
      if (c.list === v.final && v.pending === 'rerank') body = `<div class="widget__status"><span class="spinner" aria-hidden="true"></span>리랭커 준비 중…</div>`;
    }
    return `<section class="w08-col w08-col--${c.key}" aria-label="${c.name} 결과">
      <header><b>${c.name}</b><small>${escapeHtml(c.sub)}</small></header>${body}</section>`;
  }

  function formulaHtml(v, bRank, vRank) {
    if (!v.fused?.length) {
      return s.mode === 'rrf'
        ? 'RRF(d) = 1/(k + r<sub>BM25</sub>) + 1/(k + r<sub>벡터</sub>) — 목록에 없으면 그 항은 0'
        : 'score(d) = α·norm(코사인) + (1−α)·norm(BM25), norm = min-max 정규화';
    }
    const top = v.fused[0];
    const ch = byId.get(top.id);
    const b = bRank.get(top.id);
    const r = vRank.get(top.id);
    if (s.mode === 'rrf') {
      const t = (x) => (x ? `1/(${s.k}+${x})` : '0');
      return `융합 1위 <b>${escapeHtml(ch.label)}</b>: ${t(b)} + ${t(r)} = <b>${top.score.toFixed(4)}</b>
        <span class="w08-src">(BM25 ${b ?? '–'}위, 벡터 ${r ?? '–'}위)</span>`;
    }
    const nb = new Map(minMax(v.bm).map((x) => [x.id, x.score]));
    const nv = new Map(minMax(v.vec).map((x) => [x.id, x.score]));
    return `융합 1위 <b>${escapeHtml(ch.label)}</b>: ${s.alpha.toFixed(2)}×${(nv.get(top.id) ?? 0).toFixed(2)} + ${(1 - s.alpha).toFixed(2)}×${(nb.get(top.id) ?? 0).toFixed(2)} = <b>${top.score.toFixed(3)}</b>
      <span class="w08-src">(정규화한 코사인 × α + 정규화한 BM25 × (1−α))</span>`;
  }

  function verdictHtml(v, r) {
    if (!v.q) return '';
    if (!v.item) {
      return `<div class="callout"><span class="callout__title">근거 미지정 질문</span>
        직접 입력한 질문이라 정답 근거(✓)를 표시하지 않는다. 세 열의 순위가 어떻게 다른지만 비교한다. 골든셋·예시 질문을 고르면 ✓가 붙는다.</div>`;
    }
    const w = (x) => (x === Infinity ? `${DEPTH}위 밖` : `${x}위`);
    if (!v.vec) {
      const b = r.bm <= HIT_K;
      return `<div class="callout ${b ? 'callout--ok' : 'callout--danger'}"><span class="callout__title">BM25: 근거 ${w(r.bm)}</span>
        ${b ? '질문의 단어가 근거 청크에 그대로 있어 키워드 검색이 바로 찾았다.' : '질문과 근거 청크가 같은 단어를 거의 쓰지 않아 키워드 검색이 놓쳤다.'}
        ${v.pending === 'error' ? '' : ' 벡터 검색 결과는 모델이 준비되면 채워진다.'}</div>`;
    }
    const b = r.bm <= HIT_K;
    const ve = r.vec <= HIT_K;
    const h = r.hy <= HIT_K;
    const tail = `(BM25 ${w(r.bm)} · 벡터 ${w(r.vec)} · 하이브리드 ${w(r.hy)}, 상위 ${HIT_K}위 안이면 성공)`;
    let cls = 'callout--ok';
    let title;
    let body;
    if (h && b && !ve) {
      title = '벡터는 놓쳤고, BM25가 찾았고, 하이브리드가 살렸다';
      body = '코드·번호처럼 뜻이 없는 토큰은 임베딩이 구분하지 못한다. 정확히 일치하는 토큰은 BM25의 영역이다.';
    } else if (h && !b && ve) {
      title = 'BM25는 놓쳤고, 벡터가 찾았고, 하이브리드가 살렸다';
      body = '질문과 근거가 같은 뜻을 다른 단어로 말한다. 단어가 겹치지 않으면 BM25 점수는 0에 가깝다.';
    } else if (h && r.hy < Math.min(r.bm, r.vec)) {
      title = '하이브리드가 두 방법보다 위로 끌어올렸다';
      body = '근거 청크가 두 목록 모두에서 1위는 아니지만 양쪽에 다 걸쳐 있었다. 한쪽에서만 1위인 청크보다 양쪽에 걸친 청크가 융합 점수를 더 받는다.';
    } else if (h) {
      title = '세 방법 모두 찾았다';
      body = '질문이 문서의 단어를 그대로 쓰고 뜻도 가깝다. 쉬운 질문에서는 어떤 검색이든 된다.';
    } else if (!b && !ve) {
      cls = 'callout--danger';
      title = '모두 실패 — 융합으로 못 고치는 실패';
      body = '두 목록 어디에도 근거가 상위에 없으면 순위를 섞어도 나오지 않는다. 근거 청크 자체가 망가졌는지(파싱·청킹) 먼저 의심한다.';
    } else {
      cls = 'callout--danger';
      title = '한쪽은 찾았는데 하이브리드가 놓쳤다';
      body = s.mode === 'rrf'
        ? '다른 쪽 목록의 엉뚱한 청크가 점수를 나눠 가졌다. k를 줄이면 1위의 몫이 커진다. 가중합으로 바꿔 α를 움직여 본다.'
        : 'α가 근거를 찾은 쪽의 반대로 기울어 있다. α를 움직이거나 RRF로 바꿔 본다.';
    }
    return `<div class="callout ${cls}"><span class="callout__title">${title}</span>${body} <span class="w08-src">${tail}</span></div>`;
  }

  // ---------- events ----------
  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  on('[data-slot=presets]', 'click', (e) => {
    const id = e.target.closest('[data-preset]')?.dataset.preset;
    if (!id) return;
    selectItem(id);
    update();
  });
  on('[data-in=item]', 'change', (e) => {
    selectItem(e.target.value);
    update();
  });
  on('[data-in=query]', 'input', (e) => {
    s.query = e.target.value;
    // keep the evidence only while the text still matches the chosen question
    const same = items.find((it) => it.question === s.query.trim());
    s.itemId = same ? same.id : '';
    $('[data-in=item]').value = s.itemId;
    root.querySelectorAll('[data-preset]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.preset === s.itemId)));
    clearTimeout(st.timer);
    st.timer = setTimeout(update, 250);
  });
  on('[data-in=mode]', 'change', (e) => {
    s.mode = e.target.value;
    syncInputs();
    update();
  });
  on('[data-in=k]', 'input', (e) => {
    s.k = Number(e.target.value);
    $('[data-out=k]').textContent = s.k;
    update();
  });
  on('[data-in=alpha]', 'input', (e) => {
    s.alpha = Number(e.target.value);
    $('[data-out=alpha]').textContent = s.alpha.toFixed(2);
    update();
  });
  on('[data-in=rerank]', 'change', (e) => {
    s.rerank = e.target.checked;
    update();
  });
  on('[data-in=model]', 'change', (e) => {
    prepareModel(e.target.value);
  });
  // column items live in the output node (terminal card or inline host)
  out.addEventListener(
    'click',
    (e) => {
      const id = e.target.closest('.w08-item')?.dataset.id;
      if (!id) return;
      s.sel = s.sel === id ? '' : id;
      update();
    },
    { signal: ctrl.signal },
  );

  selectItem(options.preset ?? 'x05');
  $('[data-slot=status]').hidden = true;
  $('[data-slot=body]').hidden = false;
  registerOutput(outputId, {
    title: options.outputTitle ?? 'BM25 vs 벡터 vs 하이브리드',
    node: out,
    inlineHost: $('[data-slot=out-inline]'),
  });
  update(); // BM25 column appears immediately
  prepareModel(vec.model);
}

export function unmount(el) {
  const st = state.get(el);
  if (st) {
    st.alive = false;
    clearTimeout(st.timer);
    st.ctrl.abort();
    unregisterOutput(st.outputId);
  }
  state.delete(el);
  el.replaceChildren();
}

// ---------- pure helpers (exported for tests) ----------

/** Share of the query's unique BM25 tokens that also occur in the text (0..1). */
export function coverage(query, text) {
  const q = [...new Set(tokenize(query))];
  if (q.length === 0) return 0;
  const t = new Set(tokenize(text));
  return q.filter((x) => t.has(x)).length / q.length;
}

/** Arrow showing how an item moved relative to its rank in a reference list. */
export function moveHtml(prevRank, rank, refName) {
  if (!refName) return '';
  if (!prevRank) return `<span class="w08-new" title="${refName} 상위 ${DEPTH}개에 없었다">NEW</span>`;
  const d = prevRank - rank;
  if (d > 0) return `<span class="w08-up" title="${refName} ${prevRank}위 → ${rank}위">▲${d}</span>`;
  if (d < 0) return `<span class="w08-down" title="${refName} ${prevRank}위 → ${rank}위">▼${-d}</span>`;
  return `<span class="w08-same" title="${refName}와 같은 순위">＝</span>`;
}

/** Escape text and wrap every occurrence of a query token (incl. Hangul bigrams) in <mark>. */
export function markTokens(text, query) {
  const lower = text.toLowerCase();
  const hit = new Uint8Array(text.length);
  for (const t of new Set(tokenize(query ?? ''))) {
    let i = lower.indexOf(t);
    while (i !== -1) {
      hit.fill(1, i, i + t.length);
      i = lower.indexOf(t, i + 1);
    }
  }
  let html = '';
  let i = 0;
  while (i < text.length) {
    let j = i;
    while (j < text.length && hit[j] === hit[i]) j++;
    const part = escapeHtml(text.slice(i, j));
    html += hit[i] ? `<mark>${part}</mark>` : part;
    i = j;
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
