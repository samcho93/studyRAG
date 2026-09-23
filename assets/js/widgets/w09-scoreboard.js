// w09 평가 점수판
// One concept: score a retrieval configuration on the golden set (12 questions)
// with Recall@k / Hit@k / MRR / nDCG@k, save rows and compare them side by side.
// Extras that serve the same concept: chunk-level vs document-level relevance
// (how the definition of "correct" inflates numbers), a per-question breakdown
// (which questions fail and at what rank), and an optional generation check
// (Faithfulness) with a keyless heuristic + LLM-as-judge.

import { chunk, sentenceSpans, STRATEGIES } from '../core/chunker.js';
import { BM25 } from '../core/bm25.js';
import { VectorStore } from '../core/vectorstore.js';
import { rrf } from '../core/rerank.js';
import { evaluate } from '../core/metrics.js';
import { MODELS, DEFAULT_MODEL, loadModelWithUI, embedBatch, currentModel } from '../core/embed.js';
import { PROVIDERS, setKey, hasKey, clearKey, generate, LLMError } from '../core/llm.js';
import { registerOutput, unregisterOutput } from '../site/result.js';

const DATA = new URL('../../data/', import.meta.url);
const STORE_KEY = 'raglab:w09:rows';
const MAX_ROWS = 12;

export const RETRIEVERS = { bm25: 'BM25', vector: '벡터', hybrid: '하이브리드(RRF)' };
export const MODES = { chunk: '청크 단위', doc: '문서 단위' };

const PRESETS = {
  base: { label: '기본: 재귀 200 · BM25 · k=3', strategy: 'recursive', size: 200, overlap: 0, retriever: 'bm25', mode: 'chunk', k: 3 },
  cut: { label: '실패 재현: 고정 100 · 오버랩 0', strategy: 'fixed', size: 100, overlap: 0, retriever: 'bm25', mode: 'chunk', k: 3 },
  doc: { label: '같은 설정을 문서 단위로 채점', strategy: 'fixed', size: 100, overlap: 0, retriever: 'bm25', mode: 'doc', k: 3 },
  big: { label: '큰 청크: 고정 400 (만점의 함정)', strategy: 'fixed', size: 400, overlap: 0, retriever: 'bm25', mode: 'chunk', k: 3 },
};

// Answers used by the Faithfulness panel. "faithful" = golden answer.
// "unfaithful" adds a claim that is NOT in the documents (q12 contradicts them).
const UNFAITHFUL = {
  q01: '최대 12시간이다. 연장하려면 조교에게 신청하면 된다.',
  q02: '팀 프로젝트는 40%다. 그중 절반은 발표 점수다.',
  q03: '출석률 80% 이상과 팀 프로젝트 최종 발표 통과다. 필기시험 60점 이상도 필요하다.',
  q04: '저장 공간과 임베딩 비용이 겹친 비율만큼 늘어난다. 검색 속도도 두 배로 느려진다.',
  q05: '답에 필요한 정보가 두 청크로 나뉘어 어느 쪽도 검색되지 않을 수 있다. 이 문제는 리랭킹으로 완전히 해결된다.',
  q06: '질문에는 "query: ", 문서에는 "passage: " 접두어를 붙여야 한다. 접두어를 빼면 벡터 차원이 바뀐다.',
  q07: '보통 60을 쓴다. 문서 수가 많을수록 k를 줄인다.',
  q08: '답변의 각 주장이 검색된 근거로 뒷받침되는지를 측정한다. 점수는 BLEU로 계산한다.',
  q09: '가끔 가장 가까운 이웃을 놓치는 근사 검색이다. GPU가 없으면 사용할 수 없다.',
  q10: '월요일 4-6교시, 306호다.',
  q11: '수업료의 50%다. 신청은 학기 말에 한다.',
  q12: '다음 주 수업 전까지 제출하면 인정된다. 결석해도 녹화 영상으로 보강이 있다.',
};

// Stored example judge outputs, shown (clearly labelled) when no key is set.
// Written for the default setting's context (재귀 200 · BM25 · k=3); not a live model call.
const RECORDED_JUDGE = {
  q01: {
    answer: UNFAITHFUL.q01,
    claims: [
      { claim: 'GPU 서버는 한 사람이 연속으로 최대 12시간 쓸 수 있다', supported: true, reason: '근거에 "한 사람이 연속으로 사용할 수 있는 시간은 최대 12시간이다"가 있다.' },
      { claim: '연장하려면 조교에게 신청하면 된다', supported: false, reason: '근거에 사용 시간 연장 절차는 없다. 조교는 장비 고장을 알리는 대상으로만 나온다.' },
    ],
    faithfulness: 0.5,
  },
  q12: {
    answer: UNFAITHFUL.q12,
    claims: [
      { claim: '결석한 주의 실습 과제는 다음 주 수업 전까지 제출하면 인정된다', supported: true, reason: '근거의 FAQ 답변과 일치한다.' },
      { claim: '결석해도 녹화 영상으로 보강이 있다', supported: false, reason: '근거는 "녹화 영상은 제공되지 않으며"라고 해 정반대다(모순).' },
    ],
    faithfulness: 0.5,
  },
};

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w09" data-w09>
    <h3 class="widget__title">평가 점수판 · Recall@k, nDCG</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> 문서셋과 골든셋을 불러오는 중…
    </div>
    <div data-slot="body" hidden>
      <p class="w09-hint">설정을 바꾸면 골든셋 12문항 전체를 그 자리에서 다시 채점한다. BM25는 바로 계산되고, 벡터·하이브리드는 임베딩 모델이 준비된 뒤 계산된다.</p>
      <div class="w09-segs">
        <fieldset class="w09-seg" data-group="retriever">
          <legend>검색 방식</legend>
        </fieldset>
        <fieldset class="w09-seg" data-group="mode">
          <legend>정답 판정</legend>
        </fieldset>
      </div>
      <div class="widget__controls">
        <label class="field">
          <span class="field__label">청킹 전략</span>
          <select data-in="strategy"></select>
        </label>
        <label class="field">
          <span class="field__label">청크 크기 (문자) <output data-out="size"></output></span>
          <input type="range" data-in="size" min="40" max="600" step="20">
        </label>
        <label class="field">
          <span class="field__label">오버랩 (문자) <output data-out="overlap"></output></span>
          <input type="range" data-in="overlap" min="0" max="200" step="10">
        </label>
        <label class="field">
          <span class="field__label">k (상위 몇 개까지 볼까) <output data-out="k"></output></span>
          <input type="range" data-in="k" min="1" max="10" step="1">
        </label>
        <label class="field">
          <span class="field__label">임베딩 모델 (벡터·하이브리드용)</span>
          <select data-in="model"></select>
        </label>
      </div>
      <div data-slot="model" class="w09-model"></div>
      <div class="btn-row w09-presets" data-slot="presets"></div>
      <div class="btn-row w09-save">
        <button type="button" class="btn primary" data-act="save">＋ 현재 설정 저장</button>
        <button type="button" class="btn ghost" data-act="clear">저장 기록 지우기</button>
        <span class="w09-save-msg" data-slot="save-msg" role="status" aria-live="polite"></span>
      </div>

      <div data-slot="out-inline"></div>

      <details class="w09-faith" data-slot="faith">
        <summary>생성 평가 맛보기 · Faithfulness 판정 (LLM 키 선택)</summary>
        <p class="w09-hint">질문을 고르면 <b>위 설정으로 검색한 상위 k개 청크</b>가 근거(컨텍스트)가 된다. 답변의 각 주장이 이 근거로 뒷받침되는지 판정한다. 검색이 실패하면 옳은 답도 “근거 없음”이 된다는 점을 확인한다.</p>
        <div class="widget__controls">
          <label class="field">
            <span class="field__label">질문</span>
            <select data-in="fq"></select>
          </label>
        </div>
        <div class="btn-row">
          <button type="button" class="btn small" data-fill="faithful">충실한 답변 넣기</button>
          <button type="button" class="btn small" data-fill="unfaithful">근거에 없는 주장 섞기</button>
        </div>
        <label class="field w09-answer">
          <span class="field__label">채점할 답변 (직접 고쳐도 된다)</span>
          <textarea data-in="answer" rows="3" spellcheck="false"></textarea>
        </label>
        <h4>근거 (현재 설정의 상위 k개)</h4>
        <ol class="w09-ctx" data-slot="ctx"></ol>
        <h4>① 간이 휴리스틱 · 키 없이 즉시</h4>
        <p class="w09-small">문장의 핵심 단어가 모두 근거에 있으면 “뒷받침”으로 본다. 부정·모순은 잡지 못한다.</p>
        <ol class="w09-heur" data-slot="heur"></ol>
        <h4>② LLM 판정자 (LLM-as-judge)</h4>
        <div class="widget__controls">
          <label class="field">
            <span class="field__label">공급자</span>
            <select data-in="provider"></select>
          </label>
          <label class="field">
            <span class="field__label">모델</span>
            <select data-in="llm-model"></select>
          </label>
          <label class="field">
            <span class="field__label">API 키 (이 탭에만 저장)</span>
            <input type="password" data-in="key" autocomplete="off" spellcheck="false" placeholder="키를 붙여 넣고 적용">
          </label>
        </div>
        <div class="btn-row">
          <button type="button" class="btn small" data-act="key-set">키 적용</button>
          <button type="button" class="btn small ghost" data-act="key-clear">키 지우기</button>
          <button type="button" class="btn small primary" data-act="judge">판정 요청</button>
          <span class="w09-save-msg" data-slot="key-state" role="status" aria-live="polite"></span>
        </div>
        <div data-slot="judge" aria-live="polite"></div>
      </details>
    </div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w09-out">
    <div class="w09-cfg" data-slot="cfg"></div>
    <div data-slot="pending" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>
    <p class="w09-noise" data-slot="noise"></p>
    <h4>저장한 설정 비교</h4>
    <div class="w09-table-wrap" tabindex="0" aria-label="저장한 설정 비교 표">
      <table class="w09-board">
        <thead><tr><th scope="col">설정</th><th scope="col">Recall@k</th><th scope="col">Hit@k</th><th scope="col">MRR</th><th scope="col">nDCG@k</th><th scope="col">실패</th><th scope="col"><span class="visually-hidden">삭제</span></th></tr></thead>
        <tbody data-slot="board"></tbody>
      </table>
    </div>
    <p class="w09-small" data-slot="board-note"></p>
    <h4 data-slot="qhead">문항별 결과</h4>
    <ol class="w09-qlist" data-slot="qlist"></ol>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ outputTitle?: string, strategy?: string, size?: number, overlap?: number,
 *   retriever?: 'bm25'|'vector'|'hybrid', mode?: 'chunk'|'doc', k?: number, model?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w09:${++seq}`;
  const st = { ctrl, outputId, alive: true, timer: 0 };
  state.set(el, st);

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
  if (!st.alive) return;

  const docs = corpus.documents;
  const docById = Object.fromEntries(docs.map((d) => [d.id, d]));
  const items = golden.items;
  const s = {
    strategy: options.strategy ?? 'recursive',
    size: options.size ?? 200,
    overlap: options.overlap ?? 0,
    retriever: options.retriever ?? 'bm25',
    mode: options.mode ?? 'chunk',
    k: options.k ?? 3,
    model: options.model ?? DEFAULT_MODEL,
  };
  const model = { id: null, ready: false, error: null };
  const indexCache = new Map(); // chunk-config key -> { chunks, byId, bm25 }
  const vecCache = new Map(); // `${modelId}\u0000p|q\u0000${text}` -> Float32Array
  let embedQueue = Promise.resolve();
  let runSeq = 0;
  let last = null; // last scored result
  let rows = loadRows();
  const f = { q: 'q01', answer: UNFAITHFUL.q01, provider: 'anthropic', llmModel: PROVIDERS.anthropic.defaultModel, busy: false };

  // ---------- controls ----------
  $('[data-group=retriever]').insertAdjacentHTML('beforeend', radios('retriever', RETRIEVERS, outputId));
  $('[data-group=mode]').insertAdjacentHTML('beforeend', radios('mode', {
    chunk: '청크 단위 · 근거 문장을 온전히 담은 청크만 정답',
    doc: '문서 단위 · 근거 문서의 아무 청크나 정답',
  }, outputId));
  $('[data-in=strategy]').innerHTML = Object.entries(STRATEGIES).map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
  $('[data-in=model]').innerHTML = MODELS.map((m) => `<option value="${m.id}">${escapeHtml(m.label)}</option>`).join('');
  $('[data-slot=presets]').innerHTML = Object.entries(PRESETS)
    .map(([k, p]) => `<button type="button" class="btn small ghost" data-preset="${k}">${escapeHtml(p.label)}</button>`)
    .join('');
  $('[data-in=fq]').innerHTML = items.map((it) => `<option value="${it.id}">${it.id} · ${escapeHtml(it.question)}</option>`).join('');
  $('[data-in=provider]').innerHTML = Object.entries(PROVIDERS).map(([k, p]) => `<option value="${k}">${escapeHtml(p.label)}</option>`).join('');

  function syncInputs() {
    root.querySelectorAll('input[type=radio]').forEach((r) => {
      r.checked = s[r.name.split(':')[0]] === r.value;
    });
    $('[data-in=strategy]').value = s.strategy;
    $('[data-in=size]').value = s.size;
    $('[data-in=k]').value = s.k;
    $('[data-in=model]').value = s.model;
    syncOverlap();
  }

  function syncOverlap() {
    const max = Math.min(200, s.size - 20);
    s.overlap = Math.max(0, Math.min(s.overlap, max));
    const ov = $('[data-in=overlap]');
    ov.max = String(max);
    ov.value = s.overlap;
    $('[data-out=size]').textContent = s.size;
    $('[data-out=overlap]').textContent = `${s.overlap} (${Math.round((s.overlap / s.size) * 100)}%)`;
    $('[data-out=k]').textContent = s.k;
  }

  // ---------- indexing & retrieval ----------
  function getIndex() {
    const key = `${s.strategy}|${s.size}|${s.overlap}`;
    let idx = indexCache.get(key);
    if (!idx) {
      const chunks = buildChunks(docs, s);
      idx = { key, chunks, byId: new Map(chunks.map((c) => [c.id, c])), bm25: new BM25().add(chunks) };
      indexCache.set(key, idx);
      if (indexCache.size > 24) indexCache.delete(indexCache.keys().next().value);
    }
    return idx;
  }

  function bm25Lists(idx) {
    return new Map(items.map((it) => [it.id, idx.bm25.search(it.question, idx.chunks.length)]));
  }

  async function ensureVectors(texts, role, modelId, onProgress) {
    const missing = [...new Set(texts)].filter((t) => !vecCache.has(vkey(modelId, role, t)));
    if (missing.length === 0) return true;
    // serialize embedding work: one extractor, many slider events
    const job = embedQueue.then(async () => {
      const todo = missing.filter((t) => !vecCache.has(vkey(modelId, role, t)));
      if (!todo.length || !st.alive) return;
      if (currentModel().id !== modelId) throw new Error('model-switched');
      const vecs = await embedBatch(todo, { role, batchSize: 16, onProgress });
      if (currentModel().id !== modelId) throw new Error('model-switched');
      todo.forEach((t, i) => vecCache.set(vkey(modelId, role, t), vecs[i]));
    });
    embedQueue = job.catch(() => {});
    await job;
    return true;
  }

  async function vectorLists(idx, my) {
    const modelId = model.id;
    const texts = idx.chunks.map((c) => c.text);
    const need = texts.filter((t) => !vecCache.has(vkey(modelId, 'passage', t))).length;
    if (need > 0) {
      renderPending(`<div class="widget__status"><span class="spinner" aria-hidden="true"></span>
        <span data-embed-msg>청크 ${need}개 임베딩 중…</span></div>`);
    }
    await ensureVectors(texts, 'passage', modelId, (done, total) => {
      const m = out.querySelector('[data-embed-msg]');
      if (m && my === runSeq) m.textContent = `청크 임베딩 중 ${done} / ${total}`;
    });
    await ensureVectors(items.map((it) => it.question), 'query', modelId);
    if (my !== runSeq) return null;
    const store = new VectorStore();
    for (const c of idx.chunks) store.add(c.id, vecCache.get(vkey(modelId, 'passage', c.text)));
    return new Map(items.map((it) => [it.id, store.search(vecCache.get(vkey(modelId, 'query', it.question)), idx.chunks.length)]));
  }

  async function refresh() {
    const my = ++runSeq;
    syncOverlap();
    renderCfg();
    const idx = getIndex();
    let lists;
    if (s.retriever === 'bm25') {
      lists = bm25Lists(idx);
    } else {
      if (model.error) {
        renderPending(`<div class="widget__error" role="alert">임베딩 모델을 불러오지 못해 ${RETRIEVERS[s.retriever]} 점수를 계산할 수 없다.
          BM25로 바꾸면 모델 없이 채점된다. 네트워크를 확인하고 새로고침한다.</div>`, true);
        return;
      }
      if (!model.ready || model.id !== s.model) {
        renderPending(`<div class="widget__status"><span class="spinner" aria-hidden="true"></span>
          임베딩 모델 준비 중 — 준비되면 ${RETRIEVERS[s.retriever]} 점수가 자동으로 채워진다. 그동안 BM25는 바로 볼 수 있다.</div>`, true);
        return;
      }
      let vec;
      try {
        vec = await vectorLists(idx, my);
      } catch (err) {
        if (my !== runSeq || err.message === 'model-switched') return;
        renderPending(`<div class="widget__error" role="alert">임베딩 중 오류: ${escapeHtml(err.message)}</div>`, true);
        return;
      }
      if (!vec || my !== runSeq || !st.alive) return;
      lists = s.retriever === 'vector' ? vec : hybridLists(bm25Lists(idx), vec, idx.chunks.length);
    }
    last = { ...scoreConfig(items, idx, lists, s), cfg: { ...s }, idx };
    renderPending('');
    renderResults();
    renderFaith();
  }

  function scheduleRefresh() {
    clearTimeout(st.timer);
    if (s.retriever === 'bm25') refresh();
    else st.timer = setTimeout(refresh, 180);
  }

  function ensureModel() {
    const id = s.model;
    model.ready = false;
    model.error = null;
    model.id = id;
    loadModelWithUI($('[data-slot=model]'), id)
      .then(() => {
        if (!st.alive || model.id !== id) return;
        model.ready = true;
        if (s.retriever !== 'bm25') refresh();
      })
      .catch((err) => {
        if (!st.alive || model.id !== id) return;
        model.error = err;
        if (s.retriever !== 'bm25') refresh();
      });
  }

  // ---------- rendering ----------
  function renderCfg() {
    $('[data-slot=cfg]').innerHTML = `<span class="chip accent">${escapeHtml(cfgLabel(s))}</span>`;
    $('[data-slot=qhead]').textContent = `문항별 결과 · 근거가 나온 순위 (k=${s.k})`;
  }

  function renderPending(html, dim = false) {
    $('[data-slot=pending]').innerHTML = html;
    out.classList.toggle('is-stale', dim);
  }

  function renderResults() {
    if (!last) return;
    const { metrics, perQ } = last;
    const n = perQ.length;
    const hits = perQ.filter((q) => q.rank > 0 && q.rank <= s.k).length;
    const [lo, hi] = wilson(hits, n);
    $('[data-slot=stats]').innerHTML = [
      [`Recall@${s.k}`, fmt(metrics.recall)],
      [`Hit@${s.k}`, fmt(metrics.hit)],
      ['MRR', fmt(metrics.mrr)],
      [`nDCG@${s.k}`, fmt(metrics.ndcg)],
      ['성공 문항', `${hits} / ${n}`],
    ].map(([l, v]) => `<div class="stat"><span class="stat__label">${l}</span><span class="stat__value">${v}</span></div>`).join('');
    $('[data-slot=noise]').innerHTML = `문항이 ${n}개뿐이라 한 문항이 <b>${fmt(1 / n)}</b>(약 ${Math.round(100 / n)}%p)만큼 점수를 움직인다.
      Hit@${s.k}의 95% 신뢰구간은 대략 <b>${fmt(lo)}–${fmt(hi)}</b>다. 이 폭보다 작은 차이는 우연일 수 있다.`;
    renderBoard();
    renderQuestions();
  }

  function renderBoard() {
    const cur = last ? rowFrom(last) : null;
    const all = cur ? [{ ...cur, current: true }, ...rows] : rows;
    const best = {};
    for (const col of ['recall', 'hit', 'mrr', 'ndcg']) best[col] = Math.max(...all.map((r) => r[col]));
    const cell = (r, col) => `<td class="${all.length > 1 && r[col] === best[col] ? 'is-best' : ''}">${fmt(r[col])}</td>`;
    $('[data-slot=board]').innerHTML = all.map((r, i) => `
      <tr class="${r.current ? 'is-current' : ''}">
        <th scope="row">${r.current
          ? `<span class="w09-now">지금</span> ${escapeHtml(r.label)}`
          : `<button type="button" class="w09-linkbtn" data-restore="${i - (cur ? 1 : 0)}" title="이 설정으로 되돌리기">${escapeHtml(r.label)}</button>`}</th>
        ${cell(r, 'recall')}${cell(r, 'hit')}${cell(r, 'mrr')}${cell(r, 'ndcg')}
        <td class="w09-fails">${r.fails.length ? r.fails.map((q) => q.replace('q', '')).join(' ') : '—'}</td>
        <td>${r.current ? '' : `<button type="button" class="w09-x" data-del="${i - (cur ? 1 : 0)}" aria-label="${escapeHtml(r.label)} 삭제">✕</button>`}</td>
      </tr>`).join('');
    const notes = [];
    if (!rows.length) notes.push('“＋ 현재 설정 저장”을 누르면 이 표에 행이 쌓인다. 설정 이름을 누르면 그 설정으로 돌아간다. 실패 열의 숫자는 문항 번호다.');
    const ks = new Set(all.map((r) => r.k));
    const modes = new Set(all.map((r) => r.mode));
    if (ks.size > 1) notes.push('k가 다른 행이 섞여 있다. Recall@k·nDCG@k는 k가 같을 때만 직접 비교한다.');
    if (modes.size > 1) notes.push('청크 단위와 문서 단위로 채점한 행이 섞여 있다. 정답 정의가 다르면 같은 숫자라도 뜻이 다르다.');
    if (all.length >= 2) {
      const sorted = [...all].sort((a, b) => b.hit - a.hit);
      const gap = Math.round((sorted[0].hit - sorted[1].hit) * sorted[0].n);
      if (gap <= 1) notes.push(`Hit@k 1·2위의 차이가 ${gap}문항이다. 12문항에서 1문항 차이는 잡음과 구별하기 어렵다.`);
    }
    $('[data-slot=board-note]').innerHTML = notes.join('<br>');
  }

  function renderQuestions() {
    const { perQ, idx } = last;
    $('[data-slot=qlist]').innerHTML = perQ.map((q) => {
      const ok = q.rank > 0 && q.rank <= s.k;
      const mark = ok ? 'ok' : q.cut ? 'cut' : 'miss';
      const icon = ok ? '✓' : q.cut ? '✂' : '✗';
      const rankText = q.cut
        ? '근거 잘림'
        : q.rank === 0 ? '검색 안 됨' : ok ? `${q.rank}위` : `${q.rank}위 (k 밖)`;
      const ev = q.item.evidence[0];
      const broken = docById[ev.doc]?.type === 'broken';
      const top = q.ranked.slice(0, s.k).map((id, i) => {
        const c = idx.byId.get(id);
        const rel = q.relAt.has(i);
        return `<li class="${rel ? 'is-rel' : ''}"><span class="w09-src">${i + 1}. ${escapeHtml(docById[c.doc].title)} #${c.index + 1}${rel ? ' · 정답' : ''}</span>${escapeHtml(clip(c.text, 90))}</li>`;
      }).join('');
      const why = q.cut
        ? `<p class="w09-why">근거 문장 「${escapeHtml(ev.quote)}」를 온전히 담은 청크가 하나도 없다. 검색 전에 청킹에서 이미 실패했다. 오버랩을 두거나 전략을 바꿔 본다.</p>`
        : !ok && q.rank > 0
          ? `<p class="w09-why">근거 청크가 ${q.rank}위에 있다. k를 ${q.rank}까지 올리면 성공으로 바뀐다. 그 대신 LLM에 넣을 컨텍스트가 늘어난다.</p>`
          : '';
      return `<li class="w09-q ${mark}"><details>
        <summary><span class="w09-mark" aria-hidden="true">${icon}</span><b>${q.item.id}</b>
          <span class="w09-rank">${rankText}</span>${broken ? '<span class="chip warn">⚠ 망가진 문서</span>' : ''}
          <span class="w09-qtext">${escapeHtml(q.item.question)}</span></summary>
        ${why}
        ${q.item.note ? `<p class="w09-why">골든셋 메모: ${escapeHtml(q.item.note)}</p>` : ''}
        <p class="w09-small">근거: ${escapeHtml(docById[ev.doc]?.title ?? ev.doc)} — 「${escapeHtml(ev.quote)}」</p>
        <ol class="w09-top">${top || '<li>(검색 결과 없음)</li>'}</ol>
      </details></li>`;
    }).join('');
  }

  // ---------- Faithfulness panel ----------
  function contextFor(qid) {
    if (!last) return [];
    const q = last.perQ.find((x) => x.item.id === qid);
    return q ? q.ranked.slice(0, s.k).map((id) => last.idx.byId.get(id)) : [];
  }

  function renderFaith() {
    if (!last) return;
    const ctx = contextFor(f.q);
    $('[data-slot=ctx]').innerHTML = ctx.length
      ? ctx.map((c) => `<li><span class="w09-src">${escapeHtml(docById[c.doc].title)} #${c.index + 1}</span>${escapeHtml(clip(c.text, 160))}</li>`).join('')
      : '<li>(검색 결과 없음)</li>';
    const res = faithHeuristic(f.answer, ctx.map((c) => c.text).join('\n'));
    $('[data-slot=heur]').innerHTML = res.sentences.length
      ? res.sentences.map((x) => `<li class="${x.supported ? 'is-ok' : 'is-bad'}">
          <b>${x.supported ? '✓ 뒷받침' : '✗ 근거 없음'}</b> ${escapeHtml(x.sentence)}
          ${x.missing.length ? `<span class="w09-small">근거에 없는 단어: ${x.missing.map(escapeHtml).join(', ')}</span>` : ''}</li>`).join('')
        + `<li class="w09-score">휴리스틱 Faithfulness = ${res.supported} / ${res.sentences.length} = <b>${fmt(res.score)}</b></li>`
      : '<li>답변을 입력한다.</li>';
    renderKeyState();
    if (!f.busy && !hasKeySafe(f.provider)) renderRecordedJudge();
  }

  function renderKeyState() {
    $('[data-slot=key-state]').textContent = hasKeySafe(f.provider)
      ? `${PROVIDERS[f.provider].label} 키 적용됨 (이 탭에만 저장)`
      : '키 없음 — 저장된 예시 판정을 보여 준다';
  }

  function renderRecordedJudge() {
    const ex = RECORDED_JUDGE[f.q] ?? RECORDED_JUDGE.q01;
    const qid = RECORDED_JUDGE[f.q] ? f.q : 'q01';
    const matches = RECORDED_JUDGE[f.q] && f.answer.trim() === ex.answer;
    $('[data-slot=judge]').innerHTML = `<div class="w09-recorded">
      <span class="chip">예시 응답 · 수업용 작성 예시</span>
      <p class="w09-small">${matches
        ? 'API 키가 없어서 미리 저장해 둔 예시 판정을 보여 준다. 기본 설정(재귀 200 · BM25 · k=3)의 컨텍스트 기준이며, 지금 호출한 결과가 아니다.'
        : `API 키가 없어서 미리 저장해 둔 예시 판정을 보여 준다. <b>지금 입력과는 다른</b> ${qid}의 “근거에 없는 주장 섞기” 답변에 대한 판정이다.`}</p>
      <p class="w09-small">답변: ${escapeHtml(ex.answer)}</p>
      ${judgeTable(ex.claims)}
      <p>Faithfulness = ${ex.claims.filter((c) => c.supported).length} / ${ex.claims.length} = <b>${fmt(ex.faithfulness)}</b></p>
      ${qid === 'q12' ? '<p class="w09-small">휴리스틱은 이 답변을 “뒷받침”으로 본다. 단어(녹화·영상·보강)는 모두 근거에 있지만 뜻은 정반대다. 뜻을 읽는 LLM 판정자는 이런 모순을 잡을 수 있다(항상은 아니다).</p>' : ''}
    </div>`;
  }

  async function runJudge() {
    if (!hasKeySafe(f.provider)) {
      renderRecordedJudge();
      $('[data-slot=key-state]').textContent = 'API 키를 먼저 적용해야 실제 판정을 요청할 수 있다.';
      return;
    }
    const ctx = contextFor(f.q);
    if (!ctx.length || !f.answer.trim()) return;
    f.busy = true;
    const btn = $('[data-act=judge]');
    btn.disabled = true;
    $('[data-slot=judge]').innerHTML = '<div class="widget__status"><span class="spinner" aria-hidden="true"></span> LLM 판정자에게 묻는 중…</div>';
    try {
      const { text } = await generate({
        provider: f.provider,
        model: f.llmModel,
        ...judgePrompt(ctx.map((c) => c.text), f.answer),
        maxTokens: 900,
        temperature: 0,
        signal: ctrl.signal,
      });
      if (!st.alive) return;
      const parsed = parseJudge(text);
      if (!parsed) {
        $('[data-slot=judge]').innerHTML = `<div class="callout"><span class="callout__title">판정 결과를 JSON으로 읽지 못했다</span>
          <p class="w09-small">판정자가 형식을 지키지 않았다. LLM-as-judge를 쓸 때는 형식 검사와 재시도가 필요하다. 원문:</p>
          <pre class="w09-raw">${escapeHtml(text.slice(0, 1500))}</pre></div>`;
        return;
      }
      const sup = parsed.claims.filter((c) => c.supported).length;
      const score = parsed.claims.length ? sup / parsed.claims.length : 0;
      const selfScore = Number(parsed.faithfulness);
      $('[data-slot=judge]').innerHTML = `<div class="w09-live"><span class="chip ok">실제 판정 · ${escapeHtml(PROVIDERS[f.provider].label)} ${escapeHtml(f.llmModel)}</span>
        ${judgeTable(parsed.claims)}
        <p>Faithfulness = ${sup} / ${parsed.claims.length} = <b>${fmt(score)}</b></p>
        ${Number.isFinite(selfScore) && Math.abs(selfScore - score) > 0.01
          ? `<p class="w09-small">판정자가 직접 쓴 점수는 ${fmt(selfScore)}로 주장별 판정과 맞지 않는다. 산술은 LLM에 맡기지 않고 코드로 다시 계산한다.</p>` : ''}
        <p class="w09-small">같은 요청을 다시 보내 판정이 바뀌는지 확인해 본다. 판정자도 틀릴 수 있다.</p></div>`;
    } catch (err) {
      if (err.name === 'AbortError' || !st.alive) return;
      const msg = err instanceof LLMError ? err.message : 'LLM 호출 중 알 수 없는 오류가 났다.';
      $('[data-slot=judge]').innerHTML = `<div class="widget__error" role="alert">${escapeHtml(msg)}</div>`;
    } finally {
      f.busy = false;
      btn.disabled = false;
    }
  }

  function syncLlmModels() {
    const p = PROVIDERS[f.provider];
    $('[data-in=llm-model]').innerHTML = p.models.map((m) => `<option value="${m}">${m}</option>`).join('');
    f.llmModel = p.defaultModel;
    $('[data-in=llm-model]').value = f.llmModel;
  }

  // ---------- saved rows ----------
  function rowFrom(r) {
    return {
      label: cfgLabel(r.cfg),
      cfg: r.cfg,
      k: r.cfg.k,
      mode: r.cfg.mode,
      n: r.perQ.length,
      ...pick(r.metrics, ['recall', 'hit', 'mrr', 'ndcg']),
      fails: r.perQ.filter((q) => !(q.rank > 0 && q.rank <= r.cfg.k)).map((q) => q.item.id),
    };
  }

  function loadRows() {
    try {
      const v = JSON.parse(localStorage.getItem(STORE_KEY) ?? '[]');
      return Array.isArray(v) ? v.slice(0, MAX_ROWS) : [];
    } catch {
      return [];
    }
  }

  function saveRows() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(rows));
    } catch {
      /* private mode etc. — rows stay in memory */
    }
  }

  function flash(msg) {
    const m = $('[data-slot=save-msg]');
    m.textContent = msg;
    clearTimeout(st.flashTimer);
    st.flashTimer = setTimeout(() => (m.textContent = ''), 2500);
  }

  // ---------- events ----------
  const on = (target, type, fn) => target.addEventListener(type, fn, { signal: ctrl.signal });
  on(root, 'change', (e) => {
    const t = e.target;
    if (t.type === 'radio') {
      s[t.name.split(':')[0]] = t.value;
      scheduleRefresh();
    } else if (t.matches('[data-in=strategy]')) {
      s.strategy = t.value;
      scheduleRefresh();
    } else if (t.matches('[data-in=model]')) {
      s.model = t.value;
      ensureModel();
      scheduleRefresh();
    } else if (t.matches('[data-in=fq]')) {
      f.q = t.value;
      f.answer = UNFAITHFUL[f.q] ?? '';
      $('[data-in=answer]').value = f.answer;
      $('[data-slot=judge]').innerHTML = '';
      renderFaith();
    } else if (t.matches('[data-in=provider]')) {
      f.provider = t.value;
      syncLlmModels();
      $('[data-slot=judge]').innerHTML = '';
      renderFaith();
    } else if (t.matches('[data-in=llm-model]')) {
      f.llmModel = t.value;
    }
  });
  on(root, 'input', (e) => {
    const t = e.target;
    if (t.matches('[data-in=size]')) s.size = Number(t.value);
    else if (t.matches('[data-in=overlap]')) s.overlap = Number(t.value);
    else if (t.matches('[data-in=k]')) s.k = Number(t.value);
    else if (t.matches('[data-in=answer]')) {
      f.answer = t.value;
      renderFaith();
      return;
    } else return;
    syncOverlap();
    scheduleRefresh();
  });
  on(root, 'click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    if (b.dataset.preset) {
      const p = PRESETS[b.dataset.preset];
      Object.assign(s, pick(p, ['strategy', 'size', 'overlap', 'retriever', 'mode', 'k']));
      syncInputs();
      scheduleRefresh();
    } else if (b.dataset.fill) {
      f.answer = b.dataset.fill === 'faithful' ? items.find((it) => it.id === f.q)?.answer ?? '' : UNFAITHFUL[f.q] ?? '';
      $('[data-in=answer]').value = f.answer;
      renderFaith();
    } else if (b.dataset.act === 'save') {
      if (!last || out.classList.contains('is-stale')) return flash('아직 채점이 끝나지 않았다.');
      const row = rowFrom(last);
      const dup = rows.findIndex((r) => r.label === row.label);
      if (dup >= 0) rows.splice(dup, 1);
      rows.unshift(row);
      rows = rows.slice(0, MAX_ROWS);
      saveRows();
      renderBoard();
      flash(dup >= 0 ? '같은 설정을 갱신했다.' : `저장했다 (${rows.length}행).`);
    } else if (b.dataset.act === 'clear') {
      rows = [];
      saveRows();
      renderBoard();
      flash('저장 기록을 지웠다.');
    } else if (b.dataset.act === 'key-set') {
      const input = $('[data-in=key]');
      if (!input.value.trim()) return;
      try {
        setKey(f.provider, input.value);
      } catch {
        /* storage unavailable */
      }
      input.value = '';
      $('[data-slot=judge]').innerHTML = '';
      renderKeyState();
    } else if (b.dataset.act === 'key-clear') {
      try {
        clearKey(f.provider);
      } catch {
        /* ignore */
      }
      renderFaith();
    } else if (b.dataset.act === 'judge') {
      runJudge();
    }
  });
  // board buttons live in the output node (terminal or inline)
  on(out, 'click', (e) => {
    const del = e.target.closest('[data-del]');
    const res = e.target.closest('[data-restore]');
    if (del) {
      rows.splice(Number(del.dataset.del), 1);
      saveRows();
      renderBoard();
    } else if (res) {
      const r = rows[Number(res.dataset.restore)];
      if (!r?.cfg) return;
      Object.assign(s, pick(r.cfg, ['strategy', 'size', 'overlap', 'retriever', 'mode', 'k', 'model']));
      if (currentModel().id !== s.model && model.id !== s.model) ensureModel();
      syncInputs();
      scheduleRefresh();
    }
  });

  // ---------- initial render ----------
  syncInputs();
  syncLlmModels();
  $('[data-in=answer]').value = f.answer;
  $('[data-slot=status]').hidden = true;
  $('[data-slot=body]').hidden = false;
  registerOutput(outputId, {
    title: options.outputTitle ?? '평가 점수판',
    node: out,
    inlineHost: $('[data-slot=out-inline]'),
  });
  await refresh();
  if (st.alive) ensureModel();
}

export function unmount(el) {
  const st = state.get(el);
  if (st) {
    st.alive = false;
    clearTimeout(st.timer);
    clearTimeout(st.flashTimer);
    st.ctrl.abort();
    unregisterOutput(st.outputId);
  }
  state.delete(el);
  el.replaceChildren();
}

// ---------- pure helpers (exported for tests) ----------

/** Chunk every document; chunk ids are "docId#index". */
export function buildChunks(docs, cfg) {
  return docs.flatMap((d) =>
    chunk(d.text, { strategy: cfg.strategy, size: cfg.size, overlap: cfg.overlap })
      .map((c) => ({ ...c, id: `${d.id}#${c.index}`, doc: d.id })));
}

/** RRF over BM25 and vector lists, per question. */
export function hybridLists(bm25, vec, depth) {
  return new Map([...bm25.keys()].map((qid) => [qid, rrf([bm25.get(qid), vec.get(qid)], { k: 60, topK: depth })]));
}

/**
 * Turn a ranked chunk list into metric input for one golden item.
 * Each evidence unit is credited once (at its first occurrence) so overlap
 * duplicates or several chunks of the same document cannot push recall above 1.
 * @param {{ id: string, doc: string, text: string }[]} rankedChunks
 * @param {{ evidence: { doc: string, quote: string }[] }} item
 * @param {'chunk'|'doc'} mode
 */
export function labelRun(rankedChunks, item, mode) {
  const unitOf = (c) => {
    for (let i = 0; i < item.evidence.length; i++) {
      const ev = item.evidence[i];
      if (c.doc !== ev.doc) continue;
      if (mode === 'doc') return `doc:${ev.doc}`;
      if (c.text.includes(ev.quote)) return `ev:${i}`;
    }
    return null;
  };
  const relevant = [...new Set(item.evidence.map((ev, i) => (mode === 'doc' ? `doc:${ev.doc}` : `ev:${i}`)))];
  const seen = new Set();
  const relAt = new Set();
  const retrieved = rankedChunks.map((c, i) => {
    const u = unitOf(c);
    if (u && !seen.has(u)) {
      seen.add(u);
      relAt.add(i);
      return u;
    }
    return c.id;
  });
  const first = retrieved.findIndex((_, i) => relAt.has(i));
  return { retrieved, relevant, relAt, rank: first + 1 };
}

/** Score one configuration: per-question ranks + averaged metrics. */
export function scoreConfig(items, idx, lists, cfg) {
  const perQ = items.map((item) => {
    const ranked = (lists.get(item.id) ?? []).map((r) => r.id);
    const rankedChunks = ranked.map((id) => idx.byId.get(id));
    const lab = labelRun(rankedChunks, item, cfg.mode);
    // chunk mode: is there any chunk that contains the evidence at all?
    const cut = cfg.mode === 'chunk' && item.evidence.every((ev) =>
      !idx.chunks.some((c) => c.doc === ev.doc && c.text.includes(ev.quote)));
    return { item, ranked, relAt: lab.relAt, rank: lab.rank, cut, run: { retrieved: lab.retrieved, relevant: lab.relevant } };
  });
  return { perQ, metrics: evaluate(perQ.map((q) => q.run), cfg.k) };
}

/** Wilson score interval (95%) for a proportion hits/n. */
export function wilson(hits, n, z = 1.96) {
  if (n === 0) return [0, 0];
  const p = hits / n;
  const d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

const SUFFIXES = ['하려면', '으로는', '에서는', '이라면', '에게', '에서', '으로', '하면', '려면', '까지', '부터', '이며', '이고', '이다',
  '은', '는', '이', '가', '을', '를', '에', '의', '로', '와', '과', '도', '만', '다', '며', '고'];
// function words that carry no fact of their own
const STOP = new Set(['그리고', '또한', '하지만', '그러나', '따라서', '것이', '것을', '경우',
  '한다', '된다', '있다', '없다', '않다', '하는', '되는', '있는', '없는', '않는', '않을', '있을', '없을']);

function stripSuffix(w) {
  if (!/[가-힣]$/.test(w)) return w;
  for (const suf of SUFFIXES) if (w.length - suf.length >= 2 && w.endsWith(suf)) return w.slice(0, -suf.length);
  return w;
}

/** Content words of a sentence with common Korean particles/endings stripped. */
export function keyTokens(sentence) {
  const words = sentence.toLowerCase().match(/[\p{L}\p{N}%]+/gu) ?? [];
  return [...new Set(words.map(stripSuffix).filter((w) => w.length >= 2 && !STOP.has(w)))];
}

/**
 * Keyless Faithfulness heuristic: a sentence is "supported" when every key token
 * appears in the context (allowing one trailing syllable of inflection).
 * It cannot see negation or contradiction.
 */
export function faithHeuristic(answer, context) {
  const ctx = context.toLowerCase();
  const found = (t) => ctx.includes(t) || (t.length >= 3 && /[가-힣]$/.test(t) && ctx.includes(t.slice(0, -1)));
  const sentences = sentenceSpans(answer.trim())
    .map(([a, b]) => answer.trim().slice(a, b).trim())
    .filter(Boolean)
    .map((sentence) => {
      const tokens = keyTokens(sentence);
      const missing = tokens.filter((t) => !found(t));
      return { sentence, tokens, missing, supported: tokens.length > 0 && missing.length === 0 };
    });
  const supported = sentences.filter((x) => x.supported).length;
  return { sentences, supported, score: sentences.length ? supported / sentences.length : 0 };
}

/** Prompt for an LLM judge that must return per-claim JSON. */
export function judgePrompt(contexts, answer) {
  const sources = contexts.map((t, i) => `[${i + 1}] ${t}`).join('\n\n');
  return {
    system:
      '너는 RAG 답변의 Faithfulness(근거 충실도)를 채점하는 평가자다. 주어진 근거에 적힌 내용만 사실로 인정하고, ' +
      '상식이나 배경지식으로 보충하지 않는다. 근거와 모순되는 주장은 뒷받침되지 않는 것으로 본다.',
    messages: [{
      role: 'user',
      content:
        `근거:\n${sources}\n\n답변:\n${answer}\n\n` +
        '답변을 독립된 주장(claim)으로 나누고, 각 주장이 근거로 뒷받침되는지 판정하라. 다른 말 없이 다음 JSON만 출력한다.\n' +
        '{"claims":[{"claim":"...","supported":true,"reason":"근거 [번호]를 인용한 짧은 이유"}],"faithfulness":0.0}\n' +
        'faithfulness는 뒷받침되는 주장 수를 전체 주장 수로 나눈 값이다.',
    }],
  };
}

/** Extract the judge JSON from model text (tolerates code fences and chatter). */
export function parseJudge(text) {
  const a = text.indexOf('{');
  const b = text.lastIndexOf('}');
  if (a < 0 || b <= a) return null;
  try {
    const j = JSON.parse(text.slice(a, b + 1));
    if (!Array.isArray(j.claims)) return null;
    return {
      claims: j.claims.map((c) => ({ claim: String(c.claim ?? ''), supported: c.supported === true, reason: String(c.reason ?? '') })),
      faithfulness: j.faithfulness,
    };
  } catch {
    return null;
  }
}

export function cfgLabel(c) {
  const strat = { fixed: '고정', sentence: '문장', recursive: '재귀' }[c.strategy] ?? c.strategy;
  const ret = { bm25: 'BM25', vector: '벡터', hybrid: 'RRF' }[c.retriever] ?? c.retriever;
  const m = c.retriever === 'bm25' ? '' : ` (${shortModel(c.model)})`;
  return `${strat} ${c.size}/${c.overlap} · ${ret}${m} · k${c.k} · ${c.mode === 'doc' ? '문서' : '청크'}`;
}

function shortModel(id = '') {
  if (id.includes('e5')) return 'e5';
  if (id.includes('paraphrase')) return 'mMiniLM';
  if (id.includes('all-MiniLM')) return '영어 MiniLM';
  return id.split('/').pop();
}

// ---------- small utils ----------

function radios(name, entries, uid) {
  return Object.entries(entries)
    .map(([v, label]) => `<label><input type="radio" name="${name}:${uid}" value="${v}"><span>${escapeHtml(label)}</span></label>`)
    .join('');
}

function judgeTable(claims) {
  return `<ol class="w09-claims">${claims.map((c) => `<li class="${c.supported ? 'is-ok' : 'is-bad'}">
    <b>${c.supported ? '✓ 뒷받침' : '✗ 근거 없음'}</b> ${escapeHtml(c.claim)}
    <span class="w09-small">${escapeHtml(c.reason)}</span></li>`).join('')}</ol>`;
}

function hasKeySafe(provider) {
  try {
    return hasKey(provider);
  } catch {
    return false;
  }
}

const vkey = (modelId, role, text) => `${modelId}\u0000${role}\u0000${text}`;
const fmt = (v) => (Number.isFinite(v) ? v.toFixed(3) : '—');
const clip = (t, n) => {
  const x = t.replace(/\s+/g, ' ').trim();
  return x.length > n ? `${x.slice(0, n)}…` : x;
};
const pick = (o, keys) => Object.fromEntries(keys.filter((k) => o[k] !== undefined).map((k) => [k, o[k]]));

async function fetchJson(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${res.status} ${url.pathname}`);
  return res.json();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
