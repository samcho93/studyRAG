// w07 최소 RAG 파이프라인
// One concept: the whole RAG pipeline end to end, with the artifact and the
// time of every stage visible, so a wrong answer can be traced back to the
// stage that caused it (check retrieval first).
//   색인 단계: ① 문서 로드 → ② 청킹 → ③ 임베딩(+벡터스토어 저장)
//   질의 단계: ④ 검색 top-k → ⑤ 프롬프트 구성 → ⑥ 생성
// Without an API key step ⑥ shows clearly labeled saved example answers.

import { chunk } from '../core/chunker.js';
import { VectorStore } from '../core/vectorstore.js';
import { MODELS, DEFAULT_MODEL, loadModelWithUI, embed, embedBatch, currentModel, modelInfo } from '../core/embed.js';
import { PROVIDERS, setKey, hasKey, clearKey, generate, ragPrompt, LLMError } from '../core/llm.js';
import { registerOutput, unregisterOutput } from '../site/result.js';

const DATA = new URL('../../data/', import.meta.url);
const CUSTOM = '__custom';

const STEPS = [
  { id: 'load', no: '①', title: '문서 로드', sub: 'corpus.json → 문서 배열' },
  { id: 'chunk', no: '②', title: '청킹', sub: 'core/chunker.js · 재귀 분할' },
  { id: 'embed', no: '③', title: '임베딩 · 벡터스토어 저장', sub: 'core/embed.js → core/vectorstore.js' },
  { id: 'search', no: '④', title: '검색 top-k', sub: '질문 임베딩 → 코사인 유사도 정렬' },
  { id: 'prompt', no: '⑤', title: '프롬프트 구성', sub: 'core/llm.js ragPrompt()' },
  { id: 'generate', no: '⑥', title: '생성', sub: 'core/llm.js generate()' },
];

const PRESETS = {
  ok: { label: '✓ 성공: q01', q: 'q01', k: 3 },
  top1: { label: '✗ top-k=1: q03', q: 'q03', k: 1 },
  table: { label: '✗ 표 깨짐: q10', q: 'q10', k: 3 },
  ocr: { label: '✗ OCR 오류: q11', q: 'q11', k: 3 },
};

const VERDICT = {
  correct: ['ok', '정답 ✓'],
  wrong: ['warn', '오답 ✗ (틀린 근거를 그대로 옮김)'],
  refused: ['warn', '답 못함 (정답은 문서셋 안에 있다)'],
};

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget" data-w07>
    <h3 class="widget__title">최소 RAG 파이프라인</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> 문서셋과 골든셋을 불러오는 중…
    </div>
    <div data-slot="body" hidden>
      <div class="widget__controls">
        <label class="field w07-wide">
          <span class="field__label">질문 (골든셋 12문항 또는 직접 입력)</span>
          <select data-in="q"></select>
        </label>
        <label class="field w07-wide" data-slot="custom-wrap" hidden>
          <span class="field__label">직접 입력한 질문</span>
          <input type="text" data-in="custom" maxlength="200" placeholder="예: 실습실은 몇 시까지 여나?">
        </label>
        <label class="field">
          <span class="field__label">청크 크기 (문자) <output data-out="size"></output></span>
          <input type="range" data-in="size" min="60" max="400" step="20">
        </label>
        <label class="field">
          <span class="field__label">오버랩 (문자) <output data-out="overlap"></output></span>
          <input type="range" data-in="overlap" min="0" max="100" step="10">
        </label>
        <label class="field">
          <span class="field__label">top-k (LLM에 넘길 청크 수) <output data-out="k"></output></span>
          <input type="range" data-in="k" min="1" max="8" step="1">
        </label>
        <label class="field">
          <span class="field__label">임베딩 모델</span>
          <select data-in="model"></select>
        </label>
      </div>
      <div class="btn-row" data-slot="presets"></div>
      <div class="btn-row w07-run">
        <button type="button" class="btn primary" data-act="rerun">↻ 다시 실행</button>
        <span class="w07-hint">청크 크기·오버랩·모델을 바꾸면 색인부터, 질문·top-k를 바꾸면 질의 단계만 다시 돈다.</span>
      </div>
      <details data-slot="llm">
        <summary>LLM API 키 (선택 · 이 탭에만 저장)</summary>
        <div class="widget__controls w07-llm">
          <label class="field">
            <span class="field__label">공급자</span>
            <select data-in="provider"></select>
          </label>
          <label class="field">
            <span class="field__label">모델</span>
            <select data-in="llmModel"></select>
          </label>
          <label class="field w07-wide">
            <span class="field__label">API 키</span>
            <input type="password" data-in="key" autocomplete="off" spellcheck="false" placeholder="키를 붙여 넣고 저장한다">
          </label>
        </div>
        <div class="btn-row">
          <button type="button" class="btn small" data-act="save-key">키 저장 (sessionStorage)</button>
          <button type="button" class="btn small ghost" data-act="clear-key">키 지우기</button>
        </div>
        <p class="w07-keystate" data-slot="keystate" aria-live="polite"></p>
      </details>
      <div data-slot="out-inline"></div>
    </div>
  </div>`;

// Output (golden hit strip + stepper) goes to the 실행 결과 panel on wide
// screens and stays under the controls on narrow ones.
const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w07-out">
    <div class="w07-strip" data-slot="strip"></div>
    <ol class="w07-steps" data-slot="steps"></ol>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ question?: string, topK?: number, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w07:${++seq}`;
  const timers = new Set();
  const st = { ctrl, outputId, timers, genCtrl: null };
  state.set(el, st);

  let corpus;
  let golden;
  let saved;
  const fetchStart = performance.now();
  try {
    [corpus, golden, saved] = await Promise.all([
      fetchJson(new URL('corpus/corpus.json', DATA), ctrl.signal),
      fetchJson(new URL('golden/golden.json', DATA), ctrl.signal),
      fetchJson(new URL('w07/saved-answers.json', DATA), ctrl.signal),
    ]);
  } catch (err) {
    if (err.name === 'AbortError') return;
    $('[data-slot=status]').innerHTML = `<div class="widget__error" role="alert">
      문서셋을 불러오지 못했다 (${escapeHtml(err.message)}).<br>
      파일을 더블클릭해 <code>file://</code>로 열었다면 ES Module과 fetch가 동작하지 않는다.
      저장소 폴더에서 <code>python -m http.server</code>를 실행하고 <code>http://localhost:8000</code>으로 연다.</div>`;
    return;
  }
  const loadMs = performance.now() - fetchStart;

  const firstProvider = Object.keys(PROVIDERS)[0];
  const s = {
    q: options.question ?? 'q01',
    custom: '',
    size: 200,
    overlap: 0,
    k: options.topK ?? 3,
    model: DEFAULT_MODEL,
    provider: firstProvider,
    llmModel: PROVIDERS[firstProvider].defaultModel,
  };
  const docById = Object.fromEntries(corpus.documents.map((d) => [d.id, d]));
  const goldenById = Object.fromEntries(golden.items.map((it) => [it.id, it]));

  // runtime pipeline state
  let docs = null; // step ① artifact
  let index = null; // step ③ artifact: { key, store, dim, goldenVecs, ranks }
  let last = null; // step ④/⑤ artifacts for step ⑥
  let runSeq = 0; // invalidates stale async work

  // ---------- controls ----------
  $('[data-in=q]').innerHTML =
    golden.items
      .map((it) => `<option value="${it.id}">${it.id} · ${escapeHtml(it.question)}${it.note ? ' ⚠' : ''}</option>`)
      .join('') + `<option value="${CUSTOM}">✎ 직접 입력…</option>`;
  $('[data-in=model]').innerHTML = MODELS.map((m) => `<option value="${m.id}">${escapeHtml(m.label)}</option>`).join('');
  $('[data-in=provider]').innerHTML = Object.entries(PROVIDERS)
    .map(([k, p]) => `<option value="${k}">${escapeHtml(p.label)}</option>`)
    .join('');
  $('[data-slot=presets]').innerHTML = Object.entries(PRESETS)
    .map(([k, p]) => `<button type="button" class="btn small" data-preset="${k}">${p.label}</button>`)
    .join('');
  $('[data-slot=steps]').innerHTML = STEPS.map(
    (step, i) => `${i === 0 ? '<li class="w07-phase">색인 단계 · 문서가 바뀔 때만 한 번</li>' : ''}${
      i === 3 ? '<li class="w07-phase">질의 단계 · 질문마다 매번</li>' : ''
    }<li class="w07-step is-wait" data-step="${step.id}">
      <div class="w07-step__mark" aria-hidden="true">${step.no}</div>
      <div class="w07-step__main">
        <div class="w07-step__head">
          <b>${step.no} ${step.title}</b>
          <span class="w07-step__sub">${step.sub}</span>
          <span class="w07-step__time" data-time></span>
        </div>
        <div class="w07-step__body" data-body><span class="w07-muted">대기</span></div>
      </div>
    </li>`,
  ).join('');

  function syncInputs() {
    $('[data-in=q]').value = s.q;
    $('[data-slot=custom-wrap]').hidden = s.q !== CUSTOM;
    $('[data-in=size]').value = s.size;
    $('[data-in=overlap]').value = s.overlap;
    $('[data-in=k]').value = s.k;
    $('[data-in=model]').value = s.model;
    $('[data-in=provider]').value = s.provider;
    $('[data-in=llmModel]').innerHTML = PROVIDERS[s.provider].models
      .map((m) => `<option value="${m}">${m}</option>`)
      .join('');
    $('[data-in=llmModel]').value = s.llmModel;
    syncLabels();
    syncKeyState();
  }

  function syncLabels() {
    s.overlap = Math.min(s.overlap, s.size / 2);
    $('[data-in=overlap]').value = s.overlap;
    $('[data-out=size]').textContent = s.size;
    $('[data-out=overlap]').textContent = s.overlap;
    $('[data-out=k]').textContent = s.k;
  }

  function keyReady() {
    try {
      return hasKey(s.provider);
    } catch {
      return false; // sessionStorage blocked
    }
  }

  function syncKeyState() {
    $('[data-slot=keystate]').innerHTML = keyReady()
      ? `<span class="chip ok">키 있음 · ${escapeHtml(PROVIDERS[s.provider].label)}</span> ⑥ 생성 단계가 실제 LLM을 호출한다.`
      : '<span class="chip">키 없음</span> ⑥ 생성 단계는 저장된 예시 응답을 보여 준다. 검색·프롬프트까지는 키 없이 전부 실제로 실행된다.';
  }

  // ---------- stepper helpers ----------
  function setStep(id, { status, time, html }) {
    const li = $(`[data-step=${id}]`);
    if (status) li.className = `w07-step is-${status}`;
    if (time !== undefined) li.querySelector('[data-time]').textContent = time;
    if (html !== undefined) li.querySelector('[data-body]').innerHTML = html;
    return li.querySelector('[data-body]');
  }

  function resetFrom(ids, msg = '대기') {
    for (const id of ids) setStep(id, { status: 'wait', time: '', html: `<span class="w07-muted">${msg}</span>` });
  }

  const questionText = () => (s.q === CUSTOM ? s.custom.trim() : goldenById[s.q]?.question ?? '');
  const evidenceOf = (item) => {
    const ev = item?.evidence?.[0];
    if (!ev) return null;
    const start = docById[ev.doc]?.text.indexOf(ev.quote) ?? -1;
    return start < 0 ? null : { doc: ev.doc, start, end: start + ev.quote.length };
  };
  const holds = (c, ev) => ev && c.doc === ev.doc && c.start <= ev.start && c.end >= ev.end;
  const touches = (c, ev) => ev && c.doc === ev.doc && c.start < ev.end && c.end > ev.start;

  // ---------- 색인 단계 ----------
  async function runIndex({ force = false } = {}) {
    const my = ++runSeq;
    last = null;
    abortGen();
    const key = `${s.model}|${s.size}|${s.overlap}`;
    // drop a stale index right away so query-side events wait for the rebuild
    if (force || index?.key !== key) index = null;

    // ① 문서 로드
    let t0 = performance.now();
    if (!docs && !force) {
      docs = corpus.documents;
      t0 -= loadMs; // first run: report the fetch done at mount
    } else if (force) {
      setStep('load', { status: 'running', time: '', html: '<span class="w07-muted">불러오는 중…</span>' });
      try {
        docs = (await fetchJson(new URL('corpus/corpus.json', DATA), ctrl.signal)).documents;
      } catch (err) {
        if (err.name === 'AbortError') return;
        setStep('load', { status: 'error', html: `<div class="widget__error" role="alert">문서를 불러오지 못했다: ${escapeHtml(err.message)}</div>` });
        return;
      }
      if (my !== runSeq) return;
    }
    const totalChars = docs.reduce((a, d) => a + d.text.length, 0);
    const broken = docs.filter((d) => d.type === 'broken').length;
    setStep('load', {
      status: 'done',
      time: fmtMs(performance.now() - t0),
      html: `<div class="w07-facts"><span><b>${docs.length}</b>개 문서</span><span><b>${totalChars.toLocaleString()}</b>자</span><span>깨진 문서 <b>${broken}</b>개 ⚠</span></div>
        <pre class="w07-shape">docs[0] = ${escapeHtml(shape({ id: docs[0].id, title: docs[0].title, type: docs[0].type, text: docs[0].text }))}</pre>`,
    });

    // ② 청킹
    t0 = performance.now();
    const myChunks = docs.flatMap((d) =>
      chunk(d.text, { strategy: 'recursive', size: s.size, overlap: s.overlap }).map((c) => ({
        id: `${d.id}#${c.index}`,
        doc: d.id,
        title: d.title,
        start: c.start,
        end: c.end,
        text: c.text,
      })),
    );
    const chunks = myChunks;
    const chunkMs = performance.now() - t0;
    const lens = chunks.map((c) => c.end - c.start);
    const ev = evidenceOf(goldenById[s.q]);
    const sample = chunks.find((c) => touches(c, ev)) ?? chunks[0];
    setStep('chunk', {
      status: 'done',
      time: fmtMs(chunkMs),
      html: `<div class="w07-facts"><span><b>${chunks.length}</b>개 청크</span><span>평균 <b>${Math.round(lens.reduce((a, b) => a + b, 0) / lens.length)}</b>자</span><span>최소/최대 <b>${Math.min(...lens)}/${Math.max(...lens)}</b>자</span></div>
        <pre class="w07-shape">chunks[${chunks.indexOf(sample)}] = ${escapeHtml(shape(sample))}</pre>`,
    });

    // ③ 임베딩 + 벡터스토어
    if (index) {
      renderEmbedDone(index, 0, true);
    } else {
      resetFrom(['search', 'prompt', 'generate'], '임베딩이 끝나면 실행된다');
      const body = setStep('embed', { status: 'running', time: '', html: '<div data-model></div><div class="w07-muted" data-prog></div>' });
      try {
        // a model switch during loading can leave another model current: retry once
        for (let i = 0; i < 2 && currentModel().id !== s.model; i++) {
          await loadModelWithUI(body.querySelector('[data-model]'), s.model);
          if (my !== runSeq) return;
        }
      } catch (err) {
        if (my !== runSeq) return;
        setStep('embed', { status: 'error' });
        resetFrom(['search', 'prompt', 'generate'], '임베딩 모델이 없어 검색할 수 없다. 네트워크를 확인하고 ↻ 다시 실행한다.');
        return;
      }
      if (my !== runSeq) return;
      t0 = performance.now();
      const prog = body.querySelector('[data-prog]');
      let vectors;
      let goldenVecs;
      try {
        vectors = await embedBatch(
          myChunks.map((c) => c.text),
          { role: 'passage', onProgress: (d, n) => my === runSeq && (prog.textContent = `청크 임베딩 ${d}/${n}`) },
        );
        if (my !== runSeq) return;
        goldenVecs = await embedBatch(golden.items.map((it) => it.question), { role: 'query' });
      } catch (err) {
        if (my !== runSeq) return;
        setStep('embed', { status: 'error', html: `<div class="widget__error" role="alert">임베딩 중 오류: ${escapeHtml(err.message)}</div>` });
        return;
      }
      if (my !== runSeq) return;
      const store = new VectorStore();
      myChunks.forEach((c, i) => store.add(c.id, vectors[i], c));
      index = { key, store, dim: vectors[0]?.length ?? 0, sample: vectors[0], goldenVecs, chunks: myChunks };
      index.ranks = golden.items.map((it, i) => evidenceRank(store.search(goldenVecs[i], store.size), evidenceOf(it)));
      renderEmbedDone(index, performance.now() - t0, false);
    }
    await runQuery(my);
  }

  function renderEmbedDone(ix, ms, cached) {
    const m = currentModel();
    const v = Array.from(ix.sample ?? []).slice(0, 6).map((x) => x.toFixed(4)).join(', ');
    setStep('embed', {
      status: 'done',
      time: cached ? '캐시' : fmtMs(ms),
      html: `<div class="w07-facts"><span><b>${ix.store.size}</b>개 × <b>${ix.dim}</b>차원</span><span>${escapeHtml(modelInfo(s.model).id.replace('Xenova/', ''))}</span><span>${m.device === 'webgpu' ? 'WebGPU' : 'WASM'}</span>${cached ? '<span>같은 설정이라 다시 계산하지 않음</span>' : ''}</div>
        <pre class="w07-shape">store.items[0] = { id: "${escapeHtml(ix.chunks[0].id)}", vector: [${v}, …], meta: {…} }</pre>`,
    });
  }

  // ---------- 질의 단계 ----------
  async function runQuery(my) {
    if (!index) return; // index still building: runIndex calls runQuery when done
    if (my === undefined) my = ++runSeq;
    abortGen();
    const question = questionText();
    const item = s.q === CUSTOM ? null : goldenById[s.q];
    renderStrip();
    if (!question) {
      resetFrom(['search', 'prompt', 'generate'], '질문을 입력한다');
      return;
    }

    // ④ 검색
    setStep('search', { status: 'running' });
    let t0 = performance.now();
    let qv;
    try {
      qv = await embed(question, { role: 'query' });
    } catch (err) {
      if (my !== runSeq) return;
      setStep('search', { status: 'error', html: `<div class="widget__error" role="alert">질문 임베딩 실패: ${escapeHtml(err.message)}</div>` });
      return;
    }
    if (my !== runSeq) return;
    const embedMs = performance.now() - t0;
    t0 = performance.now();
    const ranked = index.store.search(qv, index.store.size);
    const searchMs = performance.now() - t0;
    const top = ranked.slice(0, s.k);
    const ev = evidenceOf(item);
    const rank = item ? evidenceRank(ranked, ev) : null;
    const hitIdx = top.findIndex((r) => holds(r.meta, ev));
    setStep('search', {
      status: 'done',
      time: `${fmtMs(embedMs + searchMs)}`,
      html: `<div class="w07-facts"><span>질문 임베딩 <b>${fmtMs(embedMs)}</b></span><span>검색 <b>${fmtMs(searchMs)}</b> (${index.store.size}개 비교)</span></div>
        ${hitVerdict(item, rank, top, ev)}
        <ol class="w07-results">${top.map((r, i) => resultHtml(r, i, ev)).join('')}</ol>`,
    });

    // ⑤ 프롬프트
    t0 = performance.now();
    const prompt = ragPrompt(question, top.map((r) => ({ text: r.meta.text.trim() })));
    const promptMs = performance.now() - t0;
    const promptText = `── system ──\n${prompt.system}\n\n── user ──\n${prompt.messages[0].content}`;
    setStep('prompt', {
      status: 'done',
      time: fmtMs(promptMs),
      html: `<div class="w07-facts"><span>근거 <b>${top.length}</b>개</span><span>총 <b>${promptText.length.toLocaleString()}</b>자</span><span>LLM이 보는 것은 이것이 전부다</span></div>
        <pre class="w07-prompt" tabindex="0" aria-label="LLM에 보내는 프롬프트 전문">${escapeHtml(promptText)}</pre>`,
    });

    last = { question, item, prompt, hitIdx };
    runGenerate(my);
  }

  // ---------- ⑥ 생성 ----------
  function abortGen() {
    st.genCtrl?.abort();
    st.genCtrl = null;
    for (const t of timers) clearTimeout(t);
    timers.clear();
  }

  function runGenerate(my) {
    if (!last) return;
    abortGen();
    const { item, hitIdx } = last;
    if (!keyReady()) {
      setStep('generate', { status: 'done', time: '저장됨', html: savedHtml(item, hitIdx) });
      return;
    }
    setStep('generate', {
      status: 'running',
      time: '',
      html: '<div class="widget__status"><span class="spinner" aria-hidden="true"></span> LLM 호출 중…</div>',
    });
    // debounce so dragging a slider does not fire a request per step
    const t = setTimeout(async () => {
      timers.delete(t);
      const gen = new AbortController();
      st.genCtrl = gen;
      const t0 = performance.now();
      try {
        const res = await generate({
          provider: s.provider,
          model: s.llmModel,
          ...last.prompt,
          maxTokens: 400,
          temperature: 0,
          signal: gen.signal,
        });
        if (my !== runSeq) return;
        setStep('generate', {
          status: 'done',
          time: fmtMs(performance.now() - t0),
          html: `<div class="w07-facts"><span class="chip ok">실제 생성 · ${escapeHtml(s.llmModel)}</span></div>
            <blockquote class="w07-answer">${citeHtml(res.text || '(빈 응답)')}</blockquote>
            ${goldenLine(item)}`,
        });
      } catch (err) {
        if (err.name === 'AbortError' || my !== runSeq) return;
        const msg = err instanceof LLMError ? err.message : 'LLM 호출 중 알 수 없는 오류가 났다.';
        setStep('generate', {
          status: 'error',
          time: '',
          html: `<div class="widget__error" role="alert">${escapeHtml(msg)}</div>${savedHtml(item, hitIdx)}`,
        });
      }
    }, 500);
    timers.add(t);
  }

  function savedHtml(item, hitIdx) {
    if (!item) {
      return `<div class="callout callout--more"><span class="callout__title">직접 입력한 질문에는 저장된 응답이 없다</span>
        <p>LLM이 받을 입력은 ④의 컨텍스트와 ⑤의 프롬프트가 전부다. 그 근거만으로 답할 수 있을지 먼저 판단해 본다. API 키를 저장하면 이 프롬프트로 실제 생성한다.</p></div>`;
    }
    const rec = saved.items[item.id] ?? {};
    const hit = hitIdx >= 0;
    const ans = hit ? rec.hit : rec.miss ?? saved.fallbackMiss;
    const text = ans.text.replaceAll('{ev}', `[${hitIdx + 1}]`);
    const [cls, label] = VERDICT[ans.verdict] ?? ['', ''];
    return `<div class="w07-facts"><span class="chip warn">예시 응답 · 수업용 작성 예시</span><span class="chip">${hit ? '근거 검색 성공을 가정한 예시' : '근거 검색 실패를 가정한 예시'}</span><span class="chip ${cls}">${label}</span></div>
      <blockquote class="w07-answer">${citeHtml(text)}</blockquote>
      ${goldenLine(item)}
      <p class="w07-muted">키가 없어 실제로 호출하지 않았다. ${escapeHtml(saved.recordedWith)} 설정과 같은 시스템 지시를 가정해 수업용으로 작성한 예시 중 현재 검색 결과(근거 ${hit ? '포함' : '없음'})에 맞는 것을 보여 준다.</p>`;
  }

  function goldenLine(item) {
    return item ? `<p class="w07-golden"><b>골든셋 정답</b> ${escapeHtml(item.answer)}</p>` : '';
  }

  // ---------- golden hit strip ----------
  function renderStrip() {
    if (!index) {
      $('[data-slot=strip]').innerHTML = '';
      return;
    }
    const hits = index.ranks.filter((r) => r > 0 && r <= s.k).length;
    $('[data-slot=strip]').innerHTML = `
      <div class="w07-strip__head"><b>골든셋 12문항 hit@${s.k} = ${hits}/12</b>
        <span class="w07-muted">근거 청크의 검색 순위 · 누르면 그 질문으로 바뀐다</span></div>
      <div class="w07-strip__chips">${golden.items
        .map((it, i) => {
          const r = index.ranks[i];
          const ok = r > 0 && r <= s.k;
          const label = r > 0 ? `${r}위` : '잘림';
          return `<button type="button" class="w07-qchip ${ok ? 'is-hit' : 'is-miss'}${s.q === it.id ? ' is-current' : ''}" data-pick="${it.id}"
            aria-label="${it.id} 근거 ${label}, ${ok ? 'hit' : 'miss'}" aria-pressed="${s.q === it.id}">${it.id} <span>${ok ? '✓' : '✗'} ${label}</span></button>`;
        })
        .join('')}</div>`;
  }

  // ---------- events ----------
  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  let customTimer = null;

  on('[data-in=q]', 'change', (e) => {
    s.q = e.target.value;
    $('[data-slot=custom-wrap]').hidden = s.q !== CUSTOM;
    if (s.q === CUSTOM) {
      if (!s.custom) s.custom = '실습실은 몇 시까지 여나?';
      $('[data-in=custom]').value = s.custom;
    }
    runQuery();
  });
  on('[data-in=custom]', 'input', (e) => {
    s.custom = e.target.value;
    clearTimeout(customTimer);
    customTimer = setTimeout(() => runQuery(), 450);
  });
  on('[data-in=size]', 'input', (e) => {
    s.size = Number(e.target.value);
    syncLabels();
  });
  on('[data-in=overlap]', 'input', (e) => {
    s.overlap = Number(e.target.value);
    syncLabels();
  });
  // re-indexing costs an embedding pass: run it when the slider is released
  on('[data-in=size]', 'change', () => runIndex());
  on('[data-in=overlap]', 'change', () => runIndex());
  on('[data-in=k]', 'input', (e) => {
    s.k = Number(e.target.value);
    syncLabels();
    runQuery();
  });
  on('[data-in=model]', 'change', (e) => {
    s.model = e.target.value;
    runIndex();
  });
  on('[data-slot=presets]', 'click', (e) => {
    const p = PRESETS[e.target.closest('[data-preset]')?.dataset.preset];
    if (!p) return;
    const reindex = s.size !== 200 || s.overlap !== 0;
    Object.assign(s, { q: p.q, k: p.k, size: 200, overlap: 0 });
    syncInputs();
    if (reindex) runIndex();
    else runQuery();
  });
  on('[data-act=rerun]', 'click', () => runIndex({ force: true }));
  on('[data-slot=strip]', 'click', (e) => {
    const id = e.target.closest('[data-pick]')?.dataset.pick;
    if (!id) return;
    s.q = id;
    syncInputs();
    runQuery();
  });
  on('[data-in=provider]', 'change', (e) => {
    s.provider = e.target.value;
    s.llmModel = PROVIDERS[s.provider].defaultModel;
    syncInputs();
    if (last) runGenerate(runSeq);
  });
  on('[data-in=llmModel]', 'change', (e) => {
    s.llmModel = e.target.value;
    if (last) runGenerate(runSeq);
  });
  on('[data-act=save-key]', 'click', () => {
    const input = $('[data-in=key]');
    const value = input.value;
    input.value = '';
    if (!value.trim()) {
      $('[data-slot=keystate]').textContent = '키가 비어 있다. 키를 붙여 넣고 저장한다.';
      return;
    }
    try {
      setKey(s.provider, value);
    } catch {
      $('[data-slot=keystate]').textContent = '이 브라우저에서는 sessionStorage를 쓸 수 없어 키를 저장하지 못했다.';
      return;
    }
    syncKeyState();
    if (last) runGenerate(runSeq);
  });
  on('[data-act=clear-key]', 'click', () => {
    try {
      clearKey(s.provider);
    } catch {
      /* storage unavailable: nothing stored */
    }
    syncKeyState();
    if (last) {
      abortGen();
      runGenerate(runSeq);
    }
  });
  ctrl.signal.addEventListener('abort', () => clearTimeout(customTimer));

  // ---------- start ----------
  syncInputs();
  $('[data-slot=status]').hidden = true;
  $('[data-slot=body]').hidden = false;
  registerOutput(outputId, {
    title: options.outputTitle ?? '최소 RAG 파이프라인',
    node: out,
    inlineHost: $('[data-slot=out-inline]'),
  });
  runIndex();
}

export function unmount(el) {
  const st = state.get(el);
  if (st) {
    st.ctrl.abort();
    st.genCtrl?.abort();
    for (const t of st.timers) clearTimeout(t);
    unregisterOutput(st.outputId);
  }
  state.delete(el);
  el.replaceChildren();
}

// ---------- pure helpers (exported for tests) ----------

/** 1-based rank of the first chunk that fully contains the evidence span; 0 if none does. */
export function evidenceRank(ranked, ev) {
  if (!ev) return 0;
  const i = ranked.findIndex((r) => r.meta.doc === ev.doc && r.meta.start <= ev.start && r.meta.end >= ev.end);
  return i + 1;
}

/** Compact one-line JSON-ish preview of an artifact with long text shortened. */
export function shape(obj, max = 48) {
  const parts = Object.entries(obj).map(([k, v]) => {
    if (typeof v === 'string') {
      const flat = v.replace(/\n/g, '⏎');
      const short = flat.length > max ? `${flat.slice(0, max)}…(${v.length}자)` : flat;
      return `${k}: "${short}"`;
    }
    return `${k}: ${v}`;
  });
  return `{ ${parts.join(', ')} }`;
}

function hitVerdict(item, rank, top, ev) {
  if (!item) {
    return `<div class="callout callout--more"><span class="callout__title">hit 판정 없음</span>
      <p>골든셋 질문이 아니라 정답 근거를 모른다. 아래 청크만 보고 답할 수 있는지 직접 판단한다.</p></div>`;
  }
  const inTop = top.some((r) => r.meta.doc === ev?.doc && r.meta.start <= ev.start && r.meta.end >= ev.end);
  const partial = !inTop && top.some((r) => ev && r.meta.doc === ev.doc && r.meta.start < ev.end && r.meta.end > ev.start);
  const where = rank > 0 ? `근거 청크는 전체 ${rank}위` : '근거 문장이 청킹 경계에서 잘려 온전한 청크가 없다';
  if (inTop) {
    const broken = item.note ? `<p>단, 근거 문서가 깨진 문서다 (${escapeHtml(item.note)}). 검색은 성공해도 LLM이 받는 글 자체가 틀렸을 수 있다.</p>` : '';
    return `<div class="callout ${item.note ? 'callout--danger' : 'callout--ok'}"><span class="callout__title">hit — ${where}, top-${top.length} 안</span>${broken}</div>`;
  }
  return `<div class="callout callout--danger"><span class="callout__title">miss — ${where}${rank > 0 ? `, top-${top.length} 밖` : ''}</span>
    <p>${partial ? '근거의 일부만 담긴 청크가 들어왔다. ' : ''}LLM은 아래 청크만 받는다. 정답 근거가 없으니 생성이 아무리 좋아도 맞힐 수 없다.</p></div>`;
}

function resultHtml(r, i, ev) {
  const c = r.meta;
  const full = ev && c.doc === ev.doc && c.start <= ev.start && c.end >= ev.end;
  const part = !full && ev && c.doc === ev.doc && c.start < ev.end && c.end > ev.start;
  let body = escapeHtml(c.text.trim());
  if (full || part) {
    const a = Math.max(ev.start, c.start) - c.start;
    const b = Math.min(ev.end, c.end) - c.start;
    body = `${escapeHtml(c.text.slice(0, a).trimStart())}<mark>${escapeHtml(c.text.slice(a, b))}</mark>${escapeHtml(c.text.slice(b).trimEnd())}`;
  }
  return `<li class="${full ? 'is-evidence' : ''}">
    <div class="w07-res__meta"><b>[${i + 1}]</b><span>${r.score.toFixed(3)}</span><span>${escapeHtml(c.title)}</span><span>${escapeHtml(c.id)}</span>
    ${full ? '<span class="chip ok">근거</span>' : part ? '<span class="chip warn">근거 일부</span>' : ''}</div>
    <div class="w07-res__text">${body}</div></li>`;
}

function citeHtml(text) {
  return escapeHtml(text).replace(/\[(\d+)\]/g, '<span class="w07-cite">[$1]</span>');
}

function fmtMs(ms) {
  return ms < 10 ? `${ms.toFixed(1)} ms` : `${Math.round(ms).toLocaleString()} ms`;
}

async function fetchJson(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${res.status} ${url.pathname}`);
  return res.json();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
