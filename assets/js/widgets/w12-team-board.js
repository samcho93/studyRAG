// w12 팀별 결과 업로드 (팀 점수판)
// One concept: every team's system is scored on the SAME golden set with the
// SAME metrics, so results become comparable. Teams export a results JSON
// (schema below); files are read locally in the browser (never uploaded),
// validated field by field, scored with core/metrics.js and shown as a
// sortable scoreboard + per-question hit heatmap.
//
// Results JSON schema (raglab team result):
// {
//   "team": "팀 이름",                        // non-empty string
//   "members": ["이름1", "이름2", "이름3"],     // 1+ non-empty strings (3~4 recommended)
//   "system": {
//     "chunking": "recursive 200 / overlap 20", // string (or object)
//     "embedding": "Xenova/multilingual-e5-small",
//     "retriever": "hybrid (BM25 + vector, RRF k=60)",
//     "k": 5,                                   // integer 1..50
//     "reranker": null,                         // string or null
//     "llm": "gpt-4.1-mini"                     // string or null
//   },
//   "golden": [                                 // one entry per golden question
//     { "id": "q01", "question": "…",           // id must exist in golden.json
//       "retrieved": ["facility-rules#1", …],   // ranked doc or chunk ids ("docId#n")
//       "answer": "…", "citations": ["facility-rules#1"] }
//   ],
//   "notes": "선택: 자유 메모"
// }

import { evaluate } from '../core/metrics.js';
import { registerOutput, unregisterOutput } from '../site/result.js';

const DATA = new URL('../../data/', import.meta.url);
const SAMPLES = ['sample-team-a.json', 'sample-team-b.json', 'sample-team-c.json'];
const MAX_BYTES = 2 * 1024 * 1024;
const K_CHOICES = [1, 3, 5, 10];

export const METRICS = [
  { key: 'recall', label: 'Recall@k', hint: '상위 k개 안에 근거 문서가 들어온 비율' },
  { key: 'mrr', label: 'MRR@k', hint: '첫 근거 문서 순위의 역수 평균 (k 밖이면 0)' },
  { key: 'ndcg', label: 'nDCG@k', hint: '근거가 위에 있을수록 높은 점수' },
  { key: 'cite', label: '인용률', hint: '인용을 하나 이상 단 답변의 비율' },
  { key: 'citeOk', label: '근거 인용률', hint: '골든셋 근거 문서를 인용한 답변의 비율' },
];

// ---------- pure logic (exported for tests and reuse) ----------

const typeName = (v) => (v === null ? 'null' : Array.isArray(v) ? 'array' : typeof v);
const KO_TYPE = { null: 'null', array: '배열', object: '객체', string: '문자열', number: '숫자', boolean: '참/거짓', undefined: '없음' };
const got = (v) => `지금: ${KO_TYPE[typeName(v)] ?? typeName(v)}`;

/** "rag-hybrid#2" → "rag-hybrid" (chunk ids are scored at document level). */
export const docOf = (id) => String(id).split('#')[0];

/**
 * Map a ranked list of doc/chunk ids to doc ids. A document that already
 * appeared keeps its first rank; later duplicates become null so they still
 * take a rank position but can never be counted twice.
 */
export function toDocRanking(ids) {
  const seen = new Set();
  return ids.map((id) => {
    const d = docOf(id);
    if (seen.has(d)) return null;
    seen.add(d);
    return d;
  });
}

/**
 * Validate a parsed results object against the schema.
 * @param {unknown} data
 * @param {{ items: { id: string, question: string }[] }} golden
 * @param {Set<string>} [docIds] known corpus document ids (optional, for warnings)
 * @returns {{ ok: boolean, errors: {path: string, msg: string}[], warnings: {path: string, msg: string}[] }}
 */
export function validateResult(data, golden, docIds) {
  const errors = [];
  const warnings = [];
  const err = (path, msg) => errors.push({ path, msg });
  const warn = (path, msg) => warnings.push({ path, msg });
  const isStr = (v) => typeof v === 'string' && v.trim() !== '';

  if (typeName(data) !== 'object') {
    err('(최상위)', `{ … } 객체여야 한다 (${got(data)})`);
    return { ok: false, errors, warnings };
  }

  if (!isStr(data.team)) err('team', `비어 있지 않은 문자열이어야 한다 (${typeof data.team === 'string' ? '지금: 빈 문자열' : got(data.team)})`);

  if (!Array.isArray(data.members)) {
    err('members', `문자열 배열이어야 한다. 예: ["김OO", "이OO"] (${got(data.members)})`);
  } else if (data.members.length === 0) {
    err('members', '팀원이 한 명 이상 있어야 한다');
  } else {
    data.members.forEach((m, i) => { if (!isStr(m)) err(`members[${i}]`, `이름 문자열이어야 한다 (${got(m)})`); });
    if (data.members.length < 3 || data.members.length > 4) warn('members', `팀은 3~4명이 기준이다 (지금 ${data.members.length}명)`);
  }

  const sys = data.system;
  if (typeName(sys) !== 'object') {
    err('system', `시스템 구성 객체여야 한다 (${got(sys)})`);
  } else {
    if (!isStr(sys.chunking) && typeName(sys.chunking) !== 'object') err('system.chunking', `청킹 설정을 문자열로 적는다. 예: "recursive 200 / overlap 20" (${got(sys.chunking)})`);
    for (const key of ['embedding', 'retriever']) {
      if (!isStr(sys[key])) err(`system.${key}`, `비어 있지 않은 문자열이어야 한다 (${got(sys[key])})`);
    }
    if (!Number.isInteger(sys.k)) {
      err('system.k', `정수여야 한다${typeof sys.k === 'string' ? ` — 따옴표를 뺀다: "k": ${sys.k.trim() || 5}` : ''} (${got(sys.k)})`);
    } else if (sys.k < 1 || sys.k > 50) {
      err('system.k', `1~50 사이여야 한다 (지금: ${sys.k})`);
    }
    for (const key of ['reranker', 'llm']) {
      if (!(key in sys)) warn(`system.${key}`, '항목이 없다. 쓰지 않았다면 null로 적는다');
      else if (sys[key] !== null && !isStr(sys[key])) err(`system.${key}`, `문자열 또는 null이어야 한다 (${got(sys[key])})`);
    }
  }

  const known = new Map(golden.items.map((it) => [it.id, it.question]));
  if (!Array.isArray(data.golden)) {
    err('golden', `문항 배열이어야 한다 (${got(data.golden)})`);
  } else if (data.golden.length === 0) {
    err('golden', '문항이 하나도 없다');
  } else {
    const seen = new Set();
    data.golden.forEach((q, i) => {
      const p = `golden[${i}]`;
      if (typeName(q) !== 'object') { err(p, `문항 객체여야 한다 (${got(q)})`); return; }
      const tag = isStr(q.id) ? `${p} (${q.id})` : p;
      if (!isStr(q.id)) err(`${p}.id`, `문항 ID 문자열이어야 한다 (${got(q.id)})`);
      else if (!known.has(q.id)) err(`${p}.id`, `골든셋에 없는 ID "${q.id}"다 (q01~q${String(known.size).padStart(2, '0')} 중 하나)`);
      else if (seen.has(q.id)) err(`${p}.id`, `"${q.id}"가 두 번 나온다. 문항마다 한 번만 적는다`);
      else seen.add(q.id);
      if (isStr(q.id) && known.has(q.id) && q.question !== undefined && q.question !== known.get(q.id)) {
        warn(`${tag}.question`, '골든셋 질문과 문구가 다르다. 채점은 id 기준으로 한다');
      }

      if (!Array.isArray(q.retrieved)) {
        err(`${tag}.retrieved`, `검색 결과 ID 배열이어야 한다. 예: ["rag-hybrid#1"] (${got(q.retrieved)})`);
      } else {
        q.retrieved.forEach((r, j) => { if (!isStr(r)) err(`${tag}.retrieved[${j}]`, `문서 또는 청크 ID 문자열이어야 한다 (${got(r)})`); });
        if (q.retrieved.length === 0) warn(`${tag}.retrieved`, '비어 있다. 이 문항은 0점이다');
        if (docIds) {
          const unknown = [...new Set(q.retrieved.filter((r) => isStr(r) && !docIds.has(docOf(r))).map(docOf))];
          if (unknown.length) warn(`${tag}.retrieved`, `코퍼스에 없는 문서 ID: ${unknown.join(', ')}`);
        }
      }
      if (typeof q.answer !== 'string') err(`${tag}.answer`, `답변 문자열이어야 한다 (${got(q.answer)})`);
      if (!Array.isArray(q.citations)) {
        err(`${tag}.citations`, `인용 ID 배열이어야 한다. 인용이 없으면 [] (${got(q.citations)})`);
      } else {
        q.citations.forEach((c, j) => { if (!isStr(c)) err(`${tag}.citations[${j}]`, `ID 문자열이어야 한다 (${got(c)})`); });
        if (Array.isArray(q.retrieved)) {
          const pool = new Set(q.retrieved.filter(isStr).map(docOf));
          const outside = q.citations.filter((c) => isStr(c) && !pool.has(docOf(c)));
          if (outside.length) warn(`${tag}.citations`, `검색되지 않은 문서를 인용했다: ${outside.join(', ')}`);
        }
      }
    });
    const missing = [...known.keys()].filter((id) => !seen.has(id));
    if (missing.length && errors.length === 0) warn('golden', `${missing.length}문항 누락(${missing.join(', ')}) — 0점으로 계산한다`);
  }

  if (data.notes !== undefined && typeof data.notes !== 'string') err('notes', `문자열이어야 한다 (${got(data.notes)})`);
  return { ok: errors.length === 0, errors, warnings };
}

/**
 * Score one (valid) team result against the golden set at cutoff k.
 * Missing questions count as empty retrieval / no citation.
 */
export function scoreTeam(result, golden, k) {
  const byId = new Map(result.golden.map((q) => [q.id, q]));
  const perQuestion = golden.items.map((item) => {
    const relevant = [...new Set(item.evidence.map((e) => e.doc))];
    const sub = byId.get(item.id);
    const ranking = toDocRanking(sub?.retrieved ?? []);
    const top = ranking.slice(0, k);
    const idx = top.findIndex((d) => d !== null && relevant.includes(d));
    const cites = (sub?.citations ?? []).map(docOf);
    return {
      id: item.id,
      question: item.question,
      relevant,
      retrieved: top,
      rank: idx === -1 ? null : idx + 1,
      answer: sub?.answer ?? '',
      cited: cites.length > 0,
      citedOk: cites.some((d) => relevant.includes(d)),
      missing: !sub,
    };
  });
  const m = evaluate(perQuestion.map((q) => ({ retrieved: q.retrieved, relevant: q.relevant })), k);
  const n = perQuestion.length || 1;
  return {
    recall: m.recall,
    mrr: m.mrr,
    ndcg: m.ndcg,
    hit: m.hit,
    cite: perQuestion.filter((q) => q.cited).length / n,
    citeOk: perQuestion.filter((q) => q.citedOk).length / n,
    perQuestion,
  };
}

/** A fill-in template built from the golden set (used by the download button). */
export function makeTemplate(golden) {
  return {
    team: '팀 이름',
    members: ['이름1', '이름2', '이름3'],
    system: {
      chunking: 'recursive 200 / overlap 20',
      embedding: 'Xenova/multilingual-e5-small',
      retriever: 'hybrid (BM25 + vector, RRF k=60)',
      k: 5,
      reranker: null,
      llm: null,
    },
    golden: golden.items.map((it) => ({ id: it.id, question: it.question, retrieved: [], answer: '', citations: [] })),
    notes: '',
  };
}

/** Parse JSON text with a friendly, positioned error message. */
export function parseJson(text) {
  try {
    return { value: JSON.parse(text) };
  } catch (e) {
    const lc = /line (\d+) column (\d+)/.exec(e.message);
    const pos = Number(/position (\d+)/.exec(e.message)?.[1]);
    let where = '';
    if (lc) where = ` (${lc[1]}행 ${lc[2]}열 근처)`;
    else if (Number.isFinite(pos)) {
      const before = text.slice(0, pos);
      const line = before.split('\n').length;
      const col = pos - before.lastIndexOf('\n');
      where = ` (${line}행 ${col}열 근처)`;
    }
    return { error: `JSON 문법 오류${where}: 쉼표·따옴표·괄호를 확인한다. 마지막 항목 뒤 쉼표, 작은따옴표, 주석은 JSON에서 허용되지 않는다.` };
  }
}

// ---------- widget ----------

const BROKEN_SYNTAX = `{
  "team": "예시 팀 Z · 문법 오류",
  "members": ["예시 학생 Z1", "예시 학생 Z2", "예시 학생 Z3"],
  'system': { "k": 5, },
}`;

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget" data-w12>
    <h3 class="widget__title">팀별 결과 업로드 · 점수판</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> 골든셋과 예시 팀 결과를 불러오는 중…
    </div>
    <div data-slot="body" hidden>
      <div class="w12-drop" data-slot="drop">
        <p><b>결과 JSON 파일을 여기에 끌어 놓거나</b></p>
        <div class="btn-row">
          <button type="button" class="btn primary" data-act="pick">📂 파일 선택 (여러 개 가능)</button>
          <button type="button" class="btn" data-act="template">⬇ 샘플 JSON 내려받기</button>
        </div>
        <input type="file" data-in="files" accept=".json,application/json" multiple hidden>
        <p class="w12-privacy">🔒 파일은 이 브라우저 안에서만 읽는다. 서버가 없으므로 어디에도 전송·저장되지 않고, 탭을 닫으면 사라진다.</p>
      </div>
      <div class="widget__controls">
        <label class="field">
          <span class="field__label">채점 기준 k (모든 팀 동일 적용)</span>
          <select data-in="k"></select>
        </label>
        <div class="field">
          <span class="field__label">실패 예시</span>
          <div class="btn-row">
            <button type="button" class="btn small ghost" data-act="broken">스키마 오류 파일</button>
            <button type="button" class="btn small ghost" data-act="syntax">JSON 문법 오류</button>
          </div>
        </div>
        <div class="field">
          <span class="field__label">목록</span>
          <div class="btn-row">
            <button type="button" class="btn small ghost" data-act="reset">예시 팀 다시 불러오기</button>
            <button type="button" class="btn small ghost" data-act="clear">모두 비우기</button>
          </div>
        </div>
      </div>
      <h4 class="w12-h">불러온 파일 <span data-out="count"></span></h4>
      <ul class="w12-files" data-slot="files" aria-live="polite"></ul>
      <details>
        <summary>결과 JSON 스키마 보기</summary>
        <pre class="w12-schema"><code>{
  "team": "팀 이름",                 // 필수 · 문자열
  "members": ["이름1", "이름2"],     // 필수 · 문자열 배열 (3~4명 권장)
  "system": {                        // 필수
    "chunking": "recursive 200 / overlap 20",
    "embedding": "Xenova/multilingual-e5-small",
    "retriever": "hybrid (BM25 + vector, RRF k=60)",
    "k": 5,                          // 정수 1~50 (따옴표 없이)
    "reranker": null,                // 문자열 또는 null
    "llm": "gpt-4.1-mini"            // 문자열 또는 null
  },
  "golden": [                        // 필수 · 골든셋 문항마다 하나
    { "id": "q01",                   // q01~q12
      "question": "…",               // 선택 (확인용)
      "retrieved": ["facility-rules#1", "dept-overview#2"],  // 순위순 문서/청크 ID
      "answer": "최대 12시간이다.",
      "citations": ["facility-rules#1"] }                    // 없으면 []
  ],
  "notes": "선택 · 자유 메모"
}</code></pre>
        <p class="w12-note">청크 ID는 <code>문서ID#번호</code>로 적는다. 채점은 문서 단위로 하며, 같은 문서의 두 번째 청크부터는 순위 자리만 차지하고 점수에는 한 번만 들어간다. 주석(<code>//</code>)은 설명용이므로 실제 파일에는 넣지 않는다.</p>
      </details>
      <div data-slot="out-inline"></div>
    </div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w12-out">
    <div class="w12-out__head">
      <b>RAG Lab 12주차 · 팀 점수판</b>
      <span data-out="caption"></span>
    </div>
    <div class="btn-row w12-out__tools">
      <button type="button" class="btn small ghost" data-act="print">🖨 인쇄</button>
      <button type="button" class="btn small ghost" data-act="full">⛶ 크게 보기</button>
    </div>
    <div data-slot="empty" class="callout" hidden><span class="callout__title">채점할 팀이 없다</span>
      검증을 통과한 파일이 없다. “예시 팀 다시 불러오기”를 누르거나 결과 JSON을 올린다.</div>
    <div data-slot="board-wrap">
      <h4>점수판 <small>(열 제목을 눌러 정렬)</small></h4>
      <div class="w12-scroll"><table class="w12-board" data-slot="board"></table></div>
      <h4>문항별 적중 지도 <small>(숫자 = 근거 문서의 순위, ✗ = 상위 k 밖)</small></h4>
      <div class="w12-scroll"><table class="w12-heat" data-slot="heat"></table></div>
      <div class="w12-legend" aria-hidden="true">
        <span><i class="w12-cell r1"></i>1위</span><span><i class="w12-cell r3"></i>2~3위</span>
        <span><i class="w12-cell rk"></i>4위 이하</span><span><i class="w12-cell miss"></i>실패</span>
        <span>칸 오른쪽 위 · = 근거 문서를 인용하지 않은 답변</span>
      </div>
      <div data-slot="allfail"></div>
      <h4>시스템 구성</h4>
      <div class="w12-scroll"><table class="w12-sys" data-slot="sys"></table></div>
    </div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ k?: number, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w12:${++seq}`;
  const st = { ctrl, outputId, restorePrint: null };
  state.set(el, st);

  let golden;
  let docIds;
  const entries = []; // { name, source: 'sample'|'upload', data?, errors, warnings }
  try {
    const [g, corpus, ...samples] = await Promise.all([
      fetchJson(new URL('golden/golden.json', DATA), ctrl.signal),
      fetchJson(new URL('corpus/corpus.json', DATA), ctrl.signal),
      ...SAMPLES.map((f) => fetchText(new URL(`w12/${f}`, DATA), ctrl.signal)),
    ]);
    golden = g;
    docIds = new Set(corpus.documents.map((d) => d.id));
    samples.forEach((text, i) => entries.push(makeEntry(SAMPLES[i], text, 'sample')));
  } catch (err) {
    if (err.name === 'AbortError') return;
    $('[data-slot=status]').innerHTML = `<div class="widget__error" role="alert">
      골든셋이나 예시 파일을 불러오지 못했다 (${escapeHtml(err.message)}).<br>
      파일을 더블클릭해 <code>file://</code>로 열었다면 ES Module과 fetch가 동작하지 않는다.
      저장소 폴더에서 <code>python -m http.server</code>를 실행하고 <code>http://localhost:8000</code>으로 연다.</div>`;
    return;
  }

  const s = { k: K_CHOICES.includes(options.k) ? options.k : 5, sort: 'ndcg', dir: -1 };
  $('[data-in=k]').innerHTML = K_CHOICES.map((k) => `<option value="${k}">k = ${k}</option>`).join('');
  $('[data-in=k]').value = String(s.k);

  function makeEntry(name, text, source) {
    const parsed = parseJson(text);
    if (parsed.error) return { name, source, errors: [{ path: '(파일)', msg: parsed.error }], warnings: [] };
    const v = validateResult(parsed.value, golden, docIds);
    return { name, source, data: v.ok ? parsed.value : null, errors: v.errors, warnings: v.warnings };
  }

  function addEntry(entry) {
    // same team name (or same file name for invalid files) replaces the older entry
    const key = entry.data?.team ?? `file:${entry.name}`;
    const i = entries.findIndex((e) => (e.data?.team ?? `file:${e.name}`) === key);
    if (i >= 0) {
      entry.replaced = true;
      entries.splice(i, 1, entry);
    } else entries.push(entry);
  }

  async function addFiles(fileList) {
    for (const file of fileList) {
      if (file.size > MAX_BYTES) {
        addEntry({ name: file.name, source: 'upload', errors: [{ path: '(파일)', msg: `파일이 너무 크다 (${Math.round(file.size / 1024)}KB). 2MB 이하 결과 JSON만 받는다` }], warnings: [] });
        continue;
      }
      if (!/\.json$/i.test(file.name) && file.type !== 'application/json') {
        addEntry({ name: file.name, source: 'upload', errors: [{ path: '(파일)', msg: '.json 파일이 아니다' }], warnings: [] });
        continue;
      }
      try {
        addEntry(makeEntry(file.name, await file.text(), 'upload'));
      } catch (e) {
        addEntry({ name: file.name, source: 'upload', errors: [{ path: '(파일)', msg: `읽을 수 없다 (${e.message})` }], warnings: [] });
      }
    }
    render();
  }

  async function loadSamples() {
    const texts = await Promise.all(SAMPLES.map((f) => fetchText(new URL(`w12/${f}`, DATA), ctrl.signal)));
    texts.forEach((t, i) => addEntry(makeEntry(SAMPLES[i], t, 'sample')));
  }

  function renderFiles() {
    const valid = entries.filter((e) => e.data).length;
    $('[data-out=count]').textContent = `(${entries.length}개 · 채점 ${valid}팀)`;
    $('[data-slot=files]').innerHTML = entries.length
      ? entries.map((e, i) => fileItemHtml(e, i)).join('')
      : '<li class="w12-file"><span class="w12-muted">아직 불러온 파일이 없다.</span></li>';
  }

  function render() {
    renderFiles();
    const teams = entries.filter((e) => e.data).map((e) => ({ entry: e, score: scoreTeam(e.data, golden, s.k) }));
    $('[data-slot=empty]').hidden = teams.length > 0;
    $('[data-slot=board-wrap]').hidden = teams.length === 0;
    $('[data-out=caption]').textContent = `골든셋 ${golden.items.length}문항 · k = ${s.k} · ${teams.length}팀 · ${new Date().toLocaleDateString('ko-KR')}`;
    if (!teams.length) return;

    const sorted = [...teams].sort((a, b) => {
      if (s.sort === 'team') return s.dir * a.entry.data.team.localeCompare(b.entry.data.team, 'ko');
      return s.dir * (a.score[s.sort] - b.score[s.sort]) || a.entry.data.team.localeCompare(b.entry.data.team, 'ko');
    });
    const best = Object.fromEntries(METRICS.map((m) => [m.key, Math.max(...teams.map((t) => t.score[m.key]))]));
    $('[data-slot=board]').innerHTML = boardHtml(sorted, best, s);
    $('[data-slot=heat]').innerHTML = heatHtml(sorted, golden);
    $('[data-slot=sys]').innerHTML = sysHtml(sorted);

    const allFail = golden.items.filter((_, qi) => sorted.every((t) => t.score.perQuestion[qi].rank === null));
    $('[data-slot=allfail]').innerHTML = allFail.length && sorted.length > 1
      ? `<div class="callout callout--danger"><span class="callout__title">모든 팀이 놓친 문항 ${allFail.length}개</span>
          ${allFail.map((it) => `<b>${it.id}</b> ${escapeHtml(it.question)}${it.note ? ` <span class="w12-muted">— ${escapeHtml(it.note)}</span>` : ''}`).join('<br>')}
          <br>한 팀만의 실수가 아니라 데이터(파싱)나 골든셋 쪽 문제일 가능성이 크다. 발표의 “실패 사례”로 다룬다.</div>`
      : '';
  }

  // ----- events -----
  const on = (target, type, fn) => target.addEventListener(type, fn, { signal: ctrl.signal });
  const fileInput = $('[data-in=files]');
  const drop = $('[data-slot=drop]');

  on(root, 'click', async (e) => {
    const act = e.target.closest('[data-act]')?.dataset.act;
    const rm = e.target.closest('[data-remove]')?.dataset.remove;
    if (rm !== undefined) {
      entries.splice(Number(rm), 1);
      render();
      return;
    }
    if (act === 'pick') fileInput.click();
    else if (act === 'template') downloadJson('team-result-template.json', makeTemplate(golden));
    else if (act === 'broken') {
      try {
        addEntry(makeEntry('sample-broken.json', await fetchText(new URL('w12/sample-broken.json', DATA), ctrl.signal), 'sample'));
      } catch (err) {
        if (err.name === 'AbortError') return;
        addEntry({ name: 'sample-broken.json', source: 'sample', errors: [{ path: '(파일)', msg: `불러오지 못했다 (${err.message})` }], warnings: [] });
      }
      render();
    } else if (act === 'syntax') {
      addEntry(makeEntry('syntax-error-example.json', BROKEN_SYNTAX, 'sample'));
      render();
    } else if (act === 'reset') {
      try {
        await loadSamples();
      } catch (err) {
        if (err.name === 'AbortError') return;
      }
      render();
    } else if (act === 'clear') {
      entries.length = 0;
      render();
    }
  });
  on(fileInput, 'change', () => {
    addFiles([...fileInput.files]);
    fileInput.value = '';
  });
  on(drop, 'dragover', (e) => {
    e.preventDefault();
    drop.classList.add('is-over');
  });
  on(drop, 'dragleave', () => drop.classList.remove('is-over'));
  on(drop, 'drop', (e) => {
    e.preventDefault();
    drop.classList.remove('is-over');
    if (e.dataTransfer?.files?.length) addFiles([...e.dataTransfer.files]);
  });
  on($('[data-in=k]'), 'change', (e) => {
    s.k = Number(e.target.value);
    render();
  });
  on(out, 'click', (e) => {
    const sortKey = e.target.closest('[data-sort]')?.dataset.sort;
    if (sortKey) {
      s.dir = s.sort === sortKey ? -s.dir : sortKey === 'team' ? 1 : -1;
      s.sort = sortKey;
      render();
      out.querySelector(`[data-sort="${sortKey}"]`)?.focus();
      return;
    }
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (act === 'print') printBoard();
    else if (act === 'full') {
      if (document.fullscreenElement) document.exitFullscreen?.();
      else out.requestFullscreen?.().catch(() => {});
    }
  });

  // Print only the scoreboard: move it to <body> while printing, then put it back.
  function printBoard() {
    if (st.restorePrint) return;
    const holder = document.createElement('div');
    holder.className = 'w12-print-root';
    holder.append(out);
    document.body.append(holder);
    document.body.classList.add('w12-printing');
    st.restorePrint = () => {
      document.body.classList.remove('w12-printing');
      holder.remove();
      st.restorePrint = null;
      registerOutput(outputId, { title: options.outputTitle ?? '팀 점수판', node: out, inlineHost: $('[data-slot=out-inline]') });
    };
    window.addEventListener('afterprint', () => st.restorePrint?.(), { once: true, signal: ctrl.signal });
    window.print();
    // print() blocks in most browsers; restore right away when it has returned
    setTimeout(() => st.restorePrint?.(), 0);
  }

  render();
  $('[data-slot=status]').hidden = true;
  $('[data-slot=body]').hidden = false;
  registerOutput(outputId, {
    title: options.outputTitle ?? '팀 점수판',
    node: out,
    inlineHost: $('[data-slot=out-inline]'),
  });
}

export function unmount(el) {
  const st = state.get(el);
  st?.restorePrint?.();
  st?.ctrl.abort();
  if (st) unregisterOutput(st.outputId);
  state.delete(el);
  el.replaceChildren();
}

// ---------- rendering helpers ----------

const pct = (v) => `${(v * 100).toFixed(0)}%`;
const num = (v) => v.toFixed(3);

function fileItemHtml(e, i) {
  const chip = e.source === 'sample' ? '<span class="chip warn">예시</span>' : '<span class="chip">업로드</span>';
  const status = e.data
    ? `<span class="chip ok">✓ 통과</span>`
    : `<span class="chip w12-bad">✗ 오류 ${e.errors.length}개</span>`;
  const title = e.data ? escapeHtml(e.data.team) : escapeHtml(e.name);
  const list = (items, cls) => items.length
    ? `<ul class="w12-issues ${cls}">${items.slice(0, 10).map((x) => `<li><code>${escapeHtml(x.path)}</code> ${escapeHtml(x.msg)}</li>`).join('')}${items.length > 10 ? `<li>… 외 ${items.length - 10}개</li>` : ''}</ul>`
    : '';
  return `<li class="w12-file ${e.data ? '' : 'is-bad'}">
    <div class="w12-file__row">${chip}${status}<b class="w12-file__name">${title}</b>
      <span class="w12-muted">${escapeHtml(e.name)}${e.replaced ? ' · 같은 팀 교체됨' : ''}</span>
      <button type="button" class="btn small ghost w12-file__rm" data-remove="${i}" aria-label="${title} 목록에서 빼기">✕</button>
    </div>
    ${list(e.errors, 'is-error')}${list(e.warnings, 'is-warn')}
  </li>`;
}

function boardHtml(sorted, best, s) {
  const arrow = (key) => (s.sort === key ? (s.dir === 1 ? ' ▲' : ' ▼') : '');
  const aria = (key) => (s.sort === key ? ` aria-sort="${s.dir === 1 ? 'ascending' : 'descending'}"` : '');
  const head = `<thead><tr><th scope="col">#</th>
    <th scope="col"${aria('team')}><button type="button" class="w12-sort" data-sort="team">팀${arrow('team')}</button></th>
    ${METRICS.map((m) => `<th scope="col"${aria(m.key)}><button type="button" class="w12-sort" data-sort="${m.key}" title="${m.hint}">${m.label}${arrow(m.key)}</button></th>`).join('')}
  </tr></thead>`;
  const body = sorted
    .map((t, i) => `<tr><td>${i + 1}</td><th scope="row">${escapeHtml(t.entry.data.team)}${t.entry.source === 'sample' ? ' <span class="chip warn">예시</span>' : ''}</th>
      ${METRICS.map((m) => {
        const v = t.score[m.key];
        const isBest = sorted.length > 1 && v === best[m.key] && v > 0;
        return `<td class="w12-num${isBest ? ' is-best' : ''}" style="--v: ${v.toFixed(3)}">${m.key.startsWith('cite') ? pct(v) : num(v)}</td>`;
      }).join('')}</tr>`)
    .join('');
  return `${head}<tbody>${body}</tbody>`;
}

function heatHtml(sorted, golden) {
  const head = `<thead><tr><th scope="col">문항</th>${sorted
    .map((t, i) => `<th scope="col" title="${escapeHtml(t.entry.data.team)}">${teamShort(t.entry.data.team, i)}</th>`)
    .join('')}<th scope="col">적중</th></tr></thead>`;
  const body = golden.items
    .map((item, qi) => {
      let hits = 0;
      const cells = sorted
        .map((t) => {
          const q = t.score.perQuestion[qi];
          if (q.rank !== null) hits++;
          const cls = q.rank === null ? 'miss' : q.rank === 1 ? 'r1' : q.rank <= 3 ? 'r3' : 'rk';
          const tip = `${t.entry.data.team} · ${item.id}: ${q.rank === null ? '상위 k 밖' : `${q.rank}위`} · 답변: ${q.answer || '(없음)'}${q.cited ? (q.citedOk ? ' · 근거 인용 ✓' : ' · 엉뚱한 문서 인용') : ' · 인용 없음'}`;
          return `<td class="w12-cell ${cls}" title="${escapeHtml(tip)}">${q.rank ?? '✗'}${q.citedOk ? '' : '<i class="w12-nocite" aria-label="근거 인용 없음">·</i>'}</td>`;
        })
        .join('');
      const all = hits === 0 && sorted.length > 1;
      return `<tr class="${all ? 'is-allfail' : ''}"><th scope="row" title="${escapeHtml(item.question)}">${item.id}<span class="w12-q">${escapeHtml(item.question)}</span></th>${cells}<td class="w12-num">${hits}/${sorted.length}</td></tr>`;
    })
    .join('');
  return `${head}<tbody>${body}</tbody>`;
}

function sysHtml(sorted) {
  const keys = ['chunking', 'embedding', 'retriever', 'k', 'reranker', 'llm'];
  const val = (v) => (v === null || v === undefined ? '—' : typeof v === 'object' ? JSON.stringify(v) : String(v));
  return `<thead><tr><th scope="col">팀</th>${keys.map((k) => `<th scope="col">${k}</th>`).join('')}</tr></thead><tbody>${sorted
    .map((t, i) => `<tr><th scope="row" title="${escapeHtml(t.entry.data.team)}">${teamShort(t.entry.data.team, i)}</th>${keys
      .map((k) => `<td>${escapeHtml(val(t.entry.data.system[k]))}</td>`)
      .join('')}</tr>`)
    .join('')}</tbody>`;
}

/** Short column label: text before " · " (e.g. "예시 팀 A"), capped in length. */
function teamShort(name, i) {
  const base = String(name).split(' · ')[0].trim() || `팀 ${i + 1}`;
  return escapeHtml(base.length > 10 ? `${base.slice(0, 9)}…` : base);
}

function downloadJson(filename, obj) {
  const blob = new Blob([`${JSON.stringify(obj, null, 2)}\n`], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function fetchJson(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${res.status} ${url.pathname}`);
  return res.json();
}

async function fetchText(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${res.status} ${url.pathname}`);
  return res.text();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
