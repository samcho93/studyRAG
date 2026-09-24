// w14 조건 검색 (메타데이터 필터 · 셀프 쿼리)
// One concept: some questions are conditions, not similarity. For each question
// the widget shows side by side
//   (A) pure search top-k over record texts (BM25 at once, vector optional),
//   (B) self-query = parsed filter (JSON, editable, validated) + search inside it,
//   (C) the exact answer set selected by the filter alone,
// with precision/recall of A and B against the exact answer set.
// Parser: transparent rule-based extractor without a key; with a key an LLM
// must emit JSON that is validated against the allowed schema before it runs.

import { BM25 } from '../core/bm25.js';
import { MODELS, loadModelWithUI, embed, embedBatch } from '../core/embed.js';
import { PROVIDERS, setKey, hasKey, clearKey, generate, LLMError } from '../core/llm.js';
import { registerOutput, unregisterOutput } from '../site/result.js';

const DATA = new URL('../../data/', import.meta.url);
const VECTOR_MODEL = MODELS[0].id; // multilingual-e5-small

// Hand-written parser outputs used when no API key is set (never recorded from a real model).
const CANNED_LLM = {
  p1: '{"query": "", "intent": "list", "filter": {"type": "timetable", "day": "화", "room": "305"}}',
  p2: '{"query": "", "intent": "count", "filter": {"type": "timetable", "semester": 2, "room": "305"}}',
  p3: '{"query": "", "intent": "list", "filter": {"type": "course", "professor": "이 교수"}}',
  p4: '{"query": "", "intent": "list", "filter": {"type": "course", "semester": 2, "credits": 3, "status": "개설"}}',
  p5: '```json\n{"query": "", "intent": "list", "filter": {"type": "timetable", "semester": 2, "periodStart": {"lte": 3}}}\n```',
  p6: '{"query": "", "intent": "list", "filter": {"type": "timetable", "semester": 2, "room": "305"}}',
  p7: '{"query": "", "intent": "list", "filter": {"type": "timetable", "day": "목", "room": "305", "professor": "조 교수"}}',
  p8: '{"query": "GPU 서버 한 사람 연속 사용 최대 시간", "intent": "search", "filter": {}}',
};

const RULES = [
  ['R1 학기', '“1학기”, “2학기” → semester = 1 | 2'],
  ['R2 요일', '“화요일” → day = "화", 요일이 여러 개면 {"in": [...]}'],
  ['R3 강의실', '세 자리 숫자 + “호” → room. 처음 나온 번호만 쓴다(“말고” 같은 부정은 모른다)'],
  ['R4 교시', '“1-3교시” → periodStart ≥ 1, periodEnd ≤ 3 · “2교시” → 2교시에 걸친 수업 · “오전” → periodEnd ≤ 3 · “오후” → periodStart ≥ 4'],
  ['R5 담당', '“김/이/박/조/최 교수” → professor, “공동” → professor = "공동"'],
  ['R6 학점', '“3학점” → credits = 3 · “3학점 이상” → credits ≥ 3'],
  ['R7 유형', '학점·과목 → type = "course" · 수업·강의·교시·요일 → "timetable" · 장학 → "scholarship"'],
  ['R8 의도', '“몇 개” → count · “모두·전부·목록·~만” → list · 조건만 있고 남은 검색어가 없으면 list · 그 밖에는 search'],
];

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget" data-w14>
    <h3 class="widget__title">조건 검색 · 셀프 쿼리</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> 구조화 레코드를 불러오는 중…
    </div>
    <div data-slot="body" hidden>
      <div class="btn-row w14-presets" data-slot="presets" role="group" aria-label="예시 질문"></div>
      <label class="field w14-q">
        <span class="field__label">질문 (예시를 고르거나 직접 입력)</span>
        <input type="text" data-in="q" maxlength="120" autocomplete="off">
      </label>
      <div class="widget__controls">
        <label class="field">
          <span class="field__label">파서</span>
          <select data-in="parser">
            <option value="rule">규칙 기반 (키 필요 없음)</option>
            <option value="llm">LLM (JSON 출력 + 검증)</option>
          </select>
        </label>
        <label class="field">
          <span class="field__label">필터 적용 시점</span>
          <select data-in="timing">
            <option value="pre">사전 필터 (필터 → 검색)</option>
            <option value="post">사후 필터 (검색 top-k → 필터)</option>
          </select>
        </label>
        <label class="field">
          <span class="field__label">순수 검색 방식</span>
          <select data-in="method">
            <option value="bm25">BM25 (바로 실행)</option>
            <option value="vector">벡터 · multilingual-e5-small (118MB)</option>
          </select>
        </label>
        <label class="field">
          <span class="field__label">top-k <output data-out="k"></output></span>
          <input type="range" data-in="k" min="1" max="8" step="1">
        </label>
      </div>
      <div data-slot="model" class="w14-model"></div>

      <div class="w14-filter">
        <div class="w14-filter__head">
          <b>셀프 쿼리 결과 (JSON · 직접 고칠 수 있다)</b>
          <span data-slot="source" class="chip"></span>
        </div>
        <div data-slot="llm-raw"></div>
        <label class="field">
          <span class="visually-hidden">셀프 쿼리 JSON</span>
          <textarea data-in="filter" spellcheck="false" rows="7" aria-describedby="w14-filter-check"></textarea>
        </label>
        <div data-slot="check" id="w14-filter-check" aria-live="polite"></div>
        <div class="btn-row">
          <button type="button" class="btn small ghost" data-act="reset">↺ 파서 결과로 되돌리기</button>
          <button type="button" class="btn small ghost" data-act="inject">🛡 허용 안 된 필터 넣어 보기</button>
        </div>
        <div data-slot="trace" class="w14-trace"></div>
      </div>

      <details>
        <summary>규칙 파서의 규칙 8개</summary>
        <ul class="w14-rules" data-slot="rules"></ul>
      </details>
      <details data-slot="llm">
        <summary>LLM 파서 설정 (선택 · 키는 이 탭에만 저장)</summary>
        <div class="widget__controls w14-llm">
          <label class="field">
            <span class="field__label">공급자</span>
            <select data-in="provider"></select>
          </label>
          <label class="field">
            <span class="field__label">모델</span>
            <select data-in="llmModel"></select>
          </label>
          <label class="field w14-wide">
            <span class="field__label">API 키</span>
            <input type="password" data-in="key" autocomplete="off" spellcheck="false" placeholder="키를 붙여 넣고 저장한다">
          </label>
        </div>
        <div class="btn-row">
          <button type="button" class="btn small" data-act="save-key">키 저장 (sessionStorage)</button>
          <button type="button" class="btn small ghost" data-act="clear-key">키 지우기</button>
          <button type="button" class="btn small primary" data-act="llm-run">LLM으로 파싱 실행</button>
        </div>
        <p class="w14-muted" data-slot="keystate" aria-live="polite"></p>
      </details>
      <details>
        <summary>레코드 살펴보기 (<span data-slot="count"></span>개)</summary>
        <p class="w14-muted">“가상” 표시는 실습을 위해 만든 레코드다. 나머지는 공통 문서셋(시간표·교과목 안내·실습실 규정·장학금 안내)에서 옮겼다.</p>
        <div class="w14-scroll" tabindex="0" role="region" aria-label="레코드 표"><table class="w14-table" data-slot="records"></table></div>
      </details>
      <div data-slot="out-inline"></div>
    </div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w14-out">
    <div data-slot="head" class="w14-head"></div>
    <div data-slot="verdict" aria-live="polite"></div>
    <section class="w14-col" data-col="a">
      <h4>A · 순수 검색 <span data-slot="a-sub" class="w14-muted"></span></h4>
      <div class="stat-row" data-slot="a-stats"></div>
      <ol class="w14-list" data-slot="a-list"></ol>
    </section>
    <section class="w14-col" data-col="b">
      <h4>B · 셀프 쿼리 <span data-slot="b-sub" class="w14-muted"></span></h4>
      <div class="stat-row" data-slot="b-stats"></div>
      <ol class="w14-list" data-slot="b-list"></ol>
    </section>
    <section class="w14-col" data-col="c">
      <h4>C · 필터만으로 고른 정확한 답 <span data-slot="c-sub" class="w14-muted"></span></h4>
      <div data-slot="c-note"></div>
      <ol class="w14-list" data-slot="c-list"></ol>
    </section>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ preset?: string, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w14:${++seq}`;
  const st = { ctrl, outputId, llmCtrl: null };
  state.set(el, st);
  // unique id so two mounted copies (page + deck) do not share aria targets
  const checkId = `w14-filter-check-${seq}`;
  $('[data-slot=check]').id = checkId;
  $('[data-in=filter]').setAttribute('aria-describedby', checkId);

  let data;
  try {
    data = await fetchJson(new URL('w14/records.json', DATA), ctrl.signal);
  } catch (err) {
    if (err.name === 'AbortError') return;
    $('[data-slot=status]').innerHTML = `<div class="widget__error" role="alert">
      레코드를 불러오지 못했다 (${escapeHtml(err.message)}).<br>
      파일을 더블클릭해 <code>file://</code>로 열었다면 ES Module과 fetch가 동작하지 않는다.
      저장소 폴더에서 <code>python -m http.server</code>를 실행하고 <code>http://localhost:8000</code>으로 연다.</div>`;
    return;
  }

  const { records, schema, questions } = data;
  const byId = new Map(records.map((r) => [r.id, r]));
  const qById = new Map(questions.map((q) => [q.id, q]));
  const bm25 = new BM25().add(records.map((r) => ({ id: r.id, text: r.text })));

  const firstProvider = Object.keys(PROVIDERS)[0];
  const startPreset = qById.has(options.preset) ? options.preset : questions[0].id;
  const s = {
    preset: startPreset,
    question: qById.get(startPreset).q,
    parser: 'rule',
    timing: 'pre',
    method: 'bm25',
    k: 3,
    provider: firstProvider,
    llmModel: PROVIDERS[firstProvider].defaultModel,
    filterText: '',
    source: '',
    trace: [],
    llmRaw: null, // { text, label }
    live: null, // { question, text } from a real LLM call
  };
  const vec = { ready: false, loading: false, recordVecs: null, cache: new Map() };
  let computeSeq = 0;

  // ---------- static controls ----------
  $('[data-slot=presets]').innerHTML = questions
    .map((q) => `<button type="button" class="btn small" data-preset="${q.id}" aria-pressed="false">${escapeHtml(q.label)}</button>`)
    .join('');
  $('[data-slot=rules]').innerHTML = RULES.map(([r, d]) => `<li><b>${r}</b> ${escapeHtml(d)}</li>`).join('');
  $('[data-in=provider]').innerHTML = Object.entries(PROVIDERS)
    .map(([key, p]) => `<option value="${key}">${escapeHtml(p.label)}</option>`)
    .join('');
  $('[data-slot=count]').textContent = records.length;
  $('[data-slot=records]').innerHTML = recordTable(records);

  function syncInputs() {
    $('[data-in=q]').value = s.question;
    $('[data-in=parser]').value = s.parser;
    $('[data-in=timing]').value = s.timing;
    $('[data-in=method]').value = s.method;
    $('[data-in=k]').value = s.k;
    $('[data-out=k]').textContent = s.k;
    $('[data-in=provider]').value = s.provider;
    $('[data-in=llmModel]').innerHTML = PROVIDERS[s.provider].models.map((m) => `<option value="${m}">${m}</option>`).join('');
    $('[data-in=llmModel]').value = s.llmModel;
    for (const b of root.querySelectorAll('[data-preset]')) b.setAttribute('aria-pressed', String(b.dataset.preset === s.preset));
    syncKeyState();
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
      ? `<span class="chip ok">키 있음 · ${escapeHtml(PROVIDERS[s.provider].label)}</span> “LLM으로 파싱 실행”을 누르면 실제 LLM이 JSON을 만든다.`
      : '<span class="chip">키 없음</span> 예시 질문은 손으로 쓴 예시 응답을 보여 준다. 직접 입력한 질문은 규칙 파서로 대신한다.';
  }

  // ---------- parsing ----------
  function reparse() {
    s.llmRaw = null;
    if (s.parser === 'llm') {
      if (s.live && s.live.question === s.question) {
        s.llmRaw = { text: s.live.text, label: `실제 LLM 응답 · ${s.live.model}` };
        s.source = 'LLM 실제 응답';
      } else if (s.preset && CANNED_LLM[s.preset]) {
        s.llmRaw = { text: CANNED_LLM[s.preset], label: '예시 응답 · 수업용 작성 예시' };
        s.source = 'LLM 예시 응답';
      }
      if (s.llmRaw) {
        s.trace = [];
        try {
          s.filterText = JSON.stringify(extractJson(s.llmRaw.text), null, 2);
        } catch (err) {
          s.filterText = s.llmRaw.text;
        }
        return;
      }
    }
    const p = parseRule(s.question, schema);
    s.trace = p.trace;
    s.filterText = JSON.stringify({ query: p.query, intent: p.intent, filter: p.filter }, null, 2);
    s.source = s.parser === 'llm' ? '규칙 파서로 대신함' : '규칙 파서';
  }

  function renderParserBox() {
    $('[data-in=filter]').value = s.filterText;
    $('[data-slot=source]').textContent = s.source;
    $('[data-slot=source]').className = `chip${s.source.startsWith('LLM') ? ' accent' : ''}`;
    $('[data-slot=llm-raw]').innerHTML = s.llmRaw
      ? `<div class="w14-raw"><span class="w14-raw__label">${escapeHtml(s.llmRaw.label)}</span><pre>${escapeHtml(s.llmRaw.text)}</pre></div>`
      : s.parser === 'llm'
        ? '<p class="w14-muted">이 질문에는 예시 응답이 없다. 키를 저장하고 “LLM으로 파싱 실행”을 누르면 실제 LLM이 파싱한다. 지금은 규칙 파서 결과를 쓴다.</p>'
        : '';
    $('[data-slot=trace]').innerHTML = s.trace.length
      ? `<span class="w14-muted">발동한 규칙:</span> ${s.trace
          .map((t) => `<span class="chip" title="${escapeHtml(t.effect)}">${escapeHtml(t.rule)} · 「${escapeHtml(t.match)}」 → ${escapeHtml(t.effect)}</span>`)
          .join(' ')}`
      : '';
  }

  // ---------- retrieval ----------
  async function rankerFor(queries) {
    if (s.method === 'vector' && vec.ready) {
      const need = queries.filter((q) => q && !vec.cache.has(q));
      for (const q of need) vec.cache.set(q, await embed(q, { role: 'query' }));
      return {
        label: '벡터 (multilingual-e5-small)',
        rankAll: (q) => {
          const qv = vec.cache.get(q);
          if (!qv) return [];
          return records
            .map((r, i) => ({ id: r.id, score: dot(qv, vec.recordVecs[i]) }))
            .sort((a, b) => b.score - a.score);
        },
      };
    }
    return {
      label: s.method === 'vector' ? 'BM25 (벡터 모델 준비 중)' : 'BM25',
      rankAll: (q) => bm25.search(q, records.length),
    };
  }

  async function compute() {
    const my = ++computeSeq;
    const gold = s.preset ? qById.get(s.preset) : null;
    let parsed = null;
    let parseError = null;
    try {
      parsed = JSON.parse(s.filterText);
    } catch (err) {
      parseError = `JSON 문법 오류: ${err.message}`;
    }
    const check = parsed ? validateSelfQuery(parsed, schema) : { ok: false, errors: [parseError], value: null };
    const sq = check.ok ? check.value : null;

    let ranker;
    try {
      ranker = await rankerFor([s.question, sq?.query?.trim()].filter(Boolean));
    } catch (err) {
      ranker = { label: 'BM25 (벡터 오류)', rankAll: (q) => bm25.search(q, records.length) };
    }
    if (my !== computeSeq) return;

    const A = ranker.rankAll(s.question).slice(0, s.k).map((r) => r.id);
    let B = null;
    let exact = null;
    if (sq) {
      B = selfQuerySearch(records, ranker.rankAll, sq, { k: s.k, timing: s.timing, fallbackQuery: s.question });
      exact = Object.keys(sq.filter).length ? applyFilter(records, sq.filter).map((r) => r.id) : null;
    }
    const answers = gold ? gold.answers : exact;
    render({ gold, check, sq, A, B, exact, answers, ranker });
  }

  // ---------- rendering ----------
  function render({ gold, check, sq, A, B, exact, answers, ranker }) {
    $('[data-slot=check]').innerHTML = check.ok
      ? '<span class="badge-ok">검증 통과</span> <span class="w14-muted">허용된 필드·연산자·값만 들어 있다.</span>'
      : `<div class="widget__error" role="alert"><b>검증 실패 — 이 필터는 실행하지 않는다.</b><ul>${check.errors
          .map((e) => `<li>${escapeHtml(e)}</li>`)
          .join('')}</ul></div>`;

    $('[data-slot=head]').innerHTML = `<p class="w14-question">「${escapeHtml(s.question)}」 ${
      gold ? `<span class="chip accent">${escapeHtml(gold.kind)} 질문</span>` : '<span class="chip">직접 입력</span>'
    }</p>${gold ? `<p class="w14-muted">${escapeHtml(gold.note)}</p>` : ''}`;

    const ansSet = answers ? new Set(answers) : null;

    // A
    const sa = ansSet ? scoreSet(A, answers) : null;
    $('[data-slot=a-sub]').textContent = `${ranker.label} · top-${s.k}`;
    $('[data-slot=a-stats]').innerHTML = stats(sa, A.length, '돌려준 개수');
    $('[data-slot=a-list]').innerHTML = A.length ? A.map((id) => item(byId.get(id), ansSet, gold?.gold.filter)).join('') : empty('검색어와 겹치는 레코드가 없다');

    // B
    if (B) {
      const sb = ansSet ? scoreSet(B.ids, answers) : null;
      const how =
        s.timing === 'post'
          ? `사후 필터 · top-${s.k} 검색 후 필터`
          : sq.intent !== 'search' && Object.keys(sq.filter).length
            ? `사전 필터 · intent=${sq.intent} → 자르지 않고 전부`
            : `사전 필터 · 필터 안에서 top-${s.k}`;
      $('[data-slot=b-sub]').textContent = how;
      $('[data-slot=b-stats]').innerHTML = stats(sb, B.ids.length, sq.intent === 'count' ? '개수 답' : '돌려준 개수');
      $('[data-slot=b-list]').innerHTML = B.ids.length
        ? B.ids.map((id) => item(byId.get(id), ansSet, gold?.gold.filter)).join('')
        : empty(s.timing === 'post' ? `top-${s.k} 안에 조건을 통과한 레코드가 하나도 없다` : '결과 없음');
    } else {
      $('[data-slot=b-sub]').textContent = '필터 검증 실패 → 순수 검색으로 대체';
      $('[data-slot=b-stats]').innerHTML = stats(ansSet ? scoreSet(A, answers) : null, A.length, '돌려준 개수');
      $('[data-slot=b-list]').innerHTML = A.map((id) => item(byId.get(id), ansSet, gold?.gold.filter)).join('');
    }

    // C
    if (!sq) {
      $('[data-slot=c-sub]').textContent = '';
      $('[data-slot=c-note]').innerHTML = '<p class="w14-muted">필터가 검증을 통과하지 못해 정확한 답을 계산하지 않았다.</p>';
      $('[data-slot=c-list]').innerHTML = gold ? gold.answers.map((id) => item(byId.get(id), null, null, true)).join('') : '';
    } else if (!exact) {
      $('[data-slot=c-sub]').textContent = '필터 없음';
      $('[data-slot=c-note]').innerHTML =
        '<p class="w14-muted">필터가 비어 있으면 “조건을 만족하는 집합”이 곧 전체다. 이런 질문은 조건 질문이 아니라 유사도 질문이다.</p>';
      $('[data-slot=c-list]').innerHTML = gold ? gold.answers.map((id) => item(byId.get(id), null, null, true)).join('') : '';
    } else {
      $('[data-slot=c-sub]').textContent = `${exact.length}개`;
      let note = '';
      if (gold) {
        const miss = gold.answers.filter((id) => !exact.includes(id));
        const extra = exact.filter((id) => !gold.answers.includes(id));
        note =
          miss.length || extra.length
            ? `<div class="callout callout--danger"><span class="callout__title">파싱한 필터가 틀렸다</span>${
                miss.length ? `빠진 정답: ${miss.map(short).join(', ')}. ` : ''
              }${extra.length ? `잘못 들어온 레코드: ${extra.map(short).join(', ')}.` : ''}<br><span class="w14-muted">정답 필터: <code>${escapeHtml(
                JSON.stringify(gold.gold.filter),
              )}</code></span></div>`
            : '<p><span class="badge-ok">정답 집합과 일치</span></p>';
      }
      $('[data-slot=c-note]').innerHTML = note;
      $('[data-slot=c-list]').innerHTML = exact.length
        ? exact.map((id) => item(byId.get(id), ansSet, gold?.gold.filter)).join('')
        : empty('조건을 모두 만족하는 레코드가 없다 — 정답은 “없다”');
    }

    $('[data-slot=verdict]').innerHTML = verdicts({ gold, check, sq, A, B, exact, answers });
  }

  function verdicts({ gold, check, sq, A, B, exact, answers }) {
    const out = [];
    if (!check.ok) {
      out.push(`<div class="callout callout--danger"><span class="callout__title">필터를 실행하지 않았다</span>
        파서가 만든 JSON이 스키마 검증을 통과하지 못했다. 허용하지 않은 필드나 연산자를 그대로 실행하면 데이터가 새거나 엉뚱한 조건이 걸린다. B는 순수 검색으로 대체했다.</div>`);
      return out.join('');
    }
    if (sq.intent === 'count' && exact) {
      out.push(`<div class="callout callout--ok"><span class="callout__title">개수 질문 → ${exact.length}개</span>
        필터로 고른 집합을 세면 답이 나온다. 순수 검색은 언제나 top-${s.k}개를 돌려줄 뿐이라 “몇 개인가”를 모른다.</div>`);
    }
    if (B?.relaxed) {
      const labels = B.relaxed.dropped.map((f) => schema[f]?.label ?? f).join(', ');
      out.push(`<div class="callout"><span class="callout__title">빈 결과 → 조건 완화 (${escapeHtml(labels)} 제거)</span>
        조건을 모두 만족하는 레코드는 0개다. B에 보이는 것은 조건을 줄여 찾은 <b>참고 결과</b>다. 답변은 “조건에 맞는 수업은 없다. 대신 …”처럼 완화했다는 사실을 먼저 밝혀야 한다.</div>`);
    }
    if (exact === null && Object.keys(sq.filter).length === 0) {
      out.push(`<div class="callout callout--more"><span class="callout__title">조건이 없는 질문</span>
        필터가 비어 있으므로 셀프 쿼리 결과는 순수 검색과 같다. 모든 질문에 필터가 필요한 것은 아니다.</div>`);
    }
    if (gold && gold.gold.filter && Object.keys(gold.gold.filter).length) {
      const bad = A.filter((id) => !answers.includes(id));
      if (bad.length) {
        const why = bad
          .map((id) => {
            const r = byId.get(id);
            const fails = Object.entries(gold.gold.filter)
              .filter(([f, c]) => !matchCond(r.meta[f], c))
              .map(([f]) => `${schema[f]?.label ?? f}=${fmtVal(r.meta[f])}`);
            return `<li><b>${escapeHtml(short(id))}</b> — 어긴 조건: ${escapeHtml(fails.join(', ') || '없음')}</li>`;
          })
          .join('');
        out.push(`<div class="callout callout--danger"><span class="callout__title">순수 검색 top-${s.k} 중 ${bad.length}개가 조건을 어겼다</span>
          단어가 겹치거나 문장이 비슷해서 올라왔을 뿐 조건은 확인하지 않았다.<ul>${why}</ul></div>`);
      } else if (answers.length > s.k) {
        out.push(`<div class="callout callout--danger"><span class="callout__title">순수 검색은 ${answers.length}개 중 ${s.k}개까지만 찾는다</span>
          정답이 k보다 많은 목록 질문은 top-k로 다 담을 수 없다.</div>`);
      }
    }
    if (s.timing === 'post' && B && exact && B.ids.length < Math.min(s.k, exact.length)) {
      out.push(`<div class="callout"><span class="callout__title">사후 필터의 함정</span>
        먼저 top-${s.k}를 뽑고 나서 거르므로 조건에 맞는 레코드가 ${exact.length}개 있어도 ${B.ids.length}개만 남았다. 사전 필터로 바꿔 비교한다.</div>`);
    }
    return out.join('');
  }

  function short(id) {
    const r = byId.get(id);
    return r ? `${id} ${r.meta.course ?? r.meta.kind ?? r.meta.place ?? ''}`.trim() : id;
  }

  function item(r, ansSet, _goldFilter, plain = false) {
    if (!r) return '';
    const hit = ansSet ? ansSet.has(r.id) : null;
    const cls = plain ? '' : hit === true ? 'is-hit' : hit === false ? 'is-miss' : '';
    const mark = plain || hit === null ? '' : hit ? '<span class="w14-mark" aria-label="정답">✓</span>' : '<span class="w14-mark" aria-label="오답">✗</span>';
    const m = r.meta;
    const chips = [
      TYPE_LABEL[m.type],
      m.semester && `${m.semester}학기`,
      m.day && `${m.day}요일`,
      m.periodStart && `${m.periodStart}~${m.periodEnd}교시`,
      m.room && (m.room === '외부 산업체' ? m.room : `${m.room}호`),
      m.professor,
      m.credits && `${m.credits}학점`,
    ].filter(Boolean);
    return `<li class="${cls}">${mark}<div><div class="w14-meta"><code>${r.id}</code>${chips
      .map((c) => `<span>${escapeHtml(c)}</span>`)
      .join('')}${r.synthetic ? '<span class="w14-syn">가상</span>' : ''}</div>${escapeHtml(r.text)}</div></li>`;
  }

  // ---------- LLM parser ----------
  async function runLlm() {
    if (!keyReady()) {
      $('[data-slot=keystate]').innerHTML = '<span class="chip warn">키 없음</span> 먼저 API 키를 저장한다. 키 없이도 예시 질문은 예시 응답으로 볼 수 있다.';
      return;
    }
    st.llmCtrl?.abort();
    st.llmCtrl = new AbortController();
    const btn = $('[data-act=llm-run]');
    btn.disabled = true;
    $('[data-slot=keystate]').innerHTML = '<span class="spinner" aria-hidden="true"></span> LLM이 JSON을 만드는 중…';
    try {
      const prompt = llmParserPrompt(s.question, schema);
      const res = await generate({ provider: s.provider, model: s.llmModel, ...prompt, maxTokens: 300, temperature: 0, signal: st.llmCtrl.signal });
      s.live = { question: s.question, text: res.text, model: s.llmModel };
      s.parser = 'llm';
      syncInputs();
      reparse();
      renderParserBox();
      compute();
    } catch (err) {
      if (err.name === 'AbortError') return;
      const msg = err instanceof LLMError ? err.message : 'LLM 호출 중 알 수 없는 오류가 났다.';
      $('[data-slot=keystate]').innerHTML = `<span class="chip warn">실패</span> ${escapeHtml(msg)}`;
      return;
    } finally {
      btn.disabled = false;
    }
    syncKeyState();
  }

  // ---------- vector model ----------
  async function ensureVector() {
    if (vec.ready || vec.loading) return;
    vec.loading = true;
    try {
      await loadModelWithUI($('[data-slot=model]'), VECTOR_MODEL);
      vec.recordVecs = await embedBatch(records.map((r) => r.text), { role: 'passage' });
      vec.ready = true;
    } catch (err) {
      // loadModelWithUI renders its own load errors; embedding errors are shown here. Stay on BM25.
      const slot = $('[data-slot=model]');
      if (!slot.querySelector('.widget__error')) {
        slot.innerHTML = `<div class="widget__error" role="alert">벡터 검색을 준비하지 못했다: ${escapeHtml(err.message)}. BM25로 계속한다.</div>`;
      }
      s.method = 'bm25';
      syncInputs();
    } finally {
      vec.loading = false;
    }
    if (!ctrl.signal.aborted) compute();
  }

  // ---------- events ----------
  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  on('[data-slot=presets]', 'click', (e) => {
    const id = e.target.closest('[data-preset]')?.dataset.preset;
    if (!id) return;
    s.preset = id;
    s.question = qById.get(id).q;
    syncInputs();
    reparse();
    renderParserBox();
    compute();
  });
  on('[data-in=q]', 'input', (e) => {
    s.question = e.target.value;
    const match = questions.find((q) => q.q === s.question.trim());
    s.preset = match ? match.id : null;
    for (const b of root.querySelectorAll('[data-preset]')) b.setAttribute('aria-pressed', String(b.dataset.preset === s.preset));
    reparse();
    renderParserBox();
    compute();
  });
  on('[data-in=parser]', 'change', (e) => {
    s.parser = e.target.value;
    if (s.parser === 'llm') $('[data-slot=llm]').open = true;
    reparse();
    renderParserBox();
    compute();
  });
  on('[data-in=timing]', 'change', (e) => {
    s.timing = e.target.value;
    compute();
  });
  on('[data-in=method]', 'change', (e) => {
    s.method = e.target.value;
    if (s.method === 'vector') ensureVector();
    compute();
  });
  on('[data-in=k]', 'input', (e) => {
    s.k = Number(e.target.value);
    $('[data-out=k]').textContent = s.k;
    compute();
  });
  on('[data-in=filter]', 'input', (e) => {
    s.filterText = e.target.value;
    s.source = '직접 수정함';
    $('[data-slot=source]').textContent = s.source;
    $('[data-slot=source]').className = 'chip warn';
    compute();
  });
  on('[data-act=reset]', 'click', () => {
    reparse();
    renderParserBox();
    compute();
  });
  on('[data-act=inject]', 'click', () => {
    let obj;
    try {
      obj = JSON.parse(s.filterText);
    } catch {
      obj = { query: '', intent: 'list', filter: {} };
    }
    obj.filter = { ...(obj.filter ?? {}), $where: 'this.professor != null || true', studentId: { regex: '.*' } };
    s.filterText = JSON.stringify(obj, null, 2);
    s.source = '직접 수정함';
    renderParserBox();
    $('[data-slot=source]').className = 'chip warn';
    compute();
  });
  on('[data-in=provider]', 'change', (e) => {
    s.provider = e.target.value;
    s.llmModel = PROVIDERS[s.provider].defaultModel;
    syncInputs();
  });
  on('[data-in=llmModel]', 'change', (e) => {
    s.llmModel = e.target.value;
  });
  on('[data-act=save-key]', 'click', () => {
    const input = $('[data-in=key]');
    if (!input.value.trim()) return;
    try {
      setKey(s.provider, input.value);
    } catch {
      $('[data-slot=keystate]').textContent = '이 브라우저에서는 sessionStorage를 쓸 수 없어 키를 저장하지 못했다.';
      return;
    }
    input.value = '';
    syncKeyState();
  });
  on('[data-act=clear-key]', 'click', () => {
    try {
      clearKey(s.provider);
    } catch {
      /* storage blocked: nothing stored */
    }
    s.live = null;
    syncKeyState();
    reparse();
    renderParserBox();
    compute();
  });
  on('[data-act=llm-run]', 'click', runLlm);

  syncInputs();
  reparse();
  renderParserBox();
  $('[data-slot=status]').hidden = true;
  $('[data-slot=body]').hidden = false;
  registerOutput(outputId, {
    title: options.outputTitle ?? '조건 검색 결과',
    node: out,
    inlineHost: $('[data-slot=out-inline]'),
  });
  compute();
}

export function unmount(el) {
  const st = state.get(el);
  if (st) {
    st.ctrl.abort();
    st.llmCtrl?.abort();
    unregisterOutput(st.outputId);
  }
  state.delete(el);
  el.replaceChildren();
}

// ---------- rendering helpers ----------

function stats(sc, n, countLabel) {
  const fmt = (v) => (v === null || v === undefined ? '—' : v.toFixed(2));
  const cells = [[countLabel, String(n)]];
  if (sc) cells.unshift(['정밀도', fmt(sc.precision)], ['재현율', fmt(sc.recall)]);
  return cells.map(([l, v]) => `<div class="stat"><span class="stat__label">${l}</span><span class="stat__value">${v}</span></div>`).join('');
}

function empty(msg) {
  return `<li class="w14-empty">${escapeHtml(msg)}</li>`;
}

function fmtVal(v) {
  return v === undefined ? '(없음)' : v === null ? 'null' : String(v);
}

function recordTable(records) {
  const head = '<thead><tr><th>id</th><th>유형</th><th>학기</th><th>요일</th><th>교시</th><th>강의실</th><th>담당</th><th>학점</th><th>가상</th><th>text</th></tr></thead>';
  const rows = records
    .map((r) => {
      const m = r.meta;
      const cell = (v) => `<td>${v === undefined || v === null ? '' : escapeHtml(String(v))}</td>`;
      return `<tr><td><code>${r.id}</code></td>${cell(m.type)}${cell(m.semester)}${cell(m.day)}${cell(
        m.periodStart ? `${m.periodStart}-${m.periodEnd}` : '',
      )}${cell(m.room)}${cell(m.professor)}${cell(m.credits)}${cell(r.synthetic ? '가상' : '')}${cell(r.text)}</tr>`;
    })
    .join('');
  return `${head}<tbody>${rows}</tbody>`;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

async function fetchJson(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${res.status} ${url.pathname}`);
  return res.json();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

// ---- logic:start (pure functions, no DOM — tested in Node)

const OPS = ['eq', 'in', 'gte', 'lte'];
const INTENTS = ['search', 'list', 'count'];
const TYPE_LABEL = { timetable: '시간표', course: '교과목', facility: '실습실', scholarship: '장학금' };

// Words that carry no search meaning once the conditions are pulled out.
const STOP = new Set(
  ('하는 있는 수업 수업은 수업을 수업이 수업의 수업만 과목 과목은 과목만 과목을 과목이 강의 강의는 ' +
    '모두 전부 모든 다 목록 알려줘 보여줘 찾아줘 알려 줘 뭐야 뭐 무엇 무슨 있나 있어 있나요 몇 개 개야 개인가 ' +
    '담당하는 담당 말고 에서 에 은 는 이 가 을 를 좀 이나 나 또는 이랑 하고 언제 교시야').split(' '),
);

/**
 * Transparent rule-based self-query parser. Every rule that fires is recorded in `trace`.
 * @returns {{ query: string, intent: string, filter: object, trace: { rule: string, match: string, effect: string }[] }}
 */
export function parseRule(question, schema) {
  const filter = {};
  const trace = [];
  let rest = ` ${question} `;
  const take = (re, rule, fn) => {
    const m = rest.match(re);
    if (!m) return null;
    const effect = fn(m);
    if (effect === null) return null;
    trace.push({ rule, match: m[0].trim(), effect });
    rest = rest.replace(m[0], ' ');
    return m;
  };

  // R1 학기
  take(/([12])\s*학기(에서|에|의|는)?/, 'R1 학기', (m) => {
    filter.semester = Number(m[1]);
    return `semester = ${m[1]}`;
  });
  // R2 요일 (several days → in)
  const dayRe = /([월화수목금])요일(에는|에|은|는|의)?/g;
  const days = [...rest.matchAll(dayRe)].map((m) => m[1]);
  if (days.length) {
    filter.day = days.length === 1 ? days[0] : { in: days };
    trace.push({ rule: 'R2 요일', match: days.map((d) => `${d}요일`).join(', '), effect: `day = ${JSON.stringify(filter.day)}` });
    rest = rest.replace(dayRe, ' ');
  }
  // R3 강의실: the FIRST three-digit room number (negation is not understood)
  take(/(\d{3})\s*호(에서|에|의|는|실)?/, 'R3 강의실', (m) => {
    filter.room = m[1];
    return `room = "${m[1]}"`;
  });
  // R4 교시 범위 / 오전 / 오후
  if (!take(/(\d)\s*[-~]\s*(\d)\s*교시(에|의)?/, 'R4 교시 범위', (m) => {
    filter.periodStart = { gte: Number(m[1]) };
    filter.periodEnd = { lte: Number(m[2]) };
    return `periodStart ≥ ${m[1]}, periodEnd ≤ ${m[2]}`;
  })) {
    take(/(\d)\s*교시(에|의)?/, 'R4 교시', (m) => {
      filter.periodStart = { lte: Number(m[1]) };
      filter.periodEnd = { gte: Number(m[1]) };
      return `periodStart ≤ ${m[1]} ≤ periodEnd`;
    });
  }
  take(/오전(에|의|에는)?/, 'R4 오전', () => {
    filter.periodEnd = { lte: 3 };
    return 'periodEnd ≤ 3 (오전 안에 끝나는 수업)';
  });
  take(/오후(에|의|에는)?/, 'R4 오후', () => {
    filter.periodStart = { gte: 4 };
    return 'periodStart ≥ 4';
  });
  // R5 담당 교수 (only names that exist in the schema)
  const initials = (schema.professor?.values ?? []).filter((p) => p.endsWith(' 교수')).map((p) => p[0]).join('');
  if (initials) {
    take(new RegExp(`([${initials}])\\s*교수(님이|님|가|의|는|이)?`), 'R5 담당', (m) => {
      filter.professor = `${m[1]} 교수`;
      return `professor = "${m[1]} 교수"`;
    });
  }
  take(/공동\s*(담당|수업)?/, 'R5 담당', () => {
    filter.professor = '공동';
    return 'professor = "공동"';
  });
  // R6 학점 (N학점 이상 → gte)
  take(/(\d)\s*학점\s*(이상)?(짜리)?/, 'R6 학점', (m) => {
    filter.credits = m[2] ? { gte: Number(m[1]) } : Number(m[1]);
    return m[2] ? `credits ≥ ${m[1]}` : `credits = ${m[1]}`;
  });
  // R7 유형
  if (filter.credits !== undefined || /과목/.test(question)) {
    filter.type = 'course';
    trace.push({ rule: 'R7 유형', match: filter.credits !== undefined ? '학점' : '과목', effect: 'type = "course"' });
  } else if (/수업|강의|교시|시간표/.test(question) || filter.day || filter.periodStart || filter.periodEnd) {
    filter.type = 'timetable';
    trace.push({ rule: 'R7 유형', match: (question.match(/수업|강의|교시|시간표/) ?? ['요일·교시'])[0], effect: 'type = "timetable"' });
  } else if (/장학/.test(question)) {
    filter.type = 'scholarship';
    trace.push({ rule: 'R7 유형', match: '장학', effect: 'type = "scholarship"' });
  }

  // R8 의도
  let intent = 'search';
  const words = rest
    .replace(/[?？!.,()]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
  const query = words.filter((w) => !STOP.has(w)).join(' ');
  if (/몇\s*(개|과목|강의|건)|개수/.test(question)) intent = 'count';
  else if (/모두|전부|모든|목록|만\s*(보여|알려|찾아)|만\s*$|만\?/.test(question)) intent = 'list';
  else if (Object.keys(filter).length > 1 && query === '') intent = 'list';
  trace.push({ rule: 'R8 의도', match: intent === 'search' ? '(기본값)' : intent === 'count' ? '몇 개' : '목록', effect: `intent = "${intent}"` });

  return { query, intent, filter, trace };
}

/**
 * Validate a self-query object (from the rule parser, an LLM or a human) against the schema.
 * Only allowed fields, allowed operators and in-range values pass; everything else is an error.
 * @returns {{ ok: boolean, errors: string[], value: { query: string, intent: string, filter: object } | null }}
 */
export function validateSelfQuery(obj, schema) {
  const errors = [];
  if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) {
    return { ok: false, errors: ['최상위 값은 { } 객체여야 한다'], value: null };
  }
  for (const key of Object.keys(obj)) {
    if (!['query', 'intent', 'filter'].includes(key)) errors.push(`허용하지 않는 최상위 키: "${key}"`);
  }
  const query = obj.query ?? '';
  if (typeof query !== 'string') errors.push('query는 문자열이어야 한다');
  const intent = obj.intent ?? 'search';
  if (!INTENTS.includes(intent)) errors.push(`intent는 ${INTENTS.join(' | ')} 중 하나여야 한다 (받은 값: ${JSON.stringify(intent)})`);
  const filter = obj.filter ?? {};
  if (filter === null || typeof filter !== 'object' || Array.isArray(filter)) {
    errors.push('filter는 { } 객체여야 한다');
    return { ok: false, errors, value: null };
  }
  const checkValue = (field, spec, v) => {
    if (spec.kind === 'enum') {
      if (!spec.values.includes(v)) errors.push(`${field}: 허용 값이 아니다 ${JSON.stringify(v)} (허용: ${spec.values.join(', ')})`);
    } else if (spec.kind === 'int') {
      if (!Number.isInteger(v) || v < spec.min || v > spec.max) errors.push(`${field}: ${spec.min}~${spec.max} 사이 정수여야 한다 (받은 값: ${JSON.stringify(v)})`);
    }
  };
  for (const [field, cond] of Object.entries(filter)) {
    const spec = schema[field];
    if (!spec) {
      errors.push(`허용하지 않는 필드: "${field}"`);
      continue;
    }
    if (cond !== null && typeof cond === 'object' && !Array.isArray(cond)) {
      const ops = Object.keys(cond);
      if (ops.length === 0) errors.push(`${field}: 빈 조건 { }`);
      for (const op of ops) {
        if (!OPS.includes(op)) {
          errors.push(`${field}: 허용하지 않는 연산자 "${op}" (허용: ${OPS.join(', ')})`);
        } else if (op === 'in') {
          if (!Array.isArray(cond.in) || cond.in.length === 0) errors.push(`${field}.in: 값이 하나 이상인 배열이어야 한다`);
          else cond.in.forEach((v) => checkValue(field, spec, v));
        } else if (op === 'gte' || op === 'lte') {
          if (spec.kind !== 'int') errors.push(`${field}: ${op}는 숫자 필드에만 쓸 수 있다`);
          else checkValue(field, spec, cond[op]);
        } else {
          checkValue(field, spec, cond.eq);
        }
      }
    } else if (Array.isArray(cond)) {
      errors.push(`${field}: 배열은 {"in": [...]} 형태로 쓴다`);
    } else {
      checkValue(field, spec, cond);
    }
  }
  return errors.length ? { ok: false, errors, value: null } : { ok: true, errors, value: { query, intent, filter } };
}

/** Does one meta value satisfy one condition (scalar = eq, or {eq, in, gte, lte})? */
export function matchCond(v, cond) {
  if (cond !== null && typeof cond === 'object' && !Array.isArray(cond)) {
    if ('eq' in cond && v !== cond.eq) return false;
    if ('in' in cond && !cond.in.includes(v)) return false;
    if ('gte' in cond && !(typeof v === 'number' && v >= cond.gte)) return false;
    if ('lte' in cond && !(typeof v === 'number' && v <= cond.lte)) return false;
    return true;
  }
  return v === cond;
}

export const matchFilter = (meta, filter) => Object.entries(filter).every(([f, c]) => matchCond(meta[f], c));
export const applyFilter = (records, filter) => records.filter((r) => matchFilter(r.meta, filter));

// Least important conditions are dropped first when a filter returns nothing. `type` is never dropped.
export const RELAX_ORDER = ['professor', 'room', 'periodStart', 'periodEnd', 'day', 'credits', 'semester'];

/** Drop conditions one by one until something matches. */
export function relaxFilter(records, filter) {
  let f = { ...filter };
  const dropped = [];
  for (const field of RELAX_ORDER) {
    if (applyFilter(records, f).length) break;
    if (field in f) {
      const { [field]: _, ...restF } = f;
      f = restF;
      dropped.push(field);
    }
  }
  return { filter: f, dropped, results: applyFilter(records, f) };
}

/**
 * Self-query retrieval.
 * @param {{ id: string, meta: object }[]} records
 * @param {(query: string) => { id: string, score: number }[]} rankAll  full ranking over all records
 * @param {{ query: string, intent: string, filter: object }} sq
 * @param {{ k: number, timing: 'pre'|'post', fallbackQuery: string }} opts
 */
export function selfQuerySearch(records, rankAll, sq, { k, timing, fallbackQuery }) {
  const hasFilter = Object.keys(sq.filter).length > 0;
  const q = sq.query.trim() || fallbackQuery;
  let filter = sq.filter;
  let relaxed = null;
  let exact = applyFilter(records, filter);
  if (hasFilter && exact.length === 0) {
    relaxed = relaxFilter(records, filter);
    filter = relaxed.filter;
  }
  const allowed = new Set(applyFilter(records, filter).map((r) => r.id));
  const limitless = sq.intent !== 'search' && hasFilter;

  if (timing === 'post') {
    // search first (top-k), then throw away what fails the filter
    const top = rankAll(q).slice(0, k);
    return { ids: top.filter((r) => allowed.has(r.id)).map((r) => r.id), relaxed, exactCount: exact.length, searched: top.length };
  }
  // pre-filter: rank only the allowed records; records without any score keep data order
  const ranked = sq.query.trim() ? rankAll(sq.query).filter((r) => allowed.has(r.id) && r.score > 0).map((r) => r.id) : [];
  const seen = new Set(ranked);
  const ordered = [...ranked, ...records.filter((r) => allowed.has(r.id) && !seen.has(r.id)).map((r) => r.id)];
  return { ids: limitless ? ordered : ordered.slice(0, k), relaxed, exactCount: exact.length, searched: allowed.size };
}

/** Precision/recall of a retrieved id list against an exact answer set. */
export function scoreSet(retrieved, gold) {
  const g = new Set(gold);
  const hits = retrieved.filter((id) => g.has(id)).length;
  if (g.size === 0) return { hits, precision: retrieved.length ? 0 : 1, recall: null };
  return { hits, precision: retrieved.length ? hits / retrieved.length : 0, recall: hits / g.size };
}

/** Pull the first JSON object out of an LLM reply (tolerates ```json fences and chatter). */
export function extractJson(text) {
  const s = String(text).replace(/```(?:json)?/gi, '');
  const a = s.indexOf('{');
  const b = s.lastIndexOf('}');
  if (a === -1 || b <= a) throw new Error('응답에서 JSON 객체를 찾지 못했다');
  return JSON.parse(s.slice(a, b + 1));
}

/** System prompt that makes an LLM act as the self-query parser. */
export function llmParserPrompt(question, schema) {
  const fields = Object.entries(schema)
    .map(([f, s]) => `- ${f} (${s.label}): ${s.kind === 'enum' ? s.values.map((v) => JSON.stringify(v)).join(', ') : `정수 ${s.min}~${s.max}`}`)
    .join('\n');
  return {
    system:
      '너는 학과 정보 검색기의 질의 파서다. 사용자 질문을 JSON 객체 하나로만 바꾼다. 설명이나 코드 블록 없이 JSON만 출력한다.\n' +
      '형식: {"query": 검색어 문자열(조건으로 옮긴 부분은 뺀다, 없으면 ""), "intent": "search" | "list" | "count", ' +
      '"filter": {필드: 값 | {"in": [값, ...]} | {"gte": 정수} | {"lte": 정수}}}\n' +
      `허용 필드와 값:\n${fields}\n` +
      '규칙: 질문에 없는 조건은 넣지 않는다. 오전은 1~3교시, 오후는 4~6교시다. 수업 시간표는 type "timetable", 학점·과목 정보는 type "course"다. ' +
      '목록을 원하면 intent "list", 개수를 물으면 "count", 그 밖에는 "search"다.',
    messages: [{ role: 'user', content: question }],
  };
}

export { TYPE_LABEL };
// ---- logic:end
