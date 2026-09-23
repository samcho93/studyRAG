// w11 인용 하이라이트 데모
// One concept: a citation [n] in an answer must point to an exact span of the
// original document. Hover / focus / tap a citation and the supporting span is
// highlighted inside the source chunk using the chunk's { start, end } offsets
// from core/chunker.js. A validator flags citation numbers that don't exist,
// sentences without a citation and citations whose source lacks the claim's
// key terms.

import { chunk, sentenceSpans } from '../core/chunker.js';
import { BM25 } from '../core/bm25.js';
import { PROVIDERS, setKey, hasKey, clearKey, generate, ragPrompt, LLMError } from '../core/llm.js';
import { registerOutput, unregisterOutput } from '../site/result.js';

const DATA = new URL('../../data/', import.meta.url);
const RECORDED_LABEL = '예시 응답 · 수업용 작성 예시';

// ------------------------------------------------------------------ pure helpers
// (exported so they can be tested in Node and reused in Challenge answers)

const CITE_RE = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
const SENT_END_RE = /[.!?。]+(?:\s*\[\d+(?:\s*,\s*\d+)*\])*(?=\s|$)|\n+/g;
const REFUSAL_RE = /찾을 수 없|알 수 없|내용이 없|정보가 없|답할 수 없/;
const TOKEN_RE = /[a-z][a-z0-9]*|\d+(?:\.\d+)?|[가-힣]+/gi;
const SUFFIXES = ['이라는', '으로', '에서', '에게', '까지', '부터', '이다', '한다', '된다', '하는', '하고', '라는',
  '이며', '이고', '에는', '에도', '와', '과', '는', '은', '이', '가', '을', '를', '에', '의', '로', '도', '만', '다'];
const STOP = new Set(['그래서', '따라서', '하지만', '그러나', '또한', '그리고', '이는', '경우', '때문', '있다', '없다',
  '한다', '된다', '이다', '같다', '있는', '없는', '하는', '되는', '대한', '위해', '통해', '모두', '만큼', '나머지',
  '다만', '먼저', '않는', '않다', '않고', '않으며', '있으면', '하면', '되면', '두면',
  'the', 'and', 'of', 'to', 'is', 'in']);

/** All citation numbers in order of appearance. "[1, 2]" yields two entries. */
export function parseCitations(text) {
  const out = [];
  for (const m of String(text).matchAll(CITE_RE)) {
    for (const n of m[1].split(',')) out.push({ n: Number(n.trim()), index: m.index, length: m[0].length });
  }
  return out;
}

export function stripCitations(text) {
  return String(text).replace(CITE_RE, '').replace(/\s+([.!?。])/g, '$1').replace(/\s{2,}/g, ' ').trim();
}

/**
 * Split an answer into sentences. A citation written after the period
 * ("…다. [1]") stays with the sentence it belongs to; "4.0" is not a boundary.
 * @returns {{ start: number, end: number, text: string }[]}
 */
export function splitSentences(text) {
  const out = [];
  const push = (s, e) => {
    while (s < e && /\s/.test(text[s])) s++;
    while (e > s && /\s/.test(text[e - 1])) e--;
    if (e <= s) return;
    const piece = text.slice(s, e);
    const prev = out[out.length - 1];
    // a line that holds only citations belongs to the previous sentence
    if (prev && stripCitations(piece) === '') {
      prev.end = e;
      prev.text = text.slice(prev.start, e);
      return;
    }
    out.push({ start: s, end: e, text: piece });
  };
  let last = 0;
  for (const m of String(text).matchAll(SENT_END_RE)) {
    const end = m.index + m[0].length;
    push(last, end);
    last = end;
  }
  push(last, text.length);
  return out;
}

/** Key terms of a claim: content words (particles stripped) and numbers. */
export function keyTerms(claim) {
  const words = [];
  const numbers = [];
  for (const [tok] of String(claim).matchAll(TOKEN_RE)) {
    if (/^\d/.test(tok)) {
      numbers.push(tok);
      continue;
    }
    let w = tok.toLowerCase();
    if (STOP.has(w)) continue;
    if (/^[가-힣]/.test(w)) {
      const suf = SUFFIXES.find((s) => w.length > s.length && w.endsWith(s));
      if (suf) w = w.slice(0, -suf.length);
    }
    if (w.length < 2 || STOP.has(w)) continue;
    if (!words.includes(w)) words.push(w);
  }
  return { words, numbers: [...new Set(numbers)] };
}

function hasWord(normSource, w) {
  if (!/^[가-힣]/.test(w) || w.length <= 2) return normSource.includes(w);
  for (let i = 0; i < w.length - 1; i++) if (normSource.includes(w.slice(i, i + 2))) return true;
  return false;
}

function hasNumber(source, num) {
  const esc = num.replace('.', '\\.');
  return new RegExp(`(?<![\\d.])${esc}(?!\\d)`).test(source);
}

/**
 * How well does `source` support `claim`? Numbers must appear exactly;
 * a Hangul word counts as present if any of its character bigrams appears.
 */
export function supportCheck(claim, source) {
  const { words, numbers } = keyTerms(claim);
  const norm = String(source).toLowerCase().replace(/\s+/g, '');
  const missingWords = words.filter((w) => !hasWord(norm, w));
  const missingNumbers = numbers.filter((n) => !hasNumber(source, n));
  const total = words.length + numbers.length;
  const found = total - missingWords.length - missingNumbers.length;
  return { ratio: total ? found / total : 1, missingWords, missingNumbers, total };
}

/** Sentence-level alignment: the sentence of `chunkText` that best supports `claim`. */
export function alignSpan(claim, chunkText) {
  let best = null;
  for (const [s0, e0] of sentenceSpans(chunkText)) {
    let s = s0;
    let e = e0;
    while (s < e && /\s/.test(chunkText[s])) s++;
    while (e > s && /\s/.test(chunkText[e - 1])) e--;
    if (e <= s) continue;
    const { ratio } = supportCheck(claim, chunkText.slice(s, e));
    if (!best || ratio > best.score) best = { start: s, end: e, score: ratio };
  }
  return best;
}

/**
 * Validate an answer against its numbered sources.
 * @param {string} answer
 * @param {{ n: number, text: string, start?: number }[]} sources  chunk text + its offset in the document
 */
export function validateAnswer(answer, sources) {
  const byN = new Map(sources.map((s) => [s.n, s]));
  const sentences = splitSentences(answer).map((sen, index) => {
    const claim = stripCitations(sen.text);
    const cites = [...new Set(parseCitations(sen.text).map((c) => c.n))];
    const refusal = REFUSAL_RE.test(claim);
    const issues = [];
    const invalid = cites.filter((n) => !byN.has(n));
    const valid = cites.filter((n) => byN.has(n));
    for (const n of invalid) issues.push({ type: 'invalid', n });
    if (!refusal && cites.length === 0) issues.push({ type: 'uncited' });

    let support = null;
    const spans = [];
    if (valid.length) {
      support = supportCheck(claim, valid.map((n) => byN.get(n).text).join('\n'));
      if (support.missingNumbers.length || support.ratio < 0.6) {
        issues.push({ type: 'unsupported', n: valid, missing: [...support.missingNumbers, ...support.missingWords] });
      } else if (support.missingWords.length) {
        issues.push({ type: 'weak', n: valid, missing: support.missingWords });
      }
      for (const n of valid) {
        const src = byN.get(n);
        const a = alignSpan(claim, src.text);
        if (a && a.score >= 0.5) {
          const off = src.start ?? 0;
          spans.push({ n, start: off + a.start, end: off + a.end, score: a.score });
        }
      }
    }
    const fail = issues.some((i) => i.type !== 'weak');
    const status = refusal && !issues.length ? 'refusal' : fail ? 'fail' : issues.length ? 'warn' : 'ok';
    return { index, ...sen, claim, cites, refusal, issues, support, spans, status };
  });

  const claims = sentences.filter((s) => !s.refusal);
  const cited = claims.filter((s) => s.cites.some((n) => byN.has(n)));
  return {
    sentences,
    stats: {
      sentences: sentences.length,
      claims: claims.length,
      cited: cited.length,
      coverage: claims.length ? cited.length / claims.length : null,
      invalid: sentences.reduce((a, s) => a + s.issues.filter((i) => i.type === 'invalid').length, 0),
      uncited: sentences.filter((s) => s.issues.some((i) => i.type === 'uncited')).length,
      unsupported: sentences.filter((s) => s.issues.some((i) => i.type === 'unsupported')).length,
      weak: sentences.filter((s) => s.status === 'warn').length,
    },
    ok: sentences.every((s) => s.status === 'ok' || s.status === 'refusal'),
  };
}

/**
 * Escape `text` and wrap each [start, end) range in a mark. Ranges may overlap:
 * the text is cut at every boundary and each piece lists the keys that cover it.
 * @param {string} text
 * @param {{ start: number, end: number, key: string }[]} spans offsets relative to `text`
 */
export function highlightHtml(text, spans) {
  const cuts = new Set([0, text.length]);
  for (const s of spans) {
    cuts.add(Math.max(0, Math.min(text.length, s.start)));
    cuts.add(Math.max(0, Math.min(text.length, s.end)));
  }
  const points = [...cuts].sort((a, b) => a - b);
  let html = '';
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];
    const keys = spans.filter((s) => s.start <= a && s.end >= b).map((s) => s.key);
    const piece = escapeHtml(text.slice(a, b));
    html += keys.length ? `<mark class="w11-mark" data-k="${escapeHtml(keys.join(' '))}">${piece}</mark>` : piece;
  }
  return html;
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

// ------------------------------------------------------------------ templates

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w11" data-w11>
    <h3 class="widget__title">인용 하이라이트 데모</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> 문서셋과 저장된 응답을 불러오는 중…
    </div>
    <div data-slot="body" hidden>
      <div class="widget__controls">
        <label class="field">
          <span class="field__label">예시 질문 (저장된 응답)</span>
          <select data-in="example"></select>
        </label>
      </div>
      <div class="btn-row" data-slot="presets" style="margin-bottom: var(--space-3)"></div>

      <div class="w11-chat" data-slot="chat" aria-live="polite"></div>
      <span data-slot="nosrc" hidden>검색된 출처 목록에 없는 번호다</span>

      <form class="w11-ask" data-slot="ask">
        <label class="field">
          <span class="field__label">직접 질문하기 <span data-slot="mode" class="w11-mode"></span></span>
          <input type="text" data-in="question" autocomplete="off" placeholder="예: 실습실은 몇 시까지 여나?">
        </label>
        <button type="submit" class="btn primary">질문하기</button>
      </form>

      <details class="w11-llm">
        <summary>LLM 설정 (API 키 · 선택)</summary>
        <div class="widget__controls" style="margin-top: var(--space-3)">
          <label class="field">
            <span class="field__label">공급자</span>
            <select data-in="provider"></select>
          </label>
          <label class="field">
            <span class="field__label">모델</span>
            <select data-in="model"></select>
          </label>
          <label class="field">
            <span class="field__label">API 키 (이 탭의 sessionStorage에만 저장)</span>
            <input type="password" data-in="key" autocomplete="off" spellcheck="false" placeholder="키를 붙여 넣는다">
          </label>
        </div>
        <div class="btn-row">
          <button type="button" class="btn small" data-act="save-key">키 저장</button>
          <button type="button" class="btn small ghost" data-act="clear-key">키 지우기</button>
          <label class="w11-check"><input type="checkbox" data-in="stream" checked> 스트리밍처럼 한 글자씩 표시</label>
        </div>
        <p class="w11-keynote" data-slot="keystate" role="status"></p>
      </details>

      <div data-slot="out-inline"></div>
    </div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w11-out">
    <div data-slot="verdict"></div>
    <div class="stat-row" data-slot="stats"></div>
    <p class="w11-hint" data-slot="hint" role="status" aria-live="polite"></p>
    <h4 data-slot="src-title">출처</h4>
    <ol class="w11-sources" data-slot="sources"></ol>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ item?: string, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const uid = `w11-${++seq}`;
  const outputId = `w11:${seq}`;
  const st = { ctrl, outputId, job: null, timer: 0 };
  state.set(el, st);

  let corpus;
  let recorded;
  try {
    [corpus, recorded] = await Promise.all([
      fetchJson(new URL('corpus/corpus.json', DATA), ctrl.signal),
      fetchJson(new URL('w11/answers.json', DATA), ctrl.signal),
    ]);
  } catch (err) {
    if (err.name === 'AbortError') return;
    $('[data-slot=status]').innerHTML = `<div class="widget__error" role="alert">
      문서셋을 불러오지 못했다 (${escapeHtml(err.message)}).<br>
      파일을 더블클릭해 <code>file://</code>로 열었다면 ES Module과 fetch가 동작하지 않는다.
      저장소 폴더에서 <code>python -m http.server</code>를 실행하고 <code>http://localhost:8000</code>으로 연다.</div>`;
    return;
  }

  $('[data-slot=nosrc]').id = `${uid}-nosrc`;

  // ---- index: chunk every document, keep offsets, BM25 over chunks
  const cfg = recorded.retrieval;
  const docs = new Map(corpus.documents.map((d) => [d.id, d]));
  const chunks = new Map();
  for (const d of corpus.documents) {
    for (const c of chunk(d.text, cfg.chunk)) {
      const id = `${d.id}#${c.index}`;
      chunks.set(id, { id, doc: d.id, title: d.title, index: c.index, start: c.start, end: c.end, text: c.text });
    }
  }
  const bm25 = new BM25().add([...chunks.values()].map((c) => ({ id: c.id, text: c.text })));
  const retrieve = (q, k = cfg.k) => bm25.search(q, k).map((r) => ({ ...chunks.get(r.id), score: r.score }));

  const items = recorded.items;
  const byId = new Map(items.map((it) => [it.id, it]));

  // ---- controls
  const exSel = $('[data-in=example]');
  const good = items.filter((it) => it.kind === 'good');
  const special = items.filter((it) => it.kind !== 'good');
  exSel.innerHTML =
    `<optgroup label="골든셋 질문 · 저장된 답변">${good
      .map((it) => `<option value="${it.id}">${it.id.toUpperCase()} · ${escapeHtml(it.question)}</option>`)
      .join('')}</optgroup>` +
    `<optgroup label="실패 · 거절 예시">${special
      .map((it) => `<option value="${it.id}">${escapeHtml(it.label)}</option>`)
      .join('')}</optgroup>`;
  $('[data-slot=presets]').innerHTML = special
    .map((it) => `<button type="button" class="btn small ghost" data-preset="${it.id}">${escapeHtml(it.label)}</button>`)
    .join('');

  const provSel = $('[data-in=provider]');
  provSel.innerHTML = Object.entries(PROVIDERS).map(([k, p]) => `<option value="${k}">${escapeHtml(p.label)}</option>`).join('');
  const fillModels = () => {
    $('[data-in=model]').innerHTML = PROVIDERS[provSel.value].models
      .map((m) => `<option value="${m}">${escapeHtml(m)}</option>`)
      .join('');
  };
  fillModels();

  function syncKeyState() {
    const has = safeHasKey(provSel.value);
    $('[data-slot=keystate]').innerHTML = has
      ? `<span class="chip ok">🔑 ${escapeHtml(PROVIDERS[provSel.value].label)} 키 저장됨 · 이 탭에서만</span> “질문하기”가 LLM을 호출한다.`
      : '<span class="chip warn">🔑 키 없음</span> 예시 질문은 저장된 응답으로, 직접 입력한 질문은 검색 결과만 보여 준다.';
    $('[data-slot=mode]').textContent = has ? '· LLM 생성' : '· 키 없음: 검색만';
  }

  // ---- rendering
  let view = null; // { sources, result }
  let pinned = null;

  function sourcesFor(item) {
    return item.sources.map((id, i) => ({ ...chunks.get(id), n: i + 1 })).filter((s) => s.id);
  }

  function withScores(question, sources) {
    const scored = new Map(bm25.search(question, 50).map((r) => [r.id, r.score]));
    return sources.map((s) => ({ ...s, score: s.score ?? scored.get(s.id) ?? 0 }));
  }

  /** Render one Q/A turn. kind: recorded | llm | nokey | empty | error */
  async function showTurn({ question, answer, sources, badge, kind, note, animate }) {
    const job = {};
    st.job = job;
    pinned = null;
    const chat = $('[data-slot=chat]');
    chat.innerHTML = `
      <div class="w11-msg w11-msg--user"><span class="visually-hidden">질문: </span>${escapeHtml(question)}</div>
      <div class="w11-msg w11-msg--bot">
        <div class="w11-msg__meta">${badge ?? ''}</div>
        <div class="w11-answer" data-slot="answer"></div>
        <div data-slot="after"></div>
      </div>`;
    const ansEl = chat.querySelector('[data-slot=answer]');
    const after = chat.querySelector('[data-slot=after]');
    const result = answer ? validateAnswer(answer, sources) : null;
    view = { sources, result };

    if (!answer) {
      renderSources(sources, null);
      renderVerdict(kind, null, note);
      ansEl.innerHTML = note ?? '';
      return;
    }
    if (animate && $('[data-in=stream]').checked && !prefersReducedMotion()) {
      // While streaming, "[1" may be half-written: show raw text, validate only at the end.
      renderSources(sources, null);
      renderVerdict('pending', null);
      ansEl.classList.add('is-streaming');
      await typewriter(ansEl, answer, () => st.job !== job, st);
      if (st.job !== job) return;
      ansEl.classList.remove('is-streaming');
    }
    renderSources(sources, result);
    renderVerdict(kind, result, note);
    ansEl.innerHTML = answerHtml(answer, result, sources.length);
    after.innerHTML = feedbackHtml();
  }

  function answerHtml(answer, result, count) {
    let html = '';
    let last = 0;
    for (const s of result.sentences) {
      html += escapeHtml(answer.slice(last, s.start));
      html += sentenceHtml(s, count);
      last = s.end;
    }
    return html + escapeHtml(answer.slice(last));
  }

  function sentenceHtml(s, count) {
    let inner = '';
    let last = 0;
    for (const m of s.text.matchAll(CITE_RE)) {
      inner += escapeHtml(s.text.slice(last, m.index));
      inner += m[1]
        .split(',')
        .map((x) => Number(x.trim()))
        .map((n) => {
          const ok = n >= 1 && n <= count;
          const desc = ok ? `${uid}-src-${n}` : `${uid}-nosrc`;
          return `<button type="button" class="w11-cite${ok ? '' : ' is-bad'}" data-n="${n}" data-sent="${s.index}"
            aria-describedby="${desc}" aria-pressed="false"><span class="visually-hidden">출처 </span>[${n}]</button>`;
        })
        .join('');
      last = m.index + m[0].length;
    }
    inner += escapeHtml(s.text.slice(last));
    const flags = s.issues.map(issueLabel).map((t, i) => `<span class="w11-flag w11-flag--${s.issues[i].type}">${t}</span>`).join('');
    return `<span class="w11-sent w11-sent--${s.status}" data-sent="${s.index}">${inner}${flags}</span>`;
  }

  function renderSources(sources, result) {
    const spansByN = new Map();
    for (const s of result?.sentences ?? []) {
      for (const sp of s.spans) {
        if (!spansByN.has(sp.n)) spansByN.set(sp.n, []);
        spansByN.get(sp.n).push({ start: sp.start, end: sp.end, key: `s${s.index}` });
      }
    }
    const citedNs = new Set((result?.sentences ?? []).flatMap((s) => s.cites));
    $('[data-slot=src-title]').textContent = sources.length
      ? `출처 · BM25 상위 ${sources.length}개 청크 (재귀 분할 ${cfg.chunk.size}자)`
      : '출처 · 검색 결과 없음';
    $('[data-slot=sources]').innerHTML = sources
      .map((src) => {
        const doc = docs.get(src.doc);
        // exact offsets: the chunk is a slice of the original document text
        const text = doc.text.slice(src.start, src.end);
        const spans = (spansByN.get(src.n) ?? []).map((sp) => ({ ...sp, start: sp.start - src.start, end: sp.end - src.start }));
        return `<li class="w11-src" data-src="${src.n}">
          <div class="w11-src__head" id="${uid}-src-${src.n}">
            <b>[${src.n}]</b> ${escapeHtml(src.title)}
            <span class="chunk-meta"><span>${escapeHtml(src.doc)}</span><span>청크 #${src.index}</span><span>문서 위치 ${src.start}–${src.end}</span><span>BM25 ${src.score.toFixed(2)}</span></span>
          </div>
          ${citedNs.has(src.n) ? '' : '<span class="chip">인용되지 않음</span>'}
          <div class="w11-src__text">${highlightHtml(text, spans)}</div>
        </li>`;
      })
      .join('');
    $('[data-slot=nosrc]').textContent = `검색된 출처 목록에 없는 번호다. 출처는 ${sources.length}개뿐이다.`;
    $('[data-slot=hint]').textContent = defaultHint(sources.length);
  }

  function renderVerdict(kind, result, note) {
    const verdict = $('[data-slot=verdict]');
    const stats = $('[data-slot=stats]');
    if (!result) {
      stats.innerHTML = '';
      verdict.innerHTML = kind === 'error'
        ? '<div class="callout callout--danger"><span class="callout__title">답변 생성 실패</span>출처는 검색됐지만 LLM 호출이 실패했다.</div>'
        : kind === 'pending'
          ? '<div class="widget__status"><span class="spinner" aria-hidden="true"></span> 답변을 기다리는 중 — 도착하면 인용을 검증한다.</div>'
          : '<div class="callout"><span class="callout__title">검증할 답변 없음</span>검색 결과만 표시한다.</div>';
      return;
    }
    const t = result.stats;
    stats.innerHTML = [
      ['문장', t.sentences],
      ['인용 커버리지', t.coverage == null ? '—' : `${Math.round(t.coverage * 100)}%`],
      ['없는 번호', t.invalid],
      ['근거 불일치', t.unsupported],
    ]
      .map(([l, v]) => `<div class="stat"><span class="stat__label">${l}</span><span class="stat__value">${v}</span></div>`)
      .join('');
    const refusal = result.sentences.length > 0 && result.sentences.every((s) => s.refusal);
    const problems = result.sentences.filter((s) => s.status === 'fail' || s.status === 'warn');
    let html;
    if (refusal) {
      html = `<div class="callout callout--ok"><span class="callout__title">거절 응답 — 인용이 필요 없다</span>
        근거가 없을 때 “찾을 수 없다”고 답하는 것은 올바른 동작이다. UI는 이것을 오류가 아니라 정상 응답으로 보여 줘야 한다.</div>`;
    } else if (result.ok) {
      html = `<div class="callout callout--ok"><span class="callout__title">모든 문장이 출처로 뒷받침된다</span>
        ${t.claims}개 문장 모두 존재하는 출처 번호를 인용했고, 인용한 청크에 핵심어가 들어 있다.</div>`;
    } else {
      const fail = problems.some((s) => s.status === 'fail');
      html = `<div class="callout ${fail ? 'callout--danger' : ''}"><span class="callout__title">${fail ? '인용 검증 실패' : '확인이 필요한 문장'} — ${problems.length}개 문장</span>
        <ul class="w11-issues">${problems
          .map((s) => `<li><b>문장 ${s.index + 1}</b> ${s.issues.map((i) => escapeHtml(issueText(i, view.sources.length))).join(' · ')}</li>`)
          .join('')}</ul></div>`;
    }
    verdict.innerHTML = html + (note && kind !== 'nokey' ? `<p class="w11-note">${note}</p>` : '');
  }

  // ---- citation ↔ source linking
  function activate(n, sent, scroll) {
    clearActive();
    const btns = root.querySelectorAll(`.w11-cite[data-n="${n}"][data-sent="${sent}"]`);
    btns.forEach((b) => b.classList.add('is-active'));
    root.querySelector(`.w11-sent[data-sent="${sent}"]`)?.classList.add('is-active');
    const card = out.querySelector(`.w11-src[data-src="${n}"]`);
    const hint = $('[data-slot=hint]');
    const s = view?.result?.sentences[sent];
    if (!card) {
      hint.innerHTML = `<span class="w11-bad">[${n}]번 출처는 없다.</span> 검색된 출처는 ${view.sources.length}개뿐이다. 모델이 번호를 지어냈거나 출처 목록과 답변이 어긋났다.`;
      return;
    }
    card.classList.add('is-active');
    const marks = [...card.querySelectorAll('.w11-mark')].filter((m) => m.dataset.k.split(' ').includes(`s${sent}`));
    marks.forEach((m) => m.classList.add('is-active'));
    const span = s?.spans.find((sp) => sp.n === n);
    const src = view.sources.find((x) => x.n === n);
    hint.innerHTML = span
      ? `문장 ${sent + 1} ↔ [${n}] ${escapeHtml(src.title)} · 문서 위치 <b>${span.start}–${span.end}</b> (청크 ${src.start}–${src.end} 안)`
      : `<span class="w11-bad">문장 ${sent + 1}을 뒷받침하는 문장을 [${n}]에서 찾지 못했다.</span> 인용 번호가 엉뚱한 출처를 가리킨다.`;
    if (scroll) (marks[0] ?? card).scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  function clearActive() {
    root.querySelectorAll('.w11-cite.is-active, .w11-sent.is-active').forEach((x) => x.classList.remove('is-active'));
    out.querySelectorAll('.is-active').forEach((x) => x.classList.remove('is-active'));
  }

  function restorePinned() {
    if (pinned) activate(pinned.n, pinned.sent, false);
    else {
      clearActive();
      $('[data-slot=hint]').textContent = defaultHint(view?.sources.length ?? 0);
    }
  }

  const on = (target, type, fn) => target.addEventListener(type, fn, { signal: ctrl.signal });
  const citeOf = (e) => e.target.closest?.('.w11-cite');

  on(root, 'pointerover', (e) => {
    const b = citeOf(e);
    if (b && e.pointerType === 'mouse') activate(+b.dataset.n, +b.dataset.sent, !narrow());
  });
  on(root, 'pointerout', (e) => {
    const b = citeOf(e);
    if (b && e.pointerType === 'mouse' && !b.contains(e.relatedTarget)) restorePinned();
  });
  on(root, 'focusin', (e) => {
    const b = citeOf(e);
    if (b) activate(+b.dataset.n, +b.dataset.sent, !narrow());
  });
  on(root, 'focusout', (e) => {
    if (citeOf(e)) restorePinned();
  });
  on(root, 'click', (e) => {
    const b = citeOf(e);
    if (b) {
      const n = +b.dataset.n;
      const sent = +b.dataset.sent;
      const same = pinned && pinned.n === n && pinned.sent === sent;
      root.querySelectorAll('.w11-cite[aria-pressed="true"]').forEach((x) => x.setAttribute('aria-pressed', 'false'));
      pinned = same ? null : { n, sent };
      if (pinned) b.setAttribute('aria-pressed', 'true');
      if (pinned) activate(n, sent, true);
      else clearActive();
      return;
    }
    const fb = e.target.closest('[data-fb]');
    if (fb) {
      fb.parentElement.querySelectorAll('[data-fb]').forEach((x) => x.setAttribute('aria-pressed', String(x === fb)));
      root.querySelector('[data-slot=fbnote]').textContent =
        fb.dataset.fb === 'up'
          ? '피드백 기록: 도움됨. 실제 서비스라면 질문·답변·출처 id와 함께 서버에 저장한다 (이 데모는 저장하지 않는다).'
          : '피드백 기록: 도움 안 됨. 어떤 문장이 틀렸는지 함께 받으면 골든셋(9주차)에 추가할 수 있다.';
      return;
    }
    const preset = e.target.closest('[data-preset]');
    if (preset) {
      exSel.value = preset.dataset.preset;
      showRecorded(preset.dataset.preset, true);
    }
  });

  on(exSel, 'change', () => showRecorded(exSel.value, true));
  on(provSel, 'change', () => {
    fillModels();
    syncKeyState();
  });
  on($('[data-act=save-key]'), 'click', () => {
    const input = $('[data-in=key]');
    const v = input.value.trim();
    input.value = ''; // never keep the key in the DOM
    if (!v) return;
    try {
      setKey(provSel.value, v);
    } catch {
      $('[data-slot=keystate]').textContent = '이 브라우저에서는 sessionStorage를 쓸 수 없어 키를 저장하지 못했다.';
      return;
    }
    syncKeyState();
  });
  on($('[data-act=clear-key]'), 'click', () => {
    try {
      clearKey(provSel.value);
    } catch {
      /* storage unavailable */
    }
    syncKeyState();
  });
  on($('[data-slot=ask]'), 'submit', (e) => {
    e.preventDefault();
    const q = $('[data-in=question]').value.trim();
    if (q) ask(q);
  });

  function showRecorded(id, animate) {
    const it = byId.get(id);
    if (!it) return;
    $('[data-in=question]').value = it.question;
    const sources = withScores(it.question, sourcesFor(it));
    const live = retrieve(it.question).map((r) => r.id).join(',');
    const drift = live !== it.sources.join(',')
      ? '⚠ 저장 당시의 검색 결과와 지금 검색 결과가 다르다. 저장된 답변은 저장 당시 출처 기준으로 표시한다.'
      : '';
    const badge = `<span class="chip ${it.kind === 'bad' ? 'warn' : 'accent'}">${RECORDED_LABEL}</span>${
      it.kind === 'bad' ? ' <span class="chip warn">일부러 망가뜨린 답변</span>' : ''}`;
    return showTurn({ question: it.question, answer: it.answer, sources, badge, kind: 'recorded', note: drift, animate });
  }

  async function ask(question) {
    const sources = retrieve(question).map((s, i) => ({ ...s, n: i + 1 }));
    const provider = provSel.value;
    if (!sources.length) {
      return showTurn({
        question, answer: '문서에서 찾을 수 없다. 검색된 문서가 하나도 없다.', sources,
        badge: '<span class="chip">검색 결과 없음 · LLM 호출 안 함</span>', kind: 'recorded', animate: true,
      });
    }
    if (!safeHasKey(provider)) {
      const match = items.find((it) => it.kind === 'good' && it.question === question);
      if (match) {
        exSel.value = match.id;
        return showRecorded(match.id, true);
      }
      return showTurn({
        question, answer: '', sources, kind: 'nokey',
        badge: '<span class="chip warn">🔑 키 없음 · 답변 생성 안 함</span>',
        note: '<div class="callout"><span class="callout__title">검색까지만 했다</span>API 키가 없어 답변을 만들지 않았다. 오른쪽 출처는 BM25로 실제 검색한 결과다. 예시 질문을 고르거나 “LLM 설정”에서 키를 입력한다.</div>',
      });
    }
    // with a key: numbered sources → generate → parse citations
    st.llm?.abort();
    const llm = new AbortController();
    st.llm = llm;
    const job = {};
    st.job = job;
    const chat = $('[data-slot=chat]');
    chat.innerHTML = `<div class="w11-msg w11-msg--user">${escapeHtml(question)}</div>
      <div class="w11-msg w11-msg--bot"><div class="widget__status" role="status"><span class="spinner" aria-hidden="true"></span> 출처 ${sources.length}개로 답변 생성 중…</div></div>`;
    renderSources(sources, null);
    renderVerdict('pending', null);
    const model = $('[data-in=model]').value;
    try {
      const { system, messages } = ragPrompt(question, sources);
      const res = await generate({ provider, model, system, messages, maxTokens: 600, temperature: 0.2, signal: llm.signal });
      if (st.job !== job) return;
      const text = (res.text ?? '').trim();
      if (!text) throw new LLMError('모델이 빈 응답을 돌려줬다. 다시 시도한다.', 'empty');
      await showTurn({
        question, answer: text, sources, animate: true, kind: 'llm',
        badge: `<span class="chip ok">LLM 생성 · ${escapeHtml(model)}</span>`,
      });
    } catch (err) {
      if (err.name === 'AbortError' || st.job !== job) return;
      const msg = err instanceof LLMError ? err.message : '알 수 없는 오류로 답변을 만들지 못했다.';
      await showTurn({
        question, answer: '', sources, kind: 'error',
        badge: '<span class="chip warn">오류</span>',
        note: `<div class="widget__error" role="alert">${escapeHtml(msg)}<br>출처는 그대로 보여 준다. 키 없이도 예시 질문은 동작한다.</div>`,
      });
    }
  }

  // ---- first render: already shows an answer with live citations
  syncKeyState();
  const first = byId.has(options.item) ? options.item : 'q01';
  exSel.value = first;
  $('[data-slot=status]').hidden = true;
  $('[data-slot=body]').hidden = false;
  registerOutput(outputId, {
    title: options.outputTitle ?? '인용 검증 · 출처',
    node: out,
    inlineHost: $('[data-slot=out-inline]'),
  });
  showRecorded(first, false);
}

export function unmount(el) {
  const st = state.get(el);
  if (st) {
    st.ctrl.abort();
    st.llm?.abort();
    st.job = null;
    clearInterval(st.timer);
    unregisterOutput(st.outputId);
  }
  state.delete(el);
  el.replaceChildren();
}

// ------------------------------------------------------------------ small helpers

function defaultHint(count) {
  return count ? '답변의 [번호]에 마우스를 올리거나 Tab으로 이동하면 근거 구간이 표시된다. 누르면 고정된다.' : '';
}

function issueLabel(i) {
  switch (i.type) {
    case 'invalid': return `없는 출처 [${i.n}]`;
    case 'uncited': return '인용 없음';
    case 'unsupported': return '출처와 불일치';
    case 'weak': return '확인 필요';
    default: return i.type;
  }
}

function issueText(i, count) {
  switch (i.type) {
    case 'invalid': return `[${i.n}]은 없는 번호다 (출처 ${count}개)`;
    case 'uncited': return '어느 출처도 인용하지 않은 주장이다';
    case 'unsupported': return `인용한 [${i.n.join(', ')}]에 핵심어가 없다: ${i.missing.join(', ')}`;
    case 'weak': return `출처에 없는 단어: ${i.missing.join(', ')}`;
    default: return i.type;
  }
}

function feedbackHtml() {
  return `<div class="w11-feedback">
    <span>이 답변이 도움이 되었나?</span>
    <button type="button" class="btn small ghost" data-fb="up" aria-pressed="false"><span aria-hidden="true">👍</span> 도움됨</button>
    <button type="button" class="btn small ghost" data-fb="down" aria-pressed="false"><span aria-hidden="true">👎</span> 아니다</button>
    <span class="w11-fbnote" data-slot="fbnote" role="status"></span>
  </div>`;
}

function typewriter(el, text, cancelled, st) {
  return new Promise((resolve) => {
    let i = 0;
    el.textContent = '';
    clearInterval(st.timer);
    st.timer = setInterval(() => {
      if (cancelled()) {
        clearInterval(st.timer);
        resolve();
        return;
      }
      i = Math.min(text.length, i + 3);
      el.textContent = text.slice(0, i);
      if (i >= text.length) {
        clearInterval(st.timer);
        resolve();
      }
    }, 16);
  });
}

// Below 860px the sources render under the widget: only scroll to them on an explicit tap/click.
function narrow() {
  return typeof matchMedia === 'function' && matchMedia('(max-width: 860px)').matches;
}

function prefersReducedMotion() {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function safeHasKey(provider) {
  try {
    return hasKey(provider);
  } catch {
    return false;
  }
}

async function fetchJson(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${res.status} ${url.pathname}`);
  return res.json();
}
