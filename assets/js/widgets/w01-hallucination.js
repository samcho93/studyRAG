// w01 환각 체험
// One concept: an LLM asked about our own department without context produces a
// plausible but unverifiable (often wrong) answer; the same LLM with the
// evidence document pasted into the prompt answers correctly and can be checked.

import { PROVIDERS, setKey, hasKey, clearKey, generate, ragPrompt, LLMError } from '../core/llm.js';
import { registerOutput, unregisterOutput } from '../site/result.js';

const DATA = new URL('../../data/', import.meta.url);
const QUESTION_IDS = ['q01', 'q02', 'q03']; // answers exist only in our corpus

const SYSTEM_NO_CONTEXT = '질문에 한국어로 간결하게 답한다.';

// Pre-recorded example answers, shown without an API key. They are labelled as
// examples in the UI. A = typical no-context answer (plausible, wrong), B = grounded.
const EXAMPLES = {
  q01: {
    A: '대학 실습실의 GPU 서버는 공정한 배분을 위해 보통 1회 4시간 정도로 사용 시간을 제한한다. 학교마다 규정이 다르므로 정확한 시간은 학과 사무실에 확인하는 것이 좋다.',
    B: '한 사람이 GPU 서버를 연속으로 사용할 수 있는 시간은 최대 12시간이다 [1]. 장시간 작업은 공용 캘린더에 사용 시간을 먼저 등록해야 한다 [1].',
  },
  q02: {
    A: '프로젝트 중심 교과목에서 팀 프로젝트는 일반적으로 성적의 30% 안팎을 차지한다. 나머지는 중간·기말고사와 과제로 구성되는 경우가 많다.',
    B: '팀 프로젝트는 성적의 40%를 차지한다 [1]. 나머지는 주차별 실습 과제 30%, 중간 점검 20%, 수업 참여 10%다 [1].',
  },
  q03: {
    A: '일반적으로 출석률 90% 이상, 모든 과제 제출, 최종 시험 합격이 수료 요건이다.',
    B: '수료 요건은 출석률 80% 이상과 팀 프로젝트 최종 발표 통과다 [1].',
  },
};

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w01-hal">
    <h3 class="widget__title">환각 체험</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> 문서셋과 골든셋을 불러오는 중…
    </div>
    <div data-slot="body" hidden>
      <div class="widget__controls">
        <label class="field">
          <span class="field__label">질문 (답은 우리 학과 문서에만 있다)</span>
          <select data-in="q"></select>
        </label>
      </div>
      <details>
        <summary>B에 붙여 넣는 근거 문서 보기</summary>
        <div class="w01-evidence" data-slot="evidence" tabindex="0" aria-label="근거 문서 원문"></div>
      </details>
      <details class="w01-key" data-slot="key-panel">
        <summary>내 API 키로 직접 비교하기 (선택)</summary>
        <div class="widget__controls">
          <label class="field">
            <span class="field__label">공급자</span>
            <select data-in="provider"></select>
          </label>
          <label class="field">
            <span class="field__label">모델</span>
            <select data-in="model"></select>
          </label>
          <label class="field">
            <span class="field__label">API 키 (이 탭에만 저장)</span>
            <input type="password" data-in="key" autocomplete="off" spellcheck="false" placeholder="키를 붙여 넣고 저장">
          </label>
        </div>
        <div class="btn-row">
          <button type="button" class="btn small" data-act="save">키 저장</button>
          <button type="button" class="btn ghost small" data-act="clear">키 지우기</button>
          <button type="button" class="btn primary small" data-act="run">▶ A · B 동시에 묻기</button>
        </div>
        <div class="widget__status" data-slot="key-status" role="status" aria-live="polite"></div>
      </details>
      <div data-slot="out-inline"></div>
    </div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w01-hal-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <p class="w01-golden" data-slot="golden"></p>
    <div class="w01-hal-cards">
      <section class="w01-card" data-card="A" aria-label="A 문맥 없이"></section>
      <section class="w01-card" data-card="B" aria-label="B 근거 문서와 함께"></section>
    </div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ question?: string, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w01-hal:${++seq}`;
  const st = { ctrl, outputId, run: null };
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

  const items = QUESTION_IDS.map((id) => golden.items.find((it) => it.id === id)).filter(Boolean);
  const docOf = (item) => corpus.documents.find((d) => d.id === item.evidence[0].doc);
  const s = {
    q: items.some((it) => it.id === options.question) ? options.question : items[0].id,
    provider: Object.keys(PROVIDERS)[0],
    live: {}, // qid -> { A, B } live results { text, model } | { error }
    running: false,
  };

  $('[data-in=q]').innerHTML = items.map((it) => `<option value="${it.id}">${escapeHtml(it.question)}</option>`).join('');
  $('[data-in=q]').value = s.q;
  $('[data-in=provider]').innerHTML = Object.entries(PROVIDERS)
    .map(([k, p]) => `<option value="${k}">${escapeHtml(p.label)}</option>`)
    .join('');

  function fillModels() {
    const p = PROVIDERS[s.provider];
    $('[data-in=model]').innerHTML = p.models.map((m) => `<option value="${m}">${escapeHtml(m)}</option>`).join('');
    $('[data-in=model]').value = p.defaultModel;
  }

  function prompts(item) {
    const doc = docOf(item);
    const b = ragPrompt(item.question, [{ text: doc.text }]);
    return {
      A: { system: SYSTEM_NO_CONTEXT, messages: [{ role: 'user', content: item.question }] },
      B: b,
    };
  }

  function syncKeyUi(message) {
    const ok = safeHasKey(s.provider);
    $('[data-act=run]').disabled = !ok || s.running;
    $('[data-act=clear]').disabled = !ok;
    $('[data-slot=key-status]').innerHTML =
      message ??
      (ok
        ? `<span class="badge-ok">키 있음</span> ${escapeHtml(PROVIDERS[s.provider].label)} 키가 이 탭에 저장되어 있다. 탭을 닫으면 사라진다.`
        : '키가 없으면 아래 결과는 <b>예시 응답</b>이다. 키를 저장하면 같은 두 프롬프트를 실제 LLM에 보낸다.');
  }

  function render() {
    const item = items.find((it) => it.id === s.q);
    const doc = docOf(item);
    const quote = item.evidence[0].quote;
    $('[data-slot=evidence]').innerHTML = markQuote(doc.text, quote);
    $('[data-slot=golden]').innerHTML = `<b>골든 정답</b> ${escapeHtml(item.answer)} <span class="w01-muted">(golden.json ${item.id} · 근거: ${escapeHtml(doc.title)})</span>`;

    const p = prompts(item);
    const live = s.live[item.id];
    const results = {};
    for (const side of ['A', 'B']) {
      const r = live?.[side];
      results[side] = r
        ? { ...r, live: true }
        : { text: EXAMPLES[item.id][side], live: false };
      renderCard(side, item, p[side], results[side]);
    }
    $('[data-slot=verdict]').innerHTML = verdictHtml(item, results);
  }

  function renderCard(side, item, prompt, r) {
    const card = $(`[data-card=${side}]`);
    const promptText = `${prompt.system ? `[system]\n${prompt.system}\n\n` : ''}[user]\n${prompt.messages[0].content}`;
    let body;
    let check = null;
    if (r.pending) {
      body = '<p class="w01-answer"><span class="spinner w01-inline-spin" aria-hidden="true"></span> 응답을 기다리는 중…</p>';
    } else if (r.error) {
      body = `<div class="widget__error" role="alert">${escapeHtml(r.error)}</div>`;
    } else {
      check = containsGoldenFact(r.text, item.answer);
      body = `<p class="w01-answer">${markAnswer(r.text, check)}</p>`;
    }
    card.classList.toggle('is-ok', Boolean(check?.ok));
    card.classList.toggle('is-bad', Boolean(check && !check.ok));
    const sourceChip = r.live
      ? `<span class="chip accent">실제 응답 · ${escapeHtml(r.model ?? '')}</span>`
      : '<span class="chip warn">예시 응답 · 수업용 작성 예시</span>';
    const factChip = check
      ? check.ok
        ? `<span class="chip ok">✓ 골든 사실 포함 (${escapeHtml(check.facts.join(', '))})</span>`
        : `<span class="chip w01-chip-bad">✗ 골든 사실 없음 (${escapeHtml(check.facts.join(', '))})</span>`
      : '';
    const groundChip = side === 'A'
      ? '<span class="chip w01-chip-bad">근거 없음 · 검증 불가</span>'
      : '<span class="chip ok">근거 문서 1개 · 대조 가능</span>';
    card.innerHTML = `
      <div class="w01-card__head">${side === 'A' ? 'A · 문맥 없이 묻기' : 'B · 근거 문서를 붙여 묻기'}</div>
      <div class="w01-card__chips">${sourceChip}${groundChip}${factChip}</div>
      ${body}
      <details><summary>보낸 프롬프트 (${[...promptText].length.toLocaleString()}자)</summary><pre>${escapeHtml(promptText)}</pre></details>`;
  }

  async function runLive() {
    const item = items.find((it) => it.id === s.q);
    const model = $('[data-in=model]').value;
    const provider = s.provider;
    const p = prompts(item);
    st.run?.abort();
    const runCtrl = new AbortController();
    st.run = runCtrl;
    s.running = true;
    s.live[item.id] = { A: { pending: true, live: true, model }, B: { pending: true, live: true, model } };
    syncKeyUi('<span class="spinner w01-inline-spin" aria-hidden="true"></span> 두 프롬프트를 동시에 보내는 중…');
    render();
    await Promise.all(
      ['A', 'B'].map(async (side) => {
        try {
          const res = await generate({ provider, model, ...p[side], maxTokens: 300, signal: runCtrl.signal });
          s.live[item.id][side] = { text: res.text.trim() || '(빈 응답)', model };
        } catch (err) {
          if (err.name === 'AbortError') return;
          s.live[item.id][side] = { error: friendlyError(err), model };
        }
        if (!ctrl.signal.aborted && s.q === item.id) render();
      }),
    );
    if (ctrl.signal.aborted || runCtrl.signal.aborted) return;
    s.running = false;
    syncKeyUi();
    render();
  }

  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  on('[data-in=q]', 'change', (e) => {
    s.q = e.target.value;
    render();
  });
  on('[data-in=provider]', 'change', (e) => {
    s.provider = e.target.value;
    fillModels();
    syncKeyUi();
  });
  on('[data-act=save]', 'click', () => {
    const input = $('[data-in=key]');
    const value = input.value.trim();
    input.value = ''; // never keep the key in the DOM
    if (!value) {
      syncKeyUi('키 입력란이 비어 있다.');
      return;
    }
    try {
      setKey(s.provider, value);
      syncKeyUi();
    } catch {
      syncKeyUi('<span class="w01-muted">이 브라우저에서는 sessionStorage를 쓸 수 없어 키를 저장하지 못했다.</span>');
    }
  });
  on('[data-in=key]', 'keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      $('[data-act=save]').click();
    }
  });
  on('[data-act=clear]', 'click', () => {
    try {
      clearKey(s.provider);
    } catch {
      /* storage unavailable */
    }
    syncKeyUi();
  });
  on('[data-act=run]', 'click', () => {
    if (!s.running) runLive();
  });

  fillModels();
  syncKeyUi();
  render();
  $('[data-slot=status]').hidden = true;
  $('[data-slot=body]').hidden = false;
  registerOutput(outputId, {
    title: options.outputTitle ?? '환각 체험 결과',
    node: out,
    inlineHost: $('[data-slot=out-inline]'),
  });
}

export function unmount(el) {
  const st = state.get(el);
  if (st) {
    st.ctrl.abort();
    st.run?.abort();
    unregisterOutput(st.outputId);
  }
  state.delete(el);
  el.replaceChildren();
}

// ---------- pure helpers (exported for tests) ----------

const CLAIM_RE = /\d+(?:\.\d+)?\s*(?:%|퍼센트|시간|교시|학점|호|명|주|회)?/g;

export function normalize(s) {
  return String(s).normalize('NFKC').toLowerCase().replace(/퍼센트/g, '%').replace(/\s+/g, '');
}

/** Key facts of a golden answer: numbers with units ("12시간", "40%"); otherwise the whole answer. */
export function goldenFacts(answer) {
  const nums = (String(answer).match(CLAIM_RE) ?? []).map(normalize);
  if (nums.length) return [...new Set(nums)];
  return [normalize(answer).replace(/[.。!]+$/, '').replace(/다$/, '')];
}

/** Does `text` contain every key fact of the golden answer? "112시간" does not count as "12시간". */
export function containsGoldenFact(text, goldenAnswer) {
  const t = normalize(text);
  const facts = goldenFacts(goldenAnswer);
  const found = facts.filter((f) => new RegExp(`(?<!\\d)${escapeRe(f)}${/\d$/.test(f) ? '(?!\\d)' : ''}`).test(t));
  return { ok: found.length === facts.length, facts, found };
}

function markAnswer(text, check) {
  let html = '';
  let last = 0;
  for (const m of text.matchAll(CLAIM_RE)) {
    const hit = check.found.includes(normalize(m[0]));
    const cls = hit ? 'fact-ok' : check.ok ? '' : 'fact-bad';
    html += escapeHtml(text.slice(last, m.index));
    html += cls ? `<mark class="${cls}">${escapeHtml(m[0])}</mark>` : escapeHtml(m[0]);
    last = m.index + m[0].length;
  }
  return html + escapeHtml(text.slice(last));
}

function markQuote(text, quote) {
  const i = text.indexOf(quote);
  if (i < 0) return escapeHtml(text);
  return `${escapeHtml(text.slice(0, i))}<mark>${escapeHtml(quote)}</mark>${escapeHtml(text.slice(i + quote.length))}`;
}

function verdictHtml(item, results) {
  const a = results.A;
  const b = results.B;
  if (a.pending || b.pending || a.error || b.error) return '';
  const okA = containsGoldenFact(a.text, item.answer).ok;
  const okB = containsGoldenFact(b.text, item.answer).ok;
  if (!okA && okB) {
    return `<div class="callout callout--danger"><span class="callout__title">A는 그럴듯하지만 틀렸다 — 같은 모델, 다른 입력</span>
      A는 우리 학과 규정을 본 적이 없어 “일반적인” 값을 지어내거나 확인을 미룬다. 어디서 나온 숫자인지 알 수 없으니 검증할 방법도 없다.
      B는 같은 질문에 근거 문서를 붙였을 뿐인데 골든 정답과 일치하고, 근거 문장과 대조할 수 있다.</div>`;
  }
  if (okA && okB) {
    return `<div class="callout callout--more"><span class="callout__title">둘 다 골든 사실을 포함했다</span>
      A가 맞았더라도 근거가 없으니 우연인지 알 수 없다. 질문을 바꿔 여러 번 비교해 본다.</div>`;
  }
  if (!okB) {
    return `<div class="callout callout--more"><span class="callout__title">B도 골든 사실을 놓쳤다</span>
      근거를 넣어도 답이 틀릴 수 있다. 프롬프트(2주차)와 검색 품질(9주차 평가)이 필요한 이유다.</div>`;
  }
  return '';
}

function friendlyError(err) {
  if (err instanceof LLMError) return err.message;
  return 'LLM 호출 중 알 수 없는 오류가 났다. 잠시 뒤 다시 시도한다.';
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

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
