// 응용 1 학과 안내 챗봇 — 멀티턴 대화형 RAG
// One concept: a follow-up question ("주말에는?") cannot be retrieved on its
// own. It has to be rewritten (condensed) into a standalone question using the
// conversation history. Every turn is retrieved twice, side by side:
//   (A) the raw follow-up only, (B) the rewritten standalone question,
// and the widget shows whether the evidence chunk made it into the top 3.
// Without an API key the rewrite is a transparent rule-based one; with a key an
// LLM condense prompt (shown in full) can be run via core/llm generate().

import { chunk, sentenceSpans } from '../core/chunker.js';
import { BM25, tokenize } from '../core/bm25.js';
import { VectorStore } from '../core/vectorstore.js';
import { DEFAULT_MODEL, loadModelWithUI, embed, embedBatch } from '../core/embed.js';
import { PROVIDERS, setKey, hasKey, clearKey, generate, ragPrompt, LLMError } from '../core/llm.js';
import { registerOutput, unregisterOutput, focusOutput } from '../site/result.js';

const DATA = new URL('../../data/', import.meta.url);
const RECORDED_LABEL = '예시 응답 · 수업용 작성 예시';
const MAX_WINDOW = 5;

// ------------------------------------------------------------------ pure helpers
// (exported so they can be tested in Node and reused in Challenge answers)

/** Topic dictionary of the rule-based rewriter. Only these words are recognised as topics. */
export const ENTITIES = [
  '성적 우수 장학금', '가계 곤란 장학금', '근로 장학금', '장학금',
  'RAG 과목', 'RAG', '검색증강생성', '딥러닝', '자연어 처리', 'MLOps', '파이썬 프로그래밍', '머신러닝 기초', '데이터 분석',
  '실습실', 'GPU 서버', '도서관', '스터디룸', '열람실', '기숙사', '통학 버스', '학과 행정실', '행정실',
  '휴학', '수강 정정', '증명서', '시간표', '수료', '노트북', '결석',
];
export const PRONOUNS = ['그 두 과목', '그 과목', '그 장학금', '거기서', '거기는', '거기에', '거기', '그곳', '그것', '그거', '그건', '그게'];
export const MARKERS = ['그러면', '그럼', '그리고', '근데', '그래서', '또'];
const PARTICLES = ['에서는', '에서도', '까지는', '으로는', '에서', '에는', '에도', '까지', '부터', '으로', '이야', '이랑', '한테', '에게',
  '은', '는', '이', '가', '을', '를', '에', '의', '도', '만', '로', '와', '과', '야'];
const STOP = new Set(['몇', '언제', '언제까지', '어디', '어디로', '어디서', '어디에', '무엇', '뭐', '뭐야', '누가', '누구', '얼마', '얼마나',
  '어떻게', '왜', '무슨', '어느', '해', '해야', '돼', '되나', '있어', '있나', '열어', '열려', '받아', '받을', '받나', '수', '내야', '알려줘',
  '마셔', '마셔도', '가져와', '먹어', '배워', '배우나', '하나', '하면', '말이야', '말', '내', '제', '좀', '다시', '있는지', '해도', '되는지',
  '가능해', '돼요', '해요', '줘', '거야', '건가']);

const reEscape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const SORTED_ENTITIES = [...ENTITIES].sort((a, b) => b.length - a.length);

/** Entities mentioned in text (longest alias first, no overlaps), in reading order. */
export function findEntities(text) {
  const found = [];
  const taken = [];
  for (const e of SORTED_ENTITIES) {
    const re = new RegExp(reEscape(e).replace(/ /g, '\\s*'), 'gi');
    for (const m of String(text).matchAll(re)) {
      const s = m.index;
      const t = s + m[0].length;
      if (taken.some(([a, b]) => s < b && t > a)) continue;
      taken.push([s, t]);
      found.push({ entity: e, surface: m[0], index: s });
    }
  }
  return found.sort((a, b) => a.index - b.index);
}

export function stripParticle(w) {
  for (const p of PARTICLES) if (w.length - p.length >= 2 && w.endsWith(p)) return w.slice(0, -p.length);
  return w;
}

/** Content words: particles stripped; question words, verbs, pronouns and markers removed. */
export function contentWords(text) {
  const out = [];
  for (const [tok] of String(text).matchAll(/[가-힣A-Za-z0-9]+/g)) {
    if (STOP.has(tok) || MARKERS.includes(tok) || PRONOUNS.includes(tok)) continue;
    const w = stripParticle(tok);
    if (w.length < 2 || STOP.has(w)) continue;
    out.push(w);
  }
  return out;
}

export function findPronoun(text) {
  for (const p of PRONOUNS) {
    const i = String(text).indexOf(p);
    if (i >= 0) return { pronoun: p, index: i };
  }
  return null;
}

export function leadingMarker(text) {
  const t = String(text).trim();
  return MARKERS.find((m) => t === m || t.startsWith(`${m} `)) ?? null;
}

function withoutEntities(text) {
  let rest = String(text);
  for (const e of findEntities(rest)) rest = rest.replace(e.surface, ' ');
  return rest;
}

/**
 * Rule-based condense (no LLM).
 * @param {string[]} history previous user questions, oldest first (already PII-masked)
 * @param {string} question the new user question
 * @param {{ window?: number }} opts how many previous user turns the rewriter may look at
 * @returns {{ query: string, changed: boolean, followUp: boolean, clarify: boolean,
 *   steps: { type: string, text: string }[], looked: number, hidden: number }}
 */
export function ruleRewrite(history, question, { window = 3 } = {}) {
  const q = String(question).trim();
  const steps = [];
  const visible = window > 0 ? history.slice(-window) : [];
  const marker = leadingMarker(q);
  const pron = findPronoun(q);
  const ents = findEntities(q);
  const rest = contentWords(withoutEntities(q));
  const followUp = Boolean(pron || marker || (ents.length === 0 && rest.length <= 2));
  const base = { query: q, changed: false, followUp, clarify: false, steps, looked: visible.length, hidden: history.length - visible.length };

  if (!followUp) {
    steps.push({ type: 'standalone', text: ents.length ? `주제어 “${ents.map((e) => e.surface).join(', ')}”가 있는 독립 질문 → 그대로 검색` : '생략이 없는 독립 질문 → 그대로 검색' });
    return base;
  }

  let body = q;
  if (marker) {
    body = body.slice(marker.length).trim();
    steps.push({ type: 'marker', text: `접속어 “${marker}” 제거` });
  }

  // the most recent previous user turn (inside the window) that names a topic
  let topic = null;
  for (let i = visible.length - 1; i >= 0; i--) {
    const e = findEntities(visible[i]);
    if (e.length) {
      topic = { surface: e[e.length - 1].surface, turnsAgo: visible.length - i };
      break;
    }
  }

  // Rule 3 — topic switch: the question names only a new topic ("그럼 딥러닝은?") → reuse the previous attribute
  if (ents.length && !pron) {
    const prev = visible[visible.length - 1];
    if (rest.length === 0 && prev) {
      let attr = prev;
      const pm = leadingMarker(attr);
      if (pm) attr = attr.slice(pm.length);
      const pp = findPronoun(attr);
      if (pp) attr = attr.replace(pp.pronoun, ' ');
      attr = withoutEntities(attr).replace(/\s+/g, ' ').trim();
      if (contentWords(attr).length) {
        const query = `${ents.map((e) => e.surface).join(' ')} ${attr}`;
        steps.push({ type: 'attr', text: `주제만 바뀐 질문 → 바로 앞 질문의 나머지 “${attr}”를 붙임` });
        return { ...base, query, changed: true };
      }
    }
    steps.push({ type: 'standalone', text: `주제어 “${ents.map((e) => e.surface).join(', ')}”가 있어 앞 주제를 붙이지 않음` });
    return { ...base, query: body, changed: body !== q };
  }

  // Rule 1 — pronoun → the most recent topic
  if (pron) {
    if (!topic) {
      steps.push({ type: 'fail', text: `대명사 “${pron.pronoun}”가 가리킬 주제어를 ${visible.length ? `최근 ${visible.length}턴` : '대화 기록'}에서 찾지 못함 → 되묻기` });
      return { ...base, clarify: true };
    }
    body = body.replace(pron.pronoun, topic.surface).replace(/\s+/g, ' ').trim();
    steps.push({ type: 'pronoun', text: `대명사 “${pron.pronoun}” → “${topic.surface}” (${topic.turnsAgo}턴 전 질문의 주제어)` });
    return { ...base, query: body, changed: true };
  }

  // Rule 2 — ellipsis → prepend the topic (+ key words of the previous follow-up)
  const carry = [];
  if (topic) {
    carry.push(topic.surface);
    steps.push({ type: 'carry', text: `주제어 “${topic.surface}” 이어 붙임 (${topic.turnsAgo}턴 전 질문)` });
  }
  // key words of the previous follow-up ("주말") only travel together with a topic
  const prev = visible[visible.length - 1];
  if (topic && prev && findEntities(prev).length === 0) {
    const kw = contentWords(prev).filter((w) => !body.includes(w) && !carry.includes(w));
    if (kw.length) {
      carry.push(...kw);
      steps.push({ type: 'carry', text: `바로 앞 질문의 핵심어 “${kw.join(', ')}” 이어 붙임` });
    }
  }
  if (!carry.length) {
    if (marker) {
      steps.push({ type: 'fail', text: `${visible.length ? `최근 ${visible.length}턴` : '대화 기록'}에서 이어 붙일 주제어를 찾지 못함 → 되묻기` });
      return { ...base, query: body, clarify: true };
    }
    steps.push({ type: 'fail', text: history.length ? `최근 ${visible.length}턴에서 이어 붙일 주제어를 찾지 못함 → 그대로 검색` : '대화 기록이 없음 → 그대로 검색' });
    return base;
  }
  return { ...base, query: `${carry.join(' ')} ${body}`, changed: true };
}

// ---- personal information: masked before it enters the history or a prompt
export const PII_RULES = [
  { name: '전화번호', re: /01[016789][-\s]?\d{3,4}[-\s]?\d{4}/g },
  { name: '주민등록번호', re: /\d{6}[-\s]?[1-4]\d{6}/g },
  { name: '이메일', re: /[\w.+-]+@[\w-]+\.[\w.-]+/g },
  { name: '학번', re: /(?<!\d)20\d{8}(?!\d)/g },
];

export function maskPII(text) {
  const found = [];
  let out = String(text);
  for (const { name, re } of PII_RULES) {
    out = out.replace(re, () => {
      found.push(name);
      return `[${name}]`;
    });
  }
  return { text: out, found };
}

export function stripCitations(text) {
  return String(text).replace(/\s*\[\d+(?:\s*,\s*\d+)*\]/g, '').replace(/\s{2,}/g, ' ').trim();
}

/** Rough token estimate used in this lab: Korean ≈ 1 token per character (tokenizers differ). */
export const estimateTokens = (text) => String(text).replace(/\s+/g, '').length;

export const CONDENSE_SYSTEM =
  '너는 학과 안내 챗봇의 질문 재작성기다. 대화 기록과 마지막 질문을 읽고, 기록 없이도 뜻이 통하는 검색용 독립 질문 한 문장으로 다시 쓴다.\n' +
  '규칙: 1) 대명사(거기, 그거)와 생략된 주제를 기록에 나온 구체적인 이름으로 바꾼다. 2) 마지막 질문이 이미 독립적이면 그대로 쓴다. ' +
  "3) 답하지 말고 질문만 출력한다. 4) 기록으로도 무엇을 묻는지 알 수 없으면 '되묻기: '로 시작하는 확인 질문을 쓴다. " +
  '5) 학번·전화번호 같은 개인정보는 질문에 넣지 않는다.';

/**
 * Condense prompt for an LLM rewriter.
 * @param {{ user: string, bot?: string }[]} turns previous turns, oldest first (PII-masked)
 * @param {string} question
 * @param {{ window?: number, botChars?: number }} opts
 */
export function buildCondensePrompt(turns, question, { window = 3, botChars = 160 } = {}) {
  const visible = window > 0 ? turns.slice(-window) : [];
  const lines = [];
  for (const t of visible) {
    lines.push(`사용자: ${t.user}`);
    if (t.bot) {
      const b = stripCitations(t.bot);
      lines.push(`챗봇: ${b.length > botChars ? `${b.slice(0, botChars)}…` : b}`);
    }
  }
  const history = lines.length ? `대화 기록 (최근 ${visible.length}턴):\n${lines.join('\n')}` : '대화 기록: (없음)';
  const content = `${history}\n\n마지막 질문: ${question}\n\n독립 질문:`;
  return {
    system: CONDENSE_SYSTEM,
    messages: [{ role: 'user', content }],
    included: visible.length,
    dropped: turns.length - visible.length,
    chars: CONDENSE_SYSTEM.length + content.length,
    historyChars: lines.join('\n').length,
  };
}

/** Clean an LLM rewrite: first line, no label/quotes. Detect "되묻기:". */
export function parseRewrite(text) {
  let line = String(text).trim().split(/\n/).find((l) => l.trim()) ?? '';
  line = line.replace(/^(독립 질문|재작성|질문)\s*[:：]\s*/, '').replace(/^["“'‘]|["”'’]$/g, '').trim();
  const clarify = /^되묻기\s*[:：]/.test(line);
  return { query: clarify ? line.replace(/^되묻기\s*[:：]\s*/, '') : line, clarify };
}

/** 1-based rank of the first hit whose chunk contains the evidence quote, or 0. */
export function evidenceRank(hits, evidence) {
  if (!evidence) return null;
  const i = hits.findIndex((h) => h.doc === evidence.doc && h.raw.includes(evidence.quote));
  return i + 1;
}

/**
 * Extractive answer: the sentence of the top chunk that shares the most
 * (idf-weighted) query tokens. Topic words (entities, document title) already
 * chose the chunk, so they are ignored when choosing the sentence.
 */
export function extractAnswer(query, hit, idf) {
  if (!hit) return null;
  const titleToks = new Set(tokenize(hit.title));
  // "몇 퍼센트야?", "언제까지?" ask for a number: prefer sentences that contain one
  const wantsNumber = /몇|얼마|퍼센트|비율|언제|기간/.test(query);
  const pick = (text) => {
    const qt = [...new Set(tokenize(text))].filter((t) => !titleToks.has(t));
    let best = null;
    for (const [s, e] of sentenceSpans(hit.raw)) {
      const sentence = hit.raw.slice(s, e).trim();
      if (!sentence) continue;
      const st = new Set(tokenize(sentence));
      let score = 0;
      for (const t of qt) if (st.has(t)) score += idf(t);
      if (score > 0 && wantsNumber && /\d/.test(sentence)) score += 1;
      if (!best || score > best.score) best = { sentence, score };
    }
    return best;
  };
  const best = pick(withoutEntities(query));
  // only topic words in the question ("성적 우수 장학금은 얼마나 받아?") → score with the whole query
  return best && best.score === 0 ? pick(query) : best;
}

/** Clarifying question that offers the titles of the retrieved documents as choices. */
export function clarifyTemplate(aHits, bHits) {
  const titles = [...new Set([...bHits, ...aHits].map((h) => h.title))].slice(0, 3);
  return titles.length
    ? `무엇에 대한 질문인지 확실하지 않다. 다음 중 어느 것을 묻는가? ${titles.map((t) => `‘${t}’`).join(', ')}`
    : '학과 안내 문서에서 관련 내용을 찾지 못했다. 과목·시설·제도 이름을 넣어 다시 물어봐 줄 수 있나?';
}

/**
 * What the bot says for one turn.
 * Order: clarify (rewrite failed or top score < clarifyBelow) → LLM answer →
 * hand-written example (only when the rewritten query retrieved the evidence) →
 * extractive sentence → clarify.
 */
export function chooseAnswer({ rw, A, B, script, llm, mode, clarifyBelow }) {
  let answer;
  const top = B.hits[0]?.score ?? 0;
  if (rw.clarify || top < clarifyBelow) {
    const why = rw.clarify ? '재작성 단계에서 주제를 확정하지 못함' : `재작성 질문의 BM25 최고 점수 ${top.toFixed(2)} < 기준 ${clarifyBelow}`;
    if (llm?.answer) answer = { kind: 'llm', text: llm.answer, why };
    else if (rw.source === 'example' && rw.clarifyText) answer = { kind: 'clarify-example', text: script.answer, why };
    else if (script?.clarify && mode === 'rule') answer = { kind: 'clarify-example', text: script.answer, why };
    else answer = { kind: 'clarify', text: clarifyTemplate(A.hits, B.hits), why };
  } else if (llm?.answer) {
    answer = { kind: 'llm', text: llm.answer };
  } else if (script?.answer && B.rank > 0) {
    answer = { kind: 'example', text: script.answer.replaceAll('[*]', `[${B.rank}]`) };
  } else if (B.extract && B.extract.score > 0) {
    answer = { kind: 'extract', text: `${B.extract.sentence} [1]` };
  } else {
    answer = { kind: 'clarify', text: clarifyTemplate(A.hits, B.hits), why: '검색된 1위 청크에 질문의 단어가 있는 문장이 없음' };
  }
  answer.plain = stripCitations(answer.text);
  const cited = [...new Set([...answer.text.matchAll(/\[(\d+)\]/g)].map((m) => +m[1]))].filter((n) => n >= 1 && n <= B.hits.length);
  answer.sources = answer.kind.startsWith('clarify') ? [] : cited.map((n) => ({ n, hit: B.hits[n - 1] }));
  return answer;
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

// ------------------------------------------------------------------ templates (created lazily: Node can import the helpers)

let tpl = null;
function templates() {
  if (tpl) return tpl;
  const root = document.createElement('template');
  root.innerHTML = `
  <div class="widget w13" data-w13>
    <h3 class="widget__title">학과 안내 챗봇 · 원문 질문 vs 재작성 질문</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> 학과 문서와 대화 시나리오를 불러오는 중…
    </div>
    <div data-slot="body" hidden>
      <div class="w13-scen">
        <span class="field__label" id="__UID__-scen">대화 시나리오 (누르면 처음부터 다시 재생)</span>
        <div class="btn-row" role="group" aria-labelledby="__UID__-scen" data-slot="scenarios"></div>
      </div>
      <div class="widget__controls">
        <label class="field">
          <span class="field__label">재작성 방식</span>
          <select data-in="mode">
            <option value="rule">규칙 기반 (키 없이 동작)</option>
            <option value="llm">LLM condense (키 없으면 예시)</option>
          </select>
        </label>
        <label class="field">
          <span class="field__label">대화 기록 창: 최근 <output data-out="window">3</output>턴</span>
          <input type="range" data-in="window" min="0" max="${MAX_WINDOW}" step="1" value="3">
        </label>
      </div>
      <div class="btn-row">
        <button type="button" class="btn small ghost" data-act="reset">↺ 대화 초기화</button>
        <button type="button" class="btn small ghost" data-act="vector" aria-pressed="false">벡터 검색도 비교 (모델 약 118MB)</button>
      </div>
      <div data-slot="model-status"></div>

      <div class="w13-chat" data-slot="chat" aria-live="polite"></div>

      <form class="w13-ask" data-slot="ask">
        <label class="field">
          <span class="field__label">이어서 질문하기 <span data-slot="modehint" class="w13-muted"></span></span>
          <input type="text" data-in="question" autocomplete="off" maxlength="200" placeholder="예: 그럼 자연어 처리는?">
        </label>
        <button type="submit" class="btn primary">보내기</button>
      </form>

      <details class="w13-llm">
        <summary>LLM 설정 (API 키 · 선택)</summary>
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
            <span class="field__label">API 키 (이 탭의 sessionStorage에만 저장)</span>
            <input type="password" data-in="key" autocomplete="off" spellcheck="false" placeholder="키를 붙여 넣는다">
          </label>
        </div>
        <div class="btn-row">
          <button type="button" class="btn small" data-act="save-key">키 저장</button>
          <button type="button" class="btn small ghost" data-act="clear-key">키 지우기</button>
        </div>
        <p class="w13-muted" data-slot="keystate" role="status"></p>
      </details>

      <div data-slot="out-inline"></div>
    </div>
  </div>`;
  const out = document.createElement('template');
  out.innerHTML = `
  <div class="w13-out">
    <div data-slot="turnhead"></div>
    <div data-slot="trace"></div>
    <div class="w13-ab" data-slot="ab"></div>
    <div data-slot="llmrun"></div>
    <details class="w13-prompt" data-slot="promptbox">
      <summary data-slot="promptsum">condense 프롬프트</summary>
      <pre class="w13-pre" data-slot="prompt"></pre>
    </details>
    <h4>시나리오 점수판 · 근거 청크가 상위 3개 안에 들어왔나</h4>
    <div class="w13-score" data-slot="score"></div>
  </div>`;
  tpl = { root, out };
  return tpl;
}

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ scenario?: string, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const T = templates();
  const uid = `w13-${++seq}`;
  const root = T.root.content.firstElementChild.cloneNode(true);
  root.innerHTML = root.innerHTML.replaceAll('__UID__', uid);
  const out = T.out.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w13:${seq}`;
  const st = { ctrl, outputId, alive: true, llm: null };
  state.set(el, st);

  let corpus;
  let extra;
  let data;
  try {
    [corpus, extra, data] = await Promise.all([
      fetchJson(new URL('corpus/corpus.json', DATA), ctrl.signal),
      fetchJson(new URL('w13/dept-docs.json', DATA), ctrl.signal),
      fetchJson(new URL('w13/scenarios.json', DATA), ctrl.signal),
    ]);
  } catch (err) {
    if (err.name === 'AbortError') return;
    $('[data-slot=status]').innerHTML = `<div class="widget__error" role="alert">
      학과 문서를 불러오지 못했다 (${escapeHtml(err.message)}).<br>
      파일을 더블클릭해 <code>file://</code>로 열었다면 ES Module과 fetch가 동작하지 않는다.
      저장소 폴더에서 <code>python -m http.server</code>를 실행하고 <code>http://localhost:8000</code>으로 연다.</div>`;
    return;
  }
  if (!st.alive) return;

  // ---- index: department documents only (shared ones + lab additions), title-prefixed chunks
  const cfg = data.retrieval;
  const docs = [...extra.shared.map((id) => corpus.documents.find((d) => d.id === id)).filter(Boolean), ...extra.documents];
  const chunks = [];
  for (const d of docs) {
    for (const c of chunk(d.text, cfg.chunk)) {
      chunks.push({ id: `${d.id}#${c.index}`, doc: d.id, title: d.title, index: c.index, raw: c.text, text: cfg.titlePrefix ? `[${d.title}] ${c.text}` : c.text });
    }
  }
  const byId = new Map(chunks.map((c) => [c.id, c]));
  const bm25 = new BM25().add(chunks.map((c) => ({ id: c.id, text: c.text })));
  const idf = (t) => bm25.idf(t);
  const search = (q) => (q ? bm25.search(q, cfg.k).map((r) => ({ ...byId.get(r.id), score: r.score })) : []);

  const scenarios = data.scenarios;
  const s = {
    scenario: scenarios.some((x) => x.id === options.scenario) ? options.scenario : scenarios[0].id,
    window: data.defaultWindow ?? 3,
    mode: 'rule',
    turns: [], // { text, pii: string[], script?: object, llm?: object }
    selected: -1,
  };
  const vec = { on: false, ready: false, error: '', store: null, qcache: new Map(), loading: false };

  // ---- controls
  $('[data-slot=scenarios]').innerHTML = scenarios
    .map((sc) => `<button type="button" class="btn small ghost" data-scen="${sc.id}" aria-pressed="false">${escapeHtml(sc.label)}</button>`)
    .join('');
  const provSel = $('[data-in=provider]');
  provSel.innerHTML = Object.entries(PROVIDERS).map(([k, p]) => `<option value="${k}">${escapeHtml(p.label)}</option>`).join('');
  const fillModels = () => {
    $('[data-in=model]').innerHTML = PROVIDERS[provSel.value].models.map((m) => `<option value="${m}">${escapeHtml(m)}</option>`).join('');
  };
  fillModels();

  const keyReady = () => safeHasKey(provSel.value);

  function syncKeyState() {
    const has = keyReady();
    $('[data-slot=keystate]').innerHTML = has
      ? `<span class="chip ok">🔑 ${escapeHtml(PROVIDERS[provSel.value].label)} 키 저장됨 · 이 탭에서만</span> “LLM condense” 방식에서 새 질문은 LLM으로 재작성·답변한다.`
      : '<span class="chip warn">🔑 키 없음</span> 규칙 기반 재작성과 예시 응답·추출식 답변으로 동작한다.';
    $('[data-slot=modehint]').textContent =
      s.mode === 'llm' ? (has ? '· LLM이 재작성하고 답한다' : '· 키 없음: 새 질문은 규칙 기반으로 대체') : '· 규칙 기반 재작성 + 추출식 답변';
  }

  // ---- conversation
  function loadScenario(id) {
    st.llm?.abort();
    const sc = scenarios.find((x) => x.id === id) ?? scenarios[0];
    s.scenario = sc.id;
    s.turns = sc.turns.map((t) => {
      const m = maskPII(t.q);
      return { text: m.text, pii: m.found, script: t };
    });
    s.selected = firstInteresting(sc);
    root.querySelectorAll('[data-scen]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.scen === sc.id)));
    render();
  }

  // open on the first follow-up that the raw question gets wrong (the failure is visible right away)
  function firstInteresting(sc) {
    const i = sc.turns.findIndex((t, k) => k > 0 && t.evidence);
    return i >= 0 ? i : sc.turns.length - 1;
  }

  /** Compute rewrite → retrieval → answer for every turn under the current settings. */
  function computeAll() {
    const res = [];
    const provider = provSel.value;
    const model = $('[data-in=model]').value;
    for (let i = 0; i < s.turns.length; i++) {
      const turn = s.turns[i];
      const script = turn.script;
      const userHistory = s.turns.slice(0, i).map((t) => t.text);
      const prompt = buildCondensePrompt(res.map((r) => ({ user: r.turn.text, bot: r.answer.plain })), turn.text, { window: s.window });
      const rule = ruleRewrite(userHistory, turn.text, { window: s.window });

      let rw;
      const llm = turn.llm && turn.llm.window === s.window && turn.llm.provider === provider && turn.llm.model === model ? turn.llm : null;
      if (s.mode === 'llm' && llm?.rewrite) {
        rw = { source: 'llm', query: llm.rewrite.query, clarify: llm.rewrite.clarify, changed: llm.rewrite.query !== turn.text, steps: [] };
      } else if (s.mode === 'llm' && script?.llmRewrite) {
        const p = parseRewrite(script.llmRewrite);
        rw = { source: 'example', query: p.clarify ? turn.text : p.query, clarifyText: p.clarify ? p.query : '', clarify: p.clarify, changed: !p.clarify && p.query !== turn.text, steps: [] };
      } else {
        rw = { source: s.mode === 'llm' ? 'fallback' : 'rule', ...rule };
      }

      const A = { query: turn.text, hits: search(turn.text) };
      const B = { query: rw.query, hits: search(rw.query) };
      A.rank = evidenceRank(A.hits, script?.evidence);
      B.rank = evidenceRank(B.hits, script?.evidence);
      A.extract = extractAnswer(A.query, A.hits[0], idf);
      B.extract = extractAnswer(B.query, B.hits[0], idf);

      const answer = chooseAnswer({ rw, A, B, script, llm: s.mode === 'llm' ? llm : null, mode: s.mode, clarifyBelow: cfg.clarifyBelow });
      res.push({ i, turn, script, rule, rw, A, B, answer, prompt, llm });
    }
    return res;
  }

  // ---- rendering
  let view = [];
  function render() {
    view = computeAll();
    if (s.selected >= view.length) s.selected = view.length - 1;
    renderChat();
    renderOutput();
    renderScore();
    syncKeyState();
  }

  const KIND = {
    example: `<span class="chip accent">${RECORDED_LABEL}</span>`,
    extract: '<span class="chip">추출식 · 검색된 문장을 그대로 인용</span>',
    llm: '<span class="chip ok">LLM 생성</span>',
    clarify: '<span class="chip warn">되묻기 · 규칙으로 만든 확인 질문</span>',
    'clarify-example': `<span class="chip warn">되묻기</span> <span class="chip accent">${RECORDED_LABEL}</span>`,
  };

  function rankChip(rank, evidence) {
    if (!evidence) return '<span class="chip">근거 판정 없음 (자유 질문)</span>';
    return rank > 0 ? `<span class="chip ok">근거 ✓ ${rank}위</span>` : '<span class="chip warn">근거 ✗ 상위 3개에 없음</span>';
  }

  function renderChat() {
    const chat = $('[data-slot=chat]');
    if (!view.length) {
      chat.innerHTML = `<div class="callout"><span class="callout__title">대화가 비었다</span>
        아래 입력창에 첫 질문을 쓰거나 위의 시나리오 버튼을 누른다. 예: <button type="button" class="btn small ghost" data-starter="실습실은 몇 시까지 열어?">실습실은 몇 시까지 열어?</button></div>`;
      return;
    }
    chat.innerHTML = view
      .map((v) => {
        const sel = v.i === s.selected;
        const rwChip = v.rw.clarify
          ? '<span class="chip warn">재작성 실패 → 되묻기</span>'
          : v.rw.changed
            ? `<span class="chip">재작성: ${escapeHtml(v.rw.query)}</span>`
            : '<span class="chip">재작성 없음 (독립 질문)</span>';
        const pii = v.turn.pii.length
          ? `<span class="chip warn">개인정보 ${v.turn.pii.length}건 가림 (${escapeHtml(v.turn.pii.join(', '))}) · 기록에는 가린 문장만 저장</span>`
          : '';
        const srcs = v.answer.sources.length
          ? `<div class="w13-srcs">${v.answer.sources.map((x) => `<span>[${x.n}] ${escapeHtml(x.hit.title)}</span>`).join('')}</div>`
          : '';
        return `<div class="w13-turn${sel ? ' is-selected' : ''}" data-turn="${v.i}">
          <div class="w13-msg w13-msg--user"><span class="visually-hidden">질문 ${v.i + 1}: </span>${escapeHtml(v.turn.text)}${pii ? `<div class="w13-meta">${pii}</div>` : ''}</div>
          <div class="w13-msg w13-msg--bot">
            <div class="w13-meta">${KIND[v.answer.kind]} ${rwChip} ${v.answer.kind.startsWith('clarify') ? '' : rankChip(v.B.rank, v.script?.evidence)}</div>
            ${v.turn.llm?.pending ? '<div class="widget__status" role="status"><span class="spinner" aria-hidden="true"></span> LLM이 재작성·답변하는 중… (아래는 그 전까지의 규칙 기반 결과)</div>' : ''}
            <div class="w13-answer">${escapeHtml(v.answer.text)}</div>
            ${srcs}
            <button type="button" class="btn small ${sel ? '' : 'ghost'} w13-pick" data-pick="${v.i}" aria-pressed="${sel}">턴 ${v.i + 1} 비교 보기${sel ? ' ✓' : ''}</button>
          </div>
        </div>`;
      })
      .join('');
  }

  function hitListHtml(col, evidence, which) {
    if (!col.hits.length) return '<p class="w13-muted">검색 결과 없음 — 질문의 단어가 어떤 청크에도 없다.</p>';
    return `<ol class="w13-hits">${col.hits
      .map((h, k) => {
        const isEv = evidence && h.doc === evidence.doc && h.raw.includes(evidence.quote);
        return `<li class="${isEv ? 'is-hit' : ''}">
          <div class="w13-hit__head"><b>${k + 1}</b> ${escapeHtml(h.title)} ${isEv ? '<span class="chip ok">근거</span>' : ''}</div>
          <div class="chunk-meta"><span>${escapeHtml(h.id)}</span><span>BM25 ${h.score.toFixed(2)}</span></div>
          <div class="w13-hit__text">${snippetHtml(h.raw, isEv ? evidence.quote : '')}</div>
        </li>`;
      })
      .join('')}</ol><div data-vec="${which}"></div>`;
  }

  function renderOutput() {
    const v = view[s.selected];
    const head = $('[data-slot=turnhead]');
    if (!v) {
      head.innerHTML = '<p class="w13-muted">대화가 비었다. 질문을 보내면 이곳에 원문 질문과 재작성 질문의 검색 결과가 나란히 나온다.</p>';
      ['trace', 'ab', 'llmrun'].forEach((k) => ($(`[data-slot=${k}]`).innerHTML = ''));
      $('[data-slot=promptbox]').hidden = true;
      return;
    }
    $('[data-slot=promptbox]').hidden = false;
    const ev = v.script?.evidence;
    head.innerHTML = `<div class="w13-turnhead"><b>턴 ${v.i + 1}</b> “${escapeHtml(v.turn.text)}”
      ${ev ? `<span class="w13-muted">· 필요한 근거: ${escapeHtml(docTitle(ev.doc))} “${escapeHtml(ev.quote)}”</span>` : '<span class="w13-muted">· 자유 질문이라 정답 근거가 지정되지 않았다</span>'}</div>`;

    // rewrite trace
    const rw = v.rw;
    let trace;
    if (rw.source === 'rule' || rw.source === 'fallback') {
      trace = `<div class="w13-trace">
        <div class="w13-trace__title">규칙 기반 재작성 · 사용자 질문 최근 ${s.window}턴을 본다${rw.hidden ? ` (창 밖 ${rw.hidden}턴은 보지 않음)` : ''}</div>
        <ol>${rw.steps.map((x) => `<li class="w13-step w13-step--${x.type}">${escapeHtml(x.text)}</li>`).join('')}</ol>
        ${rw.source === 'fallback' ? '<p class="w13-muted">LLM 결과가 없어 규칙 기반으로 대체했다. 키를 넣고 아래 버튼으로 이 턴을 LLM에 보낼 수 있다.</p>' : ''}
        <p class="w13-limit"><b>규칙의 한계</b> 주제어 사전에 있는 단어만 주제로 알아본다 · 챗봇 답변 속 대상(“거기”=행정실)은 못 찾는다 · 짧은 새 질문에도 앞 주제를 붙일 수 있다</p>
      </div>`;
    } else if (rw.source === 'example') {
      trace = `<div class="w13-trace">
        <div class="w13-trace__title">LLM condense <span class="chip accent">${RECORDED_LABEL}</span></div>
        <p>${rw.clarify ? `되묻기: ${escapeHtml(rw.clarifyText)}` : `독립 질문: <b>${escapeHtml(rw.query)}</b>`}</p>
        <p class="w13-muted">아래 condense 프롬프트를 받은 모델이 낼 법한 출력을 사람이 작성한 예시다. 키를 넣고 버튼을 누르면 실제로 호출한다.</p>
      </div>`;
    } else {
      trace = `<div class="w13-trace">
        <div class="w13-trace__title">LLM condense <span class="chip ok">LLM 생성 · ${escapeHtml(v.llm.model)}</span></div>
        <p>${rw.clarify ? `되묻기: ${escapeHtml(rw.query)}` : `독립 질문: <b>${escapeHtml(rw.query)}</b>`}</p>
      </div>`;
    }
    if (v.script?.note) trace += `<p class="w13-note">💡 ${escapeHtml(v.script.note)}</p>`;
    $('[data-slot=trace]').innerHTML = trace;

    // A | B
    const col = (label, c, which, extra) => `<section class="w13-col w13-col--${which}" aria-label="${label}">
      <h5>${label}</h5>
      <div class="w13-q"><code>${escapeHtml(c.query)}</code></div>
      <div class="w13-verdict">${rankChip(c.rank, ev)}</div>
      ${hitListHtml(c, ev, which)}
      ${extra}
    </section>`;
    const exA = v.A.extract && v.A.extract.score > 0
      ? `<p class="w13-extract"><span class="chip">추출식</span> 이 결과로 답하면: “${escapeHtml(v.A.extract.sentence)}”</p>`
      : '<p class="w13-extract w13-muted">추출할 문장이 없다.</p>';
    const exB = v.B.extract && v.B.extract.score > 0
      ? `<p class="w13-extract"><span class="chip">추출식</span> 이 결과로 답하면: “${escapeHtml(v.B.extract.sentence)}”</p>`
      : '';
    $('[data-slot=ab]').innerHTML =
      col('(A) 원문 후속 질문만', v.A, 'A', exA) +
      col(rw.clarify ? '(B) 재작성 실패 → 되묻기' : '(B) 재작성한 독립 질문', v.B, 'B', exB + (v.answer.why ? `<p class="w13-muted">되묻기 이유: ${escapeHtml(v.answer.why)}</p>` : ''));

    // LLM run button / state
    const llmSlot = $('[data-slot=llmrun]');
    const t = v.turn;
    if (t.llm?.pending) {
      llmSlot.innerHTML = '<div class="widget__status" role="status"><span class="spinner" aria-hidden="true"></span> LLM이 질문을 재작성하고 답하는 중…</div>';
    } else if (t.llm?.error) {
      llmSlot.innerHTML = `<div class="widget__error" role="alert">${escapeHtml(t.llm.error)} 검색 결과는 그대로 보여 준다.</div>${llmButton()}`;
    } else {
      llmSlot.innerHTML = s.mode === 'llm' && !v.llm ? llmButton() : '';
    }

    // condense prompt
    const p = v.prompt;
    $('[data-slot=promptsum]').innerHTML = `condense 프롬프트 · 기록 ${p.included}턴 포함${p.dropped ? `, ${p.dropped}턴 잘림` : ''} · ${p.chars.toLocaleString()}자 · 약 ${estimateTokens(p.system + p.messages[0].content).toLocaleString()}토큰`;
    $('[data-slot=prompt]').textContent = `[system]\n${p.system}\n\n[user]\n${p.messages[0].content}`;

    fillVectors(v);
  }

  function llmButton() {
    return keyReady()
      ? '<button type="button" class="btn small" data-act="llm-turn">이 턴을 LLM으로 재작성·답변</button>'
      : '<p class="w13-muted">API 키를 넣으면 이 턴을 실제 LLM으로 재작성할 수 있다 (LLM 설정).</p>';
  }

  function renderScore() {
    const rows = view.filter((v) => v.script?.evidence);
    const box = $('[data-slot=score]');
    if (!rows.length) {
      box.innerHTML = '<p class="w13-muted">정답 근거가 정해진 시나리오 턴이 없다. 시나리오를 고르면 점수판이 채워진다.</p>';
      return;
    }
    const a = rows.filter((v) => v.A.rank > 0).length;
    const b = rows.filter((v) => v.B.rank > 0).length;
    const cell = (r) => (r > 0 ? `<td class="w13-ok">✓ ${r}위</td>` : '<td class="w13-bad">✗</td>');
    box.innerHTML = `<div class="stat-row">
        <div class="stat"><span class="stat__label">(A) 원문 hit@3</span><span class="stat__value">${a}/${rows.length}</span></div>
        <div class="stat"><span class="stat__label">(B) 재작성 hit@3</span><span class="stat__value">${b}/${rows.length}</span></div>
        <div class="stat"><span class="stat__label">기록 창</span><span class="stat__value">${s.window}턴</span></div>
      </div>
      <div class="w13-table"><table>
        <thead><tr><th>턴</th><th>질문</th><th>(A) 원문</th><th>(B) 재작성</th></tr></thead>
        <tbody>${rows
          .map((v) => `<tr class="${v.i === s.selected ? 'is-selected' : ''}"><td><button type="button" class="btn small ghost" data-pick="${v.i}">${v.i + 1}</button></td><td>${escapeHtml(v.turn.text)}</td>${cell(v.A.rank)}${cell(v.B.rank)}</tr>`)
          .join('')}</tbody></table></div>`;
  }

  const docTitle = (id) => docs.find((d) => d.id === id)?.title ?? id;

  // ---- optional vector retrieval (after the model loads)
  async function enableVectors() {
    if (vec.loading || vec.ready) return;
    vec.loading = true;
    vec.error = '';
    const statusEl = $('[data-slot=model-status]');
    try {
      await loadModelWithUI(statusEl, DEFAULT_MODEL);
      if (!st.alive) return;
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
      if (!st.alive) return;
      vec.store = new VectorStore();
      chunks.forEach((c, i) => vec.store.add(c.id, vectors[i]));
      vec.ready = true;
    } catch (err) {
      vec.error = err?.message ?? String(err);
      vec.on = false;
      $('[data-act=vector]').setAttribute('aria-pressed', 'false');
    } finally {
      vec.loading = false;
    }
    if (st.alive) renderOutput();
  }

  let vecTick = 0;
  async function fillVectors(v) {
    const my = ++vecTick;
    for (const which of ['A', 'B']) {
      const slot = out.querySelector(`[data-vec=${which}]`);
      if (!slot) continue;
      if (!vec.on) {
        slot.innerHTML = '';
        continue;
      }
      slot.innerHTML = vec.ready
        ? '<p class="w13-muted"><span class="spinner" aria-hidden="true"></span> 벡터 검색 중…</p>'
        : '<p class="w13-muted">벡터 검색: 모델을 준비하는 중이다. BM25 결과는 이미 위에 있다.</p>';
    }
    if (!vec.on || !vec.ready) return;
    try {
      for (const which of ['A', 'B']) {
        const c = v[which];
        const key = c.query;
        if (!vec.qcache.has(key)) vec.qcache.set(key, await embed(key, { role: 'query' }));
        if (my !== vecTick || !st.alive) return;
        const hits = vec.store.search(vec.qcache.get(key), cfg.k).map((r) => ({ ...byId.get(r.id), score: r.score }));
        const rank = evidenceRank(hits, v.script?.evidence);
        const slot = out.querySelector(`[data-vec=${which}]`);
        if (!slot) return;
        slot.innerHTML = `<div class="w13-vec"><b>벡터 상위 3</b> ${rankChip(rank, v.script?.evidence)}
          <ol>${hits.map((h) => `<li>${escapeHtml(h.title)} <span class="chunk-meta"><span>${escapeHtml(h.id)}</span><span>cos ${h.score.toFixed(3)}</span></span></li>`).join('')}</ol></div>`;
      }
    } catch (err) {
      out.querySelectorAll('[data-vec]').forEach((x) => (x.innerHTML = `<p class="widget__error">벡터 검색 실패: ${escapeHtml(err?.message ?? err)}</p>`));
    }
  }

  // ---- LLM: condense + grounded answer for one turn
  async function runLLM(i) {
    const turn = s.turns[i];
    if (!turn) return;
    const provider = provSel.value;
    const model = $('[data-in=model]').value;
    st.llm?.abort();
    const ac = new AbortController();
    st.llm = ac;
    const prior = view.slice(0, i).map((v) => ({ user: v.turn.text, bot: v.answer.plain }));
    const prompt = buildCondensePrompt(prior, turn.text, { window: s.window });
    turn.llm = { pending: true, window: s.window, provider, model };
    s.selected = i;
    render();
    try {
      const r1 = await generate({ provider, model, system: prompt.system, messages: prompt.messages, maxTokens: 120, temperature: 0, signal: ac.signal });
      const rewrite = parseRewrite(r1.text ?? '');
      if (!rewrite.query) throw new LLMError('모델이 빈 재작성을 돌려줬다. 다시 시도한다.', 'empty');
      let answer;
      if (rewrite.clarify) {
        answer = rewrite.query;
      } else {
        const hits = search(rewrite.query);
        if (!hits.length) {
          answer = '학과 안내 문서에서 찾을 수 없다.';
        } else {
          const { system, messages } = ragPrompt(rewrite.query, hits);
          const r2 = await generate({ provider, model, system, messages, maxTokens: 400, temperature: 0.2, signal: ac.signal });
          answer = (r2.text ?? '').trim() || '문서에서 찾을 수 없다.';
        }
      }
      if (st.llm !== ac || !st.alive) return;
      turn.llm = { window: s.window, provider, model, rewrite, answer };
    } catch (err) {
      if (err.name === 'AbortError' || !st.alive) return;
      turn.llm = { window: s.window, provider, model, error: err instanceof LLMError ? err.message : 'LLM 호출 중 알 수 없는 오류가 났다.' };
    }
    if (st.alive) render();
  }

  // ---- events
  const on = (target, type, fn) => target.addEventListener(type, fn, { signal: ctrl.signal });
  const select = (i, from) => {
    s.selected = i;
    renderChat();
    renderOutput();
    renderScore();
    // re-rendering replaced the button: keep keyboard focus on its replacement
    (from === 'out' ? out : root).querySelector(`[data-pick="${i}"]`)?.focus();
    if (from !== 'out') focusOutput(outputId);
  };
  const clickHandler = (e) => {
    const scen = e.target.closest('[data-scen]');
    if (scen) return loadScenario(scen.dataset.scen);
    const pick = e.target.closest('[data-pick]');
    if (pick) return select(+pick.dataset.pick, out.contains(pick) ? 'out' : 'root');
    const starter = e.target.closest('[data-starter]');
    if (starter) return ask(starter.dataset.starter);
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (act === 'reset') {
      st.llm?.abort();
      s.turns = [];
      s.selected = -1;
      root.querySelectorAll('[data-scen]').forEach((b) => b.setAttribute('aria-pressed', 'false'));
      render();
      $('[data-in=question]').focus();
    } else if (act === 'vector') {
      vec.on = !vec.on;
      e.target.closest('[data-act]').setAttribute('aria-pressed', String(vec.on));
      if (vec.on && !vec.ready) enableVectors();
      renderOutput();
    } else if (act === 'llm-turn') {
      runLLM(s.selected);
    } else if (act === 'save-key') {
      const input = $('[data-in=key]');
      const val = input.value.trim();
      input.value = ''; // never keep the key in the DOM
      if (!val) return;
      try {
        setKey(provSel.value, val);
      } catch {
        $('[data-slot=keystate]').textContent = '이 브라우저에서는 sessionStorage를 쓸 수 없어 키를 저장하지 못했다.';
        return;
      }
      render();
    } else if (act === 'clear-key') {
      try {
        clearKey(provSel.value);
      } catch {
        /* storage unavailable */
      }
      render();
    }
  };
  // below 860px the output node sits inside root: handle its clicks only once
  on(root, 'click', (e) => {
    if (!out.contains(e.target)) clickHandler(e);
  });
  on(out, 'click', clickHandler);
  on($('[data-in=mode]'), 'change', (e) => {
    s.mode = e.target.value;
    render();
  });
  on($('[data-in=window]'), 'input', (e) => {
    s.window = +e.target.value;
    $('[data-out=window]').textContent = s.window;
    render();
  });
  on(provSel, 'change', () => {
    fillModels();
    render();
  });
  on($('[data-in=model]'), 'change', render);
  on($('[data-slot=ask]'), 'submit', (e) => {
    e.preventDefault();
    const input = $('[data-in=question]');
    const q = input.value.trim();
    if (!q) return;
    input.value = '';
    ask(q);
  });

  function ask(q) {
    const m = maskPII(q.slice(0, 200));
    s.turns.push({ text: m.text, pii: m.found });
    s.selected = s.turns.length - 1;
    render();
    if (s.mode === 'llm' && keyReady()) runLLM(s.selected);
  }

  // ---- first render: a full scenario is already on screen
  $('[data-slot=status]').hidden = true;
  $('[data-slot=body]').hidden = false;
  $('[data-in=window]').value = s.window;
  $('[data-out=window]').textContent = s.window;
  registerOutput(outputId, {
    title: options.outputTitle ?? '원문 vs 재작성 검색 비교',
    node: out,
    inlineHost: $('[data-slot=out-inline]'),
  });
  loadScenario(s.scenario);
}

export function unmount(el) {
  const st = state.get(el);
  if (st) {
    st.alive = false;
    st.ctrl.abort();
    st.llm?.abort();
    unregisterOutput(st.outputId);
  }
  state.delete(el);
  el.replaceChildren();
}

// ------------------------------------------------------------------ small helpers

function snippetHtml(text, quote) {
  const t = text.replace(/\s+/g, ' ').trim();
  if (quote) {
    const q = quote.replace(/\s+/g, ' ');
    const i = t.indexOf(q);
    if (i >= 0) {
      const from = Math.max(0, i - 30);
      const to = Math.min(t.length, i + q.length + 30);
      return `${from > 0 ? '…' : ''}${escapeHtml(t.slice(from, i))}<mark>${escapeHtml(q)}</mark>${escapeHtml(t.slice(i + q.length, to))}${to < t.length ? '…' : ''}`;
    }
  }
  return escapeHtml(t.length > 90 ? `${t.slice(0, 90)}…` : t);
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
