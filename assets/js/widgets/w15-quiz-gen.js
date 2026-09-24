// w15 RAG 퀴즈 생성기 (응용 실습 3)
// One concept: generation grounded in retrieved sources, followed by automatic
// verification. Retrieve top chunks for a topic (BM25, optional hybrid with
// vectors), generate quiz items that each carry a citation (chunk id + exact
// source sentence), then verify every item: the answer must appear in the cited
// sentence, the sentence must exist in the cited chunk, distractors must not
// also be correct, near-duplicates and items from broken (OCR) documents are
// flagged. Without an API key a transparent rule-based generator (cloze + MCQ
// distractors of the same type) runs fully offline.

import { chunk, sentenceSpans } from '../core/chunker.js';
import { BM25 } from '../core/bm25.js';
import { rrf } from '../core/rerank.js';
import { VectorStore } from '../core/vectorstore.js';
import { DEFAULT_MODEL, loadModelWithUI, embed, embedBatch } from '../core/embed.js';
import { PROVIDERS, setKey, hasKey, clearKey, generate, LLMError } from '../core/llm.js';
import { registerOutput, unregisterOutput } from '../site/result.js';

const DATA = new URL('../../data/', import.meta.url);
const EXAMPLE_LABEL = '예시 응답 · 수업용 작성 예시';
const CHUNK_CFG = { strategy: 'recursive', size: 200, overlap: 0 };

// ------------------------------------------------------------------ pure helpers
// (exported so they can be tested in Node and reused in Challenge answers)

export const STEM = '빈칸에 알맞은 말은?';
export const BLANK = '____';
export const DUP_THRESHOLD = 0.7;

/** Technical terms the rule generator may blank out, grouped so distractors share a category. */
export const TERMS = [
  ['chunk', ['고정 길이 청킹', '재귀 분할', '오버랩', '청킹', '청크']],
  ['embed', ['코사인 유사도', '임베딩', '벡터스토어', '내적', '다국어 모델']],
  ['index', ['Flat 인덱스', 'HNSW', '근사 검색']],
  ['search', ['하이브리드 검색', '키워드 검색', '벡터 검색', '크로스 인코더', '리랭킹', 'BM25', 'RRF']],
  ['eval', ['Faithfulness', 'Recall@k', '골든셋', 'nDCG', 'MRR']],
  ['llm', ['환각', 'LLM']],
  ['campus', ['학과 행정실', '공용 캘린더', '담당 교수', '조교', 'GPU']],
  ['course', ['파이썬 프로그래밍', '머신러닝 기초', '자연어 처리', '데이터 분석', '팀 프로젝트', '딥러닝']],
].flatMap(([cat, list]) => list.map((term) => ({ term, cat })))
  .sort((a, b) => b.term.length - a.term.length);

// A number token; O, I, l are accepted inside it on purpose, because OCR turns
// 0 into O and 1 into l ("5O%", "2l4") and a naive extractor happily copies them.
const N = '[0-9OIl]*\\d[0-9OIl]*(?:\\.\\d+)?';
const NB = '(?<![A-Za-z0-9.@~])';
const NUM_PATTERNS = [
  { type: 'percent', priority: 1, re: new RegExp(`${NB}${N}(?:~${N})?\\s?%`, 'g') },
  { type: 'hours', priority: 1, re: new RegExp(`${NB}${N}(?:~${N})?시간`, 'g') },
  { type: 'clock', priority: 1, re: new RegExp(`(?:오전|오후)\\s?${N}시(?!간)|${NB}${N}시(?!간)`, 'g') },
  { type: 'unit', priority: 2, re: new RegExp(`${NB}${N}(?:~${N})?(학년도|학기|학점|주차|분위|차원|층|호|명|대|월|년|주|회|개)`, 'g') },
  { type: 'number', priority: 4, re: new RegExp(`${NB}${N}(?![0-9A-Za-z%.~])`, 'g') },
];

export const TYPE_LABEL = {
  percent: '비율(%)', hours: '시간', clock: '시각', number: '숫자', term: '용어',
};
export function typeLabel(type) {
  if (type.startsWith('unit:')) return `숫자+${type.slice(5)}`;
  if (type.startsWith('term:')) return '기술 용어';
  return TYPE_LABEL[type] ?? type;
}

/**
 * Candidate answer spans in a sentence: numbers with units first, then terms,
 * then bare numbers. Overlapping matches keep the higher-priority/longer one.
 * @returns {{ text: string, start: number, end: number, type: string, priority: number }[]}
 */
export function findCandidates(sentence) {
  const found = [];
  for (const p of NUM_PATTERNS) {
    for (const m of sentence.matchAll(p.re)) {
      const type = p.type === 'unit' ? `unit:${m[1]}` : p.type;
      found.push({ text: m[0].trim(), start: m.index, end: m.index + m[0].trimEnd().length, type, priority: p.priority });
    }
  }
  for (const { term, cat } of TERMS) {
    let from = 0;
    for (;;) {
      const i = sentence.indexOf(term, from);
      if (i < 0) break;
      found.push({ text: term, start: i, end: i + term.length, type: `term:${cat}`, priority: 3 });
      from = i + term.length;
    }
  }
  found.sort((a, b) => a.priority - b.priority || (b.end - b.start) - (a.end - a.start) || a.start - b.start);
  const kept = [];
  for (const c of found) {
    if (kept.some((k) => c.start < k.end && k.start < c.end)) continue;
    kept.push(c);
  }
  return kept.sort((a, b) => a.priority - b.priority || a.start - b.start);
}

const norm = (s) => String(s ?? '').replace(/\s+/g, ' ').trim().toLowerCase();
const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** Regex for `term` inside text; numbers must not be glued to other digits ("1" is not in "12"). */
function termRegex(term, flags = '') {
  const t = norm(term);
  const body = t.split(' ').map(escRe).join('\\s+');
  const pre = /^\d/.test(t) ? '(?<![0-9.])' : '';
  const post = /\d$/.test(t) ? '(?![0-9])' : '';
  return new RegExp(`${pre}${body}${post}`, `i${flags}`);
}

/** Does `text` contain `term` (whitespace- and case-insensitive, digit-bounded)? */
export function containsTerm(text, term) {
  if (!norm(term)) return false;
  return termRegex(term).test(String(text ?? ''));
}

/** Replace every occurrence of `answer` in `sentence` with a blank. */
export function makeCloze(sentence, answer) {
  return sentence.replace(termRegex(answer, 'g'), BLANK);
}

/** OCR damage: letters inside numbers ("5O%", "2l4") and runs of spaced single syllables ("장 학 금 안 내"). */
export function ocrSuspects(text) {
  const out = [];
  for (const m of String(text ?? '').matchAll(/[0-9OIl]*(?:\d[OIl]|[OIl]\d)[0-9OIl]*%?/g)) out.push(m[0]);
  for (const m of String(text ?? '').matchAll(/(?:^|\s)((?:[가-힣] ){3,}[가-힣])(?=\s|$)/g)) out.push(m[1]);
  return [...new Set(out)];
}

/** Deterministic PRNG so "다시 섞기" is reproducible. */
export function seededRandom(seed) {
  let h = 1779033703 ^ String(seed).length;
  for (const ch of String(seed)) {
    h = Math.imul(h ^ ch.charCodeAt(0), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  let a = h >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(list, rand) {
  const a = [...list];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** Trimmed sentences of a text with offsets relative to that text. */
export function sentencesOf(text) {
  const out = [];
  for (const [s0, e0] of sentenceSpans(text)) {
    let s = s0;
    let e = e0;
    while (s < e && /\s/.test(text[s])) s++;
    while (e > s && /\s/.test(text[e - 1])) e--;
    if (e - s >= 8) out.push({ start: s, end: e, text: text.slice(s, e) });
  }
  return out;
}

/**
 * Distractor pools: every candidate value in the (non-broken) corpus, by type.
 * @param {{ text: string, type?: string }[]} docs
 * @returns {Map<string, string[]>}
 */
export function buildPools(docs) {
  const pools = new Map();
  for (const d of docs) {
    if (d.type === 'broken') continue;
    for (const sen of sentencesOf(d.text)) {
      for (const c of findCandidates(sen.text)) {
        if (ocrSuspects(c.text).length) continue;
        if (!pools.has(c.type)) pools.set(c.type, []);
        if (!pools.get(c.type).includes(c.text)) pools.get(c.type).push(c.text);
      }
    }
  }
  return pools;
}

const SHIFTS = {
  percent: [(v) => v + 10, (v) => v * 2, (v) => v - 10, (v) => v + 30, (v) => v + 50],
  hours: [(v) => v * 2, (v) => v / 2, (v) => v + 6, (v) => v * 4, (v) => v + 1],
  clock: [(v) => v + 2, (v) => v - 2, (v) => v + 3, (v) => v - 3, (v) => v + 1],
  number: [(v) => v * 2, (v) => v + 1, (v) => v * 10, (v) => v + 0.5, (v) => v + 3],
  unit: [(v) => v + 1, (v) => v * 2, (v) => v - 1, (v) => v + 2, (v) => v + 3, (v) => v * 3, (v) => v + 4],
};

/** Change every number in `text` with fn, keeping the surrounding format ("10~20%" → "20~30%"). */
function shiftNumbers(text, fn) {
  let ok = true;
  const out = text.replace(/\d+(?:\.\d+)?/g, (m) => {
    const v = fn(Number(m));
    if (!(v > 0) || !Number.isFinite(v)) ok = false;
    return Number.isInteger(v) ? String(v) : v.toFixed(1);
  });
  return ok ? out : null;
}

/**
 * Pick `count` wrong choices of the same type as the answer. A value that also
 * appears in the source sentence would be a second correct answer, so it is
 * never used. When the corpus has too few values of that type, numbers are
 * shifted instead and marked synthetic.
 * @returns {{ text: string, synthetic: boolean }[]}
 */
export function pickDistractors(answer, type, sentence, pools, count, rand = Math.random) {
  const usable = (v) =>
    norm(v) !== norm(answer) &&
    !containsTerm(sentence, v) &&
    !norm(v).includes(norm(answer)) &&
    !norm(answer).includes(norm(v)) &&
    ocrSuspects(v).length === 0;
  let pool = (pools.get(type) ?? []).filter(usable);
  if (type.startsWith('term:') && pool.length < count) {
    const other = [...pools.entries()].filter(([t]) => t.startsWith('term:') && t !== type).flatMap(([, v]) => v);
    pool = [...shuffle(pool, rand), ...shuffle(other.filter(usable), rand)];
  } else {
    pool = shuffle(pool, rand);
  }
  const picked = pool.slice(0, count).map((text) => ({ text, synthetic: false }));
  const kind = type.startsWith('unit:') ? 'unit' : type;
  for (const fn of SHIFTS[kind] ?? []) {
    if (picked.length >= count) break;
    const v = shiftNumbers(answer, fn);
    if (v && usable(v) && !picked.some((p) => p.text === v)) picked.push({ text: v, synthetic: true });
  }
  return picked;
}

/**
 * Rule-based quiz generation (no LLM). Takes sentences from the retrieved chunks
 * round-robin by rank, blanks the best candidate span and adds same-type distractors.
 * @param {{ sources: { id: string, text: string }[], pools: Map<string,string[]>,
 *   count: number, difficulty: 'easy'|'normal'|'hard', seed?: string|number }} opts
 */
export function generateRuleQuiz({ sources, pools, count, difficulty = 'normal', seed = 0 }) {
  const lanes = sources.map((src) =>
    sentencesOf(src.text)
      .map((sen) => ({ src, sen, cands: findCandidates(sen.text) }))
      .filter((x) => x.cands.length));
  const picks = [];
  for (let round = 0; picks.length < count && lanes.some((l) => l.length > round); round++) {
    for (const lane of lanes) {
      if (picks.length >= count) break;
      if (lane[round]) picks.push(lane[round]);
    }
  }
  const nChoices = difficulty === 'easy' ? 3 : 4;
  return picks.map(({ src, sen, cands }, i) => {
    const c = cands[0];
    const rand = seededRandom(`${seed}|${src.id}|${sen.start}`);
    const item = {
      id: `Q${i + 1}`,
      type: difficulty === 'hard' ? 'short' : 'mcq',
      question: `${STEM} ${makeCloze(sen.text, c.text)}`,
      answer: c.text,
      citation: { chunk: src.id, quote: sen.text },
      meta: { answerType: c.type, synthetic: [] },
    };
    if (item.type === 'mcq') {
      const ds = pickDistractors(c.text, c.type, sen.text, pools, nChoices - 1, rand);
      item.choices = shuffle([c.text, ...ds.map((d) => d.text)], rand);
      item.meta.synthetic = ds.filter((d) => d.synthetic).map((d) => d.text);
    }
    return item;
  });
}

/** Character-bigram Jaccard similarity of two questions (stem, spaces, punctuation ignored). */
export function bigramJaccard(a, b) {
  const grams = (s) => {
    const t = String(s).replace(STEM, '').replace(/[\s\p{P}\p{S}_]+/gu, '').toLowerCase();
    const g = new Set();
    for (let i = 0; i < t.length - 1; i++) g.add(t.slice(i, i + 2));
    return g;
  };
  const A = grams(a);
  const B = grams(b);
  if (!A.size && !B.size) return 1;
  let inter = 0;
  for (const x of A) if (B.has(x)) inter++;
  return inter / (A.size + B.size - inter);
}

/**
 * Parse an LLM reply that should be one JSON object. Code fences and text
 * around the object are tolerated (and reported); anything else is an error.
 * @returns {{ ok: true, data: any, notes: string[] } | { ok: false, error: string, notes: string[] }}
 */
export function parseQuizJson(text) {
  const notes = [];
  let t = String(text ?? '').trim();
  const fence = t.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fence) {
    t = fence[1].trim();
    notes.push('코드 펜스(```)를 벗겨 냈다');
  }
  const a = t.indexOf('{');
  const b = t.lastIndexOf('}');
  if (a < 0 || b < a) return { ok: false, error: 'JSON 객체({ … })를 찾지 못했다', notes };
  if (a > 0 || b < t.length - 1) notes.push('JSON 앞뒤의 설명 문장을 무시했다');
  const body = t.slice(a, b + 1);
  try {
    return { ok: true, data: JSON.parse(body), notes };
  } catch (err) {
    const pos = Number(String(err.message).match(/position (\d+)/)?.[1]);
    const near = Number.isFinite(pos) ? ` · 문제 위치 근처: “${body.slice(Math.max(0, pos - 25), pos + 5).replace(/\s+/g, ' ')}”` : '';
    return { ok: false, error: `JSON 파싱 실패: ${err.message}${near}`, notes };
  }
}

/**
 * Validate the quiz JSON schema.
 * { items: [{ type: 'mcq'|'short', question, answer, choices?: string[3..5], citation: { chunk, quote } }] }
 * @returns {{ ok: boolean, errors: { path: string, msg: string }[], itemErrors: Map<number, string[]> }}
 */
export function validateSchema(data) {
  const errors = [];
  const itemErrors = new Map();
  const add = (i, path, msg) => {
    errors.push({ path, msg });
    if (i != null) itemErrors.set(i, [...(itemErrors.get(i) ?? []), `${path.replace(/^items\[\d+\]\.?/, '') || '항목'}: ${msg}`]);
  };
  const str = (v) => typeof v === 'string' && v.trim().length > 0;
  if (!data || typeof data !== 'object' || !Array.isArray(data.items)) {
    add(null, 'items', '배열이어야 한다');
    return { ok: false, errors, itemErrors };
  }
  if (!data.items.length) add(null, 'items', '문항이 하나도 없다');
  data.items.forEach((it, i) => {
    const p = `items[${i}]`;
    if (!it || typeof it !== 'object') {
      add(i, p, '객체여야 한다');
      return;
    }
    if (!['mcq', 'short'].includes(it.type)) add(i, `${p}.type`, '"mcq" 또는 "short"여야 한다');
    if (!str(it.question)) add(i, `${p}.question`, '비어 있지 않은 문자열이어야 한다');
    if (!str(it.answer)) add(i, `${p}.answer`, '비어 있지 않은 문자열이어야 한다');
    if (it.type === 'mcq') {
      if (!Array.isArray(it.choices)) add(i, `${p}.choices`, '객관식에는 선택지 배열이 필요하다');
      else {
        if (it.choices.length < 3 || it.choices.length > 5) add(i, `${p}.choices`, `선택지는 3~5개여야 한다 (지금 ${it.choices.length}개)`);
        if (!it.choices.every(str)) add(i, `${p}.choices`, '모든 선택지는 문자열이어야 한다');
        else if (new Set(it.choices.map(norm)).size !== it.choices.length) add(i, `${p}.choices`, '같은 선택지가 두 번 있다');
        if (str(it.answer) && !it.choices.some((c) => norm(c) === norm(it.answer))) add(i, `${p}.choices`, '정답이 선택지에 없다');
      }
    }
    if (!it.citation || typeof it.citation !== 'object') add(i, `${p}.citation`, '인용 객체가 필요하다');
    else {
      if (!str(it.citation.chunk)) add(i, `${p}.citation.chunk`, '청크 id가 필요하다');
      if (!str(it.citation.quote)) add(i, `${p}.citation.quote`, '근거 문장이 비어 있다');
    }
  });
  return { ok: errors.length === 0, errors, itemErrors };
}

/**
 * Verify quiz items against the sources they were generated from.
 * @param {object[]} items
 * @param {{ chunkById: Map<string,{id:string,doc:string,text:string}>, allowed?: Set<string>|null,
 *   docById?: Map<string,{type?:string,title?:string,broken?:string}>, itemErrors?: Map<number,string[]>,
 *   dupThreshold?: number }} ctx
 * @returns {{ status: 'ok'|'warn'|'fail', checks: { level: 'ok'|'warn'|'fail', code: string, text: string }[] }[]}
 */
export function verifyQuiz(items, { chunkById, allowed = null, docById = new Map(), itemErrors = new Map(), dupThreshold = DUP_THRESHOLD }) {
  return items.map((it, i) => {
    const checks = [];
    const push = (level, code, text) => checks.push({ level, code, text });
    for (const e of itemErrors.get(i) ?? []) push('fail', 'schema', `형식 오류 · ${e}`);

    const cid = it?.citation?.chunk;
    const quote = String(it?.citation?.quote ?? '');
    const answer = String(it?.answer ?? '');
    const chunkRec = cid ? chunkById.get(cid) : null;

    // 1) the cited chunk must be one of the sources the generator was given
    if (cid && !chunkRec) push('fail', 'chunk', `존재하지 않는 청크 id “${cid}”를 인용했다`);
    else if (cid && allowed && !allowed.has(cid)) push('fail', 'chunk', `검색되지 않은 청크 ${cid}를 인용했다 (근거로 준 청크만 쓸 수 있다)`);

    // 2) the quoted span must exist in that chunk
    if (chunkRec && quote.trim()) {
      if (containsTerm(chunkRec.text, quote)) push('ok', 'span', `인용 문장이 청크 ${cid}에 그대로 있다`);
      else {
        const real = [...chunkById.values()].find((c) => containsTerm(c.text, quote));
        push('fail', 'span', real
          ? `인용 청크 오류 · 이 문장은 ${cid}가 아니라 ${real.id}에 있다`
          : `인용 문장이 청크 ${cid}에 없다 (원문을 바꿔 적었거나 지어낸 문장)`);
      }
    }

    // 3) the answer must appear in the quoted sentence
    if (answer.trim() && quote.trim()) {
      if (containsTerm(quote, answer)) push('ok', 'answer', '정답이 근거 문장 안에 있다');
      else push('fail', 'answer', `정답 “${answer}”가 근거 문장에 없다 (환각 의심)`);
    }

    // 4) no distractor may also be supported by the sentence
    if (it?.type === 'mcq' && Array.isArray(it.choices) && quote.trim()) {
      const rest = answer.trim() ? quote.replace(termRegex(answer, 'g'), ' ') : quote;
      const also = it.choices.filter((c) => typeof c === 'string' && norm(c) !== norm(answer) && containsTerm(rest, c));
      if (also.length) push('fail', 'distractor', `오답 선택지 ${also.map((c) => `“${c}”`).join(', ')}도 근거 문장에 있다 (정답이 둘)`);
      else push('ok', 'distractor', '오답 선택지가 근거 문장과 겹치지 않는다');
    }

    // 5) the question must not give the answer away
    if (answer.trim() && containsTerm(String(it?.question ?? ''), answer)) push('warn', 'leak', '문제에 정답이 그대로 들어 있다');

    // 6) broken / OCR-damaged documents
    const doc = chunkRec ? docById.get(chunkRec.doc) : null;
    const badAnswer = [answer, ...(Array.isArray(it?.choices) ? it.choices : [])].flatMap((x) => ocrSuspects(x));
    if (badAnswer.length) push('fail', 'ocr', `정답·선택지에 OCR 오류 의심 문자열: ${[...new Set(badAnswer)].join(', ')}`);
    if (doc?.type === 'broken') push('warn', 'broken', `깨진 문서에서 만든 문항이다${doc.broken ? ` (${doc.broken})` : ''}`);
    else if (ocrSuspects(quote).length) push('warn', 'broken', `근거 문장에 OCR 오류 의심: ${ocrSuspects(quote).join(', ')}`);

    // 7) near-duplicates of an earlier item
    for (let j = 0; j < i; j++) {
      const sim = bigramJaccard(it?.question ?? '', items[j]?.question ?? '');
      if (sim >= dupThreshold) {
        push('warn', 'dup', `${items[j].id ?? `Q${j + 1}`} 문항과 거의 같다 (유사도 ${sim.toFixed(2)})`);
        break;
      }
    }

    const status = checks.some((c) => c.level === 'fail') ? 'fail' : checks.some((c) => c.level === 'warn') ? 'warn' : 'ok';
    return { status, checks };
  });
}

/** Prompt for the LLM generator: retrieved chunks are the only allowed source, output is JSON only. */
export function buildQuizPrompt(sources, { count, difficulty }) {
  const format = difficulty === 'hard'
    ? '단답형 (type "short", choices 없음)'
    : `객관식 ${difficulty === 'easy' ? 3 : 4}지선다 (type "mcq", choices ${difficulty === 'easy' ? 3 : 4}개)`;
  return {
    system: [
      '너는 교사를 돕는 퀴즈 출제 도우미다. 반드시 [근거 청크]에 적힌 내용만 사용하고, 근거에 없는 사실로 문항을 만들지 않는다.',
      '출력은 설명 없이 JSON 객체 하나만 쓴다. 형식:',
      '{"items":[{"type":"mcq","question":"…","answer":"…","choices":["…","…","…","…"],"citation":{"chunk":"청크 id","quote":"청크에서 그대로 복사한 한 문장"}}]}',
      '규칙: 1) citation.quote는 해당 청크 원문에서 글자 그대로 복사한 한 문장이다. 2) answer는 quote 안에 글자 그대로 들어 있어야 한다.',
      '3) 오답 선택지는 quote에 나오지 않는 말로 만든다(정답이 둘이 되면 안 된다). 4) 같은 내용을 묻는 문항을 두 번 만들지 않는다.',
    ].join('\n'),
    messages: [{
      role: 'user',
      content: `[근거 청크]\n${sources.map((s) => `(id: ${s.id})\n${s.text.trim()}`).join('\n\n')}\n\n문항 수: ${count}\n문항 형식: ${format}\n난이도: ${{ easy: '쉬움', normal: '보통', hard: '어려움' }[difficulty]}`,
    }],
  };
}

/** Markdown export (answers + citations after the questions, for a teacher's review copy). */
export function toMarkdown(quiz) {
  const L = [`# ${quiz.title}`, '', `- 생성: ${quiz.generator}`, `- 근거 청크: ${quiz.sources.join(', ')}`,
    '- 자동 생성 문항이다. 수업에 쓰기 전에 교사가 검토한다.', ''];
  quiz.items.forEach((it, i) => {
    L.push(`## ${i + 1}. ${it.question}`, '');
    if (it.type === 'mcq') it.choices.forEach((c, k) => L.push(`${'ABCDE'[k]}) ${c}`));
    else L.push('(단답형)');
    L.push('');
  });
  L.push('---', '', '## 정답과 근거', '');
  quiz.items.forEach((it, i) => {
    L.push(`${i + 1}. **${it.answer}** — ${it.citation.chunk} · 검증 ${it.verify.status} · 교사 검토 ${it.review === 'approved' ? '승인' : '대기'}`,
      '', `   > ${it.citation.quote}`, '');
  });
  return `${L.join('\n')}\n`;
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

// ------------------------------------------------------------------ templates

const TOPICS = [
  { id: 'chunking', label: '청킹', query: '청킹 오버랩 청크 크기' },
  { id: 'facility', label: '실습실 규정', query: '실습실 규정' },
  { id: 'eval', label: 'RAG 평가', query: 'RAG 평가 골든셋 Recall@k Faithfulness' },
  { id: 'course', label: '수업·성적', query: '수업 시간 성적 평가 수료 요건' },
  { id: 'scan', label: '장학금 (스캔 깨짐)', query: '장학금 지급액' },
];

const GEN_LABEL = { rule: '규칙 기반 생성(키 없이)', llm: 'LLM 생성', example: `실패 예시 (${EXAMPLE_LABEL})` };

let templates = null;
function getTemplates() {
  if (templates) return templates;
  const main = document.createElement('template');
  main.innerHTML = `
  <div class="widget w15" data-w15>
    <h3 class="widget__title">RAG 퀴즈 생성기 · 생성 → 검증</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> 문서셋을 불러오는 중…
    </div>
    <div data-slot="body" hidden>
      <div class="btn-row w15-topics" data-slot="topics" role="group" aria-label="주제 프리셋"></div>
      <div class="widget__controls">
        <label class="field">
          <span class="field__label">생성 방식</span>
          <select data-in="gen"></select>
        </label>
        <label class="field" data-show="search">
          <span class="field__label">주제 · 검색어</span>
          <input type="text" data-in="query" autocomplete="off">
        </label>
        <label class="field" data-show="search">
          <span class="field__label">근거 범위</span>
          <select data-in="doc"></select>
        </label>
        <label class="field" data-show="search">
          <span class="field__label">검색 방식</span>
          <select data-in="retrieval">
            <option value="bm25">BM25 (즉시)</option>
            <option value="hybrid">하이브리드: BM25 + 벡터 (모델 약 118MB)</option>
          </select>
        </label>
        <label class="field" data-show="search">
          <span class="field__label">근거 청크 수 <output data-out="k"></output></span>
          <input type="range" data-in="k" min="1" max="5" step="1">
        </label>
        <label class="field" data-show="search">
          <span class="field__label">문항 수 <output data-out="count"></output></span>
          <input type="range" data-in="count" min="1" max="8" step="1">
        </label>
        <label class="field" data-show="search">
          <span class="field__label">난이도</span>
          <select data-in="difficulty">
            <option value="easy">쉬움 · 3지선다</option>
            <option value="normal">보통 · 4지선다</option>
            <option value="hard">어려움 · 단답형</option>
          </select>
        </label>
        <label class="field" data-show="example" hidden>
          <span class="field__label">예시 세트</span>
          <select data-in="example"></select>
        </label>
      </div>
      <div data-slot="model-status"></div>

      <details class="w15-llm" data-slot="llm-box">
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
        </div>
        <p class="w15-note" data-slot="keystate" role="status"></p>
      </details>

      <div class="btn-row w15-actions">
        <button type="button" class="btn primary" data-act="llm" hidden>LLM으로 생성</button>
        <span class="w15-seg" role="group" aria-label="보기">
          <button type="button" class="btn small" data-view="review" aria-pressed="true">검토 보기</button>
          <button type="button" class="btn small" data-view="solve" aria-pressed="false">퀴즈 풀기</button>
        </span>
        <button type="button" class="btn small ghost" data-act="shuffle">선택지 다시 섞기</button>
      </div>

      <div data-slot="banner" aria-live="polite"></div>
      <ol class="w15-items" data-slot="items"></ol>
      <div class="w15-solvebar" data-slot="solvebar" hidden>
        <button type="button" class="btn primary" data-act="grade">채점하기</button>
        <button type="button" class="btn small ghost" data-act="reset">다시 풀기</button>
        <span data-slot="score" role="status" aria-live="polite"></span>
      </div>

      <div class="w15-export">
        <label class="w15-check"><input type="checkbox" data-in="exclude" checked> 검증 실패 문항은 내보내지 않기</label>
        <div class="btn-row">
          <button type="button" class="btn small" data-act="json">JSON 내려받기</button>
          <button type="button" class="btn small" data-act="md">Markdown 내려받기</button>
        </div>
      </div>

      <div data-slot="out-inline"></div>
    </div>
  </div>`;
  const out = document.createElement('template');
  out.innerHTML = `
  <div class="w15-out">
    <div class="w15-pipe" data-slot="pipe" aria-label="생성 파이프라인 단계"></div>
    <div data-slot="verdict"></div>
    <div class="stat-row" data-slot="stats"></div>
    <details class="w15-raw" data-slot="raw-box" hidden>
      <summary>모델 원문 출력 (JSON)</summary>
      <div data-slot="parse"></div>
      <pre><code data-slot="raw"></code></pre>
    </details>
    <h4 data-slot="src-title">근거 청크</h4>
    <ol class="w15-sources" data-slot="sources"></ol>
  </div>`;
  templates = { main, out };
  return templates;
}

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ topic?: string, gen?: 'rule'|'example', outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const T = getTemplates();
  const root = T.main.content.firstElementChild.cloneNode(true);
  const out = T.out.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const uid = `w15-${++seq}`;
  const outputId = `w15:${seq}`;
  const st = { ctrl, outputId, alive: true, llm: null };
  state.set(el, st);

  let corpus;
  let examples;
  try {
    [corpus, examples] = await Promise.all([
      fetchJson(new URL('corpus/corpus.json', DATA), ctrl.signal),
      fetchJson(new URL('w15/llm-examples.json', DATA), ctrl.signal),
    ]);
  } catch (err) {
    if (err.name === 'AbortError') return;
    $('[data-slot=status]').innerHTML = `<div class="widget__error" role="alert">
      문서셋을 불러오지 못했다 (${escapeHtml(err.message)}).<br>
      파일을 더블클릭해 <code>file://</code>로 열었다면 ES Module과 fetch가 동작하지 않는다.
      저장소 폴더에서 <code>python -m http.server</code>를 실행하고 <code>http://localhost:8000</code>으로 연다.</div>`;
    return;
  }

  // ---- index: chunk every document (offsets kept), BM25 over chunks
  const docById = new Map(corpus.documents.map((d) => [d.id, d]));
  const chunks = [];
  for (const d of corpus.documents) {
    for (const c of chunk(d.text, CHUNK_CFG)) chunks.push({ id: `${d.id}#${c.index}`, doc: d.id, title: d.title, index: c.index, start: c.start, end: c.end, text: c.text });
  }
  const chunkById = new Map(chunks.map((c) => [c.id, c]));
  const bm25 = new BM25().add(chunks.map(({ id, text }) => ({ id, text })));
  const pools = buildPools(corpus.documents);

  const s = {
    gen: options.gen === 'example' ? 'example' : 'rule',
    topic: TOPICS.some((t) => t.id === options.topic) ? options.topic : 'chunking',
    query: '',
    doc: '',
    retrieval: 'bm25',
    k: 3,
    count: 5,
    difficulty: 'normal',
    example: examples.sets[0].id,
    seed: 0,
    view: 'review',
    exclude: true,
  };
  s.query = TOPICS.find((t) => t.id === s.topic).query;

  const vec = { ready: false, loading: false, error: '', store: null, qcache: new Map() };
  let llmResult = null; // { key, raw, model }
  let current = null; // what is on screen: { items, verify, sources, allowed, raw, parse, genLabel, kind }
  const approvals = new Map();
  const answers = new Map();
  let graded = false;

  // ---- controls
  $('[data-slot=topics]').innerHTML = TOPICS
    .map((t) => `<button type="button" class="btn small" data-topic="${t.id}" aria-pressed="false">${escapeHtml(t.label)}</button>`)
    .join('');
  $('[data-in=gen]').innerHTML = Object.entries(GEN_LABEL).map(([k, v]) => `<option value="${k}">${escapeHtml(v)}</option>`).join('');
  $('[data-in=doc]').innerHTML = '<option value="">검색으로 찾기 (상위 k개 청크)</option>' + corpus.documents
    .map((d) => `<option value="${d.id}">문서 전체: ${escapeHtml(d.title)}${d.type === 'broken' ? ' ⚠' : ''}</option>`)
    .join('');
  $('[data-in=example]').innerHTML = examples.sets.map((x) => `<option value="${x.id}">${escapeHtml(x.label)}</option>`).join('');
  const provSel = $('[data-in=provider]');
  provSel.innerHTML = Object.entries(PROVIDERS).map(([k, p]) => `<option value="${k}">${escapeHtml(p.label)}</option>`).join('');
  const fillModels = () => {
    $('[data-in=model]').innerHTML = PROVIDERS[provSel.value].models.map((m) => `<option value="${m}">${escapeHtml(m)}</option>`).join('');
  };
  fillModels();

  function syncInputs() {
    $('[data-in=gen]').value = s.gen;
    $('[data-in=query]').value = s.query;
    $('[data-in=doc]').value = s.doc;
    $('[data-in=retrieval]').value = s.retrieval;
    $('[data-in=k]').value = s.k;
    $('[data-in=count]').value = s.count;
    $('[data-in=difficulty]').value = s.difficulty;
    $('[data-in=example]').value = s.example;
    $('[data-in=exclude]').checked = s.exclude;
    $('[data-out=k]').textContent = s.doc ? '문서 전체' : s.k;
    $('[data-out=count]').textContent = s.count;
    $('[data-in=k]').disabled = Boolean(s.doc);
    root.querySelectorAll('[data-show=search]').forEach((x) => { x.hidden = s.gen === 'example'; });
    root.querySelector('[data-show=example]').hidden = s.gen !== 'example';
    $('[data-slot=topics]').hidden = s.gen === 'example';
    root.querySelectorAll('[data-topic]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.topic === s.topic && !s.doc)));
    root.querySelectorAll('[data-view]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === s.view)));
    const llmBtn = $('[data-act=llm]');
    llmBtn.hidden = s.gen !== 'llm';
    llmBtn.disabled = !safeHasKey(provSel.value);
    $('[data-act=shuffle]').hidden = s.gen !== 'rule';
  }

  function syncKeyState() {
    const has = safeHasKey(provSel.value);
    $('[data-slot=keystate]').innerHTML = has
      ? `<span class="chip ok">🔑 ${escapeHtml(PROVIDERS[provSel.value].label)} 키 저장됨 · 이 탭에서만</span> 생성 방식을 “LLM 생성”으로 두고 버튼을 누른다.`
      : '<span class="chip warn">🔑 키 없음</span> 규칙 기반 생성과 실패 예시는 키 없이 동작한다.';
    syncInputs();
  }

  // ---- retrieval
  async function retrieve() {
    if (s.doc) return { list: chunks.filter((c) => c.doc === s.doc).map((c) => ({ ...c, score: null })), note: '' };
    const q = s.query.trim();
    if (!q) return { list: [], note: '' };
    const bm = bm25.search(q, 20);
    if (s.retrieval !== 'hybrid' || !vec.ready) {
      const note = s.retrieval === 'hybrid'
        ? (vec.error ? '벡터 모델을 불러오지 못해 BM25 결과만 쓴다.' : '벡터 모델 준비 중 — 지금은 BM25 결과다.')
        : '';
      return { list: bm.slice(0, s.k).map((r) => ({ ...chunkById.get(r.id), score: r.score, how: 'BM25' })), note };
    }
    let vs;
    try {
      if (!vec.qcache.has(q)) vec.qcache.set(q, await embed(q, { role: 'query' }));
      vs = vec.store.search(vec.qcache.get(q), 20);
    } catch {
      return { list: bm.slice(0, s.k).map((r) => ({ ...chunkById.get(r.id), score: r.score, how: 'BM25' })), note: '질의 임베딩에 실패해 BM25 결과만 쓴다.' };
    }
    const fused = rrf([bm, vs], { topK: s.k });
    return { list: fused.map((r) => ({ ...chunkById.get(r.id), score: r.score, how: 'RRF' })), note: '' };
  }

  async function prepareModel() {
    if (vec.ready || vec.loading) return;
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
        onProgress: (done, total) => { note.lastElementChild.textContent = `청크 ${done} / ${total}개 임베딩 중…`; },
      });
      note.remove();
      const store = new VectorStore();
      chunks.forEach((c, i) => store.add(c.id, vectors[i]));
      vec.store = store;
      vec.ready = true;
    } catch (err) {
      vec.error = err?.message ?? String(err);
    } finally {
      vec.loading = false;
    }
    if (st.alive) update();
  }

  // ---- building what is shown
  const settingsKey = () => JSON.stringify([s.query, s.doc, s.retrieval, vec.ready, s.k, s.count, s.difficulty]);

  function fromRaw(raw, sources, allowed) {
    const parse = parseQuizJson(raw);
    if (!parse.ok) return { items: [], verify: [], parse, raw, sources, allowed };
    const schema = validateSchema(parse.data);
    const items = (Array.isArray(parse.data?.items) ? parse.data.items : []).map((it, i) => ({ ...(it ?? {}), id: `Q${i + 1}` }));
    const verify = verifyQuiz(items, { chunkById, allowed, docById, itemErrors: schema.itemErrors });
    return { items, verify, parse: { ...parse, schema }, raw, sources, allowed };
  }

  let tick = 0;
  async function update() {
    if (!st.alive) return;
    const my = ++tick;
    syncInputs();
    let view;
    if (s.gen === 'example') {
      const set = examples.sets.find((x) => x.id === s.example) ?? examples.sets[0];
      const sources = set.sources.map((id) => chunkById.get(id)).filter(Boolean).map((c) => ({ ...c, score: null }));
      const raw = set.raw ?? JSON.stringify(set.output, null, 2);
      view = { ...fromRaw(raw, sources, new Set(set.sources)), kind: 'example', genLabel: EXAMPLE_LABEL, set };
    } else {
      const { list, note } = await retrieve();
      if (my !== tick || !st.alive) return;
      const allowed = new Set(list.map((c) => c.id));
      if (s.gen === 'llm' && llmResult && llmResult.key === settingsKey()) {
        view = { ...fromRaw(llmResult.raw, list, allowed), kind: 'llm', genLabel: `LLM 생성 · ${llmResult.model}`, note };
      } else {
        const items = generateRuleQuiz({ sources: list, pools, count: s.count, difficulty: s.difficulty, seed: s.seed });
        const verify = verifyQuiz(items, { chunkById, allowed, docById });
        let banner = '';
        if (s.gen === 'llm') {
          banner = safeHasKey(provSel.value)
            ? (llmResult ? '설정이 바뀌어 이전 LLM 결과를 치웠다. “LLM으로 생성”을 다시 누른다. 아래는 비교용 규칙 기반 결과다.'
              : '“LLM으로 생성”을 누르면 아래 근거 청크만 주고 JSON으로 문항을 받는다. 아래는 비교용 규칙 기반 결과다.')
            : 'API 키가 없다. “LLM 설정”에서 키를 저장하면 생성 버튼이 켜진다. 아래는 규칙 기반 결과다.';
        }
        view = { items, verify, sources: list, allowed, kind: 'rule', genLabel: GEN_LABEL.rule, note, banner };
      }
    }
    current = view;
    answers.clear();
    graded = false;
    render();
  }

  // ---- rendering
  function counts() {
    const { items, verify } = current;
    return {
      failN: verify.filter((x) => x.status === 'fail').length,
      warnN: verify.filter((x) => x.status === 'warn').length,
      okN: verify.filter((x) => x.status === 'ok').length,
      approvedN: items.filter((it, i) => verify[i].status !== 'fail' && approvals.get(sig(it))).length,
    };
  }

  function render() {
    renderItems();
    renderPanel();
  }

  function renderItems() {
    const v = current;
    const { items, verify } = v;
    const { failN } = counts();

    // banner in the widget
    const bits = [];
    if (v.kind === 'rule') {
      bits.push(`<p class="w15-note"><span class="chip accent">${GEN_LABEL.rule}</span> 근거 문장의 숫자·기술 용어를 빈칸으로 바꾸고, 문서셋에서 같은 종류의 다른 값을 골라 오답으로 쓴다. 모델을 쓰지 않으므로 과정이 전부 보인다.</p>`);
    } else if (v.kind === 'llm') {
      bits.push(`<p class="w15-note"><span class="chip ok">${escapeHtml(v.genLabel)}</span> 근거 청크만 주고 JSON으로 받은 문항이다. 검증 결과를 믿기 전에 원문 출력도 확인한다.</p>`);
    }
    if (v.kind === 'example') {
      bits.push(`<div class="callout callout--danger"><span class="callout__title"><span class="chip warn">${EXAMPLE_LABEL}</span> 일부러 틀리게 쓴 LLM 스타일 출력</span>
        실제 모델 호출 기록이 아니다. LLM이 자주 내는 실수(정답 환각, 정답이 둘인 선택지, 엉뚱한 청크 인용, 중복, 형식 위반)를 손으로 재현했다. 검증기가 어떤 규칙으로 잡는지 본다.</div>`);
    }
    if (v.banner) bits.push(`<div class="callout"><span class="callout__title">LLM 생성 대기</span>${escapeHtml(v.banner)}</div>`);
    if (v.note) bits.push(`<p class="w15-note">${escapeHtml(v.note)}</p>`);
    if (v.parse && !v.parse.ok) {
      bits.push(`<div class="callout callout--danger"><span class="callout__title">JSON 파싱 실패 — 문항을 하나도 만들 수 없다</span>${escapeHtml(v.parse.error)}
        <br>구조화 출력은 파싱부터 통과해야 한다. 실행 결과 창의 “모델 원문 출력”에서 원문을 확인한다.</div>`);
    }
    if (!v.sources.length && v.kind !== 'example') {
      bits.push('<div class="callout"><span class="callout__title">근거가 없다</span>검색어와 겹치는 청크가 없어 문항을 만들지 않았다. 근거 없이 만들면 그것이 곧 환각이다.</div>');
    } else if (!items.length && (!v.parse || v.parse.ok)) {
      bits.push(v.kind === 'rule'
        ? '<div class="callout"><span class="callout__title">만들 문항이 없다</span>근거 청크에 숫자·기술 용어가 없어 규칙 기반 생성기가 빈칸을 고르지 못했다. 규칙 기반의 한계다.</div>'
        : '<div class="callout callout--danger"><span class="callout__title">문항이 없다</span>JSON은 읽었지만 스키마에 맞는 <code>items</code> 배열이 없다. 결과 창의 “모델 원문 출력”을 확인한다.</div>');
    }
    $('[data-slot=banner]').innerHTML = bits.join('');

    // items
    const solve = s.view === 'solve';
    const list = $('[data-slot=items]');
    const shown = solve ? items.map((it, i) => [it, i]).filter(([, i]) => verify[i].status !== 'fail') : items.map((it, i) => [it, i]);
    list.innerHTML = shown.map(([it, i]) => (solve ? solveItemHtml(it, i) : reviewItemHtml(it, i, verify[i]))).join('');
    if (solve && failN) {
      list.insertAdjacentHTML('afterbegin', `<li class="w15-skip">검증에 실패한 ${failN}개 문항은 풀기에서 뺐다.</li>`);
    }
    $('[data-slot=solvebar]').hidden = !solve || shown.length === 0;
    $('[data-slot=score]').innerHTML = graded ? scoreHtml(shown) : '';
  }

  // Output panel: pipeline, verdict, stats, raw JSON, sources. Re-rendered alone
  // when a teacher approval changes so the focused checkbox is not replaced.
  function renderPanel() {
    const v = current;
    const { items, verify } = v;
    const { failN, warnN, okN, approvedN } = counts();
    const stage = (label, val, cls = '') => `<span class="w15-pipe__step ${cls}"><b>${label}</b>${val}</span>`;
    $('[data-slot=pipe]').innerHTML = [
      stage('검색', `${v.sources.length}개 청크`),
      stage('생성', `${items.length}문항`, v.parse && !v.parse.ok ? 'is-fail' : ''),
      stage('자동 검증', `통과 ${okN} · 경고 ${warnN} · 실패 ${failN}`, failN ? 'is-fail' : warnN ? 'is-warn' : 'is-ok'),
      stage('교사 검토', `승인 ${approvedN} / ${items.length - failN}`),
    ].join('<span class="w15-pipe__arrow" aria-hidden="true">→</span>');

    $('[data-slot=stats]').innerHTML = [
      ['생성 방식', v.kind === 'rule' ? '규칙' : v.kind === 'llm' ? 'LLM' : '예시'],
      ['문항', items.length],
      ['검증 실패', failN],
      ['경고', warnN],
    ].map(([l, val]) => `<div class="stat"><span class="stat__label">${l}</span><span class="stat__value">${val}</span></div>`).join('');

    let verdict;
    if (v.parse && !v.parse.ok) {
      verdict = '<div class="callout callout--danger"><span class="callout__title">형식 검증 실패</span>모델 출력이 JSON이 아니다. 자동 검증 단계까지 가지 못했다.</div>';
    } else if (!items.length) {
      verdict = '<div class="callout"><span class="callout__title">검증할 문항 없음</span></div>';
    } else if (failN) {
      verdict = `<div class="callout callout--danger"><span class="callout__title">검증 실패 ${failN}문항 — 그대로 내보내면 안 된다</span>
        빨간 문항은 근거와 맞지 않는다. 내보내기에서 자동으로 빠진다(설정 변경 가능).</div>`;
    } else if (warnN) {
      verdict = `<div class="callout"><span class="callout__title">경고 ${warnN}문항 — 사람이 확인한다</span>
        규칙상 틀리지는 않았지만 중복이거나 깨진 문서에서 나왔다. 교사 검토에서 판단한다.</div>`;
    } else {
      verdict = `<div class="callout callout--ok"><span class="callout__title">자동 검증 통과 ${okN}문항</span>
        모든 정답이 인용 문장 안에 있고, 오답이 근거와 겹치지 않는다. 그래도 문제가 좋은지는 규칙이 판단하지 못한다 — 교사 검토가 남아 있다.</div>`;
    }
    $('[data-slot=verdict]').innerHTML = verdict;

    const rawBox = $('[data-slot=raw-box]');
    rawBox.hidden = !v.raw;
    if (v.raw) {
      $('[data-slot=raw]').textContent = v.raw;
      const p = v.parse;
      const lines = [...(p?.notes ?? []).map((n) => `<li>ℹ ${escapeHtml(n)}</li>`)];
      if (p && !p.ok) lines.push(`<li class="w15-bad">✗ ${escapeHtml(p.error)}</li>`);
      if (p?.schema) {
        lines.push(p.schema.ok ? '<li class="w15-good">✓ 스키마 검증 통과</li>'
          : p.schema.errors.map((e) => `<li class="w15-bad">✗ ${escapeHtml(e.path)} — ${escapeHtml(e.msg)}</li>`).join(''));
      }
      $('[data-slot=parse]').innerHTML = lines.length ? `<ul class="w15-parse">${lines.join('')}</ul>` : '';
      if (p && !p.ok) rawBox.open = true;
    }

    // sources with cited sentences highlighted
    $('[data-slot=src-title]').textContent = v.kind === 'example'
      ? `근거 청크 · 예시 출력에 준 ${v.sources.length}개`
      : s.doc ? `근거 청크 · 문서 전체 ${v.sources.length}개` : `근거 청크 · 상위 ${v.sources.length}개 (재귀 분할 ${CHUNK_CFG.size}자)`;
    $('[data-slot=sources]').innerHTML = v.sources.map((src) => {
      const cites = items.map((it, i) => ({ it, i })).filter(({ it }) => it?.citation?.chunk === src.id);
      const spans = [];
      for (const { it, i } of cites) {
        const q = String(it.citation.quote ?? '');
        const m = q.trim() ? src.text.match(termRegex(q)) : null;
        if (m) spans.push({ start: m.index, end: m.index + m[0].length, cls: verify[i].status, label: it.id });
      }
      const doc = docById.get(src.doc);
      return `<li class="w15-src">
        <div class="chunk-meta"><span><b>${escapeHtml(src.id)}</b></span><span>${escapeHtml(src.title)}</span>${src.score != null ? `<span>${src.how ?? 'BM25'} ${src.score.toFixed(src.how === 'RRF' ? 4 : 2)}</span>` : ''}${doc?.type === 'broken' ? '<span class="w15-bad">⚠ 깨진 문서</span>' : ''}</div>
        <div class="w15-src__cites">${cites.length ? cites.map(({ it, i }) => `<span class="w15-tag w15-tag--${verify[i].status}">${escapeHtml(it.id)}</span>`).join('') : '<span class="chip">인용되지 않음</span>'}</div>
        <div class="w15-src__text">${markSpans(src.text, spans)}</div>
      </li>`;
    }).join('');
  }

  function reviewItemHtml(it, i, vr) {
    const typeChip = it.type === 'mcq' ? `객관식 ${it.choices?.length ?? '?'}지` : it.type === 'short' ? '단답형' : escapeHtml(String(it.type ?? '형식 없음'));
    const statusChip = { ok: '<span class="badge-ok">검증 통과</span>', warn: '<span class="chip warn">경고</span>', fail: '<span class="w15-failchip">검증 실패</span>' }[vr.status];
    const syn = new Set(it.meta?.synthetic ?? []);
    const choices = Array.isArray(it.choices)
      ? `<ol class="w15-choices" type="A">${it.choices.map((c) => {
        const right = norm(c) === norm(it.answer);
        return `<li class="${right ? 'is-answer' : ''}">${escapeHtml(c)}${right ? ' <span class="w15-good">✓ 정답</span>' : ''}${syn.has(c) ? ' <span class="chip">생성된 수치</span>' : ''}</li>`;
      }).join('')}</ol>`
      : `<p class="w15-ans">정답: <b>${escapeHtml(it.answer ?? '(없음)')}</b></p>`;
    const cite = it.citation
      ? `<div class="w15-cite"><span class="w15-cite__id">${escapeHtml(it.citation.chunk ?? '(청크 없음)')}</span> ${it.citation.quote ? `「${escapeHtml(it.citation.quote)}」` : '<i>(근거 문장 없음)</i>'}</div>`
      : '<div class="w15-cite"><i>(인용 없음)</i></div>';
    const checks = vr.checks.map((c) => `<li class="w15-check--${c.level}"><span aria-hidden="true">${c.level === 'ok' ? '✓' : c.level === 'warn' ? '!' : '✗'}</span> ${escapeHtml(c.text)}</li>`).join('');
    const typeInfo = it.meta?.answerType ? `<span class="chip">빈칸: ${escapeHtml(typeLabel(it.meta.answerType))}</span>` : '';
    const approved = approvals.get(sig(it));
    return `<li class="w15-item w15-item--${vr.status}">
      <div class="w15-item__head"><b>${escapeHtml(it.id)}</b><span class="chip">${typeChip}</span>${typeInfo}${statusChip}</div>
      <p class="w15-q">${questionHtml(it.question)}</p>
      ${choices}
      ${cite}
      <ul class="w15-checks">${checks}</ul>
      <label class="w15-check w15-approve"><input type="checkbox" data-approve="${i}" ${approved ? 'checked' : ''} ${vr.status === 'fail' ? 'disabled' : ''}>
        교사 검토: 승인${vr.status === 'fail' ? ' (검증 실패 문항은 먼저 고쳐야 한다)' : ''}</label>
    </li>`;
  }

  function solveItemHtml(it, i) {
    const name = `${uid}-q${i}`;
    const res = graded ? gradeOne(it, answers.get(i)) : null;
    const body = it.type === 'mcq' && Array.isArray(it.choices)
      ? `<fieldset class="w15-fs"><legend class="visually-hidden">${escapeHtml(it.id)} 선택지</legend>${it.choices.map((c, k) => {
        const checked = answers.get(i) === c ? 'checked' : '';
        const mark = graded && norm(c) === norm(it.answer) ? ' <span class="w15-good">✓ 정답</span>' : '';
        return `<label class="w15-opt"><input type="radio" name="${name}" value="${escapeHtml(c)}" data-answer="${i}" ${checked} ${graded ? 'disabled' : ''}> ${'ABCDE'[k]}. ${escapeHtml(c)}${mark}</label>`;
      }).join('')}</fieldset>`
      : `<label class="field"><span class="field__label">답</span><input type="text" data-answer="${i}" value="${escapeHtml(answers.get(i) ?? '')}" autocomplete="off" ${graded ? 'disabled' : ''}></label>`;
    const after = graded
      ? `<p class="${res ? 'w15-good' : 'w15-bad'}">${res ? '✓ 맞았다' : `✗ 틀렸다 · 정답: ${escapeHtml(it.answer)}`}</p>
         <div class="w15-cite"><span class="w15-cite__id">${escapeHtml(it.citation?.chunk ?? '')}</span> 「${escapeHtml(it.citation?.quote ?? '')}」</div>`
      : '';
    return `<li class="w15-item w15-item--solve${graded ? (res ? ' is-right' : ' is-wrong') : ''}">
      <div class="w15-item__head"><b>${escapeHtml(it.id)}</b></div>
      <p class="w15-q">${questionHtml(it.question)}</p>
      ${body}${after}
    </li>`;
  }

  function scoreHtml(shown) {
    const n = shown.length;
    const right = shown.filter(([it, i]) => gradeOne(it, answers.get(i))).length;
    return `<span class="chip ${right === n ? 'ok' : 'accent'}">점수 ${right} / ${n}</span>`;
  }

  // ---- export
  function exportQuiz() {
    const v = current;
    const rows = v.items.map((it, i) => ({ it, vr: v.verify[i] })).filter(({ vr }) => !(s.exclude && vr.status === 'fail'));
    const topic = s.gen === 'example' ? v.set.label : s.doc ? docById.get(s.doc)?.title : s.query;
    return {
      title: `RAG 퀴즈 — ${topic}`,
      generator: v.kind === 'example' ? `${EXAMPLE_LABEL}` : v.genLabel,
      createdAt: new Date().toISOString(),
      retrieval: v.kind === 'example' ? 'example' : s.doc ? `document:${s.doc}` : `${s.retrieval}, k=${s.k}`,
      sources: v.sources.map((c) => c.id),
      notice: '자동 생성 문항이다. 수업·평가에 쓰기 전에 교사가 정답과 근거를 검토한다. 원문: RAG Lab 공통 문서셋.',
      items: rows.map(({ it, vr }) => ({
        id: it.id,
        type: it.type,
        question: it.question,
        ...(Array.isArray(it.choices) ? { choices: it.choices } : {}),
        answer: it.answer,
        citation: it.citation,
        verify: { status: vr.status, issues: vr.checks.filter((c) => c.level !== 'ok').map((c) => c.text) },
        review: approvals.get(sig(it)) ? 'approved' : 'pending',
      })),
    };
  }

  // ---- events
  const on = (target, type, fn) => target.addEventListener(type, fn, { signal: ctrl.signal });
  on($('[data-slot=topics]'), 'click', (e) => {
    const b = e.target.closest('[data-topic]');
    if (!b) return;
    const t = TOPICS.find((x) => x.id === b.dataset.topic);
    Object.assign(s, { topic: t.id, query: t.query, doc: '' });
    update();
  });
  on($('[data-in=gen]'), 'change', (e) => { s.gen = e.target.value; if (s.gen === 'llm') $('[data-slot=llm-box]').open = true; update(); });
  let qTimer = 0;
  on($('[data-in=query]'), 'input', (e) => {
    s.query = e.target.value;
    s.topic = TOPICS.find((t) => t.query === s.query)?.id ?? '';
    clearTimeout(qTimer);
    qTimer = setTimeout(update, 150);
  });
  on($('[data-in=doc]'), 'change', (e) => { s.doc = e.target.value; update(); });
  on($('[data-in=retrieval]'), 'change', (e) => {
    s.retrieval = e.target.value;
    if (s.retrieval === 'hybrid') prepareModel();
    update();
  });
  on($('[data-in=k]'), 'input', (e) => { s.k = Number(e.target.value); update(); });
  on($('[data-in=count]'), 'input', (e) => { s.count = Number(e.target.value); update(); });
  on($('[data-in=difficulty]'), 'change', (e) => { s.difficulty = e.target.value; update(); });
  on($('[data-in=example]'), 'change', (e) => { s.example = e.target.value; update(); });
  on($('[data-in=exclude]'), 'change', (e) => { s.exclude = e.target.checked; });
  on(provSel, 'change', () => { fillModels(); syncKeyState(); });
  on($('[data-act=save-key]'), 'click', () => {
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
    syncKeyState();
    update();
  });
  on($('[data-act=clear-key]'), 'click', () => {
    try { clearKey(provSel.value); } catch { /* storage unavailable */ }
    syncKeyState();
    update();
  });
  on(root, 'click', (e) => {
    const vb = e.target.closest('[data-view]');
    if (vb) {
      s.view = vb.dataset.view;
      graded = false;
      syncInputs();
      render();
      return;
    }
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (act === 'shuffle') { s.seed++; update(); }
    else if (act === 'grade') { graded = true; render(); }
    else if (act === 'reset') { graded = false; answers.clear(); render(); }
    else if (act === 'json') {
      const q = exportQuiz();
      download('rag-quiz.json', `${JSON.stringify(q, null, 2)}\n`, 'application/json');
    } else if (act === 'md') {
      download('rag-quiz.md', toMarkdown(exportQuiz()), 'text/markdown');
    } else if (act === 'llm') runLlm();
  });
  // The quiz panel lives in the 실행 결과 pane on wide screens (outside `root`),
  // so answer/approval events are listened for on both containers.
  const onChange = (e) => {
    const ap = e.target.closest('[data-approve]');
    if (ap) {
      approvals.set(sig(current.items[Number(ap.dataset.approve)]), ap.checked);
      renderPanel();
      return;
    }
    const an = e.target.closest('[data-answer]');
    if (an && an.type === 'radio') answers.set(Number(an.dataset.answer), an.value);
  };
  const onInput = (e) => {
    const an = e.target.closest('input[type=text][data-answer]');
    if (an) answers.set(Number(an.dataset.answer), an.value);
  };
  for (const host of [root, out]) {
    on(host, 'change', onChange);
    on(host, 'input', onInput);
  }

  async function runLlm() {
    const provider = provSel.value;
    if (!safeHasKey(provider)) return;
    const { list } = await retrieve();
    if (!list.length) return;
    st.llm?.abort();
    const llm = new AbortController();
    st.llm = llm;
    const model = $('[data-in=model]').value;
    const key = settingsKey();
    $('[data-slot=banner]').innerHTML = `<div class="widget__status" role="status"><span class="spinner" aria-hidden="true"></span> 근거 청크 ${list.length}개로 ${s.count}문항 생성 중…</div>`;
    $('[data-act=llm]').disabled = true;
    try {
      const { system, messages } = buildQuizPrompt(list, { count: s.count, difficulty: s.difficulty });
      const res = await generate({ provider, model, system, messages, maxTokens: 1800, temperature: 0.3, signal: llm.signal });
      if (!st.alive) return;
      llmResult = { key, raw: String(res.text ?? '').trim() || '(빈 응답)', model };
      await update();
    } catch (err) {
      if (err.name === 'AbortError' || !st.alive) return;
      const msg = err instanceof LLMError ? err.message : '알 수 없는 오류로 문항을 만들지 못했다.';
      $('[data-slot=banner]').innerHTML = `<div class="widget__error" role="alert">${escapeHtml(msg)}<br>규칙 기반 생성과 실패 예시는 키 없이 계속 쓸 수 있다.</div>`;
    } finally {
      if (st.alive) syncInputs();
    }
  }

  // ---- first render: rule-based quiz for the default topic, no key needed
  syncKeyState();
  $('[data-slot=status]').hidden = true;
  $('[data-slot=body]').hidden = false;
  registerOutput(outputId, {
    title: options.outputTitle ?? '퀴즈 검증 · 근거 청크',
    node: out,
    inlineHost: $('[data-slot=out-inline]'),
  });
  await update();
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

function sig(it) {
  return `${it?.question ?? ''}|${it?.answer ?? ''}|${it?.citation?.chunk ?? ''}`;
}

function gradeOne(it, given) {
  return given != null && norm(given).replace(/\s/g, '') === norm(it.answer).replace(/\s/g, '');
}

function questionHtml(q) {
  return escapeHtml(String(q ?? '')).split(BLANK).join(`<span class="w15-blank" aria-label="빈칸">${BLANK}</span>`);
}

/**
 * Escape text and wrap [start, end) ranges in marks. Items citing the same
 * sentence share one mark (labels joined, worst status wins); other overlaps are skipped.
 */
function markSpans(text, spans) {
  const rank = { ok: 0, warn: 1, fail: 2 };
  const merged = new Map();
  for (const sp of spans) {
    const k = `${sp.start}:${sp.end}`;
    const m = merged.get(k);
    if (!m) merged.set(k, { ...sp });
    else {
      m.label += ` ${sp.label}`;
      if (rank[sp.cls] > rank[m.cls]) m.cls = sp.cls;
    }
  }
  const sorted = [...merged.values()].sort((a, b) => a.start - b.start);
  let html = '';
  let last = 0;
  for (const sp of sorted) {
    if (sp.start < last) continue;
    html += escapeHtml(text.slice(last, sp.start));
    html += `<mark class="w15-mark w15-mark--${sp.cls}"><span class="w15-mark__tag">${escapeHtml(sp.label)}</span>${escapeHtml(text.slice(sp.start, sp.end))}</mark>`;
    last = sp.end;
  }
  return html + escapeHtml(text.slice(last));
}

function download(filename, text, type) {
  const blob = new Blob([text], { type: `${type};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
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
