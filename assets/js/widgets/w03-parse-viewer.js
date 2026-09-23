// w03 파싱 정제 뷰어
// One concept: what a loader hands you is not clean text. Toggle individual
// cleaning steps on the corpus's intentionally broken documents (or your own
// text/file) and watch whether each golden fact becomes recoverable — or gets
// destroyed by an over-aggressive rule.
//
// No PDF library is used (adding pdf.js needs approval): the corpus already
// stores the *output* of naive loaders (OCR text, flattened table, crawled
// HTML), and local .html files are converted with the browser's DOMParser.

import { registerOutput, unregisterOutput } from '../site/result.js';

const DATA = new URL('../../data/', import.meta.url);
const MAX_FILE_BYTES = 1_000_000;
const CUSTOM = 'custom';

// ------------------------------------------------------------------ cleaning steps
// Every step is a pure function (text) => { text, count }. They run in the
// order listed here; the order matters (see dedupe before header).

const DAYS = '월화수목금토일';
// Single syllables that are commonly standalone words; never glued to the next word.
const STANDALONE = new Set([...'매주각총약더잘못꼭또그이저한두세네수것등때중후전외새첫온옛헌왜좀곧늘다및']);
const BOILERPLATE = /(로그인|회원가입|사이트맵|copyright|©|all rights reserved|개인정보처리방침|이용약관|무단수집거부)/i;
const PAGE_LINE = /^\s*(?:[-–—]\s*\d{1,4}\s*[-–—]|\d{1,4}\s*\/\s*\d{1,4}|(?:page|p\.)\s*\d{1,4}(?:\s*of\s*\d{1,4})?|\d{1,3})\s*$/i;
const PAGE_TAIL = /\s+[-–—]\s*\d{1,4}\s*[-–—]\s*$/;
// A run of digits mixed with O/o/I/l that is NOT glued to other Latin letters.
const OCR_RUN = /(?<![A-Za-z])[0-9OoIl]*\d[0-9OoIl]*(?![A-Za-z])/g;
const TABLE_HEAD = /^\s*요일\s+교시\s+과목\s+강의실\s+담당\s*$/;
const ROW_START = new RegExp(`\\s+(?=[${DAYS}]\\s+\\d+-\\d+\\s)`);
const ROW = new RegExp(`^([${DAYS}])\\s+(\\d+-\\d+)\\s+(.*)$`);

export function removePageNumbers(text) {
  let count = 0;
  const lines = [];
  for (const line of text.split('\n')) {
    if (line.trim() && PAGE_LINE.test(line)) {
      count++;
      continue;
    }
    lines.push(
      line.replace(PAGE_TAIL, () => {
        count++;
        return '';
      }),
    );
  }
  return { text: lines.join('\n'), count };
}

export function dedupeParagraphs(text) {
  const seen = new Set();
  let count = 0;
  const kept = [];
  for (const para of text.split(/\n[ \t]*\n+/)) {
    const key = para.replace(/\s+/g, ' ').trim();
    if (key && seen.has(key)) {
      count++;
      continue;
    }
    if (key) seen.add(key);
    kept.push(para);
  }
  return { text: kept.join('\n\n'), count };
}

/** Lines repeated ≥ 2 times (digits ignored, i.e. page headers) + menu/copyright lines. */
export function removeBoilerplate(text) {
  const lines = text.split('\n');
  const norm = (l) => l.replace(/\d+/g, '#').replace(/\s+/g, ' ').trim();
  const freq = new Map();
  for (const l of lines) {
    const k = norm(l);
    if (k.length >= 4) freq.set(k, (freq.get(k) ?? 0) + 1);
  }
  let count = 0;
  const kept = lines.filter((l) => {
    const k = norm(l);
    const drop = (k.length >= 4 && freq.get(k) >= 2) || (k.length > 0 && k.length <= 160 && BOILERPLATE.test(k));
    if (drop) count++;
    return !drop;
  });
  return { text: kept.join('\n'), count };
}

/** Split one table row's cells after 요일·교시: 과목 … 강의실 담당. */
export function splitRowCells(rest) {
  const toks = rest.split(/\s+/).filter(Boolean);
  let who = '';
  if (toks.at(-1) === '공동') who = toks.pop();
  else if (toks.at(-1) === '교수' && toks.length >= 2) who = toks.splice(-2).join(' ');
  let roomIdx = toks.findIndex((t) => /^\d{3}호?$/.test(t));
  if (roomIdx < 0) roomIdx = toks.findIndex((t) => /^(외부|온라인|원격|강당)/.test(t));
  const subject = roomIdx < 0 ? toks.join(' ') : toks.slice(0, roomIdx).join(' ');
  const room = roomIdx < 0 ? '' : toks.slice(roomIdx).join(' ');
  return { subject, room: room || '—', who: who || '—' };
}

/** Rebuild the flattened timetable using the known column pattern. */
export function rebuildTimetable(text) {
  const lines = text.split('\n');
  const h = lines.findIndex((l) => TABLE_HEAD.test(l));
  if (h < 0 || h + 1 >= lines.length) return { text, count: 0 };
  const body = lines[h + 1].trim();
  const parts = body.split(ROW_START);
  const rows = parts.map((p) => p.match(ROW));
  if (rows.some((m) => !m)) return { text, count: 0 }; // not the pattern we know
  const out = rows.map(([, day, period, rest]) => {
    const { subject, room, who } = splitRowCells(rest);
    return [day, period, subject, room, who].join(' | ');
  });
  lines.splice(h, 2, '요일 | 교시 | 과목 | 강의실 | 담당', ...out);
  return { text: lines.join('\n'), count: out.length };
}

export function fixOcrDigits(text) {
  let count = 0;
  const out = text.replace(OCR_RUN, (run) => {
    const fixed = run.replace(/[Oo]/g, '0').replace(/[Il]/g, '1');
    if (fixed !== run) count++;
    return fixed;
  });
  return { text: out, count };
}

export function ocrEverywhere(text) {
  let count = 0;
  const out = text.replace(/[OIl]/g, (c) => {
    count++;
    return c === 'O' ? '0' : '1';
  });
  return { text: out, count };
}

const isSyllable = (t) => /^[가-힣]$/.test(t);

export function mergeSplitSyllables(text) {
  let count = 0;
  // rule 1: three or more single syllables in a row — "장 학 금 안 내"
  let t = text.replace(/(?<![가-힣])[가-힣](?: [가-힣](?![가-힣])){2,}/g, (m) => {
    count++;
    return m.replace(/ /g, '');
  });
  // rule 2: a detached ending "다" before punctuation — "지급한 다."
  t = t.replace(/([가-힣]) 다(?=[.!?]|$)/gm, (m, a) => {
    count++;
    return `${a}다`;
  });
  // rule 3: a lone syllable glued to the next Hangul word — "평 점이", "소 득"
  t = t
    .split('\n')
    .map((line) => {
      const toks = line.split(' ');
      const out = [];
      for (let i = 0; i < toks.length; i++) {
        let tok = toks[i];
        if (isSyllable(tok) && !STANDALONE.has(tok) && i + 1 < toks.length && /^[가-힣]/.test(toks[i + 1])) {
          tok += toks[++i];
          count++;
        }
        out.push(tok);
      }
      return out.join(' ');
    })
    .join('\n');
  return { text: t, count };
}

export function removeAllHangulSpaces(text) {
  let count = 0;
  const out = text.replace(/([가-힣]) +(?=[가-힣])/g, (m, a) => {
    count++;
    return a;
  });
  return { text: out, count };
}

export function tidy(text) {
  return text
    .split('\n')
    .map((l) => l.replace(/[ \t]+$/, ''))
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/^\n+|\n+$/g, '');
}

export const STEPS = [
  { id: 'pagenum', label: '페이지 번호 제거', desc: '“- 2 -”, “3 / 10”, “Page 4” 같은 쪽 번호를 지운다', fn: removePageNumbers },
  { id: 'dedupe', label: '중복 문단 제거', desc: '공백을 정규화해 같은 문단은 처음 것만 남긴다', fn: dedupeParagraphs },
  { id: 'header', label: '머리글·바닥글 제거', desc: '2번 이상 반복되는 줄(숫자 무시)과 메뉴·저작권 줄을 지운다', fn: removeBoilerplate },
  { id: 'table', label: '표 행 복원', desc: '“요일 교시 과목 강의실 담당” 다음 줄을 요일+교시 패턴에서 잘라 행으로 되돌린다', fn: rebuildTimetable },
  { id: 'ocr', label: 'OCR 숫자 교정', desc: '숫자에 붙은 O·o→0, l·I→1. 영어 단어 안은 건드리지 않는다', fn: fixOcrDigits },
  { id: 'ocrAll', label: '과도: 모든 O→0, l·I→1', desc: '문맥을 보지 않고 전부 바꾼다', fn: ocrEverywhere, aggressive: true },
  { id: 'space', label: '끊긴 단어 공백 병합', desc: '“장 학 금”, “평 점이”, “지급한 다.”처럼 음절 사이에 낀 공백을 붙인다', fn: mergeSplitSyllables },
  { id: 'spaceAll', label: '과도: 한글 사이 공백 전부 제거', desc: '띄어쓰기를 모두 없앤다', fn: removeAllHangulSpaces, aggressive: true },
];

/** Apply the enabled steps in STEPS order. */
export function cleanText(raw, enabled) {
  let text = raw.replace(/\r\n?/g, '\n');
  const counts = {};
  for (const step of STEPS) {
    if (!enabled.has(step.id)) continue;
    const r = step.fn(text);
    text = r.text;
    counts[step.id] = r.count;
  }
  return { text: tidy(text), counts };
}

const RECOMMENDED = {
  'broken-scan': ['pagenum', 'dedupe', 'header', 'ocr', 'space'],
  'broken-table': ['pagenum', 'dedupe', 'header', 'table', 'ocr'],
  'broken-mixed': ['pagenum', 'dedupe', 'header', 'ocr'],
};
const DEFAULT_RECOMMENDED = ['pagenum', 'dedupe', 'header', 'ocr'];
export const recommendedFor = (docId) => RECOMMENDED[docId] ?? DEFAULT_RECOMMENDED;

// ------------------------------------------------------------------ fact checks

const count = (t, s) => t.split(s).length - 1;

function mustInclude(label, needle, source, diagnose) {
  return {
    label,
    source,
    run: (t) => {
      if (t.includes(needle)) return { ok: true };
      const squeezed = (x) => x.replace(/\s+/g, '');
      const why =
        diagnose?.(t) ??
        (squeezed(t).includes(squeezed(needle))
          ? `글자는 남아 있지만 띄어쓰기가 달라져 「${needle}」 구간이 일치하지 않는다 — 공백 규칙이 너무 공격적이다`
          : `「${needle}」 구간을 찾을 수 없다`);
      return { ok: false, why };
    },
  };
}

const NO_ROWS = '행 경계가 없다 — “… 306 이 교수 화 1-3 검색증강생성(RAG) …”처럼 앞 행의 담당자와 다음 행의 요일이 이어져, 어느 요일·강의실이 어느 과목의 것인지 판단할 수 없다';

/** Hand-written checks for the broken documents (golden facts + controls). */
export const EXTRA_CHECKS = {
  'broken-scan': [
    mustInclude('지급액은 수업료의 50%', '지급액은 수업료의 50%', 'q11', (t) =>
      t.includes('5O%')
        ? '「5O%」 그대로다 — 숫자 0이 영문 대문자 O로 인식됐다. “50%”로 검색해도 일치하지 않고, LLM은 이 값을 그대로 인용한다'
        : /지급액은수업료|수업료의50/.test(t)
          ? '띄어쓰기가 사라져 문장이 원문과 달라졌다 — 과도한 공백 제거의 부작용'
          : null),
    mustInclude('평균 평점이 4.0 이상', '평균 평점이 4.0 이상', '추가 확인', (t) =>
      t.includes('평 점이') ? '「평 점이」 — OCR이 단어 중간에 공백을 넣었다. “평점”으로 검색하면 일치하지 않는다' : null),
    mustInclude('근로 장학: 주 10시간 이내', '주 10시간 이내', '추가 확인', (t) =>
      t.includes('1O시간') ? '「1O시간」 — 숫자 0 자리에 영문 O' : null),
    mustInclude('문의처: 내선 214', '내선 214', '추가 확인', (t) =>
      t.includes('2l4') ? '「2l4」 — 숫자 1 자리에 영문 소문자 l' : null),
    {
      label: '본문 사이에 머리글·쪽 번호 없음',
      source: '추가 확인',
      run: (t) =>
        /분당융합기술교육원|[-–]\s*\d+\s*[-–]/.test(t)
          ? { ok: false, why: '「한국폴리텍대학 분당융합기술교육원 - 2 -」가 문단 사이에 끼어 있다. 청킹하면 이 줄이 매 페이지 청크에 섞여 들어간다' }
          : { ok: true },
    },
  ],
  'broken-table': [
    mustInclude('RAG 수업: 화 | 1-3 | … | 305 | 조 교수', '화 | 1-3 | 검색증강생성(RAG) 시스템 구축 | 305 | 조 교수', 'q10', (t) =>
      t.includes('조교수')
        ? '「조 교수」가 「조교수」(직급 이름)로 붙었다 — 스캔 문서용 공백 병합을 표에 적용한 부작용'
        : t.includes(' | ')
          ? null
          : NO_ROWS),
    mustInclude('MLOps 행이 온전함', '목 | 1-3 | MLOps | 305 | 박 교수', '추가 확인', (t) =>
      t.includes('ML0ps')
        ? '「MLOps」가 「ML0ps」로 바뀌었다 — 문맥 없이 O→0을 적용하면 멀쩡한 단어가 망가진다'
        : t.includes('박교수')
          ? '「박 교수」가 「박교수」로 붙었다 — 공백 병합의 부작용'
          : t.includes(' | ')
          ? null
          : '표 행이 아직 복원되지 않았다'),
    {
      label: '8개 행이 모두 5칸으로 복원',
      source: '추가 확인',
      run: (t) => {
        const rows = t.split('\n').filter((l) => new RegExp(`^[${DAYS}] \\| `).test(l));
        const five = rows.filter((l) => l.split(' | ').length === 5);
        return rows.length === 8 && five.length === 8
          ? { ok: true, why: '단, 「수 | 1-6 | 현장 실습 | 외부 산업체 | —」는 규칙으로 추측한 행이다' }
          : { ok: false, why: `5칸으로 복원된 행 ${five.length}개 / 8개` };
      },
    },
  ],
  'broken-mixed': [
    {
      label: '노트북 Q&A가 한 번만 남음',
      source: '추가 확인',
      run: (t) => {
        const n = count(t, 'Q. 노트북을 가져와도 되나요?');
        if (n === 1) return { ok: true };
        if (n === 0 && t.includes('노트북을가져와')) return { ok: false, why: '띄어쓰기가 사라져 원래 질문 문장과 달라졌다' };
        if (n === 0) return { ok: false, why: '0번 — 두 번 반복된 줄을 “머리글”로 오인해 둘 다 지웠다. 중복 문단 제거가 먼저 돌아야 한다' };
        return { ok: false, why: `${n}번 — 같은 Q&A가 중복 저장되면 검색 상위 자리를 같은 내용이 차지한다` };
      },
    },
    {
      label: '메뉴·저작권 문구 없음',
      source: '추가 확인',
      run: (t) =>
        /로그인|Copyright/i.test(t)
          ? { ok: false, why: '「홈 | 학과소개 | … | 로그인」, 「Copyright ©…」가 본문과 함께 청킹·임베딩된다' }
          : { ok: true },
    },
    mustInclude('「결석 시 보강」 띄어쓰기 유지', '결석 시 보강이', '추가 확인', (t) =>
      t.includes('시보강') ? '「결석 시보강이」 — 스캔 문서용 공백 병합이 멀쩡한 웹 문서를 망가뜨렸다' : null),
  ],
};

/** Golden evidence quotes for a doc (skipped when a hand-written check covers that id). */
export function checksFor(docId, golden, mustKeep) {
  const extra = EXTRA_CHECKS[docId] ?? [];
  const covered = new Set(extra.map((c) => c.source));
  const auto = golden.items
    .flatMap((it) => it.evidence.filter((e) => e.doc === docId).map((e) => ({ it, e })))
    .filter(({ it }) => !covered.has(it.id))
    .map(({ it, e }) => mustInclude(it.answer, e.quote, it.id));
  const user = mustKeep ? [mustInclude(`직접 지정: 「${mustKeep}」`, mustKeep, '직접 지정')] : [];
  return [...extra, ...auto, ...user];
}

// ------------------------------------------------------------------ diff

/** Character (or token) level LCS diff → [{ type: 'eq'|'del'|'ins', text }]. */
export function diff(a, b) {
  let pre = 0;
  while (pre < a.length && pre < b.length && a[pre] === b[pre]) pre++;
  let suf = 0;
  while (suf < a.length - pre && suf < b.length - pre && a[a.length - 1 - suf] === b[b.length - 1 - suf]) suf++;
  const midA = a.slice(pre, a.length - suf);
  const midB = b.slice(pre, b.length - suf);

  let ua = [...midA];
  let ub = [...midB];
  if (ua.length * ub.length > 2_500_000) {
    ua = midA.match(/\s+|[^\s]+/g) ?? [];
    ub = midB.match(/\s+|[^\s]+/g) ?? [];
  }
  let mid;
  if (ua.length * ub.length > 6_000_000) {
    mid = [
      { type: 'del', text: midA },
      { type: 'ins', text: midB },
    ];
  } else {
    mid = lcsOps(ua, ub);
  }
  const ops = [];
  const push = (type, text) => {
    if (!text) return;
    const last = ops.at(-1);
    if (last?.type === type) last.text += text;
    else ops.push({ type, text });
  };
  push('eq', a.slice(0, pre));
  for (const o of mid) push(o.type, o.text);
  push('eq', a.slice(a.length - suf));
  return ops;
}

function lcsOps(ua, ub) {
  const n = ua.length;
  const m = ub.length;
  const w = m + 1;
  const dp = new Uint32Array((n + 1) * w);
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i * w + j] = ua[i] === ub[j] ? dp[(i + 1) * w + j + 1] + 1 : Math.max(dp[(i + 1) * w + j], dp[i * w + j + 1]);
    }
  }
  const ops = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (ua[i] === ub[j]) {
      ops.push({ type: 'eq', text: ua[i++] });
      j++;
    } else if (dp[(i + 1) * w + j] >= dp[i * w + j + 1]) ops.push({ type: 'del', text: ua[i++] });
    else ops.push({ type: 'ins', text: ub[j++] });
  }
  while (i < n) ops.push({ type: 'del', text: ua[i++] });
  while (j < m) ops.push({ type: 'ins', text: ub[j++] });
  return ops;
}

// ------------------------------------------------------------------ HTML loader

/** HTML → plain text with DOMParser (browser only). Table cells are joined with " | ". */
export function htmlToText(html) {
  const doc = new DOMParser().parseFromString(html, 'text/html');
  doc.querySelectorAll('script, style, noscript, template, svg').forEach((n) => n.remove());
  doc.querySelectorAll('tr').forEach((tr) => {
    const cells = [...tr.children]
      .filter((c) => c.tagName === 'TD' || c.tagName === 'TH')
      .map((c) => c.textContent.replace(/\s+/g, ' ').trim());
    tr.replaceWith(doc.createTextNode(`\n${cells.join(' | ')}\n`));
  });
  doc.querySelectorAll('br').forEach((br) => br.replaceWith(doc.createTextNode('\n')));
  const blocks = 'p, div, li, h1, h2, h3, h4, h5, h6, section, article, header, footer, nav, aside, main, table, ul, ol, dl, dt, dd, blockquote, pre, figure, figcaption';
  doc.querySelectorAll(blocks).forEach((el) => {
    el.before(doc.createTextNode('\n'));
    el.after(doc.createTextNode('\n'));
  });
  const text = doc.body?.textContent ?? '';
  return text
    .split('\n')
    .map((l) => l.replace(/[ \t ]+/g, ' ').trim())
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

// ------------------------------------------------------------------ templates

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget" data-w03>
    <h3 class="widget__title">파싱 정제 뷰어</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> 문서셋을 불러오는 중…
    </div>
    <div data-slot="body" hidden>
      <div class="widget__controls">
        <label class="field">
          <span class="field__label">문서 (로더가 뽑아낸 원문)</span>
          <select data-in="doc"></select>
        </label>
      </div>
      <p class="w03-source" data-slot="source"></p>
      <div class="btn-row" data-slot="presets" style="margin-bottom: var(--space-3)">
        <button type="button" class="btn small" data-preset="off">모두 끄기 (실패 재현)</button>
        <button type="button" class="btn small" data-preset="recommended">권장 설정 (이 문서)</button>
        <button type="button" class="btn small" data-preset="aggressive">전부 켜기 (과도한 정제)</button>
      </div>
      <fieldset class="w03-steps" data-slot="steps">
        <legend>정제 단계 — 위에서 아래 순서로 적용한다</legend>
      </fieldset>
      <details data-slot="custom">
        <summary>직접 붙여넣기 · 파일 불러오기 (.txt · .md · .html)</summary>
        <p class="w03-note">파일은 브라우저 안에서만 읽고 어디에도 업로드하지 않는다. HTML은 DOMParser로 태그를 해석해 텍스트로 바꾸며, 표 셀은 “ | ”로 구분한다.</p>
        <div class="w03-file">
          <label class="btn small" data-slot="file-label">📂 파일 선택<input type="file" data-in="file" accept=".txt,.md,.markdown,.html,.htm,text/plain,text/markdown,text/html" class="visually-hidden"></label>
          <span data-slot="file-name" class="w03-note"></span>
        </div>
        <div data-slot="file-error" role="alert"></div>
        <label class="field" style="margin-top: var(--space-2)">
          <span class="field__label">원문 (고치면 “직접 입력”으로 바뀐다)</span>
          <textarea data-in="text" spellcheck="false"></textarea>
        </label>
        <label class="field" style="margin-top: var(--space-2)">
          <span class="field__label">반드시 남아야 할 문구 (선택 — 사실 확인에 추가)</span>
          <input type="text" data-in="keep" placeholder="예: 50%">
        </label>
      </details>
      <div data-slot="out-inline"></div>
    </div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w03-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <ul class="w03-checks" data-slot="checks"></ul>
    <div class="stat-row" data-slot="stats"></div>
    <div class="legend" aria-hidden="true">
      <span><i class="w03-sw w03-sw--del"></i>지워진 부분</span>
      <span><i class="w03-sw w03-sw--ins"></i>새로 생기거나 바뀐 부분</span>
    </div>
    <div class="w03-cols">
      <section>
        <h4>원문 (로더 출력)</h4>
        <div class="w03-pane" data-slot="raw" tabindex="0" aria-label="원문, 지워진 부분 표시"></div>
      </section>
      <section>
        <h4>정제본 (청킹에 들어갈 텍스트)</h4>
        <div class="w03-pane" data-slot="clean" tabindex="0" aria-label="정제본, 바뀐 부분 표시"></div>
      </section>
    </div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ doc?: string, steps?: string[], outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w03:${++seq}`;
  state.set(el, { ctrl, outputId });

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

  // broken documents first — they are this week's material
  const docs = [...corpus.documents].sort((a, b) => (b.type === 'broken') - (a.type === 'broken'));
  const s = {
    doc: options.doc ?? 'broken-scan',
    text: '',
    source: '',
    steps: new Set(options.steps ?? []),
    keep: '',
  };

  const docSel = $('[data-in=doc]');
  docSel.innerHTML =
    docs.map((d) => `<option value="${d.id}">${d.type === 'broken' ? '⚠ ' : ''}${escapeHtml(d.title)}</option>`).join('') +
    `<option value="${CUSTOM}">✍ 직접 입력 · 파일</option>`;

  const stepsBox = $('[data-slot=steps]');
  stepsBox.insertAdjacentHTML(
    'beforeend',
    STEPS.map(
      (st) => `<label class="w03-step${st.aggressive ? ' w03-step--warn' : ''}">
        <input type="checkbox" data-step="${st.id}">
        <span class="w03-step__name">${st.aggressive ? '⚠ ' : ''}${escapeHtml(st.label)}</span>
        <span class="w03-step__count" data-count="${st.id}"></span>
        <span class="w03-step__desc">${escapeHtml(st.desc)}</span>
      </label>`,
    ).join(''),
  );

  function loadDoc(docId) {
    const d = docs.find((x) => x.id === docId);
    if (!d) {
      s.doc = CUSTOM;
      s.source = s.source || '직접 입력한 텍스트';
      return;
    }
    s.doc = d.id;
    s.text = d.text;
    s.source = `${d.source}${d.broken ? ` · 문제: ${d.broken}` : ''}`;
    $('[data-in=text]').value = s.text;
  }

  function syncInputs() {
    docSel.value = s.doc;
    stepsBox.querySelectorAll('[data-step]').forEach((cb) => {
      cb.checked = s.steps.has(cb.dataset.step);
    });
  }

  function render() {
    $('[data-slot=source]').textContent = `출처: ${s.source}`;
    const { text: cleaned, counts } = cleanText(s.text, s.steps);
    for (const st of STEPS) {
      const c = counts[st.id];
      $(`[data-count=${st.id}]`).textContent = c === undefined ? '' : `${c}건`;
    }

    const raw = s.text.replace(/\r\n?/g, '\n');
    const checks = checksFor(s.doc, golden, s.keep.trim()).map((c) => ({ ...c, result: c.run(cleaned) }));
    const okN = checks.filter((c) => c.result.ok).length;

    const ops = diff(raw, cleaned);
    const removed = ops.filter((o) => o.type === 'del').reduce((n, o) => n + o.text.length, 0);
    const added = ops.filter((o) => o.type === 'ins').reduce((n, o) => n + o.text.length, 0);

    $('[data-slot=verdict]').innerHTML = verdictHtml(checks.length, okN, s.steps);
    $('[data-slot=checks]').innerHTML = checks
      .map(
        (c) => `<li class="${c.result.ok ? 'is-ok' : 'is-fail'}">
          <span class="w03-check__mark" aria-label="${c.result.ok ? '복구됨' : '실패'}">${c.result.ok ? '✓' : '✗'}</span>
          <span><b>${escapeHtml(c.label)}</b> <span class="w03-check__src">${escapeHtml(c.source)}</span></span>
          ${c.result.why ? `<span class="w03-check__why">${escapeHtml(c.result.why)}</span>` : ''}
        </li>`,
      )
      .join('');
    if (s.doc === 'broken-scan') {
      $('[data-slot=checks]').insertAdjacentHTML(
        'beforeend',
        `<li class="w03-check--note"><span aria-hidden="true">ℹ</span><span class="w03-check__why">골든셋 q11의 근거 인용문은 「지급액은 수업료의 5O% 이다.」 — 망가진 원문 그대로 기록되어 있다. 파싱을 고치면 골든셋 인용문도 함께 고쳐야 평가가 맞는다.</span></li>`,
      );
    }

    $('[data-slot=stats]').innerHTML = [
      ['원문', `${raw.length}자`],
      ['정제본', `${cleaned.length}자`],
      ['지움 / 추가', `${removed} / ${added}`],
      ['사실 확인', `${okN} / ${checks.length}`],
    ]
      .map(([l, v]) => `<div class="stat"><span class="stat__label">${l}</span><span class="stat__value">${v}</span></div>`)
      .join('');

    $('[data-slot=raw]').innerHTML = ops
      .filter((o) => o.type !== 'ins')
      .map((o) => (o.type === 'del' ? `<del>${escapeHtml(o.text)}</del>` : escapeHtml(o.text)))
      .join('');
    $('[data-slot=clean]').innerHTML =
      ops
        .filter((o) => o.type !== 'del')
        .map((o) => (o.type === 'ins' ? `<ins>${escapeHtml(o.text)}</ins>` : escapeHtml(o.text)))
        .join('') || '<span class="w03-note">(빈 텍스트)</span>';
  }

  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  on('[data-in=doc]', 'change', (e) => {
    if (e.target.value === CUSTOM) {
      s.doc = CUSTOM;
      s.source = '직접 입력한 텍스트';
      $('[data-slot=custom]').open = true;
      $('[data-in=text]').focus();
    } else {
      loadDoc(e.target.value);
    }
    render();
  });
  on('[data-slot=steps]', 'change', (e) => {
    const id = e.target.dataset.step;
    if (!id) return;
    if (e.target.checked) s.steps.add(id);
    else s.steps.delete(id);
    render();
  });
  on('[data-slot=presets]', 'click', (e) => {
    const p = e.target.closest('[data-preset]')?.dataset.preset;
    if (!p) return;
    if (p === 'off') s.steps = new Set();
    if (p === 'recommended') s.steps = new Set(recommendedFor(s.doc));
    if (p === 'aggressive') s.steps = new Set(STEPS.map((st) => st.id));
    syncInputs();
    render();
  });
  on('[data-in=text]', 'input', (e) => {
    s.text = e.target.value;
    if (s.doc !== CUSTOM) {
      s.doc = CUSTOM;
      s.source = '직접 입력한 텍스트';
      docSel.value = CUSTOM;
    }
    render();
  });
  on('[data-in=keep]', 'input', (e) => {
    s.keep = e.target.value;
    render();
  });
  on('[data-in=file]', 'change', async (e) => {
    const file = e.target.files?.[0];
    const errBox = $('[data-slot=file-error]');
    errBox.innerHTML = '';
    if (!file) return;
    if (file.size > MAX_FILE_BYTES) {
      errBox.innerHTML = `<div class="widget__error">파일이 너무 크다 (${Math.round(file.size / 1024)}KB). 1MB 이하 파일만 불러온다.</div>`;
      return;
    }
    try {
      const body = await file.text();
      const isHtml = /\.html?$/i.test(file.name) || file.type === 'text/html';
      s.text = isHtml ? htmlToText(body) : body;
      s.doc = CUSTOM;
      s.source = `내 파일 ${file.name}${isHtml ? ' (DOMParser로 HTML → 텍스트)' : ''}`;
      $('[data-in=text]').value = s.text;
      $('[data-slot=file-name]').textContent = `${file.name} · ${s.text.length}자`;
      syncInputs();
      render();
    } catch (err) {
      errBox.innerHTML = `<div class="widget__error">파일을 읽지 못했다: ${escapeHtml(err.message)}</div>`;
    } finally {
      e.target.value = '';
    }
  });

  loadDoc(s.doc);
  syncInputs();
  render();
  $('[data-slot=status]').hidden = true;
  $('[data-slot=body]').hidden = false;
  registerOutput(outputId, {
    title: options.outputTitle ?? '파싱 정제 결과',
    node: out,
    inlineHost: $('[data-slot=out-inline]'),
  });
}

export function unmount(el) {
  const st = state.get(el);
  st?.ctrl.abort();
  if (st) unregisterOutput(st.outputId);
  state.delete(el);
  el.replaceChildren();
}

// ------------------------------------------------------------------ helpers

function verdictHtml(total, ok, steps) {
  if (total === 0) {
    return `<div class="callout"><span class="callout__title">확인할 사실이 없다</span>
      직접 입력한 텍스트에는 골든셋 사실이 없다. “반드시 남아야 할 문구”를 적으면 정제 후에도 남는지 확인한다.</div>`;
  }
  const harsh = [...steps].some((id) => STEPS.find((st) => st.id === id)?.aggressive);
  if (ok === total) {
    return `<div class="callout callout--ok"><span class="callout__title">사실 ${ok}/${total} 복구 — 이 텍스트는 청킹해도 된다</span>
      모든 사실이 정제본에 올바른 형태로 남아 있다.</div>`;
  }
  const lead = steps.size === 0 ? '로더 출력을 그대로 쓰고 있다.' : harsh ? '과도한 규칙이 멀쩡한 텍스트까지 바꿨다.' : '아직 고치지 못한 부분이 있다.';
  return `<div class="callout callout--danger"><span class="callout__title">사실 ${ok}/${total}만 복구</span>
    ${lead} ✗ 표시된 사실은 이후 청킹·임베딩·검색이 아무리 좋아도 올바르게 답할 수 없다.</div>`;
}

async function fetchJson(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${res.status} ${url.pathname}`);
  return res.json();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
