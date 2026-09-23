// w01 토큰 카운터
// One concept: an LLM reads tokens, not characters. The same sentence costs a
// different number of tokens depending on the language and the tokenizer, and
// that number decides how much fits in the context window (and what it costs).

import { registerOutput, unregisterOutput } from '../site/result.js';

// Same pinned build as core/embed.js
const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
const DATA = new URL('../../data/', import.meta.url);
const COLORS = 6;
const MAX_PIECES = 700; // keep the DOM light for whole-corpus input

export const TOKENIZERS = [
  { id: 'Xenova/gpt-4o', label: 'GPT-4o (o200k · 다국어)', short: 'GPT-4o', sizeMB: 9.7 },
  { id: 'Xenova/gpt2', label: 'GPT-2 (영어 중심)', short: 'GPT-2', sizeMB: 2.1 },
];

const WINDOWS = [
  { value: 1024, label: '1,024 (GPT-2의 실제 한도)' },
  { value: 8192, label: '8,192 (소형 모델)' },
  { value: 128000, label: '128,000 (GPT-4o)' },
];

// Korean sentences come from corpus.json; English lines are translations with the same meaning.
const PRESETS = {
  gpu: {
    label: 'GPU 규정 한 문장',
    ko: 'GPU 서버에 장시간 학습 작업을 실행할 때는 공용 캘린더에 사용 시간을 등록해야 하며, 한 사람이 연속으로 사용할 수 있는 시간은 최대 12시간이다.',
    en: 'When running a long training job on the GPU server, you must register the usage time on the shared calendar, and one person may use it continuously for up to 12 hours.',
  },
  grade: {
    label: '성적 비율 한 문장',
    ko: '성적 평가는 주차별 실습 과제 30%, 중간 점검 20%, 팀 프로젝트 40%, 수업 참여 10%로 구성된다.',
    en: 'The grade consists of weekly lab assignments 30%, a midterm check 20%, the team project 40%, and class participation 10%.',
  },
  doc: {
    label: '문서 한 편 (실습실 이용 규정)',
    ko: null, // filled from corpus: facility-rules
    en:
      'The lab is open until 9 p.m. on weekdays, even outside class hours. On weekends and public holidays, only students who applied in advance may use it, and applications must be made to the department office at least two days before the day of use.\n\n' +
      'Eating or drinking anything, including beverages, is prohibited inside the lab. When running long training jobs on the GPU server, you must register the usage time on the shared calendar, and one person may use it continuously for up to 12 hours.\n\n' +
      'If you find broken equipment, report it to a teaching assistant immediately. Installing personal software is not allowed; request any software you need from the professor in charge.',
  },
  corpus: {
    label: '코퍼스 전체 12문서 (실패 재현)',
    ko: null, // filled from corpus: all documents
    en: '',
  },
};

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w01-tok">
    <h3 class="widget__title">토큰 카운터</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> <span data-msg>토크나이저 준비 중…</span>
    </div>
    <div class="btn-row w01-presets" data-slot="presets"></div>
    <div class="widget__controls w01-tok__texts">
      <label class="field">
        <span class="field__label">한국어 <output data-out="ko-chars"></output></span>
        <textarea data-in="ko" spellcheck="false"></textarea>
      </label>
      <label class="field">
        <span class="field__label">English (같은 뜻) <output data-out="en-chars"></output></span>
        <textarea data-in="en" spellcheck="false"></textarea>
      </label>
    </div>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">토큰 조각을 볼 토크나이저</span>
        <select data-in="view"></select>
      </label>
      <label class="field">
        <span class="field__label">컨텍스트 창 크기 (토큰)</span>
        <select data-in="window"></select>
      </label>
    </div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w01-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="w01-table-wrap">
      <table class="w01-table">
        <thead><tr><th scope="col">토크나이저</th><th scope="col">언어</th><th scope="col">문자</th><th scope="col">토큰</th><th scope="col">토큰/문자</th><th scope="col">컨텍스트 창 사용</th></tr></thead>
        <tbody data-slot="rows"></tbody>
      </table>
    </div>
    <h4 data-slot="pieces-title">토큰 조각</h4>
    <div class="legend" aria-hidden="true">
      <span><i style="background: var(--chunk-1)"></i><i style="background: var(--chunk-2)"></i><i style="background: var(--chunk-3)"></i>토큰 1개</span>
      <span><i class="w01-legend-multi"></i>점선 = 여러 토큰이 합쳐져야 글자 하나가 됨 (숫자 = 토큰 수)</span>
    </div>
    <div class="w01-pieces" data-slot="pieces-ko" tabindex="0" aria-label="한국어 토큰 조각"></div>
    <div class="w01-pieces" data-slot="pieces-en" tabindex="0" aria-label="영어 토큰 조각"></div>
  </div>`;

const state = new WeakMap();
let seq = 0;
let libPromise = null;

function getLib() {
  if (!libPromise) {
    libPromise = import(TRANSFORMERS_URL).then((lib) => {
      lib.env.allowLocalModels = false;
      return lib;
    });
    libPromise.catch(() => (libPromise = null));
  }
  return libPromise;
}

/**
 * @param {HTMLElement} el
 * @param {{ preset?: keyof PRESETS, window?: number, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w01-tokens:${++seq}`;
  const st = { ctrl, outputId, raf: 0 };
  state.set(el, st);

  const s = {
    view: TOKENIZERS[1].id, // start on the English-centric one: the failure is visible first
    window: options.window ?? 8192,
    tok: {}, // id -> tokenizer | Error | undefined (loading)
  };

  $('[data-in=view]').innerHTML = TOKENIZERS.map((t) => `<option value="${t.id}">${escapeHtml(t.label)}</option>`).join('');
  $('[data-in=window]').innerHTML = WINDOWS.map((w) => `<option value="${w.value}">${w.label}</option>`).join('');
  $('[data-in=view]').value = s.view;
  $('[data-in=window]').value = String(s.window);
  $('[data-slot=presets]').innerHTML = Object.entries(PRESETS)
    .map(([k, p]) => `<button type="button" class="btn ghost small" data-preset="${k}" aria-pressed="false"${p.ko === null ? ' disabled' : ''}>${escapeHtml(p.label)}</button>`)
    .join('');

  function applyPreset(key) {
    const p = PRESETS[key];
    if (!p) return;
    $('[data-in=ko]').value = p.ko ?? '';
    $('[data-in=en]').value = p.en ?? '';
    root.querySelectorAll('[data-preset]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.preset === key)));
  }

  function render() {
    st.raf = 0;
    const texts = { ko: $('[data-in=ko]').value, en: $('[data-in=en]').value };
    const chars = { ko: charCount(texts.ko), en: charCount(texts.en) };
    $('[data-out=ko-chars]').textContent = `${chars.ko}자`;
    $('[data-out=en-chars]').textContent = `${chars.en}자`;

    const counts = {}; // id -> { ko, en } token ids
    const rows = [];
    for (const t of TOKENIZERS) {
      const tk = s.tok[t.id];
      counts[t.id] = {};
      for (const lang of ['ko', 'en']) {
        let tokCell;
        let ratioCell;
        let ctxCell;
        if (tk instanceof Error) {
          tokCell = ratioCell = ctxCell = '<span class="w01-muted">불러오기 실패</span>';
        } else if (!tk) {
          tokCell = '<span class="spinner w01-inline-spin" aria-hidden="true"></span><span class="visually-hidden">불러오는 중</span>';
          ratioCell = ctxCell = '<span class="w01-muted">…</span>';
        } else {
          const ids = tk.encode(texts[lang], { add_special_tokens: false });
          counts[t.id][lang] = ids;
          const n = ids.length;
          tokCell = `<b>${n.toLocaleString()}</b>`;
          ratioCell = chars[lang] ? (n / chars[lang]).toFixed(2) : '—';
          ctxCell = contextCell(n, s.window);
        }
        rows.push(`<tr${lang === 'ko' ? ' class="w01-row-first"' : ''}>
          ${lang === 'ko' ? `<th scope="row" rowspan="2">${escapeHtml(t.short)}</th>` : ''}
          <td>${lang === 'ko' ? '한국어' : '영어'}</td><td>${chars[lang].toLocaleString()}</td>
          <td>${tokCell}</td><td>${ratioCell}</td><td>${ctxCell}</td></tr>`);
      }
    }
    $('[data-slot=rows]').innerHTML = rows.join('');
    $('[data-slot=verdict]').innerHTML = verdictHtml(counts, texts, s.window);

    // token pieces for the selected tokenizer
    const view = TOKENIZERS.find((t) => t.id === s.view);
    const tk = s.tok[s.view];
    $('[data-slot=pieces-title]').textContent = `토큰 조각 · ${view.label}`;
    for (const lang of ['ko', 'en']) {
      const box = $(`[data-slot=pieces-${lang}]`);
      box.hidden = !texts[lang];
      if (!texts[lang]) continue;
      if (tk instanceof Error) box.innerHTML = '<span class="w01-muted">이 토크나이저를 불러오지 못했다. 다른 토크나이저를 고른다.</span>';
      else if (!tk) box.innerHTML = `<span class="w01-muted">${escapeHtml(view.short)} 토크나이저를 불러오는 중… 문자 수는 위 표에 이미 나와 있다.</span>`;
      else box.innerHTML = piecesHtml(tokenPieces(tk, counts[s.view][lang]));
    }
  }

  const schedule = () => {
    if (!st.raf) st.raf = requestAnimationFrame(render);
  };

  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  on('[data-in=ko]', 'input', schedule);
  on('[data-in=en]', 'input', schedule);
  on('[data-in=view]', 'change', (e) => {
    s.view = e.target.value;
    render();
  });
  on('[data-in=window]', 'change', (e) => {
    s.window = Number(e.target.value);
    render();
  });
  on('[data-slot=presets]', 'click', (e) => {
    const key = e.target.closest('[data-preset]')?.dataset.preset;
    if (!key) return;
    applyPreset(key);
    render();
  });

  // Initial state: the GPU sentence, char counts shown immediately.
  applyPreset(options.preset && PRESETS[options.preset]?.ko !== null ? options.preset : 'gpu');
  registerOutput(outputId, {
    title: options.outputTitle ?? '토큰 카운터 결과',
    node: out,
    inlineHost: $('[data-slot=out-inline]'),
  });
  render();

  // Corpus-based presets (fetch is small; failure only disables those two buttons).
  fetchJson(new URL('corpus/corpus.json', DATA), ctrl.signal)
    .then((corpus) => {
      PRESETS.doc.ko = corpus.documents.find((d) => d.id === 'facility-rules')?.text ?? '';
      PRESETS.corpus.ko = corpus.documents.map((d) => d.text).join('\n\n');
    })
    .catch((err) => {
      if (err.name === 'AbortError') return;
      root.querySelectorAll('[data-preset=doc],[data-preset=corpus]').forEach((b) => {
        b.disabled = true;
        b.title = '문서셋을 불러오지 못했다 (file://로 열었다면 로컬 서버로 연다)';
      });
    });

  // Tokenizers: load both in parallel, re-render as each arrives.
  const statusMsg = $('[data-msg]');
  const progress = {};
  const updateStatus = () => {
    const parts = TOKENIZERS.map((t) => {
      const tk = s.tok[t.id];
      if (tk instanceof Error) return `${t.short} 실패`;
      if (tk) return `${t.short} 준비됨`;
      const p = progress[t.id];
      return `${t.short} ${p ? `${Math.round(p * 100)}%` : '대기'} (${t.sizeMB}MB)`;
    });
    statusMsg.textContent = `토크나이저 내려받는 중 · ${parts.join(' · ')} · 최초 1회만`;
  };
  updateStatus();

  let lib;
  try {
    lib = await getLib();
  } catch (err) {
    if (ctrl.signal.aborted) return;
    showError(`Transformers.js 라이브러리를 불러오지 못했다 (${escapeHtml(err.message)}).`);
    TOKENIZERS.forEach((t) => (s.tok[t.id] = err));
    render();
    return;
  }

  await Promise.all(
    TOKENIZERS.map(async (t) => {
      try {
        const tk = await lib.AutoTokenizer.from_pretrained(t.id, {
          progress_callback: (e) => {
            if (e.status === 'progress' && e.file === 'tokenizer.json' && e.total) {
              progress[t.id] = e.loaded / e.total;
              if (!ctrl.signal.aborted) updateStatus();
            }
          },
        });
        s.tok[t.id] = tk;
      } catch (err) {
        s.tok[t.id] = err instanceof Error ? err : new Error(String(err));
      }
      if (ctrl.signal.aborted) return;
      updateStatus();
      render();
    }),
  );
  if (ctrl.signal.aborted) return;

  const failed = TOKENIZERS.filter((t) => s.tok[t.id] instanceof Error);
  if (failed.length) {
    showError(`${failed.map((t) => t.short).join(', ')} 토크나이저를 불러오지 못했다. 네트워크 연결을 확인하고 새로고침한다. 학교 방화벽에서 huggingface.co 접근이 막혀 있을 수 있다.`);
  } else {
    $('[data-slot=status]').innerHTML = `<span class="badge-ok">준비됨</span> ${TOKENIZERS.map((t) => t.id).join(' · ')}`;
  }

  function showError(html) {
    $('[data-slot=status]').innerHTML = `<div class="widget__error" role="alert">${html}</div>`;
  }
}

export function unmount(el) {
  const st = state.get(el);
  if (st) {
    st.ctrl.abort();
    if (st.raf) cancelAnimationFrame(st.raf);
    unregisterOutput(st.outputId);
  }
  state.delete(el);
  el.replaceChildren();
}

// ---------- pure helpers (exported for tests) ----------

/** Count user-perceived characters (code points), not UTF-16 units. */
export function charCount(text) {
  return [...text].length;
}

/**
 * Group token ids into displayable pieces. Byte-level BPE can split one Hangul
 * syllable (3 UTF-8 bytes) over several tokens; decoding those alone gives
 * U+FFFD, so we keep adding tokens until the group decodes cleanly.
 * @returns {{ text: string, n: number }[]} concatenated text === decoded input
 */
export function tokenPieces(tokenizer, ids, maxGroup = 8) {
  const pieces = [];
  let group = [];
  const decode = (g) => tokenizer.decode(g, { skip_special_tokens: false, clean_up_tokenization_spaces: false });
  for (const id of ids) {
    group.push(id);
    const text = decode(group);
    if (!text.includes('�') || group.length >= maxGroup) {
      pieces.push({ text, n: group.length });
      group = [];
    }
  }
  if (group.length) pieces.push({ text: decode(group), n: group.length });
  return pieces;
}

function piecesHtml(pieces) {
  let html = '';
  let color = 0;
  const shown = pieces.slice(0, MAX_PIECES);
  for (const p of shown) {
    const cls = `w01-tok c${color++ % COLORS}${p.n > 1 ? ' multi' : ''}`;
    const title = p.n > 1 ? ` title="토큰 ${p.n}개"` : '';
    const sup = p.n > 1 ? `<sup aria-hidden="true">${p.n}</sup>` : '';
    html += `<span class="${cls}"${title}>${escapeHtml(p.text)}${sup}</span>`;
  }
  if (pieces.length > shown.length) {
    const rest = pieces.slice(MAX_PIECES).reduce((sum, p) => sum + p.n, 0);
    html += `<span class="w01-muted"> … 이하 토큰 ${rest.toLocaleString()}개 생략</span>`;
  }
  return html;
}

function contextCell(n, windowSize) {
  const pct = (n / windowSize) * 100;
  const over = n > windowSize;
  const label = pct < 0.1 && n > 0 ? '<0.1%' : `${pct < 10 ? pct.toFixed(1) : Math.round(pct)}%`;
  return `<div class="w01-ctx${over ? ' over' : ''}">
    <div class="bar-progress" aria-hidden="true"><div class="bar-progress__fill" style="width:${Math.min(100, pct)}%"></div></div>
    <span>${label}${over ? ' · 초과' : ''}</span></div>`;
}

function verdictHtml(counts, texts, windowSize) {
  const gpt2 = counts['Xenova/gpt2'];
  const g4o = counts['Xenova/gpt-4o'];
  const overs = TOKENIZERS.filter((t) => counts[t.id].ko && counts[t.id].ko.length > windowSize);
  if (overs.length) {
    const n = counts[overs[0].id].ko.length;
    return `<div class="callout callout--danger"><span class="callout__title">컨텍스트 창 초과 — ${overs.map((t) => t.short).join(', ')}</span>
      한국어 입력이 ${n.toLocaleString()}토큰이라 창 ${windowSize.toLocaleString()}토큰에 다 들어가지 않는다.
      넘친 부분은 잘리거나 호출이 거부된다. 문서를 통째로 넣는 대신 필요한 부분만 골라 넣어야 하는 이유다.</div>`;
  }
  if (gpt2?.ko && gpt2?.en && texts.ko && texts.en && gpt2.en.length > 0) {
    const r2 = gpt2.ko.length / gpt2.en.length;
    const r4 = g4o?.ko && g4o.en.length ? (g4o.ko.length / g4o.en.length).toFixed(1) : '?';
    return `<div class="callout callout--more"><span class="callout__title">같은 뜻인데 한국어가 ${r2.toFixed(1)}배 (GPT-2 기준)</span>
      GPT-2 토크나이저는 한국어 ${gpt2.ko.length}토큰, 영어 ${gpt2.en.length}토큰이다. 다국어 어휘를 가진 GPT-4o에서는 ${r4}배로 줄어든다.
      토큰 수는 곧 비용이고, 컨텍스트 창에 들어가는 양이다.</div>`;
  }
  if (!gpt2?.ko && !g4o?.ko) return '<div class="widget__status">토크나이저가 준비되면 토큰 수가 채워진다. 문자 수는 이미 계산되어 있다.</div>';
  return '';
}

async function fetchJson(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${res.status} ${url.pathname}`);
  return res.json();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
