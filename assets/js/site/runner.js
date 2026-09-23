// Runnable code blocks: <pre data-run="c1" data-title="C1 …"><code>…</code></pre>
// becomes an editable textarea with ▶ 실행. Code runs as an ES module inside a
// Web Worker (so an infinite loop can be terminated) and console output is sent
// to the 실습 결과 panel. Root-relative paths ('/assets/…') are rewritten to the
// site root so the same snippet works locally and on GitHub Pages.

import { registerOutput, focusOutput, markClearable } from './result.js';

const ROOT = new URL('../../../', import.meta.url).href;
const TIMEOUT_MS = 10000;

const PRELUDE = `const __fmt=(v)=>{if(typeof v==='string')return v;if(v instanceof Error)return v.stack||v.message;if(ArrayBuffer.isView(v))v=Array.from(v).map(x=>+x.toFixed?.(4)||x);try{return JSON.stringify(v,(k,x)=>ArrayBuffer.isView(x)?Array.from(x):x instanceof Map?Object.fromEntries(x):x instanceof Set?[...x]:x,2)??String(v)}catch{return String(v)}};const __send=(level,args)=>self.postMessage({type:'log',level,text:args.map(__fmt).join(' ')});const console={log:(...a)=>__send('log',a),info:(...a)=>__send('info',a),warn:(...a)=>__send('warn',a),error:(...a)=>__send('error',a),table:(d)=>{try{self.postMessage({type:'table',rows:JSON.parse(JSON.stringify(d))})}catch{__send('log',[d])}},clear:()=>self.postMessage({type:'clear'})};self.addEventListener('unhandledrejection',(e)=>__send('error',[e.reason]));`;

function rewritePaths(code) {
  return code.replace(/(['"`])\/assets\//g, `$1${ROOT}assets/`);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

function tableHtml(rows) {
  const list = Array.isArray(rows) ? rows : Object.entries(rows).map(([k, v]) => ({ '(index)': k, ...(typeof v === 'object' ? v : { value: v }) }));
  const cols = [...new Set(list.flatMap((r) => (r && typeof r === 'object' ? Object.keys(r) : ['value'])))];
  const cell = (v) => escapeHtml(typeof v === 'number' ? +v.toFixed(4) : typeof v === 'object' ? JSON.stringify(v) : v ?? '');
  return `<table><thead><tr>${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join('')}</tr></thead><tbody>${list
    .map((r) => `<tr>${cols.map((c) => `<td>${cell(r && typeof r === 'object' ? r[c] : r)}</td>`).join('')}</tr>`)
    .join('')}</tbody></table>`;
}

/** Run a module source in a worker; stream console output into `out`. */
export function runModule(code, out) {
  out.innerHTML = '<div class="line-info">실행 중…</div>';
  const src = `${PRELUDE}\n${rewritePaths(code)}\n;self.postMessage({type:'done'});`;
  const url = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
  const worker = new Worker(url, { type: 'module' });
  const started = performance.now();
  let first = true;

  const append = (html) => {
    if (first) {
      out.innerHTML = '';
      first = false;
    }
    out.insertAdjacentHTML('beforeend', html);
  };

  return new Promise((resolve) => {
    const finish = (html) => {
      clearTimeout(timer);
      worker.terminate();
      URL.revokeObjectURL(url);
      append(html);
      resolve();
    };
    const timer = setTimeout(
      () => finish(`<div class="line-error">⏱ ${TIMEOUT_MS / 1000}초 안에 끝나지 않아 중단했다. 무한 루프가 없는지 확인한다 (예: overlap ≥ size).</div>`),
      TIMEOUT_MS,
    );
    worker.onmessage = ({ data }) => {
      if (data.type === 'log') append(`<div class="line-${data.level}">${escapeHtml(data.text)}</div>`);
      else if (data.type === 'table') append(tableHtml(data.rows));
      else if (data.type === 'clear') out.innerHTML = '';
      else if (data.type === 'done') {
        finish(`<div class="line-info">✓ 완료 (${Math.round(performance.now() - started)}ms)</div>`);
      }
    };
    worker.onerror = (e) => {
      e.preventDefault();
      const where = e.lineno ? ` (${Math.max(1, e.lineno - 1)}행)` : '';
      finish(`<div class="line-error">✗ ${escapeHtml(e.message || '실행 오류 — import 경로나 문법을 확인한다')}${where}</div>`);
    };
  });
}

export function initRunners(scope = document) {
  scope.querySelectorAll('pre[data-run]').forEach((pre) => {
    const id = `run:${pre.dataset.run}`;
    const title = pre.dataset.title ?? '코드 실행 결과';
    const original = pre.textContent.replace(/\n$/, '');

    const box = document.createElement('div');
    box.className = 'runner';
    box.innerHTML = `
      <textarea spellcheck="false" autocapitalize="off" autocomplete="off" aria-label="${escapeHtml(title)} 코드"></textarea>
      <div class="runner__bar">
        <button type="button" class="btn" data-act="run">▶ 실행</button>
        <button type="button" class="btn btn--ghost" data-act="reset">되돌리기</button>
        <span class="spacer"></span><span>Ctrl+Enter</span>
      </div>
      <div class="runner__inline"></div>`;
    const ta = box.querySelector('textarea');
    ta.value = original;
    ta.rows = Math.min(28, original.split('\n').length + 1);
    pre.replaceWith(box);

    const out = document.createElement('div');
    out.className = 'console-out';
    let registered = false;

    const run = async () => {
      if (!registered) {
        registerOutput(id, { title, node: out, inlineHost: box.querySelector('.runner__inline') });
        markClearable(id);
        registered = true;
      } else if (!out.isConnected) {
        registerOutput(id, { title, node: out, inlineHost: box.querySelector('.runner__inline') });
        markClearable(id);
      }
      focusOutput(id);
      await runModule(ta.value, out);
    };

    box.addEventListener('click', (e) => {
      const act = e.target.closest('[data-act]')?.dataset.act;
      if (act === 'run') run();
      if (act === 'reset') ta.value = original;
    });
    ta.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        run();
      }
    });
  });
}
