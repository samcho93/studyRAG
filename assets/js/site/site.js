// Site chrome: theme toggle, week prev/next navigation, progress tracking.
// Loaded on every page as <script type="module" src=".../site/site.js">.

import { WEEKS, weekId } from './weeks.js';

const ROOT = new URL('../../../', import.meta.url);
const THEME_KEY = 'raglab:theme';
const DONE_KEY = 'raglab:done';

// ---------- storage helpers (private mode can throw) ----------
function read(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    return v === null ? fallback : JSON.parse(v);
  } catch {
    return fallback;
  }
}

function write(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore */
  }
}

// ---------- theme ----------
const ICONS = {
  light: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
  dark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>',
  system: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/></svg>',
};
const LABELS = { light: '라이트', dark: '다크', system: '시스템 설정' };
const ORDER = ['system', 'light', 'dark'];

function applyTheme(mode) {
  if (mode === 'system') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', mode);
}

function initTheme() {
  let mode = read(THEME_KEY, 'system');
  applyTheme(mode);
  const btn = document.querySelector('[data-theme-toggle]');
  if (!btn) return;
  const render = () => {
    btn.innerHTML = ICONS[mode];
    btn.setAttribute('aria-label', `테마: ${LABELS[mode]} (눌러서 변경)`);
    btn.title = `테마: ${LABELS[mode]}`;
  };
  render();
  btn.addEventListener('click', () => {
    mode = ORDER[(ORDER.indexOf(mode) + 1) % ORDER.length];
    write(THEME_KEY, mode);
    applyTheme(mode);
    render();
  });
}

// ---------- progress ----------
export function doneWeeks() {
  return new Set(read(DONE_KEY, []));
}

function setDone(no, done) {
  const set = doneWeeks();
  if (done) set.add(no);
  else set.delete(no);
  write(DONE_KEY, [...set].sort((a, b) => a - b));
}

function initDoneToggle() {
  const box = document.querySelector('[data-done-toggle]');
  if (!box) return;
  const no = Number(box.dataset.doneToggle);
  box.checked = doneWeeks().has(no);
  box.addEventListener('change', () => setDone(no, box.checked));
}

function initProgressBadges() {
  const done = doneWeeks();
  document.querySelectorAll('[data-week-progress]').forEach((el) => {
    if (done.has(Number(el.dataset.weekProgress))) {
      el.hidden = false;
    }
  });
  const summary = document.querySelector('[data-progress-summary]');
  if (summary) {
    const ready = WEEKS.filter((w) => w.ready).length;
    summary.textContent = `완료 ${done.size}주 · 공개 ${ready} / ${WEEKS.length}주`;
  }
}

// ---------- prev/next ----------
function initWeekNav() {
  const nav = document.querySelector('[data-week-nav]');
  if (!nav) return;
  const no = Number(nav.dataset.weekNav);
  const page = nav.dataset.page ?? 'index.html';
  const prev = [...WEEKS].reverse().find((w) => w.no < no && w.ready);
  const next = WEEKS.find((w) => w.no > no && w.ready);
  const link = (w, cls, label) =>
    w
      ? `<a class="${cls}" href="${new URL(`weeks/${weekId(w.no)}/${page}`, ROOT)}"><small>${label}</small>${w.no}주차 · ${w.title}</a>`
      : `<span class="${cls}"><small>${label}</small>${cls === 'prev' ? '처음 공개된 주차' : '다음 주차 준비 중'}</span>`;
  nav.innerHTML = link(prev, 'prev', '← 이전') + link(next, 'next', '다음 →');
}

// ---------- left TOC (week list + sections of the current page) ----------
function initToc() {
  const toc = document.querySelector('[data-toc]');
  if (!toc) return;
  const current = Number(toc.dataset.week);
  const role = toc.dataset.role ?? 'student';
  const page = role === 'teacher' ? 'teacher.html' : '';
  const done = doneWeeks();

  const sections =
    role === 'student'
      ? [...document.querySelectorAll('.shell-main section[id] > h2')].map((h) => {
          const num = h.querySelector('.section-num')?.textContent;
          const text = [...h.childNodes].filter((n) => !n.classList?.contains('section-num')).map((n) => n.textContent).join('').trim();
          return { href: `#${h.parentElement.id}`, label: num ? `${num}. ${text}` : text };
        })
      : [];

  const items = WEEKS.map((w) => {
    const no = String(w.no).padStart(2, '0');
    const isCur = w.no === current;
    if (!w.ready) {
      return `<li><span class="toc-disabled"><span class="toc-no">${no}</span>${w.title}</span></li>`;
    }
    const sub = isCur
      ? `<ol class="toc-list toc-sub" data-toc-sub>${sections
          .map((s) => `<li><a href="${s.href}">${s.label}</a></li>`)
          .join('')}</ol>`
      : '';
    return `<li><a href="${new URL(`weeks/${weekId(w.no)}/${page}`, ROOT)}"${isCur ? ' aria-current="page"' : ''}>
      <span class="toc-no">${no}</span>${w.title}${done.has(w.no) ? '<span class="toc-done" aria-label="완료">✓</span>' : ''}</a>${sub}</li>`;
  }).join('');

  toc.innerHTML = `
    <p class="toc-title">${role === 'teacher' ? '교사용 · 주차 슬라이드' : '학생용 · 주차 목차'}</p>
    <ol class="toc-list">${items}</ol>
    <p class="toc-title">바로가기</p>
    <ul class="toc-list">
      <li><a href="${ROOT}">과정 소개</a></li>
      <li><a href="${new URL('teacher/', ROOT)}">교수자용 허브</a></li>
    </ul>`;

  const shell = toc.closest('.shell');
  const toggle = document.querySelector('[data-toc-toggle]');
  const setOpen = (open) => {
    shell.classList.toggle('toc-open', open);
    toggle?.setAttribute('aria-expanded', String(open));
  };
  toggle?.addEventListener('click', () => setOpen(!shell.classList.contains('toc-open')));
  shell.querySelector('.shell-backdrop')?.addEventListener('click', () => setOpen(false));
  toc.addEventListener('click', (e) => {
    if (e.target.closest('a, button')) setOpen(false);
  });
}

initTheme();
initToc();
initDoneToggle();
initProgressBadges();
initWeekNav();
