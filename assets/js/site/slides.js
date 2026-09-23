// PPT-style slide deck for teacher pages.
// Slides are authored in HTML as <section class="slide" data-title="…"> with an
// optional <aside class="notes"> (teacher notes, never shown on the slide).
//
// Layout: left TOC lists the slides, middle holds the 16:9 stage + notes,
// right is the shared 실습 결과 panel (demo widgets / code slides write there).
// Keys: ← → / Space / PageUp PageDown · Home End · F fullscreen · N notes
//       B blackout · T timer · P presenter window
// A second window opened with P (?presenter) stays in sync via BroadcastChannel.

const W = 1280;
const H = 720;

const ICON = {
  prev: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="m15 18-6-6 6-6"/></svg>',
  next: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg>',
  notes: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M4 4h16v16H4z"/><path d="M8 9h8M8 13h8M8 17h5"/></svg>',
  timer: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="13" r="8"/><path d="M12 9v4l2 2M9 2h6"/></svg>',
  full: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5"/></svg>',
  presenter: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><rect x="2" y="4" width="13" height="10" rx="1"/><path d="M18 8h4v12H9v-3"/></svg>',
};

export function initDeck(root = document.querySelector('[data-deck]')) {
  if (!root) return;
  const isPresenter = new URLSearchParams(location.search).has('presenter');
  document.body.classList.toggle('is-presenter', isPresenter);

  const viewport = root.querySelector('.deck-viewport');
  const stage = root.querySelector('.deck-stage');
  const slides = [...stage.querySelectorAll(':scope > .slide')];
  const notesPanel = root.querySelector('.deck-notes');
  const deckTitle = root.dataset.deckTitle ?? document.title;
  const channel = 'BroadcastChannel' in window ? new BroadcastChannel(`raglab-deck:${location.pathname}`) : null;

  // ---------- chrome ----------
  const bar = root.querySelector('.deck-bar__controls');
  bar.innerHTML = `
    <span class="deck-bar__counter" aria-live="polite"></span>
    <span class="deck-bar__timer" data-running="false" title="수업 경과 시간 (T)">00:00</span>
    <button type="button" class="icon-btn" data-act="timer" aria-label="타이머 시작/정지 (T)" title="타이머 (T)">${ICON.timer}</button>
    <button type="button" class="icon-btn" data-act="notes" aria-label="교사 노트 (N)" title="교사 노트 (N)">${ICON.notes}</button>
    <button type="button" class="icon-btn hide-sm" data-act="presenter" aria-label="발표자 창 열기 (P)" title="발표자 창 (P)">${ICON.presenter}</button>
    <button type="button" class="icon-btn hide-sm" data-act="full" aria-label="전체 화면 (F)" title="전체 화면 (F)">${ICON.full}</button>`;
  viewport.insertAdjacentHTML('beforeend', `
    <button type="button" class="icon-btn deck-nav-btn deck-nav-btn--prev" data-act="prev" aria-label="이전 슬라이드">${ICON.prev}</button>
    <button type="button" class="icon-btn deck-nav-btn deck-nav-btn--next" data-act="next" aria-label="다음 슬라이드">${ICON.next}</button>
    <div class="deck-blackout" hidden></div>`);
  const counter = bar.querySelector('.deck-bar__counter');
  const timerEl = bar.querySelector('.deck-bar__timer');
  const blackout = viewport.querySelector('.deck-blackout');

  // Footer with deck title + page number on every slide
  slides.forEach((s, i) => {
    if (s.classList.contains('slide--title') || s.classList.contains('slide--demo')) return;
    s.insertAdjacentHTML('beforeend', `<div class="slide-footer" aria-hidden="true"><span>${deckTitle}</span><span>${i + 1}</span></div>`);
  });

  // ---------- slide list in the left TOC ----------
  const tocSub = document.querySelector('[data-toc-sub]');
  const label = (s) => s.dataset.title ?? s.querySelector('h1,h2')?.textContent ?? '';
  if (tocSub) {
    tocSub.innerHTML = slides
      .map((s, i) => `<li><button type="button" data-go="${i}">${i + 1}. ${label(s)}</button></li>`)
      .join('');
    tocSub.addEventListener('click', (e) => {
      const b = e.target.closest('[data-go]');
      if (b) go(Number(b.dataset.go));
    });
  }

  // ---------- scaling ----------
  const fit = () => {
    const r = viewport.getBoundingClientRect();
    const scale = Math.min(r.width / W, r.height / H);
    const x = (r.width - W * scale) / 2;
    const y = (r.height - H * scale) / 2;
    stage.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
  };
  new ResizeObserver(fit).observe(viewport);

  // ---------- state ----------
  let index = 0;
  const readHash = () => {
    const n = parseInt(location.hash.slice(1), 10);
    return Number.isFinite(n) ? Math.min(Math.max(n - 1, 0), slides.length - 1) : 0;
  };

  function go(i, { broadcast = true } = {}) {
    index = Math.min(Math.max(i, 0), slides.length - 1);
    slides.forEach((s, j) => {
      s.classList.toggle('is-active', j === index);
      s.setAttribute('aria-hidden', j === index ? 'false' : 'true');
      s.inert = j !== index;
    });
    counter.textContent = `${index + 1} / ${slides.length}`;
    history.replaceState(null, '', `${location.search}#${index + 1}`);
    renderNotes();
    tocSub?.querySelectorAll('[data-go]').forEach((b) => b.setAttribute('aria-current', String(Number(b.dataset.go) === index)));
    if (broadcast) channel?.postMessage({ type: 'go', index });
  }

  function renderNotes() {
    const s = slides[index];
    const notes = s.querySelector(':scope > .notes');
    const next = slides[index + 1];
    notesPanel.innerHTML = `<h2>교사 노트 · ${index + 1}. ${label(s)}</h2>
      ${notes ? notes.innerHTML : '<p class="muted">이 슬라이드에는 노트가 없다.</p>'}
      <p class="deck-notes__next">다음 → ${next ? label(next) : '마지막 슬라이드'}</p>`;
  }

  // ---------- notes toggle ----------
  const NOTES_KEY = 'raglab:deck-notes';
  let notesOn = isPresenter || readPref(NOTES_KEY, true);
  const setNotes = (on) => {
    notesOn = on;
    notesPanel.hidden = !on;
    bar.querySelector('[data-act=notes]').setAttribute('aria-pressed', String(on));
    if (!isPresenter) writePref(NOTES_KEY, on);
    fit();
  };

  // ---------- timer ----------
  let elapsed = 0;
  let startedAt = null;
  let tick = null;
  const renderTimer = () => {
    const total = Math.floor((elapsed + (startedAt ? Date.now() - startedAt : 0)) / 1000);
    timerEl.textContent = `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
  };
  const toggleTimer = () => {
    if (startedAt) {
      elapsed += Date.now() - startedAt;
      startedAt = null;
      clearInterval(tick);
    } else {
      startedAt = Date.now();
      tick = setInterval(renderTimer, 500);
    }
    timerEl.dataset.running = String(Boolean(startedAt));
    bar.querySelector('[data-act=timer]').setAttribute('aria-pressed', String(Boolean(startedAt)));
    renderTimer();
  };

  const toggleFull = () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else viewport.requestFullscreen?.();
  };

  const openPresenter = () => {
    const url = new URL(location.href);
    url.searchParams.set('presenter', '');
    window.open(url, 'raglab-presenter', 'width=1200,height=760');
  };

  const actions = {
    prev: () => go(index - 1),
    next: () => go(index + 1),
    notes: () => setNotes(!notesOn),
    timer: toggleTimer,
    full: toggleFull,
    presenter: openPresenter,
  };

  root.addEventListener('click', (e) => {
    const act = e.target.closest('[data-act]')?.dataset.act;
    // runner buttons inside slides share the data-act attribute; only handle deck actions
    if (Object.hasOwn(actions, act)) actions[act]();
  });
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    // Let widgets inside demo slides keep their own keyboard handling.
    if (e.target.closest('input, select, textarea, [contenteditable], summary, .shell-toc')) return;
    const k = e.key;
    if (k === 'ArrowRight' || k === 'PageDown' || (k === ' ' && !e.target.closest('button'))) {
      e.preventDefault();
      go(index + 1);
    } else if (k === 'ArrowLeft' || k === 'PageUp') {
      e.preventDefault();
      go(index - 1);
    } else if (k === 'Home') go(0);
    else if (k === 'End') go(slides.length - 1);
    else if (k === 'f' || k === 'F') toggleFull();
    else if (k === 'n' || k === 'N') setNotes(!notesOn);
    else if (k === 'b' || k === 'B' || k === '.') blackout.hidden = !blackout.hidden;
    else if (k === 't' || k === 'T') toggleTimer();
    else if (k === 'p' || k === 'P') openPresenter();
  });

  // swipe on touch screens
  let touchX = null;
  viewport.addEventListener('touchstart', (e) => { touchX = e.touches[0].clientX; }, { passive: true });
  viewport.addEventListener('touchend', (e) => {
    if (touchX === null || e.target.closest('.widget')) return;
    const dx = e.changedTouches[0].clientX - touchX;
    if (Math.abs(dx) > 60) go(index + (dx < 0 ? 1 : -1));
    touchX = null;
  });

  channel?.addEventListener('message', (e) => {
    if (e.data?.type === 'go') go(e.data.index, { broadcast: false });
    if (e.data?.type === 'hello' && !isPresenter) channel.postMessage({ type: 'go', index });
  });
  window.addEventListener('hashchange', () => go(readHash()));

  setNotes(notesOn);
  go(readHash(), { broadcast: false });
  if (isPresenter) channel?.postMessage({ type: 'hello' });
  fit();
}

function readPref(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    return v === null ? fallback : JSON.parse(v);
  } catch {
    return fallback;
  }
}

function writePref(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore */
  }
}
