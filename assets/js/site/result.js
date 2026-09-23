// Right-hand "실습 결과" panel.
// Widgets and code runners register an output node with an inline fallback host.
// On wide screens the node lives in the panel as a card; below 1000px it moves
// back next to its source so phones never need a second column.

const mq = matchMedia('(max-width: 999px)');
const cards = new Map();

function panelBody() {
  return document.querySelector('[data-result-body]');
}

function syncEmpty() {
  const body = panelBody();
  if (!body) return;
  const empty = body.querySelector('.result-empty');
  if (empty) empty.hidden = body.querySelector('.result-card') !== null;
}

function place(entry) {
  const body = panelBody();
  if (!mq.matches && body) {
    entry.content.append(entry.node);
    // newest output on top, right under the panel header
    if (!entry.card.isConnected) body.querySelector('.result-empty')?.after(entry.card) ?? body.prepend(entry.card);
  } else {
    entry.inlineHost.append(entry.node);
    entry.card.remove();
  }
  syncEmpty();
}

/**
 * Register an output node.
 * @param {string} id stable key (re-registering replaces the node)
 * @param {{ title: string, node: HTMLElement, inlineHost: HTMLElement }} opts
 */
export function registerOutput(id, { title, node, inlineHost }) {
  let entry = cards.get(id);
  if (!entry) {
    const card = document.createElement('details');
    card.className = 'result-card';
    card.open = true;
    card.innerHTML = '<summary></summary><div class="result-card__body"></div>';
    entry = { card, content: card.querySelector('.result-card__body'), inlineHost, node };
    cards.set(id, entry);
  }
  entry.card.querySelector('summary').textContent = title;
  entry.node = node;
  entry.inlineHost = inlineHost;
  entry.content.replaceChildren();
  place(entry);
  return entry;
}

/** Remove a registered output (e.g. on widget unmount). */
export function unregisterOutput(id) {
  const entry = cards.get(id);
  if (!entry) return;
  entry.card.remove();
  entry.node.remove();
  cards.delete(id);
  syncEmpty();
}

/** Bring a card into view and flash it briefly. */
export function focusOutput(id) {
  const entry = cards.get(id);
  if (!entry || mq.matches) return;
  entry.card.open = true;
  entry.card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  entry.card.classList.add('is-flash');
  setTimeout(() => entry.card.classList.remove('is-flash'), 900);
}

mq.addEventListener('change', () => cards.forEach(place));

document.addEventListener('click', (e) => {
  if (!e.target.closest('[data-result-clear]')) return;
  for (const [id, entry] of cards) {
    if (entry.card.dataset.clearable !== undefined) unregisterOutput(id);
  }
});

export function markClearable(id) {
  const entry = cards.get(id);
  if (entry) entry.card.dataset.clearable = '';
}
