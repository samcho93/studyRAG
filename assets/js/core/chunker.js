// Text chunking strategies. Every chunk keeps its character offsets
// ({ start, end }) into the source text so widgets can highlight it.

/**
 * @typedef {{ index: number, start: number, end: number, text: string }} Chunk
 */

export const STRATEGIES = {
  fixed: '고정 길이',
  sentence: '문장 단위',
  recursive: '재귀 분할',
};

const DEFAULT_SEPARATORS = ['\n\n', '\n', '. ', '다. ', ' ', ''];

/**
 * Split text into chunks.
 * @param {string} text
 * @param {{ strategy?: 'fixed'|'sentence'|'recursive', size?: number, overlap?: number, separators?: string[] }} opts
 * @returns {Chunk[]}
 */
export function chunk(text, opts = {}) {
  const { strategy = 'recursive', size = 300, overlap = 0, separators = DEFAULT_SEPARATORS } = opts;
  const safeSize = Math.max(1, Math.floor(size));
  const safeOverlap = Math.max(0, Math.min(Math.floor(overlap), safeSize - 1));

  let spans;
  switch (strategy) {
    case 'fixed':
      spans = fixedSpans(0, text.length, safeSize, safeOverlap);
      break;
    case 'sentence':
      spans = mergeSpans(
        sentenceSpans(text).flatMap(([s, e]) =>
          e - s > safeSize ? fixedSpans(s, e, safeSize, 0) : [[s, e]]),
        safeSize,
        safeOverlap,
      );
      break;
    case 'recursive':
      spans = recursiveSpans(text, 0, text.length, safeSize, safeOverlap, separators);
      break;
    default:
      throw new Error(`Unknown chunking strategy: ${strategy}`);
  }

  return spans
    .filter(([s, e]) => text.slice(s, e).trim().length > 0)
    .map(([start, end], index) => ({ index, start, end, text: text.slice(start, end) }));
}

/** Fixed-size character windows with overlap. */
export function fixedSpans(start, end, size, overlap) {
  const spans = [];
  const step = Math.max(1, size - overlap);
  for (let s = start; s < end; s += step) {
    const e = Math.min(s + size, end);
    spans.push([s, e]);
    if (e === end) break;
  }
  return spans;
}

/**
 * Sentence boundaries: ., !, ?, 。 followed by whitespace or end of text, or a line break.
 * Requiring whitespace after the punctuation keeps decimals like "4.0" intact.
 */
export function sentenceSpans(text) {
  const spans = [];
  const re = /[.!?。]+["')\]]?(?=\s|$)\s*|\n+/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    const end = m.index + m[0].length;
    if (end > last) spans.push([last, end]);
    last = end;
  }
  if (last < text.length) spans.push([last, text.length]);
  return spans;
}

/** LangChain-style recursive splitting: try coarse separators first, then finer ones. */
export function recursiveSpans(text, start, end, size, overlap, separators) {
  if (end - start <= size) return [[start, end]];
  const [sep, ...rest] = separators;
  if (sep === undefined || sep === '') return fixedSpans(start, end, size, overlap);

  const segment = text.slice(start, end);
  if (!segment.includes(sep)) return recursiveSpans(text, start, end, size, overlap, rest);

  const pieces = [];
  let s = start;
  let idx;
  while ((idx = text.indexOf(sep, s)) !== -1 && idx < end) {
    const e = Math.min(end, idx + sep.length);
    if (e > s) pieces.push([s, e]);
    s = e;
  }
  if (s < end) pieces.push([s, end]);

  const atoms = pieces.flatMap(([ps, pe]) =>
    pe - ps > size ? recursiveSpans(text, ps, pe, size, overlap, rest) : [[ps, pe]]);
  return mergeSpans(atoms, size, overlap);
}

/**
 * Greedily pack contiguous atoms into chunks of at most `size` chars.
 * When a chunk closes, trailing atoms totalling <= overlap chars are carried over.
 */
export function mergeSpans(atoms, size, overlap) {
  const out = [];
  let cur = [];
  for (const atom of atoms) {
    if (cur.length && atom[1] - cur[0][0] > size) {
      out.push([cur[0][0], cur[cur.length - 1][1]]);
      const keep = [];
      const lastEnd = cur[cur.length - 1][1];
      for (let i = cur.length - 1; i >= 0; i--) {
        if (lastEnd - cur[i][0] > overlap || atom[1] - cur[i][0] > size) break;
        keep.unshift(cur[i]);
      }
      cur = keep;
    }
    cur.push(atom);
  }
  if (cur.length) out.push([cur[0][0], cur[cur.length - 1][1]]);
  return out;
}

/** Summary statistics for a chunk list. */
export function chunkStats(chunks, textLength) {
  if (chunks.length === 0) return { count: 0, avg: 0, min: 0, max: 0, duplicated: 0 };
  const lens = chunks.map((c) => c.end - c.start);
  const total = lens.reduce((a, b) => a + b, 0);
  return {
    count: chunks.length,
    avg: Math.round(total / chunks.length),
    min: Math.min(...lens),
    max: Math.max(...lens),
    // share of characters stored more than once because of overlap
    duplicated: textLength ? Math.max(0, total - textLength) / textLength : 0,
  };
}
