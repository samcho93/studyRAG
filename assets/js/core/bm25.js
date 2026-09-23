// BM25 keyword retrieval in plain JS.
// Korean has no spaces between morphemes and particles ("RAG는", "RAG를"),
// so Hangul words are additionally indexed as character bigrams.

const HANGUL = /[가-힣]/;

/** Lowercase, split on non-word chars, add Hangul bigrams. */
export function tokenize(text, { bigrams = true } = {}) {
  const words = text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
  const tokens = [];
  for (const w of words) {
    tokens.push(w);
    if (bigrams && HANGUL.test(w) && w.length > 2) {
      for (let i = 0; i < w.length - 1; i++) tokens.push(w.slice(i, i + 2));
    }
  }
  return tokens;
}

export class BM25 {
  /**
   * @param {{ k1?: number, b?: number, tokenizer?: (t: string) => string[] }} opts
   */
  constructor({ k1 = 1.5, b = 0.75, tokenizer = tokenize } = {}) {
    this.k1 = k1;
    this.b = b;
    this.tokenizer = tokenizer;
    this.docs = [];
    this.df = new Map();
    this.avgdl = 0;
  }

  /** @param {{ id: string, text: string }[]} docs */
  add(docs) {
    for (const { id, text, ...meta } of docs) {
      const tokens = this.tokenizer(text);
      const tf = new Map();
      for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
      for (const t of tf.keys()) this.df.set(t, (this.df.get(t) ?? 0) + 1);
      this.docs.push({ id, text, meta, tf, len: tokens.length });
    }
    this.avgdl = this.docs.reduce((a, d) => a + d.len, 0) / (this.docs.length || 1);
    return this;
  }

  idf(term) {
    const n = this.df.get(term) ?? 0;
    const N = this.docs.length;
    return Math.log(1 + (N - n + 0.5) / (n + 0.5));
  }

  /** @returns {{ id: string, score: number, text: string, meta: object }[]} */
  search(query, k = 5) {
    const qTerms = [...new Set(this.tokenizer(query))];
    const results = [];
    for (const d of this.docs) {
      let score = 0;
      for (const t of qTerms) {
        const f = d.tf.get(t);
        if (!f) continue;
        const norm = f + this.k1 * (1 - this.b + (this.b * d.len) / (this.avgdl || 1));
        score += this.idf(t) * ((f * (this.k1 + 1)) / norm);
      }
      if (score > 0) results.push({ id: d.id, score, text: d.text, meta: d.meta });
    }
    return results.sort((a, b) => b.score - a.score).slice(0, k);
  }
}
