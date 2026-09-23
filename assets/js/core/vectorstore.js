// In-memory vector store with exact (flat) cosine similarity search.

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a) {
  return Math.sqrt(dot(a, a));
}

export function cosine(a, b) {
  const d = norm(a) * norm(b);
  return d === 0 ? 0 : dot(a, b) / d;
}

function normalize(v) {
  const n = norm(v) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

export class VectorStore {
  constructor() {
    this.items = [];
    this.dim = null;
  }

  get size() {
    return this.items.length;
  }

  /** Add one vector. Vectors are stored L2-normalized so search is a dot product. */
  add(id, vector, meta = {}) {
    if (this.dim === null) this.dim = vector.length;
    if (vector.length !== this.dim) {
      throw new Error(`Dimension mismatch: expected ${this.dim}, got ${vector.length}`);
    }
    this.items.push({ id, vector: normalize(vector), meta });
  }

  addMany(entries) {
    for (const { id, vector, meta } of entries) this.add(id, vector, meta);
  }

  /** @returns {{ id: string, score: number, meta: object }[]} */
  search(query, k = 5) {
    if (this.size === 0) return [];
    const q = normalize(query);
    return this.items
      .map(({ id, vector, meta }) => ({ id, score: dot(q, vector), meta }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  clear() {
    this.items = [];
    this.dim = null;
  }
}
