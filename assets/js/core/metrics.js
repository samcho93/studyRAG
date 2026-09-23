// Retrieval metrics. `retrieved` is a ranked id list, `relevant` a set/array of ids.

const toSet = (r) => (r instanceof Set ? r : new Set(r));

/** Fraction of relevant items found in the top k. */
export function recallAtK(retrieved, relevant, k) {
  const rel = toSet(relevant);
  if (rel.size === 0) return 0;
  const hits = retrieved.slice(0, k).filter((id) => rel.has(id)).length;
  return hits / rel.size;
}

/** Fraction of the top k that is relevant. */
export function precisionAtK(retrieved, relevant, k) {
  const rel = toSet(relevant);
  const top = retrieved.slice(0, k);
  if (top.length === 0) return 0;
  return top.filter((id) => rel.has(id)).length / k;
}

/** 1 if any relevant item appears in the top k. */
export function hitAtK(retrieved, relevant, k) {
  const rel = toSet(relevant);
  return retrieved.slice(0, k).some((id) => rel.has(id)) ? 1 : 0;
}

/** Reciprocal rank of the first relevant item (0 if none). */
export function reciprocalRank(retrieved, relevant) {
  const rel = toSet(relevant);
  const i = retrieved.findIndex((id) => rel.has(id));
  return i === -1 ? 0 : 1 / (i + 1);
}

/**
 * nDCG@k with binary relevance by default; pass a Map id→gain for graded relevance.
 * @param {string[]} retrieved
 * @param {Iterable<string>|Map<string, number>} relevant
 */
export function ndcgAtK(retrieved, relevant, k) {
  const gains = relevant instanceof Map ? relevant : new Map([...toSet(relevant)].map((id) => [id, 1]));
  const dcg = retrieved
    .slice(0, k)
    .reduce((acc, id, i) => acc + (gains.get(id) ?? 0) / Math.log2(i + 2), 0);
  const ideal = [...gains.values()]
    .sort((a, b) => b - a)
    .slice(0, k)
    .reduce((acc, g, i) => acc + g / Math.log2(i + 2), 0);
  return ideal === 0 ? 0 : dcg / ideal;
}

/**
 * Average metrics over a set of queries.
 * @param {{ retrieved: string[], relevant: string[] }[]} runs
 * @param {number} k
 */
export function evaluate(runs, k = 5) {
  const n = runs.length || 1;
  const sum = { recall: 0, precision: 0, hit: 0, mrr: 0, ndcg: 0 };
  for (const { retrieved, relevant } of runs) {
    sum.recall += recallAtK(retrieved, relevant, k);
    sum.precision += precisionAtK(retrieved, relevant, k);
    sum.hit += hitAtK(retrieved, relevant, k);
    sum.mrr += reciprocalRank(retrieved, relevant);
    sum.ndcg += ndcgAtK(retrieved, relevant, k);
  }
  return Object.fromEntries(Object.entries(sum).map(([key, v]) => [key, v / n]));
}
