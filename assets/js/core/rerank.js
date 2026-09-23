// Result fusion and reranking for hybrid search.

/**
 * Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_i(d)).
 * Uses ranks only, so BM25 and cosine scores need no calibration.
 * @param {{ id: string }[][]} lists ranked result lists (best first)
 * @param {{ k?: number, topK?: number }} opts
 */
export function rrf(lists, { k = 60, topK = 10 } = {}) {
  const fused = new Map();
  lists.forEach((list, li) => {
    list.forEach((item, rank) => {
      const entry = fused.get(item.id) ?? { ...item, score: 0, ranks: [] };
      entry.score += 1 / (k + rank + 1);
      entry.ranks[li] = rank + 1;
      fused.set(item.id, entry);
    });
  });
  return [...fused.values()].sort((a, b) => b.score - a.score).slice(0, topK);
}

/** Min-max normalize scores within a list to [0, 1]. */
export function minMax(list) {
  if (list.length === 0) return [];
  const scores = list.map((r) => r.score);
  const lo = Math.min(...scores);
  const hi = Math.max(...scores);
  return list.map((r) => ({ ...r, score: hi === lo ? 1 : (r.score - lo) / (hi - lo) }));
}

/**
 * Weighted score fusion: alpha * vector + (1 - alpha) * keyword, after min-max.
 * @param {{ id: string, score: number }[]} vectorList
 * @param {{ id: string, score: number }[]} keywordList
 */
export function weightedFusion(vectorList, keywordList, { alpha = 0.5, topK = 10 } = {}) {
  const fused = new Map();
  for (const r of minMax(vectorList)) fused.set(r.id, { ...r, score: alpha * r.score });
  for (const r of minMax(keywordList)) {
    const entry = fused.get(r.id) ?? { ...r, score: 0 };
    entry.score += (1 - alpha) * r.score;
    fused.set(r.id, entry);
  }
  return [...fused.values()].sort((a, b) => b.score - a.score).slice(0, topK);
}

/**
 * Rerank candidates with an arbitrary (possibly async) scoring function,
 * e.g. cosine against a stronger embedding model or an LLM relevance judge.
 */
export async function rerank(query, candidates, scoreFn, { topK = 5 } = {}) {
  const scored = await Promise.all(
    candidates.map(async (c) => ({ ...c, rerankScore: await scoreFn(query, c) })),
  );
  return scored.sort((a, b) => b.rerankScore - a.rerankScore).slice(0, topK);
}
