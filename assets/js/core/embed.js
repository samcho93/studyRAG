// Transformers.js wrapper: model loading (with progress), caching and embedding.
// The model is downloaded once from the Hugging Face CDN and then served from
// the browser cache. Inference runs on WASM (see pickDevice for why not WebGPU).

const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

export const MODELS = [
  {
    id: 'Xenova/multilingual-e5-small',
    label: 'multilingual-e5-small (다국어, 384차원)',
    sizeMB: 118,
    // e5 models expect role prefixes
    queryPrefix: 'query: ',
    passagePrefix: 'passage: ',
  },
  {
    id: 'Xenova/paraphrase-multilingual-MiniLM-L12-v2',
    label: 'paraphrase-multilingual-MiniLM (다국어, 384차원)',
    sizeMB: 118,
    queryPrefix: '',
    passagePrefix: '',
  },
  {
    id: 'Xenova/all-MiniLM-L6-v2',
    label: 'all-MiniLM-L6-v2 (영어 전용 · 한국어 실패 비교용)',
    sizeMB: 23,
    queryPrefix: '',
    passagePrefix: '',
  },
];

export const DEFAULT_MODEL = MODELS[0].id;

let lib = null;
let current = { id: null, extractor: null, device: null };
const loading = new Map();

async function getLib() {
  if (!lib) {
    lib = await import(TRANSFORMERS_URL);
    lib.env.allowLocalModels = false;
  }
  return lib;
}

// WebGPU is intentionally not used: with the q8 weights it returns wrong
// vectors (measured 2026-09 with Transformers.js 3.8.1 — unrelated Korean
// sentences scored 0.93, above paraphrases), and fp16 on WebGPU is correct but
// doubles the download (~235MB). WASM + q8 is correct, ~118MB and fast enough
// for this course's corpus.
async function pickDevice() {
  return 'wasm';
}

export function modelInfo(id = current.id) {
  return MODELS.find((m) => m.id === id) ?? { id, queryPrefix: '', passagePrefix: '' };
}

export function currentModel() {
  return { id: current.id, device: current.device, ready: Boolean(current.extractor) };
}

/**
 * Load (or switch to) an embedding model.
 * @param {string} modelId
 * @param {(p: { status: string, progress: number, file?: string, loaded?: number, total?: number }) => void} [onProgress]
 *   progress is 0..1 aggregated over all files.
 */
export async function loadModel(modelId = DEFAULT_MODEL, onProgress = () => {}) {
  if (current.id === modelId && current.extractor) return current;
  if (loading.has(modelId)) return loading.get(modelId);

  const task = (async () => {
    const { pipeline } = await getLib();
    const device = await pickDevice();
    const files = new Map();
    const report = (status, file) => {
      let loaded = 0;
      let total = 0;
      for (const f of files.values()) {
        loaded += f.loaded;
        total += f.total;
      }
      onProgress({ status, file, loaded, total, progress: total ? loaded / total : 0 });
    };

    const progress_callback = (e) => {
      if (e.status === 'progress' && e.total) {
        files.set(e.file, { loaded: e.loaded, total: e.total });
        report('downloading', e.file);
      } else if (e.status === 'done' && files.has(e.file)) {
        const f = files.get(e.file);
        files.set(e.file, { loaded: f.total, total: f.total });
        report('downloading', e.file);
      }
    };

    onProgress({ status: 'init', progress: 0 });
    const extractor = await pipeline('feature-extraction', modelId, { device, dtype: 'q8', progress_callback });

    if (current.extractor?.dispose) await current.extractor.dispose();
    current = { id: modelId, extractor, device };
    onProgress({ status: 'ready', progress: 1 });
    return current;
  })();

  loading.set(modelId, task);
  try {
    return await task;
  } finally {
    loading.delete(modelId);
  }
}

function assertReady() {
  if (!current.extractor) throw new Error('임베딩 모델이 아직 로드되지 않았다. loadModel()을 먼저 호출한다.');
}

/**
 * Embed one string. Returns a normalized Float32Array.
 * @param {string} text
 * @param {{ role?: 'query'|'passage' }} opts
 */
export async function embed(text, { role = 'query' } = {}) {
  const [vec] = await embedBatch([text], { role });
  return vec;
}

/**
 * Embed many strings in batches.
 * @param {string[]} texts
 * @param {{ role?: 'query'|'passage', batchSize?: number, onProgress?: (done: number, total: number) => void }} opts
 */
export async function embedBatch(texts, { role = 'passage', batchSize = 16, onProgress } = {}) {
  assertReady();
  const info = modelInfo();
  const prefix = role === 'query' ? info.queryPrefix : info.passagePrefix;
  const out = [];
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize).map((t) => prefix + t);
    const tensor = await current.extractor(batch, { pooling: 'mean', normalize: true });
    const [n, dim] = tensor.dims;
    for (let j = 0; j < n; j++) out.push(tensor.data.slice(j * dim, (j + 1) * dim));
    onProgress?.(Math.min(i + batchSize, texts.length), texts.length);
  }
  return out;
}

/**
 * Render a standard loading UI into `el` and load the model.
 * Returns the loaded model state; throws on failure after showing the error.
 */
export async function loadModelWithUI(el, modelId = DEFAULT_MODEL) {
  const info = modelInfo(modelId);
  el.innerHTML = `
    <div class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span>
      <span data-msg>모델 준비 중… (${info.sizeMB ?? '?'}MB, 최초 1회만 다운로드)</span>
    </div>
    <div class="bar-progress" aria-hidden="true"><div class="bar-progress__fill"></div></div>`;
  const msg = el.querySelector('[data-msg]');
  const bar = el.querySelector('.bar-progress__fill');
  try {
    const state = await loadModel(modelId, ({ status, progress, loaded, total }) => {
      bar.style.width = `${Math.round(progress * 100)}%`;
      if (status === 'downloading' && total) {
        msg.textContent = `다운로드 중 ${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)}MB`;
      }
    });
    el.innerHTML = `<div class="widget__status"><span class="badge-ok">준비됨</span>
      ${info.id} · ${state.device === 'webgpu' ? 'WebGPU' : 'WASM'}</div>`;
    return state;
  } catch (err) {
    el.innerHTML = `<div class="widget__error" role="alert">모델을 불러오지 못했다: ${escapeHtml(err.message)}<br>
      네트워크 연결을 확인하고 새로고침한다. 학교 방화벽에서 huggingface.co 접근이 막혀 있을 수 있다.</div>`;
    throw err;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
