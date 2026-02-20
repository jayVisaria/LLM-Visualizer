import axios from 'axios';

const BASE = import.meta.env.VITE_API_URL ?? '';

const api = axios.create({
  baseURL: `${BASE}/api`,
  timeout: 120000,
});

// ─── Tokenization ──────────────────────────────────────────────────────────
export const tokenize = (text: string) =>
  api.post('/tokenize', { text }).then(r => r.data);

export const getBpeMerges = (limit = 100, offset = 0) =>
  api.get('/bpe-merges', { params: { limit, offset } }).then(r => r.data);

export const getVocabStats = () =>
  api.get('/vocab-stats').then(r => r.data);

export const trainTokenizer = (num_merges: number) =>
  api.post('/train-tokenizer', { num_merges }).then(r => r.data);

// ─── Embeddings ─────────────────────────────────────────────────────────────
export const getEmbeddings = (text: string) =>
  api.post('/embeddings', { text }).then(r => r.data);

export const getEmbedPCA = (text: string) =>
  api.post('/embed-pca', { text }).then(r => r.data);

export const getEmbeddingMatrix = (rows = 50) =>
  api.get('/embedding-matrix', { params: { rows } }).then(r => r.data);

// ─── Positional Encoding ──────────────────────────────────────────────────
export const getPositionalEncoding = (seq_len = 64, d_model = 64) =>
  api.get('/positional-encoding', { params: { seq_len, d_model } }).then(r => r.data);

export const getPositionSimilarity = (seq_len = 32, d_model = 64) =>
  api.get('/position-similarity', { params: { seq_len, d_model } }).then(r => r.data);

// ─── Attention ──────────────────────────────────────────────────────────────
export const getSelfAttention = (text: string, show_masking = true) =>
  api.post('/self-attention', { text, show_masking }).then(r => r.data);

export const getMultiHeadAttention = (text: string, n_heads = 4, show_masking = true) =>
  api.post('/multi-head-attention', { text, n_heads, show_masking }).then(r => r.data);

// ─── Transformer ────────────────────────────────────────────────────────────
export const getTransformerBlock = (text: string) =>
  api.post('/transformer-block', { text }).then(r => r.data);

export const getModelArchitecture = () =>
  api.get('/model-architecture').then(r => r.data);

export const getForwardPass = (text: string) =>
  api.post('/forward-pass', { text }).then(r => r.data);

// ─── Training ───────────────────────────────────────────────────────────────
export const getTrainingStatus = () =>
  api.get('/training-status').then(r => r.data);

export const getLossHistory = () =>
  api.get('/loss-history').then(r => r.data);

export const stopTraining = () =>
  api.post('/stop-training').then(r => r.data);

export const getGradientDemo = () =>
  api.post('/gradient-demo').then(r => r.data);

export const getGradientFlow = (text: string) =>
  api.post('/gradient-flow', { text }).then(r => r.data);

// ─── Generation ─────────────────────────────────────────────────────────────
export interface GenerateParams {
  prompt: string;
  max_tokens: number;
  temperature: number;
  top_k: number | null;
  top_p: number | null;
  strategy: string;
}

export const generateSync = (params: GenerateParams) =>
  api.post('/generate-sync', params).then(r => r.data);

// SSE helper for streaming generation
export function streamGenerate(
  params: GenerateParams,
  onToken: (data: any) => void,
  onDone: (data: any) => void,
  onError?: (err: any) => void,
): () => void {
  const body = JSON.stringify(params);
  const ctrl = new AbortController();

  fetch(`${BASE}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
    signal: ctrl.signal,
  })
    .then(async (response) => {
      const reader = response.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'token') onToken(data);
              else if (data.type === 'done') onDone(data);
            } catch { /* skip */ }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') onError?.(err);
    });

  return () => ctrl.abort();
}

// SSE helper for streaming training
export function streamTraining(
  params: any,
  onStep: (data: any) => void,
  onSample: (data: any) => void,
  onDone: (data: any) => void,
  onError?: (err: any) => void,
): () => void {
  const body = JSON.stringify(params);
  const ctrl = new AbortController();

  fetch(`${BASE}/api/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
    signal: ctrl.signal,
  })
    .then(async (response) => {
      const reader = response.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'step') onStep(data);
              else if (data.type === 'sample') onSample(data);
              else if (data.type === 'done') onDone(data);
            } catch { /* skip */ }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') onError?.(err);
    });

  return () => ctrl.abort();
}

// ─── Health ─────────────────────────────────────────────────────────────────
export const getHealth = () =>
  api.get('/health').then(r => r.data);

// ─── Config & Model Management ─────────────────────────────────────────────
export const getConfig = () =>
  api.get('/config').then(r => r.data);

export const updateConfig = (data: {
  model_config_data?: {
    d_model?: number;
    n_heads?: number;
    n_layers?: number;
    context_len?: number;
    dropout?: number;
  };
  train_config?: {
    max_steps?: number;
    batch_size?: number;
    learning_rate?: number;
    weight_decay?: number;
    warmup_steps?: number;
    grad_clip?: number;
  };
}) => api.post('/config', data).then(r => r.data);

export const rebuildModel = (config: {
  d_model: number;
  n_heads: number;
  n_layers: number;
  context_len: number;
  dropout: number;
}) => api.post('/rebuild-model', config).then(r => r.data);

export const getStatus = () =>
  api.get('/status').then(r => r.data);

// ─── Datasets ───────────────────────────────────────────────────────────────
export const getDatasets = () =>
  api.get('/datasets').then(r => r.data);

export const selectDataset = (name: string) =>
  api.post('/select-dataset', { name }).then(r => r.data);

export const uploadData = (file: File) => {
  const form = new FormData();
  form.append('file', file);
  return api.post('/upload-data', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data);
};

export const uploadText = (text: string) => {
  const form = new FormData();
  form.append('text', text);
  return api.post('/upload-data', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data);
};

// ─── Snapshots ──────────────────────────────────────────────────────────────
export const saveSnapshot = (name: string, description = '') =>
  api.post('/snapshot', { name, description }).then(r => r.data);

export const listSnapshots = () =>
  api.get('/snapshots').then(r => r.data);

export const loadSnapshot = (name: string) =>
  api.post('/load-snapshot', { name }).then(r => r.data);
