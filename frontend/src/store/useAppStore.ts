import { create } from 'zustand';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface ModelConfig {
  d_model: number;
  n_heads: number;
  n_layers: number;
  context_len: number;
  dropout: number;
  vocab_size: number;
}

export interface TrainConfig {
  max_steps: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  warmup_steps: number;
  grad_clip: number;
}

export interface TokenizerInfo {
  vocab_size: number;
  num_merges: number;
}

export interface SnapshotInfo {
  name: string;
  description: string;
  timestamp: string;
  model_params: number;
}

export type AppPhase =
  | 'idle'
  | 'training'
  | 'generating'
  | 'rebuilding'
  | 'loading-snapshot';

// ─── Store ──────────────────────────────────────────────────────────────────

interface AppState {
  // Model configuration
  modelConfig: ModelConfig;
  trainConfig: TrainConfig;
  tokenizer: TokenizerInfo;
  modelParams: number | null;
  datasetName: string;

  // Runtime status
  phase: AppPhase;
  currentStep: number;
  currentLoss: number | null;
  isModelReady: boolean;

  // UI state
  configPanelOpen: boolean;
  theme: 'dark' | 'light';
  sharedText: string;       // text shared across pages for pipeline continuity
  lastError: string | null;

  // Snapshots
  snapshots: SnapshotInfo[];

  // Actions
  setModelConfig: (config: Partial<ModelConfig>) => void;
  setTrainConfig: (config: Partial<TrainConfig>) => void;
  setTokenizer: (info: TokenizerInfo) => void;
  setModelParams: (params: number) => void;
  setDatasetName: (name: string) => void;
  setPhase: (phase: AppPhase) => void;
  setTrainingProgress: (step: number, loss: number) => void;
  setModelReady: (ready: boolean) => void;
  toggleConfigPanel: () => void;
  setConfigPanelOpen: (open: boolean) => void;
  toggleTheme: () => void;
  setSharedText: (text: string) => void;
  setError: (error: string | null) => void;
  setSnapshots: (snapshots: SnapshotInfo[]) => void;

  // Bulk update from server config
  loadServerConfig: (data: any) => void;
}

export const useAppStore = create<AppState>((set) => ({
  // Defaults (match backend GPTConfig)
  modelConfig: {
    d_model: 64,
    n_heads: 4,
    n_layers: 4,
    context_len: 128,
    dropout: 0.1,
    vocab_size: 756,
  },
  trainConfig: {
    max_steps: 2000,
    batch_size: 32,
    learning_rate: 3e-4,
    weight_decay: 0.1,
    warmup_steps: 100,
    grad_clip: 1.0,
  },
  tokenizer: { vocab_size: 756, num_merges: 500 },
  modelParams: null,
  datasetName: 'tiny_shakespeare',

  phase: 'idle',
  currentStep: 0,
  currentLoss: null,
  isModelReady: false,

  configPanelOpen: false,
  theme: (localStorage.getItem('llm-theme') as 'dark' | 'light') || 'dark',
  sharedText: 'To be, or not to be, that is the question',
  lastError: null,

  snapshots: [],

  // Actions
  setModelConfig: (config) =>
    set((s) => ({ modelConfig: { ...s.modelConfig, ...config } })),

  setTrainConfig: (config) =>
    set((s) => ({ trainConfig: { ...s.trainConfig, ...config } })),

  setTokenizer: (info) => set({ tokenizer: info }),
  setModelParams: (params) => set({ modelParams: params }),
  setDatasetName: (name) => set({ datasetName: name }),
  setPhase: (phase) => set({ phase }),
  setTrainingProgress: (step, loss) =>
    set({ currentStep: step, currentLoss: loss }),
  setModelReady: (ready) => set({ isModelReady: ready }),
  toggleConfigPanel: () =>
    set((s) => ({ configPanelOpen: !s.configPanelOpen })),
  setConfigPanelOpen: (open) => set({ configPanelOpen: open }),
  toggleTheme: () =>
    set((s) => {
      const next = s.theme === 'dark' ? 'light' : 'dark';
      localStorage.setItem('llm-theme', next);
      document.documentElement.setAttribute('data-theme', next);
      return { theme: next };
    }),
  setSharedText: (text) => set({ sharedText: text }),
  setError: (error) => set({ lastError: error }),
  setSnapshots: (snapshots) => set({ snapshots }),

  loadServerConfig: (data) =>
    set((s) => {
      const updates: Partial<AppState> = { isModelReady: true };
      if (data.model) {
        updates.modelConfig = { ...s.modelConfig, ...data.model };
      }
      if (data.model_params) {
        updates.modelParams = data.model_params;
      }
      if (data.tokenizer) {
        updates.tokenizer = data.tokenizer;
      }
      if (data.training) {
        updates.trainConfig = { ...s.trainConfig, ...data.training };
      }
      if (data.dataset) {
        updates.datasetName = data.dataset;
      }
      return updates;
    }),
}));
