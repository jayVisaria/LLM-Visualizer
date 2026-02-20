import { useState, useEffect, useRef } from 'react';
import { NavLink } from 'react-router-dom';
import Plot from 'react-plotly.js';
import PageLayout from '../components/layout/PageLayout';
import { useAppStore } from '../store/useAppStore';
import { getConfig, getStatus, getLossHistory, getVocabStats, getHealth } from '../api/client';

const ROADMAP = [
  { num: 1, label: 'Tokenization', path: '/tokenization', icon: '✂️',
    objective: 'Understand how text becomes numbers using Byte Pair Encoding',
    difficulty: 'beginner' as const, time: '~8 min' },
  { num: 2, label: 'Embeddings', path: '/embeddings', icon: '🔢',
    objective: 'Learn how token IDs become meaningful vector representations',
    difficulty: 'beginner' as const, time: '~8 min' },
  { num: 3, label: 'Positional Encoding', path: '/positional-encoding', icon: '📍',
    objective: 'See why word order matters and how transformers encode it',
    difficulty: 'beginner' as const, time: '~7 min' },
  { num: 4, label: 'Self-Attention', path: '/self-attention', icon: '🔍',
    objective: 'Discover how tokens learn to "look at" each other — the core of transformers',
    difficulty: 'intermediate' as const, time: '~12 min' },
  { num: 5, label: 'Multi-Head Attention', path: '/multi-head-attention', icon: '🧩',
    objective: 'Understand why parallel attention heads capture richer language patterns',
    difficulty: 'intermediate' as const, time: '~10 min' },
  { num: 6, label: 'Transformer Block', path: '/transformer-block', icon: '🧱',
    objective: 'Assemble the core building block: attention + feed-forward + normalization',
    difficulty: 'intermediate' as const, time: '~10 min' },
  { num: 7, label: 'Full Model', path: '/full-model', icon: '🤖',
    objective: 'Stack blocks into a complete GPT architecture and inspect parameters',
    difficulty: 'intermediate' as const, time: '~8 min' },
  { num: 8, label: 'Training', path: '/training', icon: '📈',
    objective: 'Train the model live and watch it learn language patterns from Shakespeare',
    difficulty: 'advanced' as const, time: '~15 min' },
  { num: 9, label: 'Backpropagation', path: '/backpropagation', icon: '🔄',
    objective: 'Understand how gradient descent updates every weight in the model',
    difficulty: 'advanced' as const, time: '~12 min' },
  { num: 10, label: 'Generation', path: '/generation', icon: '✨',
    objective: 'Generate text with different sampling strategies and see the model in action',
    difficulty: 'advanced' as const, time: '~10 min' },
  { num: 11, label: 'Modern Architectures', path: '/modern-architectures', icon: '⚡',
    objective: 'Explore RMSNorm, RoPE, GQA, MLA, SwiGLU and MoE — the 6 upgrades powering LLaMA, DeepSeek and Mistral',
    difficulty: 'advanced' as const, time: '~15 min' },
];

export default function DashboardPage() {
  const { modelConfig, trainConfig, datasetName,
          currentStep, currentLoss, tokenizer, loadServerConfig, setModelReady } = useAppStore();

  const [lossHistory, setLossHistory] = useState<any[]>([]);
  const [vocabStats, setVocabStats] = useState<any>(null);
  const [_backendUp, setBackendUp] = useState<boolean | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      getHealth().catch(() => null),
      getConfig().catch(() => null),
      getLossHistory().catch(() => null),
      getVocabStats().catch(() => null),
    ]).then(([health, config, loss, vocab]) => {
      if (cancelled) return;
      setBackendUp(!!health);
      if (config) { loadServerConfig(config); setModelReady(true); }
      if (loss?.loss_history) setLossHistory(loss.loss_history);
      if (vocab) setVocabStats(vocab);
    });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const poll = () => {
      getStatus()
        .then(() => setBackendUp(true))
        .catch(() => setBackendUp(false));
      getLossHistory()
        .then(d => { if (d?.loss_history) setLossHistory(d.loss_history); })
        .catch(() => {});
    };
    pollRef.current = window.setInterval(poll, 5000);
    return () => { if (pollRef.current) window.clearInterval(pollRef.current); };
  }, []);

  const darkLayout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(22,27,34,0.8)',
    font: { color: '#8b949e', size: 11 },
    margin: { t: 30, r: 20, b: 40, l: 50 },
  };

  return (
    <PageLayout
      title="Build an LLM from Scratch"
      subtitle="An interactive, end-to-end guide to understanding how Large Language Models work — from raw text to generated output."
      step={0}
    >
      {/* ── What is an LLM? ── */}
      <div className="card">
        <h3>🤖 What is a Large Language Model?</h3>
        <p className="roadmap-intro">
          A Large Language Model (LLM) like ChatGPT is, at its core, a <strong>next-token prediction machine</strong>.
          Given a sequence of text, it predicts the most probable next word (or subword). It does this by learning
          statistical patterns from massive amounts of text data.
        </p>
        <p className="roadmap-intro" style={{ marginTop: 12 }}>
          Under the hood, an LLM is built from a <strong>transformer architecture</strong> — a specific type of neural
          network invented in 2017. The transformer's key innovation is <strong>self-attention</strong>: a mechanism
          that lets every word in a sentence "look at" every other word to understand context.
        </p>
        <p className="roadmap-intro" style={{ marginTop: 12 }}>
          In this interactive guide, you'll build every component of a working LLM from scratch. You'll see the real
          math, run real code, and train a real model — not just read about it. The model is small (~550K parameters
          vs. GPT-3's 175 billion), but the <strong>architecture is identical</strong>. Every concept here scales
          directly to the largest models.
        </p>
      </div>

      {/* ── How to Use This App ── */}
      <div className="card">
        <h3>🧭 How This App Works</h3>
        <p className="roadmap-intro">
          This is not a passive tutorial — it's a <strong>live laboratory</strong>. A real PyTorch model runs on
          the backend. When you type text on any page, it flows through the actual model pipeline.
          When you train, real gradients update real weights. When you generate, the model produces
          real predictions.
        </p>
        <div className="quick-start-box">
          <h4>🚀 Quick Start</h4>
          <p>
            Work through <strong>Steps 1–10 in order</strong>. Each page explains the concept, shows the math,
            lets you try it interactively, and provides the Python implementation.
            Use the <strong>⚙ Config Panel</strong> (top-right button) to change model architecture,
            upload custom data, or save snapshots of your trained model.
          </p>
          <p style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid rgba(255,255,255,0.08)', color: 'var(--text-muted)', fontSize: 13 }}>
            ⚡ After completing the 10 steps, explore <strong>Modern Architectures</strong> in the sidebar to see the
            exact formulas (RoPE, RMSNorm, SwiGLU, GQA, MLA, MoE) powering LLaMA 3, DeepSeek-V3, and Claude today.
          </p>
        </div>
      </div>

      {/* ── Learning Roadmap ── */}
      <div className="card">
        <h3>🗺️ Learning Roadmap</h3>
        <p className="card-subtitle">
          11 steps from raw text to frontier architectures. Each step builds on the previous one.
          Click any step to begin.
        </p>
        <div className="roadmap-timeline">
          {ROADMAP.map((step) => (
            <NavLink key={step.num} to={step.path} className="roadmap-card">
              <div className={`roadmap-card-dot ${step.difficulty}`} />
              <div className="roadmap-card-num">{step.num}</div>
              <div style={{ fontSize: 20, flexShrink: 0 }}>{step.icon}</div>
              <div className="roadmap-card-body">
                <div className="roadmap-card-title">{step.label}</div>
                <div className="roadmap-card-objective">{step.objective}</div>
                <div className="roadmap-card-meta">
                  <span className={`difficulty-badge ${step.difficulty}`}>{step.difficulty}</span>
                  <span className="roadmap-time">{step.time}</span>
                </div>
              </div>
            </NavLink>
          ))}
        </div>
      </div>

      {/* ── Model & Data Overview ── */}
      <div className="two-col">
        <div className="card">
          <h3>🏗️ Model Architecture</h3>
          <table className="merge-table" style={{ fontSize: 13 }}>
            <tbody>
              <tr><td style={{ color: 'var(--text-muted)' }}>Embedding Dim</td><td>{modelConfig.d_model}</td></tr>
              <tr><td style={{ color: 'var(--text-muted)' }}>Attention Heads</td><td>{modelConfig.n_heads}</td></tr>
              <tr><td style={{ color: 'var(--text-muted)' }}>Transformer Layers</td><td>{modelConfig.n_layers}</td></tr>
              <tr><td style={{ color: 'var(--text-muted)' }}>Context Length</td><td>{modelConfig.context_len}</td></tr>
              <tr><td style={{ color: 'var(--text-muted)' }}>Vocab Size</td><td>{modelConfig.vocab_size}</td></tr>
              <tr><td style={{ color: 'var(--text-muted)' }}>Dropout</td><td>{modelConfig.dropout}</td></tr>
              <tr><td style={{ color: 'var(--text-muted)' }}>Head Dim</td><td>{modelConfig.d_model / modelConfig.n_heads}</td></tr>
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3>📊 Training Status</h3>
          <table className="merge-table" style={{ fontSize: 13 }}>
            <tbody>
              <tr><td style={{ color: 'var(--text-muted)' }}>Dataset</td><td>{datasetName}</td></tr>
              <tr><td style={{ color: 'var(--text-muted)' }}>Max Steps</td><td>{trainConfig.max_steps}</td></tr>
              <tr><td style={{ color: 'var(--text-muted)' }}>Batch Size</td><td>{trainConfig.batch_size}</td></tr>
              <tr><td style={{ color: 'var(--text-muted)' }}>Learning Rate</td><td>{trainConfig.learning_rate}</td></tr>
              <tr><td style={{ color: 'var(--text-muted)' }}>Current Step</td><td>{currentStep || '—'}</td></tr>
              <tr><td style={{ color: 'var(--text-muted)' }}>Current Loss</td><td>{currentLoss != null ? currentLoss.toFixed(4) : '—'}</td></tr>
              <tr><td style={{ color: 'var(--text-muted)' }}>Tokenizer Merges</td><td>{tokenizer.num_merges}</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Loss Chart ── */}
      {lossHistory.length > 0 && (
        <div className="card">
          <h3>📉 Training Loss History</h3>
          <p className="card-subtitle">
            Loss measures how wrong the model's predictions are. Lower is better.
            A decreasing curve means the model is learning.
          </p>
          <div className="plot-container">
            <Plot
              data={[{
                x: lossHistory.map((p: any) => p.step),
                y: lossHistory.map((p: any) => p.loss),
                type: 'scatter',
                mode: 'lines',
                line: { color: '#58a6ff', width: 2 },
                name: 'Loss',
              }]}
              layout={{
                ...darkLayout,
                xaxis: { title: 'Step', color: '#8b949e', gridcolor: '#21262d' },
                yaxis: { title: 'Loss', color: '#8b949e', gridcolor: '#21262d' },
                height: 250,
                showlegend: false,
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      )}

      {/* ── Vocab Stats ── */}
      {vocabStats && (
        <div className="card">
          <h3>🔤 Vocabulary Overview</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{vocabStats.vocab_size}</div>
              <div className="stat-label">Vocab Size</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{vocabStats.num_merges}</div>
              <div className="stat-label">BPE Merges</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{vocabStats.avg_token_length?.toFixed(1) || '—'}</div>
              <div className="stat-label">Avg Token Len</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{vocabStats.max_token_length || '—'}</div>
              <div className="stat-label">Max Token Len</div>
            </div>
          </div>
        </div>
      )}

    </PageLayout>
  );
}
