import { useState, useCallback } from 'react';
import Plot from 'react-plotly.js';
import PageLayout from '../components/layout/PageLayout';
import MathBlock from '../components/shared/MathBlock';
import FormulaBlock from '../components/shared/FormulaBlock';
import CodeBlock from '../components/shared/CodeBlock';
import { getMultiHeadAttention } from '../api/client';
import { useAppStore } from '../store/useAppStore';

export default function MultiHeadAttentionPage() {
  const sharedText = useAppStore(s => s.sharedText);
  const setSharedText = useAppStore(s => s.setSharedText);
  const modelConfig = useAppStore(s => s.modelConfig);
  const [text, setText] = useState(sharedText);
  const handleTextChange = (v: string) => { setText(v); setSharedText(v); };
  const [nHeads, setNHeads] = useState(modelConfig.n_heads);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getMultiHeadAttention(text, nHeads);
      setResult(data);
    } catch (e: any) {
      console.error('Multi-head error:', e);
      setError(e?.response?.data?.detail || e?.message || 'API call failed');
    }
    setLoading(false);
  }, [text, nHeads]);

  return (
    <PageLayout
      title="Multi-Head Attention"
      subtitle="Run multiple attention heads in parallel — each head can learn different patterns (syntax, semantics, position), then combine their outputs."
      step={5}
      pipelineStep={5}
      learningContext={{
        prerequisites: [{ label: 'Self-Attention (Step 4)', path: '/self-attention' }],
        plainEnglish: "One attention head can only focus on one type of relationship at a time. Multiple heads let the model attend to different things simultaneously — one head might track grammar, another might track meaning, another might track which 'it' refers to.",
        realWorld: "GPT-3 uses 96 attention heads. Researchers have found that different heads specialize: some handle syntax, some handle rare words, some track long-range dependencies. Head pruning (removing unused heads after training) is a popular optimization technique.",
      }}
      takeaway={{
        bullets: [
          'Multi-head attention: split → parallel attention on each head → concatenate → project',
          'Each head operates on a smaller slice: d_k = d_model / n_heads dimensions',
          'More heads = richer attention patterns, but each head has less capacity (fewer dimensions)',
          'The output projection W^O combines all heads back to d_model dimensions',
        ],
        prevStep: { label: 'Self-Attention', path: '/self-attention' },
        nextStep: { label: 'Transformer Block', path: '/transformer-block', teaser: 'Wrap attention in a complete block with feed-forward network and normalization' },
        tryThis: "Change the number of heads from 1 to 8. With 1 head, you see a single attention pattern. With 4 heads (d_model=64 → d_k=16 each), notice how different heads focus on different word relationships.",
      }}
    >
      <div className="card">
        <h3>Why Multiple Heads?</h3>
        <p className="card-subtitle">
          A single attention head can only focus on one type of relationship. Multiple heads
          allow the model to attend to different aspects simultaneously.
        </p>
        <FormulaBlock
          title="Multi-Head Attention"
          formula={String.raw`\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O \quad \text{where } \text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)`}
          source="Vaswani et al. 2017, §3.2.2"
          sourceUrl="https://arxiv.org/abs/1706.03762"
          intuition="Instead of one attention pattern, run multiple attention heads in parallel — each head can focus on a different relationship (syntax, semantics, position), then combine their insights."
          terms={[
            { symbol: 'h', name: 'Number of heads', meaning: 'How many parallel attention computations to run (e.g. 8 in the original paper)' },
            { symbol: '\\text{head}_i', name: 'Individual head', meaning: 'The output of scaled dot-product attention for head i, operating on a d_k-dim slice' },
            { symbol: 'W_i^Q, W_i^K, W_i^V', name: 'Per-head projections', meaning: 'Learned matrices that project Q, K, V from d_model to d_k = d_model/h dimensions for head i' },
            { symbol: 'W^O', name: 'Output projection', meaning: 'Learned matrix (h·d_v × d_model) that combines concatenated heads back to d_model dimensions' },
            { symbol: '\\text{Concat}', name: 'Concatenation', meaning: 'Stack all head outputs side by side: h heads × d_k dims = d_model dims' },
            { symbol: 'd_k', name: 'Head dimension', meaning: 'd_model / h — each head operates on this many dimensions (64 when d_model=512, h=8)' },
          ]}
          steps={[
            { label: 'Project to per-head Q, K, V', math: String.raw`Q_i = QW_i^Q, \; K_i = KW_i^K, \; V_i = VW_i^V`, text: 'Each head gets its own view of the queries, keys, and values — different learned projections into a smaller d_k-dimensional space.' },
            { label: 'Run attention in parallel', math: String.raw`\text{head}_i = \text{Attention}(Q_i, K_i, V_i)`, text: 'Each head independently computes scaled dot-product attention on its own slice. Different heads learn different patterns.' },
            { label: 'Concatenate all heads', math: String.raw`\text{multi} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)`, text: 'Stack all head outputs along the last dimension. Result shape: (seq_len, h × d_k) = (seq_len, d_model).' },
            { label: 'Final linear projection', math: String.raw`\text{output} = \text{multi} \cdot W^O`, text: 'Mix the concatenated head outputs through a learned projection to combine the different attention patterns.' },
          ]}
        />
      </div>

      <div className="card">
        <h3>Try It</h3>
        <div className="controls">
          <div className="control-group" style={{ flex: 1 }}>
            <label>Input Text</label>
            <input
              type="text"
              value={text}
              onChange={(e) => handleTextChange(e.target.value)}
              style={{ width: '100%' }}
            />
          </div>
          <div className="control-group">
            <label>Number of Heads</label>
            <select
              value={nHeads}
              onChange={(e) => setNHeads(Number(e.target.value))}
            >
              <option value={1}>1 head</option>
              <option value={2}>2 heads</option>
              <option value={4}>4 heads</option>
              <option value={8}>8 heads</option>
            </select>
          </div>
          <button onClick={run} disabled={loading}>
            {loading ? 'Computing…' : 'Compute Multi-Head'}
          </button>
        </div>
        {error && <div style={{ marginTop: 12, padding: '10px 14px', background: '#3d1f1f', border: '1px solid #f85149', borderRadius: 6, color: '#f85149', fontSize: 13 }}>Error: {error}</div>}
      </div>

      {result && (
        <>
          {/* Stats */}
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{result.n_heads}</div>
              <div className="stat-label">Heads</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{result.d_k}</div>
              <div className="stat-label">d_k per head</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{result.d_model}</div>
              <div className="stat-label">d_model</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {result.concatenated_shape[0]}×{result.concatenated_shape[1]}
              </div>
              <div className="stat-label">Output Shape</div>
            </div>
          </div>

          {/* Per-head attention heatmaps */}
          <div className="card">
            <h3>Attention Patterns by Head</h3>
            <p className="card-subtitle">
              Each head learns different attention patterns. Notice how they focus on
              different relationships between tokens.
            </p>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${Math.min(result.n_heads, 2)}, 1fr)`,
                gap: '16px',
              }}
            >
              {result.head_attention_weights.map((weights: number[][], h: number) => (
                <div key={h} className="plot-container">
                  <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '8px', color: 'var(--accent)' }}>
                    Head {h + 1}
                  </div>
                  <Plot
                    data={[
                      {
                        z: weights,
                        x: result.token_labels,
                        y: result.token_labels,
                        type: 'heatmap',
                        colorscale: [
                          [0, '#0d1117'],
                          [0.5, '#1f6feb'],
                          [1, '#58a6ff'],
                        ],
                        showscale: false,
                        hovertemplate:
                          'Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>',
                      },
                    ]}
                    layout={{
                      height: 300,
                      margin: { t: 5, b: 60, l: 60, r: 5 },
                      paper_bgcolor: 'transparent',
                      plot_bgcolor: 'transparent',
                      font: { color: '#8b949e', size: 9 },
                      xaxis: { tickangle: -45 },
                      yaxis: { autorange: 'reversed' },
                    }}
                    config={{ responsive: true }}
                    style={{ width: '100%' }}
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Explanation */}
          <div className="card">
            <h3>How Outputs are Combined</h3>
            <div className="explanation">{result.explanation}</div>
            <MathBlock
              math={String.raw`\text{Output} = \text{Concat}\!\left(\underbrace{\text{head}_1}_{${result.d_k}\text{d}}, \ldots, \underbrace{\text{head}_${result.n_heads}}_{${result.d_k}\text{d}}\right) W^O \in \mathbb{R}^{${result.d_model}}`}
            />
          </div>
        </>
      )}

      <CodeBlock
        title="Multi-Head Attention — Python Implementation"
        collapsible
        code={`import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4, context_len=128):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Combined Q, K, V projection (more efficient)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # Causal mask
        mask = torch.tril(torch.ones(context_len, context_len))
        self.register_buffer("mask", mask.view(1, 1, context_len, context_len))

    def forward(self, x):
        B, T, C = x.shape

        # Project to Q, K, V
        q, k, v = self.qkv(x).split(C, dim=2)

        # Reshape to (B, n_heads, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Attention: (B, n_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)

        # Weighted sum → (B, n_heads, T, d_k)
        out = weights @ v

        # Concatenate heads → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)`}
      />
    </PageLayout>
  );
}
