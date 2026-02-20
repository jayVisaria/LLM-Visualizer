import { useState, useCallback } from 'react';
import Plot from 'react-plotly.js';
import PageLayout from '../components/layout/PageLayout';
import MathBlock from '../components/shared/MathBlock';
import FormulaBlock from '../components/shared/FormulaBlock';
import CodeBlock from '../components/shared/CodeBlock';
import { getSelfAttention } from '../api/client';
import { useAppStore } from '../store/useAppStore';

export default function SelfAttentionPage() {
  const sharedText = useAppStore(s => s.sharedText);
  const setSharedText = useAppStore(s => s.setSharedText);
  const [text, setText] = useState(sharedText);
  const handleTextChange = (v: string) => { setText(v); setSharedText(v); };
  const [showMasking, setShowMasking] = useState(true);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getSelfAttention(text, showMasking);
      setResult(data);
    } catch (e: any) {
      console.error('Attention error:', e);
      setError(e?.response?.data?.detail || e?.message || 'API call failed');
    }
    setLoading(false);
  }, [text, showMasking]);

  return (
    <PageLayout
      title="Self-Attention"
      subtitle="The core mechanism that allows each token to 'look at' every other token and decide what's relevant — this is what makes transformers powerful."
      step={4}
      pipelineStep={4}
      learningContext={{
        prerequisites: [
          { label: 'Positional Encoding (Step 3)', path: '/positional-encoding' },
          { label: 'Embeddings (Step 2)', path: '/embeddings' },
        ],
        plainEnglish: "Self-attention lets each word 'ask a question' to every other word: 'How relevant are you to me?' The answer determines how much each word influences the others. In 'The cat sat on the mat', when processing 'sat', attention helps the model focus on 'cat' (the actor) more than 'the'.",
        realWorld: "This is the mechanism that made ChatGPT possible. The 'Attention Is All You Need' paper (2017) replaced recurrent networks with this. The causal mask you can toggle here is why GPT can only look backward — it prevents 'cheating' by looking at future tokens during training.",
      }}
      takeaway={{
        bullets: [
          'Attention computes: Q·K^T → scale by √d_k → mask (optional) → softmax → multiply by V',
          'The three heatmaps show each transformation step: raw scores → scaled → weighted',
          'Causal masking makes this a decoder (GPT-style) — disabling it makes it an encoder (BERT-style)',
          'Scaling by √d_k prevents dot products from growing too large and causing vanishing gradients through softmax',
        ],
        prevStep: { label: 'Positional Encoding', path: '/positional-encoding' },
        nextStep: { label: 'Multi-Head Attention', path: '/multi-head-attention', teaser: 'Run multiple attention computations in parallel for richer representations' },
        tryThis: "Toggle the causal mask on and off. With the mask, each token can only attend to previous tokens (GPT-style). Without it, every token sees every other token (BERT-style). Notice how the attention pattern changes dramatically.",
      }}
    >
      <div className="card">
        <h3>Scaled Dot-Product Attention</h3>
        <p className="card-subtitle">
          Self-attention computes a weighted sum of all token representations, where the weights
          are determined by the similarity between tokens. Each token generates three vectors:
          a Query (what am I looking for?), Key (what do I contain?), and Value (what do I output?).
        </p>
        <FormulaBlock
          title="Scaled Dot-Product Attention"
          formula={String.raw`\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V`}
          source="Vaswani et al. 2017, Eq. 1"
          sourceUrl="https://arxiv.org/abs/1706.03762"
          intuition="Each token asks 'who is relevant to me?' (Query × Keys), normalizes the answers into probabilities (softmax), then collects a weighted mix of Values."
          terms={[
            { symbol: 'Q', name: 'Query matrix', meaning: 'What each token is looking for — computed as Q = X·W_Q, shape (seq_len, d_k)' },
            { symbol: 'K', name: 'Key matrix', meaning: 'What each token advertises about itself — computed as K = X·W_K, shape (seq_len, d_k)' },
            { symbol: 'V', name: 'Value matrix', meaning: 'The actual content each token offers — computed as V = X·W_V, shape (seq_len, d_v)' },
            { symbol: 'QK^T', name: 'Score matrix', meaning: 'Dot product of every query with every key — measures pairwise similarity, shape (seq_len, seq_len)' },
            { symbol: '\\sqrt{d_k}', name: 'Scale factor', meaning: 'Prevents dot products from growing too large (which pushes softmax into regions with tiny gradients)' },
            { symbol: '\\text{softmax}', name: 'Softmax', meaning: 'Converts raw scores to probabilities — each row sums to 1, so each token distributes attention across all others' },
          ]}
          steps={[
            { label: 'Project input to Q, K, V', math: String.raw`Q = XW^Q,\; K = XW^K,\; V = XW^V`, text: 'Multiply input embeddings X by learned weight matrices to get three different views of the same tokens.' },
            { label: 'Compute similarity scores', math: String.raw`\text{scores} = QK^T`, text: 'Dot product of each query with all keys. High score = high similarity. Shape is (seq_len × seq_len).' },
            { label: 'Scale scores', math: String.raw`\text{scores} = \frac{QK^T}{\sqrt{d_k}}`, text: 'Without scaling, large d_k causes dot products to grow, pushing softmax to extreme (near 0 or 1) values with vanishing gradients.' },
            { label: 'Apply causal mask (for decoders)', math: String.raw`\text{scores}_{ij} = -\infty \text{ if } j > i`, text: 'For autoregressive models (GPT), mask future positions so token i can only attend to tokens 1…i.' },
            { label: 'Softmax to get attention weights', math: String.raw`\alpha = \text{softmax}(\text{scores})`, text: 'Each row becomes a probability distribution — the model\'s "attention pattern" for that query token.' },
            { label: 'Weighted sum of values', math: String.raw`\text{output} = \alpha \cdot V`, text: 'Each token\'s new representation is a weighted mix of all value vectors, weighted by attention.' },
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
            <label>Causal Masking</label>
            <select
              value={showMasking ? 'yes' : 'no'}
              onChange={(e) => setShowMasking(e.target.value === 'yes')}
            >
              <option value="yes">Enabled (GPT-style)</option>
              <option value="no">Disabled (BERT-style)</option>
            </select>
          </div>
          <button onClick={run} disabled={loading}>
            {loading ? 'Computing…' : 'Compute Attention'}
          </button>
        </div>
        {error && <div style={{ marginTop: 12, padding: '10px 14px', background: '#3d1f1f', border: '1px solid #f85149', borderRadius: 6, color: '#f85149', fontSize: 13 }}>Error: {error}</div>}
      </div>

      {result && (
        <>
          {/* Step-by-step math */}
          {result.math && (
            <div className="card">
              <h3>Computation Steps</h3>
              {Object.entries(result.math).map(([key, desc]) => (
                <div key={key} style={{ padding: '6px 0', fontSize: '14px' }}>
                  <span style={{ color: 'var(--accent)', fontWeight: 600 }}>{key}:</span>{' '}
                  {desc as string}
                </div>
              ))}
            </div>
          )}

          {/* Raw scores heatmap */}
          <div className="two-col">
            <div className="card">
              <h3>Raw Scores (Q·Kᵀ)</h3>
              <div className="plot-container">
                <Plot
                  data={[
                    {
                      z: result.raw_scores,
                      x: result.token_labels,
                      y: result.token_labels,
                      type: 'heatmap',
                      colorscale: 'RdBu',
                      reversescale: true,
                    },
                  ]}
                  layout={{
                    height: 400,
                    margin: { t: 10, b: 80, l: 80, r: 20 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#8b949e', size: 10 },
                    xaxis: { title: 'Key', tickangle: -45 },
                    yaxis: { title: 'Query', autorange: 'reversed' },
                  }}
                  config={{ responsive: true }}
                  style={{ width: '100%' }}
                />
              </div>
            </div>

            <div className="card">
              <h3>Scaled Scores (Q·Kᵀ / √d_k)</h3>
              <div className="plot-container">
                <Plot
                  data={[
                    {
                      z: result.scaled_scores,
                      x: result.token_labels,
                      y: result.token_labels,
                      type: 'heatmap',
                      colorscale: 'RdBu',
                      reversescale: true,
                    },
                  ]}
                  layout={{
                    height: 400,
                    margin: { t: 10, b: 80, l: 80, r: 20 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#8b949e', size: 10 },
                    xaxis: { title: 'Key', tickangle: -45 },
                    yaxis: { title: 'Query', autorange: 'reversed' },
                  }}
                  config={{ responsive: true }}
                  style={{ width: '100%' }}
                />
              </div>
            </div>
          </div>

          {/* Attention weights */}
          <div className="card">
            <h3>
              Attention Weights (after softmax{result.causal_masking ? ' + causal mask' : ''})
            </h3>
            <p className="card-subtitle">
              Each row shows how much attention a query token pays to each key token.
              {result.causal_masking && ' The upper triangle is zero — tokens cannot attend to future positions.'}
            </p>
            <div className="plot-container">
              <Plot
                data={[
                  {
                    z: result.attention_weights,
                    x: result.token_labels,
                    y: result.token_labels,
                    type: 'heatmap',
                    colorscale: [
                      [0, '#0d1117'],
                      [0.5, '#1f6feb'],
                      [1, '#58a6ff'],
                    ],
                    hovertemplate:
                      'Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>',
                  },
                ]}
                layout={{
                  height: 500,
                  margin: { t: 10, b: 100, l: 100, r: 60 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { color: '#8b949e', size: 11 },
                  xaxis: { title: 'Key (attending to)', tickangle: -45 },
                  yaxis: { title: 'Query (attending from)', autorange: 'reversed' },
                }}
                config={{ responsive: true }}
                style={{ width: '100%' }}
              />
            </div>
          </div>

          <MathBlock
            math={String.raw`\text{scale factor} = \sqrt{d_k} = \sqrt{${result.d_k}} \approx ${result.scale_factor.toFixed(2)}`}
          />
        </>
      )}

      <CodeBlock
        title="Self-Attention — Python Implementation"
        collapsible
        code={`import numpy as np
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K: (seq_len, d_k) — queries and keys
    V:    (seq_len, d_v) — values
    mask: (seq_len, seq_len) — 1=keep, 0=mask

    Returns: (seq_len, d_v) — context vectors
    """
    d_k = Q.shape[-1]

    # Step 1: Dot product of queries and keys
    scores = Q @ K.T  # (seq, seq)

    # Step 2: Scale by √d_k
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply causal mask (prevent looking ahead)
    if mask is not None:
        scores = np.where(mask == 1, scores, -1e9)

    # Step 4: Softmax → attention probabilities
    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    weights = softmax(scores)  # each row sums to 1

    # Step 5: Weighted sum of values
    output = weights @ V  # (seq, d_v)

    return output, weights

# Causal mask for autoregressive generation:
seq_len = 6
causal_mask = np.tril(np.ones((seq_len, seq_len)))
# [[1, 0, 0, 0, 0, 0],
#  [1, 1, 0, 0, 0, 0],
#  [1, 1, 1, 0, 0, 0],  ← token 3 can see tokens 1-3, not 4-6
#  [1, 1, 1, 1, 0, 0],
#  [1, 1, 1, 1, 1, 0],
#  [1, 1, 1, 1, 1, 1]]`}
      />
    </PageLayout>
  );
}
