import { useState, useCallback } from 'react';
import Plot from 'react-plotly.js';
import PageLayout from '../components/layout/PageLayout';
import FormulaBlock from '../components/shared/FormulaBlock';
import CodeBlock from '../components/shared/CodeBlock';
import { getTransformerBlock } from '../api/client';
import { useAppStore } from '../store/useAppStore';

export default function TransformerBlockPage() {
  const sharedText = useAppStore(s => s.sharedText);
  const setSharedText = useAppStore(s => s.setSharedText);
  const [text, setText] = useState(sharedText);
  const handleTextChange = (v: string) => { setText(v); setSharedText(v); };
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getTransformerBlock(text);
      setResult(data);
    } catch (e: any) {
      console.error('Transformer block error:', e);
      setError(e?.response?.data?.detail || e?.message || 'API call failed');
    }
    setLoading(false);
  }, [text]);

  return (
    <PageLayout
      title="Transformer Block"
      subtitle="The building block of the transformer — combines multi-head attention, feed-forward network, layer normalization, and residual connections."
      step={6}
      pipelineStep={6}
      learningContext={{
        prerequisites: [
          { label: 'Multi-Head Attention (Step 5)', path: '/multi-head-attention' },
          { label: 'Embeddings (Step 2)', path: '/embeddings' },
        ],
        plainEnglish: "A transformer block is a sandwich: attention (so tokens can talk to each other) + feed-forward network (so each token can 'think' independently) + skip connections (so information and gradients can flow freely) + normalization (so numbers stay in a reasonable range). Stack 4 of these and you have our model.",
        realWorld: "Every modern LLM is just N of these blocks stacked. GPT-3 has 96 blocks, LLaMA-70B has 80. The pre-norm pattern shown here (LayerNorm before attention) is now standard — the original 2017 paper used post-norm, which was harder to train at scale.",
      }}
      takeaway={{
        bullets: [
          'Block formula: x + Attention(LN(x)), then x + FFN(LN(x))',
          'Residual connections (x + sublayer(x)) are critical — without them, gradients vanish and deep models cannot train',
          'LayerNorm stabilizes activations, preventing them from growing or shrinking across layers',
          'The FFN expands to 4× the model dimension (d_model → 4·d_model → d_model) adding non-linear capacity',
        ],
        prevStep: { label: 'Multi-Head Attention', path: '/multi-head-attention' },
        nextStep: { label: 'Full Model', path: '/full-model', teaser: 'Stack these blocks into a complete GPT architecture' },
      }}
    >
      <div className="card">
        <h3>Block Architecture (Pre-Norm)</h3>
        <p className="card-subtitle">
          Each transformer block has two sub-layers, each wrapped with a residual connection
          and preceded by Layer Normalization.
        </p>
        <FormulaBlock
          title="Pre-Norm Transformer Block"
          formula={String.raw`x = x + \text{MultiHeadAttn}(\text{LayerNorm}(x)), \quad x = x + \text{FFN}(\text{LayerNorm}(x))`}
          source="Vaswani et al. 2017, §3.1 (pre-norm variant)"
          sourceUrl="https://arxiv.org/abs/1706.03762"
          intuition="A transformer block is a sandwich: normalize → attend (so tokens talk to each other) → add back the original (residual) → normalize → feed-forward (so each token thinks independently) → add back again."
          terms={[
            { symbol: 'x', name: 'Residual stream', meaning: 'The running representation passed between layers — residual connections add to it rather than replace it' },
            { symbol: '\\text{LayerNorm}', name: 'Layer normalization', meaning: 'Normalizes each token\'s feature vector to zero mean and unit variance before each sub-layer' },
            { symbol: '\\text{MultiHeadAttn}', name: 'Multi-head attention', meaning: 'Allows each token to attend to all other tokens via multiple parallel attention heads' },
            { symbol: '\\text{FFN}', name: 'Feed-forward network', meaning: 'Two-layer MLP applied independently to each token — adds non-linear transformation capacity' },
            { symbol: 'x + \\text{sublayer}(x)', name: 'Residual connection', meaning: 'Adds the sub-layer output to the input — creates a "gradient highway" so deep networks can train' },
          ]}
          steps={[
            { label: 'Normalize input', math: String.raw`\hat{x} = \text{LayerNorm}(x)`, text: 'Stabilize activations before attention. Pre-norm (shown here, GPT-2 style) normalizes before the sub-layer; post-norm (original paper) normalizes after.' },
            { label: 'Multi-head self-attention', math: String.raw`\text{attn\_out} = \text{MultiHeadAttn}(\hat{x})`, text: 'Tokens communicate — each token computes attention over all other tokens (see Step 5).' },
            { label: 'First residual add', math: String.raw`x = x + \text{attn\_out}`, text: 'Add attention output back to the original x. This residual connection ensures the model can always "pass through" information unchanged.' },
            { label: 'Normalize again', math: String.raw`\hat{h} = \text{LayerNorm}(x)`, text: 'Normalize again before the feed-forward sub-layer.' },
            { label: 'Feed-forward network', math: String.raw`\text{ffn\_out} = \text{FFN}(\hat{h})`, text: 'Each token is independently processed through a 2-layer MLP with expansion ratio 4×.' },
            { label: 'Second residual add', math: String.raw`x = x + \text{ffn\_out}`, text: 'Add FFN output back to x. The block output is the updated residual stream, ready for the next block.' },
          ]}
        />
      </div>

      {/* LayerNorm */}
      <div className="card">
        <h3>Layer Normalization</h3>
        <p className="card-subtitle">
          Normalizes each token's features to zero mean and unit variance.
        </p>
        <FormulaBlock
          title="Layer Normalization"
          formula={String.raw`\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta`}
          source="Ba et al. 2016 (arXiv:1607.06450)"
          sourceUrl="https://arxiv.org/abs/1607.06450"
          intuition="Centers and scales each token's feature vector so numbers stay in a stable range — like converting test scores to z-scores before comparing."
          terms={[
            { symbol: 'x', name: 'Input vector', meaning: 'One token\'s feature vector of size d_model' },
            { symbol: '\\mu', name: 'Mean', meaning: 'Mean of x across the d_model dimension — computed per token' },
            { symbol: '\\sigma^2', name: 'Variance', meaning: 'Variance of x across the d_model dimension — computed per token' },
            { symbol: '\\epsilon', name: 'Epsilon', meaning: 'Small constant (typically 10⁻⁵) to prevent division by zero' },
            { symbol: '\\gamma', name: 'Scale (gain)', meaning: 'Learned parameter vector (size d_model) — lets the model undo the normalization if needed' },
            { symbol: '\\beta', name: 'Shift (bias)', meaning: 'Learned parameter vector (size d_model) — lets the model shift the output' },
          ]}
        />
      </div>

      {/* Feed-Forward Network */}
      <div className="card">
        <h3>Feed-Forward Network (FFN)</h3>
        <p className="card-subtitle">
          A two-layer MLP applied independently to each position.
        </p>
        <FormulaBlock
          title="Position-wise Feed-Forward Network"
          formula={String.raw`\text{FFN}(x) = \text{GELU}(x W_1 + b_1)\, W_2 + b_2`}
          source="Vaswani et al. 2017, Eq. 2 (ReLU → GELU in GPT-2)"
          sourceUrl="https://arxiv.org/abs/1706.03762"
          intuition="A two-layer mini neural network applied to each token independently — expands to 4× the model dimension (to have room to learn complex patterns), then squeezes back down."
          terms={[
            { symbol: 'x', name: 'Input', meaning: 'One token\'s d_model-dimensional vector (after attention + residual)' },
            { symbol: 'W_1', name: 'Expand weights', meaning: 'Weight matrix (d_model × 4·d_model) — projects to a larger intermediate space' },
            { symbol: 'W_2', name: 'Compress weights', meaning: 'Weight matrix (4·d_model × d_model) — projects back to d_model dimensions' },
            { symbol: '\\text{GELU}', name: 'Activation function', meaning: 'Gaussian Error Linear Unit — smooth version of ReLU used by GPT-2/3. The original paper used ReLU.' },
            { symbol: '4 \\times d_{\\text{model}}', name: 'Expansion factor', meaning: 'The hidden layer is 4× larger than d_model (e.g. 512 → 2048) to provide more capacity for learning non-linear transformations' },
          ]}
          steps={[
            { label: 'Expand', math: String.raw`h = xW_1 + b_1`, text: 'Project from d_model to 4×d_model dimensions. This gives the network more room to compute.' },
            { label: 'Apply non-linearity', math: String.raw`h = \text{GELU}(h)`, text: 'Introduce non-linearity. Without this, stacking linear layers would be equivalent to a single linear layer.' },
            { label: 'Compress back', math: String.raw`\text{out} = hW_2 + b_2`, text: 'Project back to d_model dimensions. The result is added to the residual stream.' },
          ]}
        />
      </div>

      {/* Interactive */}
      <div className="card">
        <h3>Trace Through a Block</h3>
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
          <button onClick={run} disabled={loading}>
            {loading ? 'Tracing…' : 'Trace Block'}
          </button>
        </div>
        {error && <div style={{ marginTop: 12, padding: '10px 14px', background: '#3d1f1f', border: '1px solid #f85149', borderRadius: 6, color: '#f85149', fontSize: 13 }}>Error: {error}</div>}
      </div>

      {result?.pytorch_trace && (
        <>
          {/* Activation stats through the block */}
          <div className="card">
            <h3>Activation Statistics Through the Block</h3>
            <p className="card-subtitle">
              How tensor statistics change at each sublayer.
            </p>
            <div className="plot-container">
              <Plot
                data={[
                  {
                    x: Object.keys(result.pytorch_trace),
                    y: Object.values(result.pytorch_trace).map((s: any) => s.mean),
                    name: 'Mean',
                    type: 'bar',
                    marker: { color: '#58a6ff' },
                  },
                  {
                    x: Object.keys(result.pytorch_trace),
                    y: Object.values(result.pytorch_trace).map((s: any) => s.std),
                    name: 'Std Dev',
                    type: 'bar',
                    marker: { color: '#3fb950' },
                  },
                ]}
                layout={{
                  height: 350,
                  barmode: 'group',
                  margin: { t: 20, b: 100, l: 60, r: 20 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { color: '#8b949e' },
                  xaxis: { tickangle: -30 },
                  yaxis: { title: 'Value', gridcolor: '#21262d' },
                  legend: { x: 0, y: 1, font: { color: '#8b949e' } },
                }}
                config={{ responsive: true }}
                style={{ width: '100%' }}
              />
            </div>
          </div>

          {/* Distribution histograms */}
          {result.distributions && (
            <div className="card">
              <h3>Activation Distributions</h3>
              <p className="card-subtitle">
                How LayerNorm and residual connections affect the distribution of values.
              </p>
              <div className="plot-container">
                <Plot
                  data={[
                    {
                      x: result.distributions.input,
                      name: 'Input',
                      type: 'histogram',
                      opacity: 0.6,
                      marker: { color: '#f85149' },
                      nbinsx: 40,
                    },
                    {
                      x: result.distributions.after_ln1,
                      name: 'After LayerNorm',
                      type: 'histogram',
                      opacity: 0.6,
                      marker: { color: '#58a6ff' },
                      nbinsx: 40,
                    },
                    {
                      x: result.distributions.after_residual_2,
                      name: 'After Full Block',
                      type: 'histogram',
                      opacity: 0.6,
                      marker: { color: '#3fb950' },
                      nbinsx: 40,
                    },
                  ]}
                  layout={{
                    height: 350,
                    barmode: 'overlay',
                    margin: { t: 20, b: 40, l: 60, r: 20 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#8b949e' },
                    xaxis: { title: 'Activation Value', gridcolor: '#21262d' },
                    yaxis: { title: 'Count', gridcolor: '#21262d' },
                    legend: { x: 0.7, y: 1, font: { color: '#8b949e' } },
                  }}
                  config={{ responsive: true }}
                  style={{ width: '100%' }}
                />
              </div>
            </div>
          )}
        </>
      )}

      {/* LayerNorm NumPy demo */}
      {result?.layernorm_demo && (
        <div className="card">
          <h3>LayerNorm Step-by-Step (NumPy)</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{result.layernorm_demo.input_stats.global_mean.toFixed(3)}</div>
              <div className="stat-label">Input Mean</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{result.layernorm_demo.input_stats.global_std.toFixed(3)}</div>
              <div className="stat-label">Input Std</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{result.layernorm_demo.output_stats.global_mean.toFixed(3)}</div>
              <div className="stat-label">Output Mean</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{result.layernorm_demo.output_stats.global_std.toFixed(3)}</div>
              <div className="stat-label">Output Std</div>
            </div>
          </div>
          <div className="explanation">{result.layernorm_demo.explanation}</div>
        </div>
      )}

      <CodeBlock
        title="Transformer Block — Python Implementation"
        collapsible
        code={`import torch.nn as nn

class TransformerBlock(nn.Module):
    """Pre-norm transformer block (GPT-2 style)."""

    def __init__(self, d_model=64, n_heads=4, context_len=128, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, context_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Sub-layer 1: Attention with residual
        x = x + self.attn(self.ln1(x))
        # Sub-layer 2: FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x

# LayerNorm in NumPy:
def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mu) / np.sqrt(var + eps)
    return gamma * x_norm + beta`}
      />
    </PageLayout>
  );
}
