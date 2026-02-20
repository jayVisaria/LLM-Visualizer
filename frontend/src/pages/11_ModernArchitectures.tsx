import PageLayout from '../components/layout/PageLayout';
import FormulaBlock from '../components/shared/FormulaBlock';
import CodeBlock from '../components/shared/CodeBlock';

// Small badge row for "used by" models
function ModelBadges({ models }: { models: string[] }) {
  return (
    <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      <span style={{ fontSize: 11, color: 'var(--text-muted)', alignSelf: 'center' }}>Used by:</span>
      {models.map(m => (
        <span
          key={m}
          style={{
            background: 'rgba(88,166,255,0.1)',
            border: '1px solid rgba(88,166,255,0.25)',
            borderRadius: 4,
            padding: '2px 8px',
            fontSize: 11,
            color: '#58a6ff',
            fontWeight: 500,
          }}
        >
          {m}
        </span>
      ))}
    </div>
  );
}

// Row in the adoption table
const ADOPTION = [
  { model: 'LLaMA 3 (Meta)',    rmsnorm: true,  rope: true,  gqa: true,  mla: false, swiglu: true,  moe: false },
  { model: 'DeepSeek-V3',       rmsnorm: true,  rope: true,  gqa: false, mla: true,  swiglu: true,  moe: true  },
  { model: 'Mistral 7B',        rmsnorm: true,  rope: true,  gqa: true,  mla: false, swiglu: true,  moe: false },
  { model: 'Mixtral 8×7B',      rmsnorm: true,  rope: true,  gqa: false, mla: false, swiglu: true,  moe: true  },
  { model: 'Qwen 2.5',          rmsnorm: true,  rope: true,  gqa: true,  mla: false, swiglu: true,  moe: false },
  { model: 'Gemma 2 (Google)',   rmsnorm: true,  rope: true,  gqa: true,  mla: false, swiglu: true,  moe: false },
  { model: 'Claude 3 (Anthropic)', rmsnorm: true, rope: true, gqa: true, mla: false, swiglu: true,  moe: false },
];

const COLS = ['RMSNorm', 'RoPE', 'GQA', 'MLA', 'SwiGLU', 'MoE'];
type AdoptionKey = 'rmsnorm' | 'rope' | 'gqa' | 'mla' | 'swiglu' | 'moe';
const COL_KEYS: AdoptionKey[] = ['rmsnorm', 'rope', 'gqa', 'mla', 'swiglu', 'moe'];

function Check({ yes }: { yes: boolean }) {
  return yes
    ? <span style={{ color: '#3fb950', fontSize: 14 }}>✓</span>
    : <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>—</span>;
}

export default function ModernArchitecturesPage() {
  return (
    <PageLayout
      title="Modern LLM Architectures"
      subtitle="The 6 key upgrades powering today's frontier models — from LLaMA 3 and DeepSeek-V3 to Mistral and Claude."
      step={11}
      learningContext={{
        prerequisites: [
          { label: 'Transformer Block (Step 6)', path: '/transformer-block' },
          { label: 'Multi-Head Attention (Step 5)', path: '/multi-head-attention' },
        ],
        plainEnglish:
          "The vanilla transformer (2017) was a breakthrough, but today's production LLMs are built on 6 evolved components. " +
          "Each upgrade targets a specific bottleneck: faster normalization, better position encoding, smaller KV caches, " +
          "more expressive FFNs, and sparse computation. Together they let a model like DeepSeek-V3 match GPT-4 performance " +
          "while activating only 37B of its 671B parameters per token.",
        realWorld:
          "Every open-source frontier model released since 2023 uses RMSNorm + RoPE + SwiGLU as a baseline. " +
          "GQA and MLA solve the KV cache bottleneck that limits how long a context window you can serve. " +
          "MoE (Mixture of Experts) lets you scale total parameter count without proportionally scaling compute.",
      }}
      takeaway={{
        bullets: [
          'RMSNorm drops mean-centering from LayerNorm — 7–64% faster, same quality',
          'RoPE encodes position as rotation, giving relative distances for free via dot-product geometry',
          'GQA shares KV heads across groups of Q heads — 8× KV cache reduction in LLaMA 3 vs. MHA',
          'MLA (DeepSeek) compresses KV into a low-rank latent — 93.3% KV cache reduction vs. MHA',
          'SwiGLU replaces GELU FFN with a gated product — consistently +1–2% on benchmarks',
          'MoE routes each token to a few of N expert FFNs — DeepSeek-V3 activates 37B of 671B params',
        ],
        prevStep: { label: 'Text Generation', path: '/generation' },
      }}
    >

      {/* ── Model Adoption Table ── */}
      <div className="card">
        <h3>Which Models Use Which Techniques?</h3>
        <p className="card-subtitle">
          Every formula below is validated from the original research paper. Here's how they map
          to models you likely use today.
        </p>
        <div style={{ overflowX: 'auto', marginTop: 12 }}>
          <table className="merge-table" style={{ fontSize: 12, minWidth: 520 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', fontWeight: 600, color: 'var(--text-muted)', paddingBottom: 8 }}>Model</th>
                {COLS.map(c => (
                  <th key={c} style={{ textAlign: 'center', fontWeight: 600, color: 'var(--text-muted)', paddingBottom: 8 }}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ADOPTION.map(row => (
                <tr key={row.model}>
                  <td style={{ color: 'var(--text-primary)', fontWeight: 500, paddingRight: 16 }}>{row.model}</td>
                  {COL_KEYS.map(k => (
                    <td key={k} style={{ textAlign: 'center' }}><Check yes={row[k]} /></td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 12 }}>
          Formulas follow the forward-pass order through a modern transformer layer: normalize → position-encode queries/keys → attention → FFN.
        </p>
      </div>

      {/* ══════════════════════════════════════════════════
          1. RMSNorm
      ══════════════════════════════════════════════════ */}
      <div className="card">
        <h3>① RMSNorm — Root Mean Square Normalization</h3>
        <p className="card-subtitle">
          Applied before every attention and FFN sub-layer (Pre-Norm). Replaces LayerNorm in nearly
          all modern LLMs by dropping the mean-centering step — achieving 7–64% speedup with no quality loss.
        </p>
        <FormulaBlock
          title="RMSNorm"
          formula={String.raw`\text{RMSNorm}(\mathbf{x}) = \dfrac{\mathbf{x}}{\mathrm{RMS}(\mathbf{x})} \cdot \boldsymbol{\gamma}, \qquad \mathrm{RMS}(\mathbf{x}) = \sqrt{\dfrac{1}{n}\sum_{i=1}^{n} x_i^2}`}
          source="Zhang & Sennrich 2019, arXiv:1910.07467"
          sourceUrl="https://arxiv.org/abs/1910.07467"
          intuition="Rescale by the root-mean-square — no mean subtraction needed. Simpler, faster, just as stable."
          terms={[
            { symbol: String.raw`\mathbf{x}`, name: 'input', meaning: 'The d-dimensional hidden state vector for one token' },
            { symbol: String.raw`n`, name: 'dimension', meaning: 'Number of elements in x (= d_model)' },
            { symbol: String.raw`\mathrm{RMS}(\mathbf{x})`, name: 'root mean square', meaning: '√(mean of squares of all elements) — measures the "energy" of the vector' },
            { symbol: String.raw`\boldsymbol{\gamma}`, name: 'gain', meaning: 'Learnable per-dimension scale parameter (shape: d_model), initialized to 1' },
          ]}
          steps={[
            { label: 'LayerNorm (for reference)', math: String.raw`\text{LN}(\mathbf{x}) = \dfrac{\mathbf{x} - \mu}{\sigma} \cdot \boldsymbol{\gamma} + \boldsymbol{\beta}`, text: 'Standard LayerNorm subtracts the mean μ and divides by std σ — two statistics to compute.' },
            { label: 'Drop mean-centering', math: String.raw`\text{RMSNorm}(\mathbf{x}) = \dfrac{\mathbf{x}}{\mathrm{RMS}(\mathbf{x})} \cdot \boldsymbol{\gamma}`, text: 'RMSNorm removes μ and β entirely. The hypothesis: re-centering is redundant — re-scaling is what matters for training stability.' },
            { label: 'Why it works', text: 'Zhang & Sennrich show that the invariance property of LayerNorm comes from re-scaling, not re-centering. Removing mean subtraction saves one pass over the vector and eliminates the β parameter (halving the number of norm params).' },
          ]}
        />
        <ModelBadges models={['LLaMA 2/3', 'DeepSeek-V2/V3', 'Mistral', 'Mixtral', 'Qwen 2.5', 'Gemma 2', 'Falcon']} />
        <CodeBlock
          title="RMSNorm — PyTorch (from LLaMA)"
          collapsible
          code={`import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    Zhang & Sennrich 2019  (arXiv:1910.07467)
    Used verbatim in LLaMA 2/3, Mistral, DeepSeek.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # γ, learned scale

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x * (1 / RMS(x))  — rsqrt fuses the sqrt + divide
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability, then cast back
        return self._norm(x.float()).type_as(x) * self.weight

# ---------- usage ----------
norm = RMSNorm(dim=4096)          # one per attention / FFN sub-layer
x    = torch.randn(2, 128, 4096)  # (batch, seq_len, d_model)
out  = norm(x)                    # same shape, RMS-normalised + scaled
print(out.shape)  # torch.Size([2, 128, 4096])`}
        />
      </div>

      {/* ══════════════════════════════════════════════════
          2. RoPE
      ══════════════════════════════════════════════════ */}
      <div className="card">
        <h3>② RoPE — Rotary Position Embedding</h3>
        <p className="card-subtitle">
          Applied to each query and key vector before computing attention scores. Encodes absolute position
          as a rotation so that the dot product Q·Kᵀ naturally produces relative positional information.
        </p>
        <FormulaBlock
          title="RoPE Rotation"
          formula={String.raw`f(\mathbf{q}, m) = \mathbf{R}_{\Theta,m}\,\mathbf{q}, \qquad \mathbf{R}_{\Theta,m}^{(j)} = \begin{pmatrix}\cos m\theta_j & -\sin m\theta_j \\ \sin m\theta_j & \cos m\theta_j\end{pmatrix}, \quad \theta_j = 10000^{-2j/d}`}
          source="Su et al. 2021, arXiv:2104.09864 (RoFormer)"
          sourceUrl="https://arxiv.org/abs/2104.09864"
          intuition="Rotate each query/key vector by an angle proportional to its position. Relative angles cancel out perfectly in the dot product."
          terms={[
            { symbol: String.raw`\mathbf{q}`, name: 'query', meaning: 'The query vector for one attention head (dimension d_h)' },
            { symbol: String.raw`m`, name: 'position', meaning: 'The absolute token position index (0, 1, 2, …, T−1)' },
            { symbol: String.raw`\mathbf{R}_{\Theta,m}`, name: 'rotation matrix', meaning: 'Block-diagonal rotation matrix built from d_h/2 independent 2×2 rotations, one per dimension pair' },
            { symbol: String.raw`\theta_j`, name: 'frequency', meaning: 'Rotation frequency for the j-th dimension pair — same geometric schedule as sinusoidal PE (base 10000)' },
            { symbol: String.raw`d`, name: 'head dim', meaning: 'Dimension of the query/key vector (= d_model / n_heads)' },
          ]}
          steps={[
            { label: 'Pair up dimensions', text: 'Group the d_h elements of q into d_h/2 consecutive pairs: (q_0, q_1), (q_2, q_3), … Each pair is rotated independently.' },
            { label: 'Rotate pair j at position m', math: String.raw`\begin{pmatrix} q_{2j}' \\ q_{2j+1}' \end{pmatrix} = \begin{pmatrix}\cos m\theta_j & -\sin m\theta_j \\ \sin m\theta_j & \cos m\theta_j\end{pmatrix} \begin{pmatrix} q_{2j} \\ q_{2j+1} \end{pmatrix}`, text: 'Apply a 2D rotation with angle mθ_j. Lower j → higher frequency, higher j → lower frequency (slower rotation).' },
            { label: 'Relative position in the dot product', math: String.raw`(\mathbf{R}_{\Theta,m}\mathbf{q})^\top (\mathbf{R}_{\Theta,n}\mathbf{k}) = \mathbf{q}^\top \mathbf{R}_{\Theta,m-n}\mathbf{k}`, text: 'Because rotations compose, the dot product of rotated q at position m with rotated k at position n only depends on the relative offset m−n. No extra learned embedding table needed.' },
          ]}
        />
        <ModelBadges models={['LLaMA 2/3', 'DeepSeek-V2/V3', 'Mistral', 'Mixtral', 'Qwen 2.5', 'Gemma 2', 'PaLM 2', 'GPT-NeoX']} />
        <CodeBlock
          title="RoPE — PyTorch (from LLaMA)"
          collapsible
          code={`import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute complex-valued rotation frequencies.
    Returns tensor of shape (end, dim/2) in complex64.
    """
    # θ_j = 10000^(-2j/dim)  for j = 0 .. dim/2-1
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)          # position indices 0 .. T-1
    freqs = torch.outer(t, freqs)  # (T, dim/2)  — angle mθ_j
    # Store as complex numbers: e^{imθ_j} = cos(mθ_j) + i·sin(mθ_j)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cis: torch.Tensor):
    """Rotate Q and K by their position's frequency vector."""
    # View last dim as complex pairs: (d_h,) → (d_h/2,) complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Broadcast freqs_cis over batch and head dimensions
    shape = [d if i == 1 or i == xq_.ndim - 1 else 1
             for i, d in enumerate(xq_.shape)]
    freqs_cis = freqs_cis.view(*shape)

    # Multiply in complex space = 2D rotation; convert back to real
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ---------- usage ----------
T, n_heads, d_h = 128, 32, 128      # LLaMA 3-8B head shape
freqs = precompute_freqs_cis(d_h, T)
q = torch.randn(1, T, n_heads, d_h) # (batch, seq, heads, head_dim)
k = torch.randn(1, T, n_heads, d_h)
# After rotation: dot(q[m], k[n]) encodes relative offset m-n
q_rot, k_rot = apply_rotary_emb(q, k, freqs)`}
        />
      </div>

      {/* ══════════════════════════════════════════════════
          3. GQA
      ══════════════════════════════════════════════════ */}
      <div className="card">
        <h3>③ GQA — Grouped-Query Attention</h3>
        <p className="card-subtitle">
          Standard multi-head attention requires one KV pair per query head. GQA shares a single KV pair
          across a group of query heads — dramatically shrinking the KV cache during inference.
        </p>
        <FormulaBlock
          title="Grouped-Query Attention"
          formula={String.raw`\text{head}_i = \text{Attn}(Q_i,\; K_{g(i)},\; V_{g(i)}), \qquad g(i) = \Bigl\lfloor \dfrac{i \cdot G}{n_h} \Bigr\rfloor`}
          source="Ainslie et al. 2023, arXiv:2305.13245"
          sourceUrl="https://arxiv.org/abs/2305.13245"
          intuition="Why give every query head its own K and V when neighbouring heads learn similar things? Group them — massive KV cache savings, near-zero quality loss."
          terms={[
            { symbol: String.raw`n_h`, name: 'query heads', meaning: 'Total number of query heads (e.g. 32 for LLaMA 3-8B)' },
            { symbol: String.raw`G`, name: 'KV groups', meaning: 'Number of key-value head groups (e.g. 8 for LLaMA 3-8B — 4× KV reduction vs. MHA)' },
            { symbol: String.raw`g(i)`, name: 'group index', meaning: 'Which KV group query head i belongs to — heads i=0..3 share group 0, heads i=4..7 share group 1, etc.' },
            { symbol: String.raw`Q_i`, name: 'query', meaning: 'Query projection for head i — unchanged from MHA' },
            { symbol: String.raw`K_{g(i)},\, V_{g(i)}`, name: 'shared KV', meaning: 'The key and value projections shared by all heads in the same group g(i)' },
          ]}
          steps={[
            { label: 'MHA (baseline)', text: 'n_h query heads × n_h KV heads = n_h × 2d_h elements cached per token per layer. Grows linearly with heads.' },
            { label: 'MQA (extreme)', text: 'G=1: all query heads share a single KV pair. Minimum KV cache, but noticeable quality drop on long contexts.' },
            { label: 'GQA (sweet spot)', math: String.raw`\text{KV cache} = G \times 2d_h \text{ per token per layer}`, text: 'G=8 (LLaMA 3-8B): 4× KV cache reduction vs. MHA. Quality matches MHA. Ainslie et al. find G≈n_h/4 is Pareto-optimal.' },
          ]}
        />
        <ModelBadges models={['LLaMA 2/3', 'Mistral 7B', 'Mixtral', 'Qwen 2.5', 'Gemma 2', 'Falcon-40B']} />
        <CodeBlock
          title="GQA — PyTorch"
          collapsible
          code={`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads.
    Equivalent to torch.repeat_interleave(x, n_rep, dim=1).
    x: (batch, n_kv_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return x  # MHA — no repeating needed
    B, n_kv, T, d = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, T, d) \\
             .reshape(B, n_kv * n_rep, T, d)

class GroupedQueryAttention(nn.Module):
    """GQA as used in LLaMA 3-8B: n_heads=32, n_kv_heads=8.
    Ainslie et al. 2023  (arXiv:2305.13245)
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads  # heads per KV group
        self.d_h        = d_model // n_heads

        self.wq = nn.Linear(d_model, n_heads    * self.d_h, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.d_h, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.d_h, bias=False)
        self.wo = nn.Linear(d_model, d_model,                bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads,    self.d_h).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.d_h).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.d_h).transpose(1, 2)

        # Expand KV heads to match Q heads (memory-efficient broadcast)
        k = repeat_kv(k, self.n_rep)  # (B, n_heads, T, d_h)
        v = repeat_kv(v, self.n_rep)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_h)
        out = F.softmax(scores, dim=-1) @ v          # (B, n_heads, T, d_h)
        out = out.transpose(1, 2).reshape(B, T, -1)  # (B, T, d_model)
        return self.wo(out)

# LLaMA 3-8B config: 32 Q heads, 8 KV heads → 4× KV cache reduction
attn = GroupedQueryAttention(d_model=4096, n_heads=32, n_kv_heads=8)
x = torch.randn(1, 128, 4096)
print(attn(x).shape)  # torch.Size([1, 128, 4096])`}
        />
      </div>

      {/* ══════════════════════════════════════════════════
          4. MLA
      ══════════════════════════════════════════════════ */}
      <div className="card">
        <h3>④ MLA — Multi-Head Latent Attention (DeepSeek)</h3>
        <p className="card-subtitle">
          Instead of caching full-dimension K and V vectors, MLA compresses them into a tiny low-rank
          latent vector. During inference this latent is all that needs to be stored — 93.3% fewer KV elements
          than MHA, with better performance.
        </p>
        <FormulaBlock
          title="MLA Low-Rank KV Compression"
          formula={String.raw`\mathbf{c}_t^{KV} = W^{DKV}\mathbf{h}_t, \quad \mathbf{k}_t^C = W^{UK}\mathbf{c}_t^{KV}, \quad \mathbf{v}_t^C = W^{UV}\mathbf{c}_t^{KV}`}
          source="DeepSeek-V2 2024, arXiv:2405.04434, §2.1.2"
          sourceUrl="https://arxiv.org/abs/2405.04434"
          intuition="Project KV down into a tiny bottleneck vector. Only cache the bottleneck — reconstruct K and V on the fly, or absorb the up-projection into the attention weights."
          terms={[
            { symbol: String.raw`\mathbf{h}_t`, name: 'hidden state', meaning: 'The full d-dimensional hidden state for token t at this layer' },
            { symbol: String.raw`\mathbf{c}_t^{KV}`, name: 'KV latent', meaning: 'Compressed latent vector of dimension d_c ≪ n_h·d_h. This is the only thing cached at inference time.' },
            { symbol: String.raw`W^{DKV}`, name: 'down-projection', meaning: 'Learnable matrix ℝ^{d_c × d} that compresses the hidden state into the KV latent' },
            { symbol: String.raw`W^{UK}`, name: 'K up-projection', meaning: 'Learnable matrix ℝ^{n_h·d_h × d_c} that reconstructs all key vectors from the latent' },
            { symbol: String.raw`W^{UV}`, name: 'V up-projection', meaning: 'Learnable matrix ℝ^{n_h·d_h × d_c} that reconstructs all value vectors from the latent' },
          ]}
          steps={[
            { label: 'Standard MHA KV cache cost', math: String.raw`2 \cdot n_h \cdot d_h \cdot L \text{ elements per token}`, text: 'Where L is the number of layers. For DeepSeek-V2 (n_h=128, d_h=128, L=60): 1.97M elements per token.' },
            { label: 'MLA cache: only the latent', math: String.raw`d_c \cdot L \text{ elements per token}`, text: 'DeepSeek-V2 sets d_c=512 (vs. n_h·d_h=16384). Cache is 512×60 = 30,720 elements — 93.3% reduction.' },
            { label: 'The absorption trick', math: String.raw`W^{UQ \leftarrow Q} = W^{UQ}\cdot W^{UK}, \quad W^{UO \leftarrow O} = W^{UV}\cdot W^O`, text: 'W^{UK} can be absorbed into W^Q, and W^{UV} into W^O. This means we never need to materialize full K or V tensors — attention is computed directly on the latent. Only the latent c_t^{KV} is cached.' },
          ]}
        />
        <ModelBadges models={['DeepSeek-V2', 'DeepSeek-V3', 'DeepSeek-R1']} />
        <CodeBlock
          title="MLA — PyTorch (simplified)"
          collapsible
          code={`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadLatentAttention(nn.Module):
    """Simplified MLA as described in DeepSeek-V2 §2.1.2.
    Key idea: compress KV into a low-rank latent c_t^{KV};
    only cache the latent (d_c << n_h * d_h elements).
    arXiv:2405.04434
    """
    def __init__(self, d_model: int, n_heads: int,
                 d_c: int = 512,       # KV latent dim (bottleneck)
                 d_rope: int = 64):    # decoupled RoPE dim per head
        super().__init__()
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.d_c  = d_c

        # KV compression: h_t → c_t^KV  (this is all that's cached)
        self.W_dkv = nn.Linear(d_model, d_c, bias=False)   # down-proj
        self.W_uk  = nn.Linear(d_c, n_heads * self.d_h, bias=False)  # K up
        self.W_uv  = nn.Linear(d_c, n_heads * self.d_h, bias=False)  # V up

        # Q compression (saves activations but not part of KV cache)
        self.W_dq = nn.Linear(d_model, d_c, bias=False)
        self.W_uq = nn.Linear(d_c, n_heads * self.d_h, bias=False)

        self.W_o  = nn.Linear(n_heads * self.d_h, d_model, bias=False)
        self.norm_k = nn.RMSNorm(self.d_h)
        self.norm_q = nn.RMSNorm(self.d_h)

    def forward(self, x: torch.Tensor,
                kv_cache: torch.Tensor | None = None):
        B, T, _ = x.shape

        # ── Compress KV ──────────────────────────────────────
        c_kv = self.W_dkv(x)          # (B, T, d_c)  ← only this is cached
        k = self.W_uk(c_kv).view(B, T, self.n_heads, self.d_h)
        v = self.W_uv(c_kv).view(B, T, self.n_heads, self.d_h)

        # ── Compress Q ────────────────────────────────────────
        c_q = self.W_dq(x)
        q = self.W_uq(c_q).view(B, T, self.n_heads, self.d_h)

        # Normalize after up-projection (DeepSeek detail)
        q = self.norm_q(q);  k = self.norm_k(k)

        # ── Attention ─────────────────────────────────────────
        q = q.transpose(1, 2);  k = k.transpose(1, 2);  v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_h)
        out = F.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.W_o(out), c_kv   # return latent for KV cache storage

# DeepSeek-V2: d_model=5120, n_heads=128, d_c=512 → 93.3% KV cache saving
mla = MultiHeadLatentAttention(d_model=512, n_heads=8, d_c=64)
x = torch.randn(1, 32, 512)
out, latent = mla(x)
print(out.shape, latent.shape)  # [1,32,512]  [1,32,64]`}
        />
      </div>

      {/* ══════════════════════════════════════════════════
          5. SwiGLU
      ══════════════════════════════════════════════════ */}
      <div className="card">
        <h3>⑤ SwiGLU — Swish-Gated Linear Unit</h3>
        <p className="card-subtitle">
          The feed-forward sublayer in every modern LLM uses SwiGLU instead of the original GELU FFN.
          Two parallel linear projections — one acts as a gate that controls how much of the other passes through.
        </p>
        <FormulaBlock
          title="SwiGLU Feed-Forward"
          formula={String.raw`\text{SwiGLU}(\mathbf{x}) = \bigl(\text{Swish}(\mathbf{x}W_1) \odot \mathbf{x}V\bigr)W_2, \qquad \text{Swish}(z) = z \cdot \sigma(z)`}
          source="Shazeer 2020, arXiv:2002.05202 (GLU Variants)"
          sourceUrl="https://arxiv.org/abs/2002.05202"
          intuition="Two linear projections — one passes through Swish, the other is a gate. Their element-wise product lets the network control information flow continuously, not with an on/off ReLU switch."
          terms={[
            { symbol: String.raw`\mathbf{x}`, name: 'input', meaning: 'Hidden state, shape (T, d_model), one row per token' },
            { symbol: String.raw`W_1`, name: 'gate proj', meaning: 'First linear projection, shape (d_model, d_ff), where d_ff ≈ 8d/3 (rounded). Applied to x, then Swish-activated.' },
            { symbol: String.raw`V`, name: 'up proj', meaning: 'Second linear projection, same shape as W_1. Acts as the "gate value" multiplied element-wise.' },
            { symbol: String.raw`W_2`, name: 'down proj', meaning: 'Output projection, shape (d_ff, d_model). Projects the gated representation back down.' },
            { symbol: String.raw`\odot`, name: 'hadamard', meaning: 'Element-wise (Hadamard) product — multiplies two vectors position by position' },
            { symbol: String.raw`\sigma(z)`, name: 'sigmoid', meaning: '1/(1+e^{−z}) — squashes to (0,1), used inside Swish' },
          ]}
          steps={[
            { label: 'Original FFN (GELU)', math: String.raw`\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x}W_1)W_2`, text: 'Two matrices, one activation. Simple but fixed nonlinearity — GELU applies the same transformation regardless of input.' },
            { label: 'Gated mechanism', math: String.raw`\text{gate}(\mathbf{x}) = \text{Swish}(\mathbf{x}W_1) \odot \mathbf{x}V`, text: 'The Swish branch decides how much of the Up branch passes through. This data-dependent gating is what gives SwiGLU its edge.' },
            { label: 'FFN dim adjustment', math: String.raw`d_{ff} = \tfrac{8}{3}\,d \approx 2.67d`, text: 'SwiGLU needs 3 matrices (W_1, V, W_2) vs. 2 in standard FFN. To keep the total parameter count equal, d_ff is reduced from 4d to 8d/3.' },
          ]}
        />
        <ModelBadges models={['LLaMA 2/3', 'DeepSeek-V2/V3', 'PaLM 2', 'Mistral', 'Mixtral', 'Qwen 2.5', 'Gemma 2']} />
        <CodeBlock
          title="SwiGLU FFN — PyTorch (from LLaMA)"
          collapsible
          code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFeedForward(nn.Module):
    """Feed-forward block with SwiGLU activation.
    Shazeer 2020  (arXiv:2002.05202)
    Identical to FeedForward in meta-llama/llama/blob/main/llama/model.py
    """
    def __init__(self, d_model: int, multiple_of: int = 256,
                 ffn_dim_multiplier: float | None = None):
        super().__init__()
        # d_ff = 8/3 * d_model, rounded up to nearest multiple_of
        # (keeps param count ≈ 2-matrix GELU FFN with d_ff=4*d_model)
        d_ff = int(2 * d_model * 4 / 3)
        if ffn_dim_multiplier:
            d_ff = int(ffn_dim_multiplier * d_ff)
        d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate branch
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # up   branch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (Swish(xW1) ⊙ xW3) W2
        # F.silu is Swish:  silu(z) = z * sigmoid(z)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# ---------- comparison ----------
class GELUFeedForward(nn.Module):
    """Original Transformer FFN (Vaswani et al. 2017)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

d_model = 4096
swiglu = SwiGLUFeedForward(d_model)     # 3 matrices, d_ff ≈ 10,752
gelu   = GELUFeedForward(d_model, d_model * 4)  # 2 matrices, d_ff = 16,384
x = torch.randn(2, 128, d_model)
print(swiglu(x).shape)   # torch.Size([2, 128, 4096])
print(gelu(x).shape)     # torch.Size([2, 128, 4096])`}
        />
      </div>

      {/* ══════════════════════════════════════════════════
          6. Mixture of Experts (DeepSeekMoE)
      ══════════════════════════════════════════════════ */}
      <div className="card">
        <h3>⑥ MoE — Mixture of Experts (DeepSeekMoE)</h3>
        <p className="card-subtitle">
          Replace each FFN with N expert FFNs. A router selects the top-K experts for each token.
          DeepSeekMoE adds fine-grained expert segmentation and always-on shared experts for
          common knowledge — enabling DeepSeek-V3 (671B total) to match dense models while
          activating only 37B parameters per token.
        </p>
        <FormulaBlock
          title="DeepSeekMoE Routing & Output"
          formula={String.raw`\mathbf{h}' = \mathbf{u} + \sum_{i=1}^{N_s}\text{FFN}_i^{(s)}(\mathbf{u}) + \sum_{i=1}^{N_r} g_{i}\,\text{FFN}_i^{(r)}(\mathbf{u})`}
          source="DeepSeek-V2 2024, arXiv:2405.04434, §2.2.1"
          sourceUrl="https://arxiv.org/abs/2405.04434"
          intuition="Most tokens don't need every FFN. Route each token to the few experts most relevant to it — compute only what's needed, skip the rest."
          terms={[
            { symbol: String.raw`\mathbf{u}`, name: 'FFN input', meaning: 'Hidden state entering the FFN sublayer, shape (d_model,)' },
            { symbol: String.raw`N_s`, name: 'shared experts', meaning: 'Always-activated expert FFNs (2 in DeepSeek-V2) — handles common knowledge not specialized to any topic' },
            { symbol: String.raw`N_r`, name: 'routed experts', meaning: 'Expert FFNs available for routing (160 in DeepSeek-V2); only K_r are activated per token' },
            { symbol: String.raw`K_r`, name: 'top-K', meaning: 'Number of routed experts activated per token (6 in DeepSeek-V2, 8 in DeepSeek-V3)' },
            { symbol: String.raw`g_i`, name: 'gate value', meaning: 'Weight for routed expert i — equals s_i if i is in Top-K, else 0' },
            { symbol: String.raw`s_i`, name: 'affinity score', meaning: String.raw`s_i = \text{Softmax}_i(\mathbf{u} \cdot \mathbf{e}_i)` + " where e_i is the centroid vector for expert i" },
            { symbol: String.raw`\text{FFN}^{(s)}/\text{FFN}^{(r)}`, name: 'expert FFN', meaning: 'A standard SwiGLU FFN — shared experts and routed experts have the same structure, just different weights' },
          ]}
          steps={[
            { label: 'Compute affinity scores', math: String.raw`s_{i,t} = \text{Softmax}_i\bigl(\mathbf{u}_t \cdot \mathbf{e}_i\bigr) \quad \forall i \in \{1,\ldots,N_r\}`, text: "Dot the token's hidden state with each expert's learned centroid vector e_i, then softmax to get a probability over experts." },
            { label: 'Select top-K experts', math: String.raw`g_{i,t} = \begin{cases} s_{i,t} & \text{if } s_{i,t} \in \text{TopK}(\{s_{j,t}\}_{j=1}^{N_r},\, K_r) \\ 0 & \text{otherwise} \end{cases}`, text: 'Only the K_r highest-affinity routed experts contribute. Experts outside TopK get weight 0 — their computation is skipped entirely.' },
            { label: 'Aggregate outputs', math: String.raw`\mathbf{h}' = \mathbf{u} + \underbrace{\sum_{i=1}^{N_s}\text{FFN}_i^{(s)}(\mathbf{u})}_{\text{shared (always)}} + \underbrace{\sum_{i=1}^{N_r} g_{i,t}\,\text{FFN}_i^{(r)}(\mathbf{u})}_{\text{sparse (top-}K_r\text{ only)}}`, text: 'Shared experts always run. Routed experts are weighted by their gate value. The residual (u) is also added. In DeepSeek-V2: 2 shared + 6 routed of 160 = 8 expert FLOPs per token instead of 160.' },
          ]}
        />
        <ModelBadges models={['DeepSeek-V2', 'DeepSeek-V3', 'DeepSeek-R1', 'Mixtral 8×7B/22B', 'Switch Transformer', 'Grok-1']} />
        <CodeBlock
          title="MoE with Top-K Routing — PyTorch"
          collapsible
          code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """One SwiGLU expert FFN (all experts have identical structure,
    different learned weights)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff,   d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoELayer(nn.Module):
    """Sparse Mixture-of-Experts FFN layer.
    DeepSeekMoE variant: N_s shared (always-on) + N_r routed experts.
    DeepSeek-V2 2024  (arXiv:2405.04434, §2.2.1)
    """
    def __init__(self, d_model: int, d_ff: int,
                 n_routed: int = 8,    # N_r — total routed experts
                 n_shared: int = 2,    # N_s — always-activated experts
                 top_k: int = 2):      # K_r — experts used per token
        super().__init__()
        self.top_k   = top_k
        self.n_shared = n_shared
        # Shared experts — always run
        self.shared = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_shared)])
        # Routed experts — only top_k activated per token
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_routed)])
        # Router: maps hidden state → expert affinity scores
        self.router  = nn.Linear(d_model, n_routed, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        flat = x.view(-1, d)                        # (B*T, d_model)

        # ── Routing ──────────────────────────────────
        logits  = self.router(flat)                 # (B*T, n_routed)
        scores  = F.softmax(logits, dim=-1)         # affinity probabilities
        topk_w, topk_idx = scores.topk(self.top_k, dim=-1)  # top-K
        # Renormalise selected weights so they sum to 1
        topk_w  = topk_w / topk_w.sum(dim=-1, keepdim=True)

        # ── Shared experts (always run) ───────────────
        out = sum(e(flat) for e in self.shared)     # (B*T, d_model)

        # ── Routed experts (sparse) ───────────────────
        # Iterate over the K selected experts for each token
        expert_out = torch.zeros_like(flat)
        for k in range(self.top_k):
            idx = topk_idx[:, k]   # which expert each token chose
            w   = topk_w[:, k]     # gate weight for that expert
            for e_id in idx.unique():
                mask = (idx == e_id)          # tokens routed to expert e_id
                expert_out[mask] += w[mask, None] * self.experts[e_id](flat[mask])
        out = out + expert_out
        return out.view(B, T, d)

# DeepSeek-V3 scale (mini demo): 8 routed, 2 shared, top-2 per token
moe = MoELayer(d_model=256, d_ff=512, n_routed=8, n_shared=2, top_k=2)
x = torch.randn(2, 16, 256)
print(moe(x).shape)   # torch.Size([2, 16, 256])
# Only 2+2=4 of 10 expert FFNs run per token → 60% FLOP savings`}
        />
      </div>

      {/* ── Summary ── */}
      <div className="card">
        <h3>Putting It All Together</h3>
        <p className="card-subtitle">
          A modern LLM forward pass through one layer looks like this:
        </p>
        <div style={{ overflowX: 'auto', marginTop: 12 }}>
          <table className="merge-table" style={{ fontSize: 13 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', fontWeight: 600, color: 'var(--text-muted)' }}>Stage</th>
                <th style={{ textAlign: 'left', fontWeight: 600, color: 'var(--text-muted)' }}>Vanilla Transformer (2017)</th>
                <th style={{ textAlign: 'left', fontWeight: 600, color: 'var(--text-muted)' }}>Modern LLM (2024)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ color: 'var(--text-muted)' }}>Normalize</td>
                <td>LayerNorm (mean + var)</td>
                <td style={{ color: '#3fb950' }}>RMSNorm (var only, faster)</td>
              </tr>
              <tr>
                <td style={{ color: 'var(--text-muted)' }}>Position</td>
                <td>Sinusoidal PE (additive, fixed)</td>
                <td style={{ color: '#3fb950' }}>RoPE (multiplicative, relative)</td>
              </tr>
              <tr>
                <td style={{ color: 'var(--text-muted)' }}>Attention KV</td>
                <td>Full KV per head (MHA)</td>
                <td style={{ color: '#3fb950' }}>Shared KV groups (GQA) or latent (MLA)</td>
              </tr>
              <tr>
                <td style={{ color: 'var(--text-muted)' }}>FFN</td>
                <td>GELU: xW₁ → GELU → W₂</td>
                <td style={{ color: '#3fb950' }}>SwiGLU: (Swish(xW₁) ⊙ xV) W₂</td>
              </tr>
              <tr>
                <td style={{ color: 'var(--text-muted)' }}>FFN Architecture</td>
                <td>Dense: same FFN for every token</td>
                <td style={{ color: '#3fb950' }}>MoE: route token to top-K of N expert FFNs</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

    </PageLayout>
  );
}
