import { useState, useCallback } from 'react';
import Plot from 'react-plotly.js';
import PageLayout from '../components/layout/PageLayout';
import FormulaBlock from '../components/shared/FormulaBlock';
import CodeBlock from '../components/shared/CodeBlock';
import { getPositionalEncoding, getPositionSimilarity } from '../api/client';
import { useAppStore } from '../store/useAppStore';

export default function PositionalEncodingPage() {
  const modelConfig = useAppStore(s => s.modelConfig);
  const [seqLen, setSeqLen] = useState(modelConfig.context_len);
  const [dModel, setDModel] = useState(modelConfig.d_model);
  const [peData, setPeData] = useState<any>(null);
  const [simData, setSimData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [pe, sim] = await Promise.all([
        getPositionalEncoding(seqLen, dModel),
        getPositionSimilarity(Math.min(seqLen, 32), dModel),
      ]);
      setPeData(pe);
      setSimData(sim);
    } catch (e: any) {
      console.error('PE error:', e);
      setError(e?.response?.data?.detail || e?.message || 'API call failed');
    }
    setLoading(false);
  }, [seqLen, dModel]);

  return (
    <PageLayout
      title="Positional Encoding"
      subtitle="Give the model information about token order using sinusoidal position signals — without this, a transformer can't distinguish 'dog bites man' from 'man bites dog'."
      step={3}
      pipelineStep={3}
      learningContext={{
        prerequisites: [{ label: 'Embeddings (Step 2)', path: '/embeddings' }],
        plainEnglish: "The model processes all words at once (not left-to-right like reading). So we add a unique position signal to each word's vector: 'I'm the 1st word', 'I'm the 5th word'. Without this, 'dog bites man' and 'man bites dog' would look identical to the model.",
        realWorld: "Original transformers (2017) used the sinusoidal encoding shown here. GPT-2 switched to learned position embeddings. LLaMA uses Rotary Position Embeddings (RoPE) which encode relative position and generalize better to longer sequences than the model was trained on.",
      }}
      takeaway={{
        bullets: [
          'Transformers process tokens in parallel — they need explicit position information',
          'Sinusoidal PE uses sin/cos waves at different frequencies to create unique position vectors',
          'Nearby positions have similar encodings (high cosine similarity), enabling the model to learn relative distance',
          'The final input to the transformer is token_embedding + position_encoding (element-wise addition)',
        ],
        prevStep: { label: 'Embeddings', path: '/embeddings' },
        nextStep: { label: 'Self-Attention', path: '/self-attention', teaser: 'The mechanism that lets tokens communicate with each other' },
      }}
    >
      <div className="card">
        <h3>Why Position Matters</h3>
        <p className="card-subtitle">
          Unlike RNNs, transformers process all tokens in parallel — they have no built-in
          notion of order. Positional encoding injects position information by adding a
          unique vector to each token's embedding.
        </p>
        <FormulaBlock
          title="Sinusoidal Positional Encoding"
          formula={String.raw`PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)`}
          source="Vaswani et al. 2017, §3.5"
          sourceUrl="https://arxiv.org/abs/1706.03762"
          intuition="Each position gets a unique fingerprint made of sine and cosine waves at different frequencies — nearby positions have similar fingerprints."
          terms={[
            { symbol: 'pos', name: 'Position', meaning: 'The index of the token in the sequence (0, 1, 2, …)' },
            { symbol: 'i', name: 'Dimension index', meaning: 'Which dimension of the encoding vector (0, 1, …, d_model/2−1)' },
            { symbol: 'd_{\\text{model}}', name: 'Model dimension', meaning: 'Size of the embedding vector (e.g. 512 in the original paper)' },
            { symbol: '10000', name: 'Base frequency', meaning: 'Controls the range of wavelengths — from 2π to 10000·2π' },
            { symbol: '\\sin / \\cos', name: 'Trig functions', meaning: 'Even dimensions use sine, odd dimensions use cosine — together they give unique position codes' },
          ]}
          steps={[
            { label: 'Compute the frequency divisor', math: String.raw`\text{div} = 10000^{2i/d_{\text{model}}}`, text: 'Each dimension pair (2i, 2i+1) has a different frequency. Low dimensions = fast oscillation, high dimensions = slow oscillation.' },
            { label: 'Divide position by frequency', math: String.raw`\theta = \frac{pos}{\text{div}}`, text: 'This gives the angle for the sinusoid at this position and dimension.' },
            { label: 'Apply sin (even dims) and cos (odd dims)', math: String.raw`PE_{(pos,2i)} = \sin(\theta), \quad PE_{(pos,2i+1)} = \cos(\theta)`, text: 'The sin/cos pair lets the model learn relative positions: PE(pos+k) can be written as a linear function of PE(pos).' },
          ]}
        />
      </div>

      <div className="card">
        <h3>Configure</h3>
        <div className="controls">
          <div className="control-group">
            <label>Sequence Length</label>
            <input
              type="number"
              value={seqLen}
              onChange={(e) => setSeqLen(Number(e.target.value))}
              min={8}
              max={256}
            />
          </div>
          <div className="control-group">
            <label>d_model</label>
            <input
              type="number"
              value={dModel}
              onChange={(e) => setDModel(Number(e.target.value))}
              min={8}
              max={256}
              step={8}
            />
          </div>
          <button onClick={run} disabled={loading}>
            {loading ? 'Computing…' : 'Generate PE'}
          </button>
        </div>
        {error && <div style={{ marginTop: 12, padding: '10px 14px', background: '#3d1f1f', border: '1px solid #f85149', borderRadius: 6, color: '#f85149', fontSize: 13 }}>Error: {error}</div>}
      </div>

      {/* PE heatmap */}
      {peData && (
        <div className="card">
          <h3>Positional Encoding Matrix ({peData.shape[0]} positions × {peData.shape[1]} dims)</h3>
          <p className="card-subtitle">
            Each row is a position's encoding vector. The characteristic sinusoidal wave pattern
            is visible — low-frequency waves on the left, high-frequency on the right.
          </p>
          <div className="plot-container">
            <Plot
              data={[
                {
                  z: peData.pe_matrix,
                  type: 'heatmap',
                  colorscale: 'RdBu',
                  reversescale: true,
                  zmin: -1,
                  zmax: 1,
                },
              ]}
              layout={{
                height: 450,
                margin: { t: 20, b: 50, l: 60, r: 20 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#8b949e' },
                xaxis: { title: 'Embedding Dimension' },
                yaxis: { title: 'Position', autorange: 'reversed' },
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      )}

      {/* Similarity matrix */}
      {simData && (
        <div className="card">
          <h3>Position Cosine Similarity</h3>
          <p className="card-subtitle">
            Cosine similarity between position encodings. Notice how nearby positions
            (along the diagonal) have higher similarity — this encodes relative distance.
          </p>
          <div className="plot-container">
            <Plot
              data={[
                {
                  z: simData.similarity_matrix,
                  type: 'heatmap',
                  colorscale: 'Viridis',
                },
              ]}
              layout={{
                height: 450,
                margin: { t: 20, b: 50, l: 60, r: 20 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#8b949e' },
                xaxis: { title: 'Position' },
                yaxis: { title: 'Position', autorange: 'reversed' },
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </div>
          <div className="explanation">{simData.explanation}</div>
        </div>
      )}

      {/* Combined embedding */}
      <div className="card">
        <h3>Combining Token + Position Embeddings</h3>
        <FormulaBlock
          title="Input Embedding Sum"
          formula={String.raw`x_i = \text{TokenEmbed}(token_i) + \text{PosEmbed}(i)`}
          source="Vaswani et al. 2017, §3.4"
          sourceUrl="https://arxiv.org/abs/1706.03762"
          intuition="Each token's final representation carries both 'what it is' (semantic meaning) and 'where it is' (position) by adding the two vectors element-wise."
          terms={[
            { symbol: 'x_i', name: 'Input vector', meaning: 'The final input to the transformer for the token at position i' },
            { symbol: '\\text{TokenEmbed}', name: 'Token embedding', meaning: 'Learned lookup table that converts a token ID to a dense vector (captures meaning)' },
            { symbol: '\\text{PosEmbed}', name: 'Position encoding', meaning: 'Sinusoidal or learned vector that encodes where the token sits in the sequence' },
          ]}
        />
      </div>

      <CodeBlock
        title="Positional Encoding — Python Implementation"
        collapsible
        code={`import numpy as np
import math

def sinusoidal_positional_encoding(seq_len, d_model):
    """
    Compute sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)

    # Division term: 10000^(2i/d_model)
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = np.sin(position * div_term)  # even dims
    pe[:, 1::2] = np.cos(position * div_term)  # odd dims
    return pe

# In PyTorch (learned positional embeddings, as in GPT-2):
import torch.nn as nn

class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, context_len, d_model):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_len, d_model)  # learned!

    def forward(self, token_ids):
        B, T = token_ids.shape
        tok = self.token_embed(token_ids)
        pos = self.pos_embed(torch.arange(T, device=token_ids.device))
        return tok + pos  # (B, T, d_model)`}
      />
    </PageLayout>
  );
}
