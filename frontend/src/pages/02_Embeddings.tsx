import { useState, useCallback } from 'react';
import Plot from 'react-plotly.js';
import PageLayout from '../components/layout/PageLayout';
import FormulaBlock from '../components/shared/FormulaBlock';
import CodeBlock from '../components/shared/CodeBlock';
import { getEmbeddings, getEmbedPCA, getEmbeddingMatrix } from '../api/client';
import { useAppStore } from '../store/useAppStore';

export default function EmbeddingsPage() {
  const sharedText = useAppStore(s => s.sharedText);
  const setSharedText = useAppStore(s => s.setSharedText);
  const [text, setText] = useState(sharedText);
  const handleTextChange = (v: string) => { setText(v); setSharedText(v); };
  const [result, setResult] = useState<any>(null);
  const [pca, setPca] = useState<any>(null);
  const [matrix, setMatrix] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [embResult, pcaResult, matResult] = await Promise.all([
        getEmbeddings(text),
        getEmbedPCA(text),
        getEmbeddingMatrix(30),
      ]);
      setResult(embResult);
      setPca(pcaResult);
      setMatrix(matResult);
    } catch (e: any) {
      console.error('Embeddings error:', e);
      setError(e?.response?.data?.detail || e?.message || 'API call failed');
    }
    setLoading(false);
  }, [text]);

  return (
    <PageLayout
      title="Token Embeddings"
      subtitle="Convert discrete token IDs into continuous vector representations that capture semantic meaning."
      step={2}
      pipelineStep={2}
      learningContext={{
        prerequisites: [{ label: 'Tokenization (Step 1)', path: '/tokenization' }],
        plainEnglish: "Each token ID is used to look up a row in a big table of numbers. That row becomes the token's 'meaning vector' — a list of 64 numbers that the model learns to associate with that token's role in language.",
        realWorld: "GPT-3's embedding table has 50,257 tokens × 12,288 dimensions = 617 million numbers just for embeddings. Our model: 756 × 64 = 48,384. Word2Vec and GloVe pioneered this idea of learned vector representations — transformers made it standard.",
      }}
      takeaway={{
        bullets: [
          'Token ID → table lookup → dense vector (d_model dimensions)',
          'Before training, embedding vectors are random. After training, similar tokens cluster together in the vector space',
          'The embedding matrix E ∈ R^(V × d_model) is a learnable parameter — it gets updated during training',
          'PCA projection shows how the high-dimensional space can be visualized in 2D',
        ],
        prevStep: { label: 'Tokenization', path: '/tokenization' },
        nextStep: { label: 'Positional Encoding', path: '/positional-encoding', teaser: 'Add word-order information to embedding vectors' },
      }}
    >
      <div className="card">
        <h3>What are Embeddings?</h3>
        <p className="card-subtitle">
          Each token ID is used as an index to look up a row in a learnable matrix.
          This maps discrete tokens into a continuous vector space where similar tokens
          end up close together.
        </p>
        <FormulaBlock
          title="Token Embedding Lookup"
          formula={String.raw`\text{embed}(x) = E[x], \quad E \in \mathbb{R}^{V \times d_{\text{model}}}`}
          source="Vaswani et al. 2017, arXiv:1706.03762 §3.4"
          sourceUrl="https://arxiv.org/abs/1706.03762"
          intuition="A token ID is just a row index into a learnable table — embed(x) is the x-th row, a learned dense vector."
          terms={[
            { symbol: 'x', name: 'token ID', meaning: 'Integer index (0 to V-1) identifying a single subword token in the vocabulary' },
            { symbol: 'E', name: 'embedding matrix', meaning: 'Learnable weight matrix of shape (V, d_model) — one row per vocabulary token, updated by gradient descent' },
            { symbol: 'V', name: 'vocabulary size', meaning: 'Total number of unique tokens the tokenizer can produce (our model: ~756)' },
            { symbol: String.raw`d_{\text{model}}`, name: 'embedding dimension', meaning: 'Size of each embedding vector — the width of the matrix rows (our model: 64)' },
          ]}
          steps={[
            { label: 'Tokenize', text: 'Convert the input string into a sequence of integer token IDs via the BPE tokenizer' },
            { label: 'Index lookup', math: String.raw`\mathbf{e}_x = E[x, :]`, text: 'Select row x from the embedding matrix — a vector of d_model floats' },
            { label: 'Forward propagation', text: 'Pass the embedding vector through the rest of the model; gradients update E during backprop so similar tokens converge in vector space' },
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
          <button onClick={run} disabled={loading}>
            {loading ? 'Computing…' : 'Embed'}
          </button>
        </div>
        {error && <div style={{ marginTop: 12, padding: '10px 14px', background: '#3d1f1f', border: '1px solid #f85149', borderRadius: 6, color: '#f85149', fontSize: 13 }}>Error: {error}</div>}
      </div>

      {/* PCA scatter plot */}
      {pca && (
        <div className="card">
          <h3>Token Embeddings in 2D (PCA Projection)</h3>
          <p className="card-subtitle">
            64-dimensional embeddings projected to 2D using PCA.
            Explained variance: {(pca.explained_variance[0] * 100).toFixed(1)}% + {(pca.explained_variance[1] * 100).toFixed(1)}%
          </p>
          <div className="plot-container">
            <Plot
              data={[
                {
                  x: pca.points.map((p: any) => p.x),
                  y: pca.points.map((p: any) => p.y),
                  text: pca.points.map((p: any) => `'${p.label}' (id: ${p.id})`),
                  mode: 'markers+text',
                  type: 'scatter',
                  textposition: 'top center',
                  textfont: { size: 11, color: '#8b949e' },
                  marker: {
                    size: 10,
                    color: pca.points.map((_: any, i: number) => i),
                    colorscale: 'Viridis',
                  },
                },
              ]}
              layout={{
                height: 450,
                margin: { t: 20, b: 40, l: 60, r: 20 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#8b949e' },
                xaxis: { title: 'PC 1', gridcolor: '#21262d', zerolinecolor: '#30363d' },
                yaxis: { title: 'PC 2', gridcolor: '#21262d', zerolinecolor: '#30363d' },
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      )}

      {/* Embedding matrix heatmap */}
      {matrix && (
        <div className="card">
          <h3>Embedding Matrix (first {matrix.shape[0]} tokens × {matrix.shape[1]} dims)</h3>
          <p className="card-subtitle">
            Each row is one token's embedding vector. This matrix is the lookup table.
          </p>
          <div className="plot-container">
            <Plot
              data={[
                {
                  z: matrix.matrix,
                  y: matrix.row_labels.map((l: string, i: number) =>
                    `${i}: '${l.replace(/\n/g, '↵')}'`
                  ),
                  type: 'heatmap',
                  colorscale: 'RdBu',
                  reversescale: true,
                },
              ]}
              layout={{
                height: Math.max(400, matrix.shape[0] * 18),
                margin: { t: 20, b: 40, l: 100, r: 20 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#8b949e', size: 10 },
                xaxis: { title: 'Embedding Dimension' },
                yaxis: { autorange: 'reversed' },
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      )}

      {/* Lookup demo */}
      {result && (
        <div className="card">
          <h3>Embedding Lookup</h3>
          <p className="card-subtitle">
            Token IDs [{result.token_ids.slice(0, 10).join(', ')}{result.token_ids.length > 10 ? '…' : ''}] →
            index into the ({result.numpy_demo.embedding_matrix_shape[0]} × {result.numpy_demo.embedding_matrix_shape[1]}) matrix
          </p>
          <div className="explanation">
            {result.numpy_demo.explanation}
          </div>
        </div>
      )}

      <CodeBlock
        title="Embedding — Python Implementation"
        collapsible
        code={`import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # Learnable embedding matrix: (vocab_size, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids):
        # token_ids: (batch, seq_len) → integers
        # output:    (batch, seq_len, d_model) → vectors
        return self.embedding(token_ids)

# NumPy equivalent (no learning, just lookup):
import numpy as np

def embed_numpy(token_ids, vocab_size=756, d_model=64):
    # Random embedding matrix (in practice, learned)
    E = np.random.randn(vocab_size, d_model) * 0.02
    # Lookup: just index into the matrix
    return E[token_ids]  # shape: (len(token_ids), d_model)`}
      />
    </PageLayout>
  );
}
