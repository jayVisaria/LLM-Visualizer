import { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import PageLayout from '../components/layout/PageLayout';
import FormulaBlock from '../components/shared/FormulaBlock';
import CodeBlock from '../components/shared/CodeBlock';
import { getModelArchitecture, getForwardPass } from '../api/client';
import { useAppStore } from '../store/useAppStore';

export default function FullModelPage() {
  const sharedText = useAppStore(s => s.sharedText);
  const setSharedText = useAppStore(s => s.setSharedText);
  const [arch, setArch] = useState<any>(null);
  const [text, setText] = useState(sharedText);
  const handleTextChange = (v: string) => { setText(v); setSharedText(v); };
  const [trace, setTrace] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getModelArchitecture().then(setArch).catch(console.error);
  }, []);

  const runTrace = async () => {
    setLoading(true);
    try {
      const data = await getForwardPass(text);
      setTrace(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  return (
    <PageLayout
      title="Full GPT Model"
      subtitle="Stack multiple transformer blocks to form a complete GPT-style language model — a decoder-only transformer that predicts the next token."
      step={7}
      pipelineStep={7}
      learningContext={{
        prerequisites: [
          { label: 'Transformer Block (Step 6)', path: '/transformer-block' },
          { label: 'Tokenization (Step 1)', path: '/tokenization' },
        ],
        plainEnglish: "Now we put it all together: token embeddings + position embeddings → stack of transformer blocks → final normalization → vocabulary projection. The output is a probability distribution over all possible next tokens. This is the complete architecture of GPT-2 — only the scale differs.",
        realWorld: "Our model has ~550K parameters. GPT-2 Small has 117M (same architecture, just bigger). GPT-3 has 175B. The architecture is identical — only the dimensions and number of layers change. Weight tying (sharing embedding and output weights) saves parameters and is used in almost all production LLMs.",
      }}
      takeaway={{
        bullets: [
          'The full GPT architecture: Embed → Position → N×Block → LayerNorm → Linear head',
          'The parameter distribution shows most model capacity lives in attention and FFN layers',
          'Weight tying shares the embedding matrix with the output projection, saving ~48K parameters in our model',
          'The forward pass trace shows how tensor statistics evolve through each layer — watch for vanishing or exploding values',
        ],
        prevStep: { label: 'Transformer Block', path: '/transformer-block' },
        nextStep: { label: 'Training', path: '/training', teaser: 'Teach this model to generate coherent text through gradient descent' },
      }}
    >
      <div className="card">
        <h3>Model Architecture</h3>
        <p className="card-subtitle">
          The full forward pass composes every component — from raw token IDs to next-token logits — in a single nested function.
        </p>
        <FormulaBlock
          title="GPT Forward Pass"
          formula={String.raw`\text{GPT}(x) = \text{Linear}\!\left(\text{LN}\!\left(\text{Block}_N\!\left(\cdots\text{Block}_1\!\left(\text{Embed}(x) + \text{PE}\right)\right)\right)\right)`}
          source="Radford et al. 2019 (GPT-2), Language Models are Unsupervised Multitask Learners"
          sourceUrl="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"
          intuition="Every GPT uses this exact pipeline — only the width (d_model) and depth (N) change between GPT-2 and GPT-4."
          terms={[
            { symbol: String.raw`x`, name: 'token ID sequence', meaning: 'Sequence of integer token IDs produced by the BPE tokenizer (length T)' },
            { symbol: String.raw`\text{Embed}(x)`, name: 'token embedding', meaning: 'Matrix multiply: looks up each token ID in E ∈ ℝ^(V×d_model) to get d_model-dimensional vectors' },
            { symbol: String.raw`\text{PE}`, name: 'positional embedding', meaning: 'Learned position vectors (shape T×d_model) added element-wise to tell the model token order' },
            { symbol: String.raw`\text{Block}_i`, name: 'transformer block i', meaning: 'One causal self-attention + feed-forward sublayer, each with residual connections and layer norms' },
            { symbol: 'N', name: 'number of blocks', meaning: 'Model depth — our model: 4; GPT-2 Small: 12; GPT-3: 96' },
            { symbol: String.raw`\text{LN}`, name: 'final layer norm', meaning: 'Pre-output normalisation that stabilises the magnitude of hidden states before projection' },
            { symbol: String.raw`\text{Linear}`, name: 'LM head', meaning: 'Projects each d_model vector to vocab_size logits; shared weights with Embed (weight tying)' },
          ]}
          steps={[
            { label: 'Embed tokens + positions', math: String.raw`\mathbf{h}_0 = \text{Embed}(x) + \text{PE}`, text: 'Map every token ID to a dense vector, then add a positional signal so the model knows word order' },
            { label: 'Apply N transformer blocks', math: String.raw`\mathbf{h}_i = \text{Block}_i(\mathbf{h}_{i-1})`, text: 'Each block mixes information across positions (attention) and processes each position independently (FFN)' },
            { label: 'Normalise and project', math: String.raw`\text{logits} = \text{Linear}(\text{LN}(\mathbf{h}_N))`, text: 'Stabilise activations, then project to vocab size — apply softmax to get next-token probabilities' },
          ]}
        />
        <div className="explanation">
          <strong>Architecture flow:</strong><br />
          1. <strong>Token Embedding</strong>: token IDs → d_model vectors<br />
          2. <strong>Position Embedding</strong>: add learned position vectors<br />
          3. <strong>Dropout</strong>: regularization<br />
          4. <strong>N × Transformer Blocks</strong>: self-attention + FFN (repeated)<br />
          5. <strong>Final LayerNorm</strong>: normalize before output<br />
          6. <strong>Linear Head</strong>: project to vocabulary size → logits<br /><br />
          The same weight matrix is shared between the token embedding and the output head
          (weight tying) — this reduces parameters and improves performance.
        </div>
      </div>

      {arch && (
        <>
          {/* Config table */}
          <div className="card">
            <h3>Model Configuration</h3>
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-value">{arch.config.d_model}</div>
                <div className="stat-label">d_model</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{arch.config.n_heads}</div>
                <div className="stat-label">Heads</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{arch.config.n_layers}</div>
                <div className="stat-label">Layers</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{arch.config.context_len}</div>
                <div className="stat-label">Context Len</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{arch.config.vocab_size.toLocaleString()}</div>
                <div className="stat-label">Vocab Size</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{arch.parameters.total.toLocaleString()}</div>
                <div className="stat-label">Total Params</div>
              </div>
            </div>
          </div>

          {/* Parameter pie chart */}
          <div className="card">
            <h3>Parameter Distribution</h3>
            <p className="card-subtitle">
              Where are the parameters? Most are in the transformer blocks' attention & FFN layers.
            </p>
            <div className="plot-container">
              <Plot
                data={[
                  {
                    labels: Object.keys(arch.parameters.breakdown),
                    values: Object.values(arch.parameters.breakdown),
                    type: 'pie',
                    hole: 0.4,
                    textinfo: 'label+percent',
                    textfont: { color: '#e6edf3', size: 12 },
                    marker: {
                      colors: ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#bc8cff', '#f0883e'],
                    },
                  },
                ]}
                layout={{
                  height: 400,
                  margin: { t: 20, b: 20, l: 20, r: 20 },
                  paper_bgcolor: 'transparent',
                  font: { color: '#8b949e' },
                  showlegend: true,
                  legend: { font: { color: '#8b949e' } },
                }}
                config={{ responsive: true }}
                style={{ width: '100%' }}
              />
            </div>
          </div>

          {/* Layer breakdown */}
          <div className="card">
            <h3>Layer-by-Layer Breakdown</h3>
            <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
              <table className="merge-table">
                <thead>
                  <tr>
                    <th>Layer</th>
                    <th>Type</th>
                    <th>Parameters</th>
                    <th>Details</th>
                  </tr>
                </thead>
                <tbody>
                  {arch.layers.map((layer: any, i: number) => (
                    <tr key={i}>
                      <td>{layer.name}</td>
                      <td>{layer.type}</td>
                      <td>
                        {typeof layer.params === 'number'
                          ? layer.params.toLocaleString()
                          : layer.params}
                      </td>
                      <td style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
                        {layer.shape || (layer.sublayers && layer.sublayers.join(' → '))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Forward pass trace */}
      <div className="card">
        <h3>Forward Pass Trace</h3>
        <p className="card-subtitle">
          Watch how tensor shapes and statistics change as data flows through the model.
        </p>
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
          <button onClick={runTrace} disabled={loading}>
            {loading ? 'Running…' : 'Run Forward Pass'}
          </button>
        </div>
      </div>

      {trace && (
        <>
          <div className="card">
            <h3>Tensor Flow</h3>
            <div className="plot-container">
              <Plot
                data={[
                  {
                    x: trace.trace.map((t: any) => t.stage),
                    y: trace.trace.map((t: any) => t.std),
                    name: 'Std Dev',
                    type: 'scatter',
                    mode: 'lines+markers',
                    marker: { color: '#58a6ff', size: 8 },
                    line: { color: '#58a6ff' },
                  },
                ]}
                layout={{
                  height: 300,
                  margin: { t: 20, b: 80, l: 60, r: 20 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { color: '#8b949e' },
                  xaxis: { tickangle: -30 },
                  yaxis: { title: 'Std Dev', gridcolor: '#21262d' },
                }}
                config={{ responsive: true }}
                style={{ width: '100%' }}
              />
            </div>
          </div>

          {/* Top predictions */}
          {trace.next_token_predictions && (
            <div className="card">
              <h3>Next Token Predictions</h3>
              <p className="card-subtitle">
                The model's probability distribution over the vocabulary for the next token
                after "{trace.token_labels[trace.token_labels.length - 1]}".
              </p>
              <div className="plot-container">
                <Plot
                  data={[
                    {
                      x: trace.next_token_predictions.map(
                        (t: any) => `'${t.text.replace(/\n/g, '↵')}'`
                      ),
                      y: trace.next_token_predictions.map((t: any) => t.probability),
                      type: 'bar',
                      marker: { color: '#3fb950' },
                    },
                  ]}
                  layout={{
                    height: 300,
                    margin: { t: 10, b: 80, l: 60, r: 20 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#8b949e' },
                    xaxis: { tickangle: -45, title: 'Token' },
                    yaxis: { title: 'Probability', gridcolor: '#21262d' },
                  }}
                  config={{ responsive: true }}
                  style={{ width: '100%' }}
                />
              </div>
            </div>
          )}
        </>
      )}

      <CodeBlock
        title="Full GPT Model — Python Implementation"
        collapsible
        code={`import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4,
                 n_layers=4, context_len=128, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, context_len, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.token_embed.weight = self.lm_head.weight

    def forward(self, token_ids, targets=None):
        B, T = token_ids.shape
        tok = self.token_embed(token_ids)          # (B, T, d)
        pos = self.pos_embed(torch.arange(T))      # (T, d)
        x = self.dropout(tok + pos)                 # (B, T, d)

        for block in self.blocks:
            x = block(x)                            # (B, T, d)

        x = self.ln_final(x)                        # (B, T, d)
        logits = self.lm_head(x)                    # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss`}
      />
    </PageLayout>
  );
}
