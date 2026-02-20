import { useState, useRef, useCallback } from 'react';
import Plot from 'react-plotly.js';
import PageLayout from '../components/layout/PageLayout';
import FormulaBlock from '../components/shared/FormulaBlock';
import CodeBlock from '../components/shared/CodeBlock';
import { streamGenerate, generateSync } from '../api/client';
import { useAppStore } from '../store/useAppStore';
import StepAnimator from '../components/shared/StepAnimator';

export default function GenerationPage() {
  const setPhase = useAppStore(s => s.setPhase);
  const [prompt, setPrompt] = useState('ROMEO:\nO, ');
  const [maxTokens, setMaxTokens] = useState(200);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(50);
  const [topP, setTopP] = useState(0.9);
  const [strategy, setStrategy] = useState<'top_k' | 'nucleus' | 'greedy'>('top_k');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedText, setGeneratedText] = useState('');
  const [tokenProbs, setTokenProbs] = useState<any[]>([]);
  const [currentProbs, setCurrentProbs] = useState<any>(null);
  const cancelRef = useRef<(() => void) | null>(null);

  const startGenerate = useCallback(() => {
    setIsGenerating(true);
    setPhase('generating');
    setGeneratedText(prompt);
    setTokenProbs([]);
    setCurrentProbs(null);

    const cancel = streamGenerate(
      {
        prompt,
        max_tokens: maxTokens,
        temperature,
        top_k: strategy === 'top_k' ? topK : 0,
        top_p: strategy === 'nucleus' ? topP : 1.0,
        strategy,
      },
      (token) => {
        setGeneratedText((prev) => prev + token.token_text);
        if (token.top_predictions) {
          setTokenProbs((prev) => [...prev, token]);
          setCurrentProbs(token);
        }
      },
      () => {
        setIsGenerating(false);
        setPhase('idle');
      },
      (err) => {
        console.error('Generation error:', err);
        setIsGenerating(false);
        setPhase('idle');
      },
    );

    cancelRef.current = cancel;
  }, [prompt, maxTokens, temperature, topK, topP, strategy]);

  const stopGenerate = () => {
    cancelRef.current?.();
    setIsGenerating(false);
    setPhase('idle');
  };

  const generateAll = async () => {
    setIsGenerating(true);
    setPhase('generating');
    setGeneratedText(prompt);
    setTokenProbs([]);
    setCurrentProbs(null);
    try {
      const data = await generateSync({
        prompt,
        max_tokens: maxTokens,
        temperature,
        top_k: strategy === 'top_k' ? topK : 0,
        top_p: strategy === 'nucleus' ? topP : 1.0,
        strategy,
      });
      setGeneratedText(data.generated_text || '');
      if (data.tokens?.length) {
        setTokenProbs(data.tokens);
        setCurrentProbs(data.tokens[data.tokens.length - 1]);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsGenerating(false);
      setPhase('idle');
    }
  };

  return (
    <PageLayout
      title="Text Generation"
      subtitle="Watch your trained GPT model generate Shakespeare-style text one token at a time."
      step={10}
      pipelineStep={10}
      learningContext={{
        prerequisites: [
          { label: 'Training (Step 8)', path: '/training' },
          { label: 'Full Model (Step 7)', path: '/full-model' },
        ],
        plainEnglish: "Generation is the payoff — making the model write. We give it a starting prompt, it predicts one token, we append that token and predict again. Each prediction is a probability distribution: the model says 'there\'s a 30% chance the next word is the, 15% chance it\'s a, 8% chance it\'s he...' We pick from these probabilities using different strategies.",
        realWorld: "ChatGPT uses nucleus (top-p) sampling with temperature. The temperature slider here works exactly the same way. Low temperature (0.3) = factual/repetitive. High temperature (1.5) = creative/chaotic. Most production APIs default to temperature=0.7, top_p=0.9.",
      }}
      takeaway={{
        bullets: [
          'Autoregressive generation: predict one token → append → predict again → repeat',
          'Temperature scales the logits before softmax: lower = sharper (confident), higher = flatter (random)',
          'Greedy = always pick the highest probability token (deterministic but repetitive)',
          'Top-k = sample from the k most likely tokens; Top-p (nucleus) = sample from the smallest set covering p probability mass',
          'You\'ve now seen the complete LLM pipeline: raw text → tokens → vectors → attention → training → generation',
        ],
        prevStep: { label: 'Backpropagation', path: '/backpropagation' },
        tryThis: "Compare the three strategies: set temperature to 0.5, try greedy (always picks the same tokens), then top-k with k=10, then nucleus with p=0.9. Notice how each produces different levels of diversity in the output.",
      }}
    >
      <div className="card">
        <h3>How Text Generation Works</h3>
        <p className="card-subtitle">
          The model generates text autoregressively — it predicts one token, appends it to the input,
          and repeats. Sampling strategies control the creativity vs. coherence trade-off.
        </p>
        <FormulaBlock
          title="Temperature Sampling"
          formula={String.raw`P(x_{t+1} \mid x_{\le t}) = \text{softmax}\!\left(\dfrac{\text{logits}_t}{\tau}\right)`}
          source="Fan et al. 2018, arXiv:1805.04833; Ackley et al. 1985 (temperature in softmax)"
          sourceUrl="https://arxiv.org/abs/1805.04833"
          intuition="Dividing logits by τ stretches or squashes the probability distribution before sampling — the single dial that controls creativity vs. coherence."
          terms={[
            { symbol: String.raw`x_{t+1}`, name: 'next token', meaning: 'The token to be sampled (appended to the context for the next step)' },
            { symbol: String.raw`\text{logits}_t`, name: 'raw logits', meaning: 'Unnormalised scores output by the LM head for every vocabulary token at step t' },
            { symbol: String.raw`\tau`, name: 'temperature', meaning: 'Scaling factor. τ=1: unchanged; τ→0: argmax (greedy); τ→∞: uniform random' },
            { symbol: String.raw`\text{softmax}(z)_i`, name: 'softmax', meaning: 'Converts logits to probabilities: exp(z_i) / Σ exp(z_j)' },
          ]}
          steps={[
            { label: 'Scale logits by 1/τ', math: String.raw`z_t' = \text{logits}_t / \tau`, text: 'τ < 1 amplifies differences between logits (sharper), τ > 1 compresses them (flatter)' },
            { label: 'Convert to probabilities', math: String.raw`P = \text{softmax}(z_t')`, text: 'Exponentiate and normalise — now every entry is in (0,1) and they sum to 1' },
            { label: 'Sample next token', math: String.raw`x_{t+1} \sim P`, text: 'Draw from the distribution; optionally apply Top-k or Top-p filtering first' },
          ]}
        />
      </div>

      {/* Controls */}
      <div className="card">
        <h3>Generation Settings</h3>
        <div style={{ marginBottom: '12px' }}>
          <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, marginBottom: '4px', color: 'var(--text-secondary)' }}>
            Prompt
          </label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            disabled={isGenerating}
            style={{
              width: '100%',
              background: 'var(--bg-primary)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              padding: '8px 12px',
              fontFamily: 'var(--font-mono)',
              fontSize: '13px',
              resize: 'vertical',
            }}
          />
        </div>

        <div className="controls">
          <div className="control-group">
            <label>Strategy</label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as any)}
              disabled={isGenerating}
            >
              <option value="top_k">Top-k</option>
              <option value="nucleus">Top-p (nucleus)</option>
              <option value="greedy">Greedy</option>
            </select>
          </div>
          <div className="control-group">
            <label>Max Tokens</label>
            <input
              type="number"
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
              min={10}
              max={500}
              disabled={isGenerating}
            />
          </div>
          <div className="control-group">
            <label>Temperature: {temperature.toFixed(2)}</label>
            <input
              type="range"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              min={0.1}
              max={2.0}
              step={0.05}
              disabled={isGenerating}
            />
          </div>
          {strategy === 'top_k' && (
            <div className="control-group">
              <label>Top-k: {topK}</label>
              <input
                type="range"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                min={1}
                max={200}
                disabled={isGenerating}
              />
            </div>
          )}
          {strategy === 'nucleus' && (
            <div className="control-group">
              <label>Top-p: {topP.toFixed(2)}</label>
              <input
                type="range"
                value={topP}
                onChange={(e) => setTopP(Number(e.target.value))}
                min={0.1}
                max={1.0}
                step={0.05}
                disabled={isGenerating}
              />
            </div>
          )}
        </div>

        <div style={{ display: 'flex', gap: '8px', marginTop: '16px', alignItems: 'center' }}>
          {isGenerating ? (
            <button className="danger" onClick={stopGenerate}>
              Stop
            </button>
          ) : (
            <>
              <button onClick={startGenerate}>
                Generate (Streaming)
              </button>
              <button className="secondary" onClick={generateAll}>
                Generate (All at Once)
              </button>
            </>
          )}
        </div>
      </div>

      {/* Generated output */}
      {generatedText && (
        <div className="card">
          <h3>Generated Text</h3>
          <div className="generation-output">
            <span style={{ color: 'var(--accent)' }}>
              {prompt}
            </span>
            <span>
              {generatedText.slice(prompt.length)}
            </span>
            {isGenerating && <span className="cursor-blink">▌</span>}
          </div>
        </div>
      )}

      {/* Step-through token inspector */}
      {tokenProbs.length > 0 && (
        <div className="card">
          <h3>Step-by-Step Token Generation</h3>
          <p className="card-subtitle">
            Walk through each generated token with its probability and alternatives.
          </p>
          <StepAnimator
            steps={tokenProbs}
            stepLabel="Token"
            autoPlaySpeed={500}
            renderSteps={(visible, currentIndex) => (
              <div>
                {/* Token sequence so far */}
                <div className="token-display" style={{ marginBottom: 12 }}>
                  {visible.map((tok: any, i: number) => (
                    <span key={i}
                      className={`token token-${i % 6} ${i === currentIndex ? '' : ''}`}
                      style={i === currentIndex ? { outline: '2px solid var(--accent)', outlineOffset: 1 } : {}}
                      title={`Step ${i + 1}`}>
                      {tok.token_text?.replace(/ /g, '·').replace(/\n/g, '↵') || '?'}
                    </span>
                  ))}
                </div>
                {/* Current token details */}
                {visible[currentIndex] && (
                  <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 13 }}>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Chosen: </span>
                      <span style={{ color: 'var(--accent)', fontFamily: 'var(--font-mono)' }}>
                        "{visible[currentIndex].token_text}"
                      </span>
                    </div>
                    {visible[currentIndex].top_predictions?.slice(0, 5).map((p: any, j: number) => (
                      <div key={j} style={{ color: p.text === visible[currentIndex].token_text ? 'var(--green)' : 'var(--text-secondary)' }}>
                        "{p.text}" → {(p.probability * 100).toFixed(1)}%
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          />
        </div>
      )}

      {/* Token probability distribution */}
      {currentProbs && (
        <div className="card">
          <h3>Last Token Probability Distribution</h3>
          <p className="card-subtitle">
            Top predicted tokens for the most recently generated position.
            The selected token is highlighted.
          </p>
          <div className="plot-container">
            <Plot
              data={[
                {
                  x: currentProbs.top_predictions.map((p: any) => p.text),
                  y: currentProbs.top_predictions.map((p: any) => p.probability),
                  type: 'bar',
                  marker: {
                    color: currentProbs.top_predictions.map((p: any) =>
                      p.text === currentProbs.token_text ? '#58a6ff' : '#30363d'
                    ),
                  },
                },
              ]}
              layout={{
                height: 350,
                margin: { t: 20, b: 80, l: 60, r: 20 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#8b949e' },
                xaxis: {
                  title: 'Token',
                  gridcolor: '#21262d',
                  tickangle: -45,
                  tickfont: { family: 'monospace' },
                },
                yaxis: {
                  title: 'Probability',
                  gridcolor: '#21262d',
                  range: [0, 1],
                },
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      )}

      {/* Entropy over time */}
      {tokenProbs.length > 2 && (
        <div className="card">
          <h3>Model Confidence Over Time</h3>
          <p className="card-subtitle">
            The probability assigned to each selected token. Higher = model was more confident.
          </p>
          <div className="plot-container">
            <Plot
              data={[
                {
                  x: tokenProbs.map((_: any, i: number) => i),
                  y: tokenProbs.map((t: any) => {
                    const chosen = t.top_predictions?.find((p: any) => p.text === t.token_text);
                    return chosen?.probability ?? 0;
                  }),
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'P(chosen token)',
                  line: { color: '#58a6ff', width: 1.5 },
                  marker: { size: 3 },
                },
              ]}
              layout={{
                height: 250,
                margin: { t: 20, b: 40, l: 60, r: 20 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#8b949e' },
                xaxis: { title: 'Generation Step', gridcolor: '#21262d' },
                yaxis: { title: 'Probability', gridcolor: '#21262d', range: [0, 1] },
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      )}

      <div className="card">
        <h3>Sampling Strategies Compared</h3>
        <p className="card-subtitle">
          Three different rules for which token to pick next — each trades off determinism, diversity, and adaptability.
        </p>
        <FormulaBlock
          title="Greedy Decoding"
          formula={String.raw`x_{t+1} = \arg\max_{x \in V}\, P(x \mid x_{\le t})`}
          intuition="Always pick the single most probable next token — fast and deterministic, but prone to repetitive loops."
          terms={[
            { symbol: String.raw`\arg\max`, name: 'argmax', meaning: 'Returns the token x that maximises the probability — no randomness involved' },
            { symbol: String.raw`P(x \mid x_{\le t})`, name: 'conditional probability', meaning: 'Model probability of token x given everything generated so far' },
          ]}
          steps={[
            { label: 'Compute probabilities', math: String.raw`P = \text{softmax}(\text{logits}/\tau)`, text: 'With τ→0, this distribution becomes a one-hot on the argmax token' },
            { label: 'Select max', math: String.raw`x_{t+1} = \arg\max P`, text: 'No sampling — always deterministic. Identical prompt always gives identical output' },
          ]}
        />
        <FormulaBlock
          title="Top-k Sampling"
          formula={String.raw`P_k(x) = \begin{cases} \dfrac{P(x)}{\displaystyle\sum_{x' \in V_k} P(x')} & x \in V_k \\[6pt] 0 & \text{otherwise} \end{cases}`}
          source="Fan et al. 2018, arXiv:1805.04833"
          sourceUrl="https://arxiv.org/abs/1805.04833"
          intuition="Zero out the tail of the distribution, renormalise over the top k tokens, then sample — limits wild picks while preserving randomness."
          terms={[
            { symbol: String.raw`V_k`, name: 'top-k set', meaning: 'The k tokens with highest probability before renormalisation (k is a fixed hyperparameter)' },
            { symbol: String.raw`P(x)`, name: 'original probability', meaning: 'Softmax probability for token x before truncation' },
            { symbol: 'k', name: 'k (hyperparameter)', meaning: 'Controls diversity: k=1 is greedy; k=vocab is pure sampling. Typical value: 50' },
          ]}
          steps={[
            { label: 'Find top-k tokens', text: 'Sort all vocab tokens by probability, keep the k largest — discard the rest (set to −∞ before softmax)' },
            { label: 'Renormalise', math: String.raw`P_k(x) = P(x) \,/\, \textstyle\sum_{x' \in V_k} P(x')`, text: 'Redistribute the zeroed-out probability mass so the k tokens sum to 1' },
            { label: 'Sample', math: String.raw`x_{t+1} \sim P_k`, text: 'Draw from the truncated distribution' },
          ]}
        />
        <FormulaBlock
          title="Top-p (Nucleus) Sampling"
          formula={String.raw`V_p = \min\!\left\{\, V' \subseteq V : \sum_{x \in V'} P(x) \geq p \,\right\}`}
          source="Holtzman et al. 2020, arXiv:1904.09751"
          sourceUrl="https://arxiv.org/abs/1904.09751"
          intuition="Use the smallest set of tokens that together cover at least p of the probability mass — the nucleus adapts its size to match distribution sharpness."
          terms={[
            { symbol: String.raw`V_p`, name: 'nucleus', meaning: 'The minimal set of highest-probability tokens whose cumulative probability first reaches p' },
            { symbol: 'p', name: 'p (hyperparameter)', meaning: 'Probability threshold. Typical value: 0.9 — keeps tokens covering 90% of the mass' },
            { symbol: String.raw`V'`, name: 'candidate set', meaning: 'Any subset of the vocabulary — minimised to find the tightest nucleus' },
          ]}
          steps={[
            { label: 'Sort by probability', text: 'Rank all vocabulary tokens from most to least probable' },
            { label: 'Build nucleus greedily', text: 'Add tokens one at a time (highest-first) until cumulative probability ≥ p — that minimal set is V_p' },
            { label: 'Renormalise and sample', math: String.raw`x_{t+1} \sim P(x)/\textstyle\sum_{x' \in V_p}P(x')`, text: 'When the model is confident, V_p is small (peaked distribution); when uncertain, it grows (flat distribution)' },
          ]}
        />
      </div>

      <CodeBlock
        title="Text Generation — Python Implementation"
        collapsible
        code={`@torch.no_grad()
def generate(model, prompt_ids, max_tokens=200,
             temperature=0.8, top_k=50):
    """Autoregressive generation with top-k sampling."""
    ids = prompt_ids.unsqueeze(0)  # (1, seq_len)

    for _ in range(max_tokens):
        # Crop to context window
        context = ids[:, -model.config.context_len:]

        # Forward pass → logits for next token
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature  # (1, vocab)

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')

        # Convert to probabilities and sample
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        # Append and continue
        ids = torch.cat([ids, next_id], dim=1)

    return ids[0]  # Generated token sequence`}
      />
    </PageLayout>
  );
}
