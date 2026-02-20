import { useState, useRef, useCallback } from 'react';
import Plot from 'react-plotly.js';
import PageLayout from '../components/layout/PageLayout';
import FormulaBlock from '../components/shared/FormulaBlock';
import CodeBlock from '../components/shared/CodeBlock';
import { streamTraining, stopTraining, getLossHistory } from '../api/client';
import { useAppStore } from '../store/useAppStore';

export default function TrainingPage() {
  const trainConfig = useAppStore(s => s.trainConfig);
  const setPhase = useAppStore(s => s.setPhase);
  const setTrainingProgress = useAppStore(s => s.setTrainingProgress);
  const [maxSteps, setMaxSteps] = useState(trainConfig.max_steps);
  const [batchSize, setBatchSize] = useState(trainConfig.batch_size);
  const [lr, setLr] = useState(trainConfig.learning_rate);
  const [isTraining, setIsTraining] = useState(false);
  const [losses, setLosses] = useState<any[]>([]);
  const [samples, setSamples] = useState<any[]>([]);
  const [latestStep, setLatestStep] = useState<any>(null);
  const cancelRef = useRef<(() => void) | null>(null);

  const startTraining = useCallback(() => {
    setIsTraining(true);
    setPhase('training');
    setLosses([]);
    setSamples([]);

    const cancel = streamTraining(
      {
        max_steps: maxSteps,
        batch_size: batchSize,
        learning_rate: lr,
        log_interval: 50,
        sample_interval: 500,
      },
      (step) => {
        setLosses((prev) => [...prev, step]);
        setLatestStep(step);
        if (step.step != null && step.loss != null) setTrainingProgress(step.step, step.loss);
      },
      (sample) => {
        setSamples((prev) => [...prev, sample]);
      },
      () => {
        setIsTraining(false);
        setPhase('idle');
      },
      (err) => {
        console.error('Training error:', err);
        setIsTraining(false);
        setPhase('idle');
      },
    );

    cancelRef.current = cancel;
  }, [maxSteps, batchSize, lr]);

  const handleStop = async () => {
    try {
      await stopTraining();
    } catch { /* ignore */ }
    cancelRef.current?.();
    setIsTraining(false);
    setPhase('idle');
  };

  const loadHistory = async () => {
    try {
      const data = await getLossHistory();
      if (data.loss_history?.length) setLosses(data.loss_history);
      if (data.samples?.length) setSamples(data.samples);
    } catch { /* ignore */ }
  };

  return (
    <PageLayout
      title="Training"
      subtitle="Train the GPT model on Tiny Shakespeare — watch the loss decrease in real time and see generated samples improve step by step."
      step={8}
      pipelineStep={8}
      learningContext={{
        prerequisites: [
          { label: 'Full Model (Step 7)', path: '/full-model' },
        ],
        plainEnglish: "Training is how the model learns. We show it millions of text examples and say 'predict the next word.' When it gets it wrong, we adjust all ~550K parameters slightly in the right direction. Repeat thousands of times. The loss number going down means the model is getting better at prediction.",
        realWorld: "GPT-3 was trained on 300 billion tokens costing ~$4.6M in compute. Our model trains on ~1M characters of Shakespeare in seconds. The training loop is identical — batch of text, forward pass, compute loss, backward pass, update weights. Scale is the only difference.",
      }}
      takeaway={{
        bullets: [
          'The training loop: forward pass → compute cross-entropy loss → backward pass → optimizer step',
          'Cross-entropy loss measures how far the model\'s predicted distribution is from the actual next token',
          'Perplexity (e^loss) gives an intuitive measure: "the model is as confused as choosing between N options"',
          'Learning rate scheduling (warmup + cosine decay) is critical for stable training',
          'Watch sample quality improve: early = random characters → middle = word-like → late = pseudo-Shakespeare',
        ],
        prevStep: { label: 'Full Model', path: '/full-model' },
        nextStep: { label: 'Backpropagation', path: '/backpropagation', teaser: 'Understand HOW parameter updates are computed via the chain rule' },
        tryThis: "Train for 500 steps, note the loss, then train for 2000 steps. Watch how sample quality improves — early samples are random characters, later samples form words, then recognizable sentence structures.",
      }}
    >
      <div className="card">
        <h3>How Training Works</h3>
        <p className="card-subtitle">
          The model learns by predicting the next token, computing how wrong it was (loss),
          and adjusting weights to do better next time.
        </p>
        <FormulaBlock
          title="Cross-Entropy Loss (Next-Token Prediction)"
          formula={String.raw`\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_{t+1} \mid x_1, \ldots, x_t)`}
          source="Standard cross-entropy for language modeling"
          sourceUrl="https://arxiv.org/abs/1706.03762"
          intuition="For each position, measure how surprised the model is by the actual next token — high probability = low loss. Average over all positions in the batch."
          terms={[
            { symbol: '\\mathcal{L}', name: 'Loss', meaning: 'The training objective to minimize — lower means the model predicts the next token more accurately' },
            { symbol: 'N', name: 'Number of predictions', meaning: 'Total number of next-token predictions in the batch (batch_size × sequence_length)' },
            { symbol: 'P(x_{t+1} \\mid x_1, \\ldots, x_t)', name: 'Predicted probability', meaning: 'The probability the model assigns to the correct next token, given all previous tokens' },
            { symbol: '\\log', name: 'Natural logarithm', meaning: 'Converts probability (0–1) to log space — penalizes very wrong predictions much more than slightly wrong ones' },
            { symbol: '-', name: 'Negation', meaning: 'log(p) is negative when p < 1, so negation makes the loss positive — higher probability = lower loss' },
          ]}
          steps={[
            { label: 'Forward pass produces logits', math: String.raw`z = \text{model}(x_1, \ldots, x_t) \in \mathbb{R}^{|V|}`, text: 'The model outputs a raw score (logit) for every token in the vocabulary at each position.' },
            { label: 'Softmax converts to probabilities', math: String.raw`P(x_{t+1}=w) = \frac{e^{z_w}}{\sum_{j=1}^{|V|} e^{z_j}}`, text: 'Softmax normalizes logits to a probability distribution over the vocabulary.' },
            { label: 'Cross-entropy measures surprise', math: String.raw`\ell_t = -\log P(x_{t+1})`, text: 'If the model assigns 90% probability to the correct token, loss is -log(0.9) ≈ 0.11. If only 1%, loss is -log(0.01) ≈ 4.6.' },
            { label: 'Average over all positions', math: String.raw`\mathcal{L} = \frac{1}{N}\sum_{t=1}^{N} \ell_t`, text: 'Average the per-position losses. A random model on vocab=756 has loss ≈ ln(756) ≈ 6.6.' },
          ]}
        />
        <FormulaBlock
          title="Perplexity"
          formula={String.raw`\text{Perplexity} = e^{\mathcal{L}}`}
          source="Standard language model evaluation metric"
          sourceUrl="https://en.wikipedia.org/wiki/Perplexity"
          intuition="Perplexity is the effective number of choices the model is 'confused' between at each step — a perplexity of 10 means the model is as uncertain as randomly picking from 10 options."
          terms={[
            { symbol: 'e^{\\mathcal{L}}', name: 'Exponential of loss', meaning: 'Converts cross-entropy (log scale) back to a count — easier to interpret than raw loss' },
            { symbol: '\\mathcal{L}', name: 'Cross-entropy loss', meaning: 'The average negative log probability from the formula above' },
          ]}
        />
      </div>

      {/* Training controls */}
      <div className="card">
        <h3>Training Configuration</h3>
        <div className="controls">
          <div className="control-group">
            <label>Max Steps</label>
            <input
              type="number"
              value={maxSteps}
              onChange={(e) => setMaxSteps(Number(e.target.value))}
              min={100}
              max={10000}
              step={100}
              disabled={isTraining}
            />
          </div>
          <div className="control-group">
            <label>Batch Size</label>
            <input
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(Number(e.target.value))}
              min={4}
              max={128}
              disabled={isTraining}
            />
          </div>
          <div className="control-group">
            <label>Learning Rate</label>
            <input
              type="number"
              value={lr}
              onChange={(e) => setLr(Number(e.target.value))}
              min={1e-5}
              max={1e-2}
              step={1e-4}
              disabled={isTraining}
            />
          </div>
          {isTraining ? (
            <button className="danger" onClick={handleStop}>
              Stop Training
            </button>
          ) : (
            <button onClick={startTraining}>Start Training</button>
          )}
          <button className="secondary" onClick={loadHistory}>
            Load History
          </button>
        </div>
      </div>

      {/* Live stats */}
      {latestStep && (
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">{latestStep.step}</div>
            <div className="stat-label">Step</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{latestStep.loss}</div>
            <div className="stat-label">Loss</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{latestStep.perplexity}</div>
            <div className="stat-label">Perplexity</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{latestStep.lr}</div>
            <div className="stat-label">Learning Rate</div>
          </div>
          {latestStep.steps_per_sec && (
            <div className="stat-card">
              <div className="stat-value">{latestStep.steps_per_sec}</div>
              <div className="stat-label">Steps/sec</div>
            </div>
          )}
        </div>
      )}

      {/* Loss curve */}
      {losses.length > 0 && (
        <div className="card">
          <h3>Loss Curve</h3>
          <p className="card-subtitle">
            The loss should decrease over time — from ~{Math.log(latestStep?.perplexity ? 756 : 756).toFixed(1)} (random)
            toward &lt;2.0 (learned patterns).
          </p>
          <div className="plot-container">
            <Plot
              data={[
                {
                  x: losses.map((l) => l.step),
                  y: losses.map((l) => l.loss),
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Loss',
                  line: { color: '#58a6ff', width: 2 },
                },
              ]}
              layout={{
                height: 400,
                margin: { t: 20, b: 50, l: 60, r: 20 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#8b949e' },
                xaxis: { title: 'Step', gridcolor: '#21262d' },
                yaxis: { title: 'Loss', gridcolor: '#21262d' },
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      )}

      {/* LR schedule */}
      {losses.length > 0 && (
        <div className="card">
          <h3>Learning Rate Schedule</h3>
          <p className="card-subtitle">
            Cosine decay with linear warmup — starts slow, ramps up, then gradually decreases.
          </p>
          <div className="plot-container">
            <Plot
              data={[
                {
                  x: losses.map((l) => l.step),
                  y: losses.map((l) => l.lr),
                  type: 'scatter',
                  mode: 'lines',
                  name: 'LR',
                  line: { color: '#d29922', width: 2 },
                },
              ]}
              layout={{
                height: 250,
                margin: { t: 20, b: 50, l: 80, r: 20 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#8b949e' },
                xaxis: { title: 'Step', gridcolor: '#21262d' },
                yaxis: { title: 'Learning Rate', gridcolor: '#21262d' },
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      )}

      {/* Generated samples */}
      {samples.length > 0 && (
        <div className="card">
          <h3>Sample Generations During Training</h3>
          <p className="card-subtitle">
            Text generated by the model at different training steps — watch it improve from gibberish to Shakespeare-ish.
          </p>
          {samples.map((s, i) => (
            <div
              key={i}
              style={{
                marginBottom: '16px',
                padding: '12px',
                background: 'var(--bg-primary)',
                borderRadius: '6px',
                border: '1px solid var(--border)',
              }}
            >
              <div style={{ fontSize: '12px', color: 'var(--accent)', fontWeight: 600, marginBottom: '6px' }}>
                Step {s.step}
              </div>
              <pre style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', whiteSpace: 'pre-wrap', color: 'var(--text-secondary)' }}>
                {s.text}
              </pre>
            </div>
          ))}
        </div>
      )}

      <CodeBlock
        title="Training Loop — Python Implementation"
        collapsible
        code={`import torch

def train(model, data, steps=2000, lr=3e-4, batch_size=32):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=0.1, betas=(0.9, 0.95))

    for step in range(steps):
        # Sample random batch
        idx = torch.randint(len(data) - context_len, (batch_size,))
        x = torch.stack([data[i:i+context_len] for i in idx])
        y = torch.stack([data[i+1:i+context_len+1] for i in idx])

        # Forward pass
        logits, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Learning rate schedule (cosine with warmup)
        if step < 100:
            lr_t = lr * step / 100
        else:
            progress = (step - 100) / (steps - 100)
            lr_t = lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_t

        if step % 100 == 0:
            print(f"step {step}: loss={loss.item():.4f}")`}
      />
    </PageLayout>
  );
}
