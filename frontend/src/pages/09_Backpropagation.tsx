import { useState } from 'react';
import Plot from 'react-plotly.js';
import PageLayout from '../components/layout/PageLayout';
import FormulaBlock from '../components/shared/FormulaBlock';
import CodeBlock from '../components/shared/CodeBlock';
import { getGradientDemo, getGradientFlow } from '../api/client';
import { useAppStore } from '../store/useAppStore';

export default function BackpropagationPage() {
  const sharedText = useAppStore(s => s.sharedText);
  const [gradDemo, setGradDemo] = useState<any>(null);
  const [gradFlow, setGradFlow] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const runGradDemo = async () => {
    setLoading(true);
    try {
      const data = await getGradientDemo();
      setGradDemo(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const runGradFlow = async () => {
    setLoading(true);
    try {
      const raw = await getGradientFlow(sharedText || 'First Citizen:\nBefore we proceed any further, hear me speak.');
      setGradFlow({
        layers: raw.gradient_flow.map((g: any) => g.name),
        avg_grads: raw.gradient_flow.map((g: any) => g.mean_grad),
        max_grads: raw.gradient_flow.map((g: any) => g.max_grad),
        loss: raw.loss,
        explanation: raw.explanation,
      });
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  return (
    <PageLayout
      title="Backpropagation"
      subtitle="Understanding how gradients flow through the transformer — the chain rule in action."
      step={9}
      pipelineStep={9}
      learningContext={{
        prerequisites: [
          { label: 'Training (Step 8)', path: '/training' },
          { label: 'Transformer Block (Step 6)', path: '/transformer-block' },
        ],
        plainEnglish: "Backpropagation answers: 'How should I change each of the ~550K parameters to make the model slightly better?' It works backward through the model — from the loss to each weight — computing how much each weight contributed to the error. Then we nudge each weight in the direction that reduces the error.",
        realWorld: "PyTorch handles backprop automatically with loss.backward(). The manual chain rule computation shown here is exactly what autograd does under the hood. Understanding it helps debug vanishing/exploding gradients — a common failure mode in deep learning. The AdamW optimizer shown here is the same one used to train GPT-3.",
      }}
      takeaway={{
        bullets: [
          'The chain rule decomposes ∂L/∂w into a product of local gradients through each layer',
          'Residual connections create a "gradient highway" — the identity term means gradients can skip layers',
          'AdamW tracks running averages of gradients (momentum) and squared gradients (adaptive learning rate)',
          'Gradient clipping prevents exploding gradients from destabilizing training',
          'The gradient flow visualization shows whether gradients are healthy (similar magnitude) or vanishing (shrinking across layers)',
        ],
        prevStep: { label: 'Training', path: '/training' },
        nextStep: { label: 'Text Generation', path: '/generation', teaser: 'Use the trained model to actually generate new text' },
      }}
    >
      <div className="card">
        <h3>The Chain Rule</h3>
        <p className="card-subtitle">
          Backpropagation computes ∂Loss/∂w for every weight w by applying the chain rule
          from the loss back through each layer.
        </p>
        <FormulaBlock
          title="Chain Rule (Backpropagation)"
          formula={String.raw`\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial z_n} \cdot \frac{\partial z_n}{\partial z_{n-1}} \cdots \frac{\partial z_1}{\partial w}`}
          source="Rumelhart, Hinton & Williams 1986"
          sourceUrl="https://en.wikipedia.org/wiki/Backpropagation"
          intuition="To find how much a single weight affects the final loss, trace backward through every layer — multiplying local gradients along the way."
          terms={[
            { symbol: '\\partial \\mathcal{L} / \\partial w', name: 'Gradient of loss w.r.t. weight', meaning: 'How much the loss changes when this specific weight is nudged slightly — this tells the optimizer which direction to update' },
            { symbol: '\\partial \\mathcal{L} / \\partial z_n', name: 'Output gradient', meaning: 'Gradient at the final layer — for softmax + cross-entropy, this simplifies to (predicted − actual)' },
            { symbol: '\\partial z_k / \\partial z_{k-1}', name: 'Local Jacobian', meaning: 'How layer k\'s output changes when its input changes — a matrix for each layer' },
            { symbol: 'w', name: 'A weight parameter', meaning: 'Any single trainable parameter in the model (there are ~550K of them)' },
          ]}
          steps={[
            { label: 'Start from the loss', math: String.raw`\nabla_{\text{output}} = \frac{\partial \mathcal{L}}{\partial z_n}`, text: 'Compute the gradient at the model output. For cross-entropy + softmax this is simply: softmax(logits) − one_hot(target).' },
            { label: 'Propagate backward through each layer', math: String.raw`\nabla_{k-1} = \nabla_k \cdot \frac{\partial z_k}{\partial z_{k-1}}`, text: 'At each layer, multiply the incoming gradient by the local derivative. This is why it\'s called "backpropagation" — gradients flow backward.' },
            { label: 'Accumulate weight gradients', math: String.raw`\frac{\partial \mathcal{L}}{\partial W_k} = \nabla_k \cdot \frac{\partial z_k}{\partial W_k}`, text: 'At each layer, also compute how the loss changes with respect to that layer\'s weights. These gradients are stored for the optimizer.' },
          ]}
        />
      </div>

      <div className="card">
        <h3>Gradient Flow Through a Transformer Block</h3>
        <FormulaBlock
          title="Residual Connection Gradients"
          formula={String.raw`\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial \text{out}} \cdot \left(I + \frac{\partial \text{FFN}(\text{LN}(h))}{\partial h}\right) \cdot \left(I + \frac{\partial \text{Attn}(\text{LN}(x))}{\partial x}\right)`}
          source="Derived from the pre-norm transformer block"
          sourceUrl="https://arxiv.org/abs/1706.03762"
          intuition="The identity matrices (I) in the gradient formula mean gradients can flow straight through via the skip connection — even if the attention or FFN gradients vanish, the signal survives."
          terms={[
            { symbol: 'I', name: 'Identity matrix', meaning: 'Represents the residual (skip) connection — gradients pass through unchanged, creating a "gradient highway"' },
            { symbol: '\\partial \\mathcal{L} / \\partial \\text{out}', name: 'Downstream gradient', meaning: 'Gradient flowing back from later layers (or the loss)' },
            { symbol: '\\partial \\text{Attn} / \\partial x', name: 'Attention Jacobian', meaning: 'How attention output changes w.r.t. its input — can be small, but the +I rescues the gradient' },
            { symbol: '\\partial \\text{FFN} / \\partial h', name: 'FFN Jacobian', meaning: 'How FFN output changes w.r.t. its input — similarly, +I ensures gradients don\'t vanish' },
          ]}
        />
      </div>

      {/* Gradient Demo */}
      <div className="card">
        <h3>Manual vs. Autograd Gradients</h3>
        <p className="card-subtitle">
          We compute gradients by hand for a simple function and verify against PyTorch autograd.
          This proves the chain rule works exactly as expected.
        </p>
        <button onClick={runGradDemo} disabled={loading}>
          {loading ? 'Computing...' : 'Run Gradient Demo'}
        </button>

        {gradDemo && (
          <div style={{ marginTop: '16px' }}>
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-value">{gradDemo.loss?.toFixed(4) ?? '—'}</div>
                <div className="stat-label">Cross-Entropy Loss</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{gradDemo.setup?.seq_len ?? '—'}</div>
                <div className="stat-label">Sequence Length</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{gradDemo.setup?.d_model ?? '—'}</div>
                <div className="stat-label">Model Dimension</div>
              </div>
            </div>

            <h4 style={{ marginTop: '16px' }}>Chain Rule Steps</h4>
            {gradDemo.chain_rule_steps?.map((step: string, i: number) => (
              <div
                key={i}
                style={{
                  padding: '10px 14px',
                  margin: '6px 0',
                  background: 'var(--bg-primary)',
                  borderRadius: '6px',
                  borderLeft: '3px solid var(--accent)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '13px',
                }}
              >
                {step}
              </div>
            ))}

            <h4 style={{ marginTop: '16px' }}>Gradient Norms</h4>
            {gradDemo.gradients?.gradient_stats && (
              <div className="plot-container">
                <Plot
                  data={[
                    {
                      x: Object.keys(gradDemo.gradients.gradient_stats),
                      y: Object.values(gradDemo.gradients.gradient_stats),
                      type: 'bar',
                      marker: { color: ['#58a6ff', '#f78166', '#3fb950'] },
                    },
                  ]}
                  layout={{
                    height: 300,
                    margin: { t: 20, b: 80, l: 60, r: 20 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#8b949e' },
                    xaxis: { title: 'Parameter', gridcolor: '#21262d', tickangle: -20 },
                    yaxis: { title: 'Gradient Norm', gridcolor: '#21262d' },
                  }}
                  config={{ responsive: true }}
                  style={{ width: '100%' }}
                />
              </div>
            )}

            {/* Gradient heatmaps */}
            {gradDemo.gradients?.['d_loss/d_x'] && (
              <>
                <h4 style={{ marginTop: '16px' }}>∂L/∂x (Input Gradients)</h4>
                <div className="plot-container">
                  <Plot
                    data={[
                      {
                        z: gradDemo.gradients['d_loss/d_x'],
                        type: 'heatmap',
                        colorscale: [
                          [0, '#0d1117'],
                          [0.5, '#161b22'],
                          [1, '#58a6ff'],
                        ],
                        showscale: true,
                      },
                    ]}
                    layout={{
                      height: 250,
                      margin: { t: 20, b: 40, l: 60, r: 20 },
                      paper_bgcolor: 'transparent',
                      plot_bgcolor: 'transparent',
                      font: { color: '#8b949e' },
                      xaxis: { title: 'd_model dimension' },
                      yaxis: { title: 'Sequence position' },
                    }}
                    config={{ responsive: true }}
                    style={{ width: '100%' }}
                  />
                </div>
              </>
            )}

            {gradDemo.gradients?.['d_loss/d_W_q'] && (
              <>
                <h4 style={{ marginTop: '16px' }}>∂L/∂W_q (Query Weight Gradients)</h4>
                <div className="plot-container">
                  <Plot
                    data={[
                      {
                        z: gradDemo.gradients['d_loss/d_W_q'],
                        type: 'heatmap',
                        colorscale: [
                          [0, '#0d1117'],
                          [0.5, '#161b22'],
                          [1, '#f78166'],
                        ],
                        showscale: true,
                      },
                    ]}
                    layout={{
                      height: 250,
                      margin: { t: 20, b: 40, l: 60, r: 20 },
                      paper_bgcolor: 'transparent',
                      plot_bgcolor: 'transparent',
                      font: { color: '#8b949e' },
                      xaxis: { title: 'Column' },
                      yaxis: { title: 'Row' },
                    }}
                    config={{ responsive: true }}
                    style={{ width: '100%' }}
                  />
                </div>
              </>
            )}

            <div className="explanation" style={{ marginTop: '16px' }}>
              <strong>Explanation:</strong> {gradDemo.explanation}
            </div>
          </div>
        )}
      </div>

      {/* Gradient Flow */}
      <div className="card">
        <h3>Gradient Flow Through the Model</h3>
        <p className="card-subtitle">
          Visualize the gradient norms at each layer of the trained (or untrained) model.
          Healthy training shows gradients that don't vanish or explode.
        </p>
        <button onClick={runGradFlow} disabled={loading}>
          {loading ? 'Computing...' : 'Compute Gradient Flow'}
        </button>

        {gradFlow && (
          <div style={{ marginTop: '16px' }}>
            <div className="plot-container">
              <Plot
                data={[
                  {
                    x: gradFlow.layers,
                    y: gradFlow.avg_grads,
                    name: 'Avg Gradient',
                    type: 'bar',
                    marker: { color: '#58a6ff' },
                  },
                  {
                    x: gradFlow.layers,
                    y: gradFlow.max_grads,
                    name: 'Max Gradient',
                    type: 'bar',
                    marker: { color: '#f78166', opacity: 0.5 },
                  },
                ]}
                layout={{
                  height: 400,
                  barmode: 'overlay',
                  margin: { t: 20, b: 100, l: 60, r: 20 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { color: '#8b949e', size: 10 },
                  xaxis: {
                    title: 'Layer',
                    gridcolor: '#21262d',
                    tickangle: -45,
                  },
                  yaxis: {
                    title: 'Gradient Norm',
                    gridcolor: '#21262d',
                    type: 'log',
                  },
                  legend: { x: 0.01, y: 0.99, bgcolor: 'transparent' },
                }}
                config={{ responsive: true }}
                style={{ width: '100%' }}
              />
            </div>
            <div className="explanation">
              <strong>What to look for:</strong><br />
              • Gradients should be roughly the same order of magnitude across layers<br />
              • Vanishing gradients (very small at early layers) → model can't learn deep features<br />
              • Exploding gradients (very large) → training becomes unstable<br />
              • Residual connections + LayerNorm help maintain healthy gradient flow
            </div>
          </div>
        )}
      </div>

      <CodeBlock
        title="Backpropagation — Manual Gradient Computation"
        collapsible
        code={`import numpy as np

# Manual backprop through softmax + cross-entropy
def backward_cross_entropy_softmax(logits, target):
    """
    Combined backward pass for softmax + cross-entropy.
    The gradient has a beautifully simple form:
    dL/d(logits) = softmax(logits) - one_hot(target)
    """
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    # Gradient: p - y (probability minus one-hot target)
    grad = probs.copy()
    grad[target] -= 1.0
    return grad

# Manual backprop through linear layer
def backward_linear(grad_output, x, W):
    """
    z = Wx + b
    dL/dW = grad_output @ x^T
    dL/db = grad_output
    dL/dx = W^T @ grad_output
    """
    grad_W = np.outer(grad_output, x)
    grad_b = grad_output
    grad_x = W.T @ grad_output
    return grad_W, grad_b, grad_x

# Verify against PyTorch autograd
import torch
x = torch.randn(10, requires_grad=True)
W = torch.randn(5, 10, requires_grad=True)
b = torch.randn(5, requires_grad=True)
target = torch.tensor(2)

logits = W @ x + b
loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0),
                                          target.unsqueeze(0))
loss.backward()
# W.grad, b.grad, x.grad now contain autograd gradients
# They should match our manual computation!`}
      />

      <div className="card">
        <h3>Optimizer: AdamW</h3>
        <p className="card-subtitle">
          Extends Adam with decoupled weight decay — each parameter gets its own adaptive learning rate while being gently pulled toward zero.
        </p>
        <FormulaBlock
          title="AdamW Update Rule"
          formula={String.raw`\begin{aligned}
            m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t \\
            v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \\
            \hat{m}_t &= \tfrac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \tfrac{v_t}{1-\beta_2^t} \\
            w_t &= w_{t-1} - \eta\!\left(\dfrac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\, w_{t-1}\right)
          \end{aligned}`}
          source="Loshchilov & Hutter 2019, arXiv:1711.05101"
          sourceUrl="https://arxiv.org/abs/1711.05101"
          intuition="Adam's per-parameter adaptive rates let fast-changing weights slow down and slow-changing ones speed up. Decoupling weight decay from the gradient step (AdamW vs Adam) gives cleaner regularization."
          terms={[
            { symbol: String.raw`g_t`, name: 'gradient', meaning: 'Derivative of the loss with respect to weight w at step t' },
            { symbol: String.raw`m_t`, name: 'first moment (momentum)', meaning: 'Exponential moving average of gradients — smooths noisy gradient signal (like momentum in SGD)' },
            { symbol: String.raw`v_t`, name: 'second moment (variance)', meaning: 'Exponential moving average of squared gradients — estimates per-parameter gradient variance' },
            { symbol: String.raw`\beta_1, \beta_2`, name: 'decay rates', meaning: 'Controls how quickly old moment estimates fade; typical values β₁=0.9, β₂=0.95' },
            { symbol: String.raw`\hat{m}_t, \hat{v}_t`, name: 'bias-corrected moments', meaning: 'Early steps have cold-started moments close to 0 — dividing by (1−β^t) corrects this initialisation bias' },
            { symbol: String.raw`\eta`, name: 'learning rate', meaning: 'Global step size scalar; typical value 3e-4 for transformers' },
            { symbol: String.raw`\epsilon`, name: 'numerical stability', meaning: 'Small constant (1e-8) added to denominator to prevent division by zero' },
            { symbol: String.raw`\lambda`, name: 'weight decay', meaning: 'Regularization coefficient that directly shrinks weights each step — decoupled from the gradient in AdamW (typical: 0.1)' },
          ]}
          steps={[
            { label: 'Track gradient momentum', math: String.raw`m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t`, text: 'Smooth out noisy gradients using an exponential moving average — equivalent to momentum in SGD' },
            { label: 'Track gradient variance', math: String.raw`v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2`, text: 'Track how much each gradient fluctuates — large v_t → small effective step for that parameter' },
            { label: 'Correct initialisation bias', math: String.raw`\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}`, text: 'Moments start at 0; dividing rescales them to unbiased estimates during the first few hundred steps' },
            { label: 'Update weight with decoupled decay', math: String.raw`w_t = w_{t-1} - \eta\!\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda w_{t-1}\right)`, text: 'Apply adaptive gradient step then separately subtract a fraction of the weight itself — this is the "decoupled" part that makes AdamW better than Adam+L2' },
          ]}
        />
      </div>
    </PageLayout>
  );
}
