import { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import PageLayout from '../components/layout/PageLayout';
import { listSnapshots } from '../api/client';
import { useAppStore } from '../store/useAppStore';

interface SnapshotMeta {
  name: string;
  description: string;
  created_at: number;
  config: Record<string, any>;
  parameters: number;
  steps: number;
  loss: number | null;
  dataset: string;
  vocab_size: number;
}

export default function ComparisonPage() {
  const modelConfig = useAppStore(s => s.modelConfig);
  const modelParams = useAppStore(s => s.modelParams);
  const currentLoss = useAppStore(s => s.currentLoss);
  const currentStep = useAppStore(s => s.currentStep);

  const [snapshots, setSnapshots] = useState<SnapshotMeta[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  useEffect(() => {
    listSnapshots()
      .then(d => setSnapshots(d.snapshots || []))
      .catch(() => {});
  }, []);

  const toggle = (name: string) => {
    setSelected(prev => {
      const n = new Set(prev);
      if (n.has(name)) n.delete(name);
      else if (n.size < 4) n.add(name); // max 4 comparisons
      return n;
    });
  };

  // Build comparison items: current model + selected snapshots
  const currentModel = {
    name: '⚡ Current Model',
    config: modelConfig,
    parameters: modelParams ?? 0,
    steps: currentStep,
    loss: currentLoss,
    isCurrent: true,
  };

  const comparisons = [
    currentModel,
    ...snapshots.filter(s => selected.has(s.name)).map(s => ({
      name: `💾 ${s.name}`,
      config: s.config,
      parameters: s.parameters,
      steps: s.steps,
      loss: s.loss,
      isCurrent: false,
    })),
  ];

  const configKeys = ['d_model', 'n_heads', 'n_layers', 'context_len', 'dropout', 'vocab_size'];

  const darkLayout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(22,27,34,0.8)',
    font: { color: '#8b949e', size: 11 },
    margin: { t: 30, r: 20, b: 40, l: 80 },
  };

  const colors = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#bc8cff'];

  return (
    <PageLayout
      title="Model Comparison"
      subtitle="Compare your current model with saved snapshots — architecture, parameter counts, and training progress side by side."
      step={0}
    >
      {/* Snapshot Selector */}
      <div className="card">
        <h3>💾 Select Snapshots to Compare</h3>
        <p className="card-subtitle">
          Select up to 4 snapshots to compare with your current model.
          {snapshots.length === 0 && ' No snapshots saved yet — save one from the ⚙ config panel.'}
        </p>
        {snapshots.length > 0 && (
          <div className="comparison-selector">
            {snapshots.map(s => (
              <button key={s.name}
                className={`comparison-chip ${selected.has(s.name) ? 'active' : ''}`}
                onClick={() => toggle(s.name)}>
                <span className="chip-name">{s.name}</span>
                <span className="chip-meta">
                  {s.steps} steps • {s.parameters > 1e6 ? `${(s.parameters / 1e6).toFixed(1)}M` : `${(s.parameters / 1e3).toFixed(0)}K`} params
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Config Comparison Table */}
      <div className="card">
        <h3>📊 Architecture Comparison</h3>
        <div style={{ overflowX: 'auto' }}>
          <table className="merge-table comparison-table">
            <thead>
              <tr>
                <th>Parameter</th>
                {comparisons.map((c, i) => (
                  <th key={i} style={{ color: colors[i] }}>{c.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {configKeys.map(key => (
                <tr key={key}>
                  <td style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{key}</td>
                  {comparisons.map((c, i) => (
                    <td key={i} style={{ fontFamily: 'var(--font-mono)' }}>
                      {(c.config as any)[key] ?? '—'}
                    </td>
                  ))}
                </tr>
              ))}
              <tr>
                <td style={{ color: 'var(--text-muted)' }}>Total Params</td>
                {comparisons.map((c, i) => (
                  <td key={i} style={{ fontWeight: 600, color: colors[i] }}>
                    {c.parameters > 1e6 ? `${(c.parameters / 1e6).toFixed(1)}M` : `${(c.parameters / 1e3).toFixed(0)}K`}
                  </td>
                ))}
              </tr>
              <tr>
                <td style={{ color: 'var(--text-muted)' }}>Training Steps</td>
                {comparisons.map((c, i) => (
                  <td key={i}>{c.steps ?? '—'}</td>
                ))}
              </tr>
              <tr>
                <td style={{ color: 'var(--text-muted)' }}>Loss</td>
                {comparisons.map((c, i) => (
                  <td key={i} style={{ color: c.loss != null ? 'var(--green)' : 'var(--text-muted)' }}>
                    {c.loss != null ? c.loss.toFixed(4) : '—'}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Visual Comparison Charts */}
      {comparisons.length > 1 && (
        <div className="two-col">
          <div className="card">
            <h3>📐 Parameter Count</h3>
            <div className="plot-container">
              <Plot
                data={[{
                  x: comparisons.map(c => c.name),
                  y: comparisons.map(c => c.parameters),
                  type: 'bar',
                  marker: { color: comparisons.map((_, i) => colors[i]) },
                }]}
                layout={{
                  ...darkLayout,
                  height: 250,
                  xaxis: { color: '#8b949e' },
                  yaxis: { title: 'Parameters', color: '#8b949e', gridcolor: '#21262d' },
                  showlegend: false,
                }}
                config={{ responsive: true, displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>
          </div>

          <div className="card">
            <h3>📉 Loss Comparison</h3>
            <div className="plot-container">
              <Plot
                data={[{
                  x: comparisons.map(c => c.name),
                  y: comparisons.map(c => c.loss ?? 0),
                  type: 'bar',
                  marker: { color: comparisons.map((_, i) => colors[i]) },
                }]}
                layout={{
                  ...darkLayout,
                  height: 250,
                  xaxis: { color: '#8b949e' },
                  yaxis: { title: 'Loss', color: '#8b949e', gridcolor: '#21262d' },
                  showlegend: false,
                }}
                config={{ responsive: true, displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Scale comparison radar-like chart */}
      {comparisons.length > 1 && (
        <div className="card">
          <h3>🎯 Architecture Radar</h3>
          <p className="card-subtitle">Normalized comparison of architecture dimensions.</p>
          <div className="plot-container">
            <Plot
              data={comparisons.map((c, i) => {
                const vals = ['d_model', 'n_heads', 'n_layers', 'context_len'].map(k => (c.config as any)[k] ?? 0);
                const maxVals = ['d_model', 'n_heads', 'n_layers', 'context_len'].map(k =>
                  Math.max(...comparisons.map(cc => (cc.config as any)[k] ?? 1)));
                const normalized = vals.map((v, j) => v / maxVals[j]);
                return {
                  type: 'scatterpolar' as const,
                  r: [...normalized, normalized[0]], // close the polygon
                  theta: ['d_model', 'n_heads', 'n_layers', 'context_len', 'd_model'],
                  fill: 'toself',
                  fillcolor: colors[i] + '20',
                  line: { color: colors[i] },
                  name: c.name,
                };
              })}
              layout={{
                ...darkLayout,
                polar: {
                  bgcolor: 'transparent',
                  radialaxis: { visible: true, range: [0, 1], color: '#30363d' },
                  angularaxis: { color: '#8b949e' },
                },
                height: 350,
                showlegend: true,
                legend: { font: { color: '#8b949e' } },
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      )}

    </PageLayout>
  );
}
