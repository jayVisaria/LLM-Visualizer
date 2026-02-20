import { useState, useCallback } from 'react';
import { useAppStore } from '../../store/useAppStore';
import { rebuildModel, updateConfig, getConfig, getDatasets, selectDataset, uploadData, uploadText,
         saveSnapshot, listSnapshots, loadSnapshot } from '../../api/client';

// ─── Slider helper ──────────────────────────────────────────────────────────
function Slider({ label, value, min, max, step, onChange, unit, tooltip }: {
  label: string; value: number; min: number; max: number;
  step: number; onChange: (v: number) => void; unit?: string; tooltip?: string;
}) {
  const [showTip, setShowTip] = useState(false);
  return (
    <div className="config-slider"
      onMouseEnter={() => setShowTip(true)}
      onMouseLeave={() => setShowTip(false)}
    >
      <div className="config-slider-header">
        <span className="config-slider-label">{label}</span>
        <span className="config-slider-value">{value}{unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step}
        value={value} onChange={e => onChange(Number(e.target.value))} />
      {tooltip && showTip && (
        <div className="config-tooltip">{tooltip}</div>
      )}
    </div>
  );
}

// ─── Main Component ─────────────────────────────────────────────────────────
export default function ConfigPanel() {
  const {
    modelConfig, trainConfig, datasetName, phase, snapshots, configPanelOpen,
    setModelConfig, setTrainConfig, setDatasetName, setPhase, setError,
    toggleConfigPanel, loadServerConfig, setSnapshots, setModelParams,
  } = useAppStore();

  const [section, setSection] = useState<'architecture' | 'training' | 'data' | 'snapshots'>('architecture');
  const [datasets, setDatasets] = useState<any[]>([]);
  const [snapshotName, setSnapshotName] = useState('');
  const [snapshotDesc, setSnapshotDesc] = useState('');
  const [rebuilding, setRebuilding] = useState(false);
  const [saving, setSaving] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<{ char_count: number; preview: string } | null>(null);
  const [pasteText, setPasteText] = useState('');
  const [dragOver, setDragOver] = useState(false);

  // Load datasets on first open of data tab
  const loadDatasets = useCallback(async () => {
    try {
      const data = await getDatasets();
      setDatasets(data.datasets || []);
    } catch { /* ignore */ }
  }, []);

  const loadSnapshotList = useCallback(async () => {
    try {
      const data = await listSnapshots();
      setSnapshots(data.snapshots || []);
    } catch { /* ignore */ }
  }, [setSnapshots]);

  // ── Handlers ──
  const handleRebuild = async () => {
    const { d_model, n_heads, n_layers, context_len, dropout } = modelConfig;
    if (d_model % n_heads !== 0) {
      setError(`d_model (${d_model}) must be divisible by n_heads (${n_heads})`);
      return;
    }
    setRebuilding(true);
    setPhase('rebuilding');
    setError(null);
    try {
      const result = await rebuildModel({ d_model, n_heads, n_layers, context_len, dropout });
      // Refresh config from server
      const cfg = await getConfig();
      loadServerConfig(cfg);
      if (result.model_params) setModelParams(result.model_params);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Rebuild failed');
    } finally {
      setRebuilding(false);
      setPhase('idle');
    }
  };

  const handleUpdateTraining = async () => {
    try {
      await updateConfig({ train_config: trainConfig });
    } catch { /* ignore */ }
  };

  const handleSelectDataset = async (name: string) => {
    try {
      await selectDataset(name);
      setDatasetName(name);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to switch dataset');
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadResult(null);
    setError(null);
    try {
      const result = await uploadData(file);
      setDatasetName('custom');
      setUploadResult({ char_count: result.char_count, preview: result.preview });
      loadDatasets();
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleUploadText = async () => {
    if (pasteText.trim().length < 100) {
      setError('Text must be at least 100 characters');
      return;
    }
    setUploading(true);
    setUploadResult(null);
    setError(null);
    try {
      const result = await uploadText(pasteText);
      setDatasetName('custom');
      setUploadResult({ char_count: result.char_count, preview: result.preview });
      setPasteText('');
      loadDatasets();
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadResult(null);
    setError(null);
    try {
      const result = await uploadData(file);
      setDatasetName('custom');
      setUploadResult({ char_count: result.char_count, preview: result.preview });
      loadDatasets();
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleSaveSnapshot = async () => {
    if (!snapshotName.trim()) return;
    setSaving(true);
    try {
      await saveSnapshot(snapshotName.trim(), snapshotDesc);
      setSnapshotName('');
      setSnapshotDesc('');
      loadSnapshotList();
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Save failed');
    } finally {
      setSaving(false);
    }
  };

  const handleLoadSnapshot = async (name: string) => {
    setPhase('loading-snapshot');
    try {
      await loadSnapshot(name);
      const cfg = await getConfig();
      loadServerConfig(cfg);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Load failed');
    } finally {
      setPhase('idle');
    }
  };

  if (!configPanelOpen) return null;

  return (
    <div className="config-panel">
      <div className="config-panel-header">
        <h3>Configuration</h3>
        <button className="config-close-btn" onClick={toggleConfigPanel}>✕</button>
      </div>

      {/* Section tabs */}
      <div className="config-tabs">
        {(['architecture', 'training', 'data', 'snapshots'] as const).map(s => (
          <button key={s} className={`config-tab ${section === s ? 'active' : ''}`}
            onClick={() => {
              setSection(s);
              if (s === 'data') loadDatasets();
              if (s === 'snapshots') loadSnapshotList();
            }}>
            {s === 'architecture' ? '🏗️' : s === 'training' ? '📈' : s === 'data' ? '📄' : '💾'}
            <span>{s.charAt(0).toUpperCase() + s.slice(1)}</span>
          </button>
        ))}
      </div>

      <div className="config-body">

        {/* ── Architecture ── */}
        {section === 'architecture' && (
          <>
            <Slider label="d_model" value={modelConfig.d_model}
              min={32} max={512} step={32} onChange={v => setModelConfig({ d_model: v })}
              tooltip="The size of each token's vector representation. Larger = more capacity to capture meaning, but more parameters. 64 is tiny (educational), GPT-2 uses 768, GPT-3 uses 12,288." />
            <Slider label="n_heads" value={modelConfig.n_heads}
              min={1} max={16} step={1} onChange={v => setModelConfig({ n_heads: v })}
              tooltip="Number of parallel attention heads. Must evenly divide d_model. More heads = richer attention patterns, but each head gets fewer dimensions. 4 is a good starting point." />
            <Slider label="n_layers" value={modelConfig.n_layers}
              min={1} max={12} step={1} onChange={v => setModelConfig({ n_layers: v })}
              tooltip="Number of stacked transformer blocks. Deeper = more capacity for complex patterns, but harder to train and more parameters. GPT-2 has 12, GPT-3 has 96." />
            <Slider label="context_len" value={modelConfig.context_len}
              min={32} max={512} step={32} onChange={v => setModelConfig({ context_len: v })}
              tooltip="Maximum number of tokens the model can see at once. Longer = handles longer text, but attention has O(n²) memory cost. GPT-3 uses 2048, GPT-4 uses 128K." />
            <Slider label="dropout" value={modelConfig.dropout}
              min={0} max={0.5} step={0.05} onChange={v => setModelConfig({ dropout: v })}
              tooltip="Randomly zeros this fraction of values during training to prevent overfitting. 0.1 is standard. Set to 0 for inference/generation." />

            {modelConfig.d_model % modelConfig.n_heads !== 0 && (
              <div className="config-warning">
                ⚠ d_model ({modelConfig.d_model}) must be divisible by n_heads ({modelConfig.n_heads})
              </div>
            )}

            <button className="config-rebuild-btn" onClick={handleRebuild}
              disabled={rebuilding || modelConfig.d_model % modelConfig.n_heads !== 0}>
              {rebuilding ? '⏳ Rebuilding…' : '🔨 Rebuild Model'}
            </button>
            <p className="config-note">
              Rebuilding resets all weights. Save a snapshot first if needed.
            </p>
          </>
        )}

        {/* ── Training ── */}
        {section === 'training' && (
          <>
            <Slider label="max_steps" value={trainConfig.max_steps}
              min={100} max={10000} step={100} onChange={v => setTrainConfig({ max_steps: v })}
              tooltip="Total number of training iterations. More steps = lower loss, but diminishing returns. 2000 is good for this dataset size." />
            <Slider label="batch_size" value={trainConfig.batch_size}
              min={4} max={128} step={4} onChange={v => setTrainConfig({ batch_size: v })}
              tooltip="Number of text sequences processed per training step. Larger = more stable gradients but more memory. 32 is a good default for small models." />
            <Slider label="learning_rate" value={trainConfig.learning_rate}
              min={1e-5} max={1e-2} step={1e-5}
              onChange={v => setTrainConfig({ learning_rate: v })} unit=""
              tooltip="How much to adjust weights per step. Too high = unstable training, too low = slow learning. 3e-4 is the 'magic number' for Adam optimizers." />
            <Slider label="weight_decay" value={trainConfig.weight_decay}
              min={0} max={0.5} step={0.01} onChange={v => setTrainConfig({ weight_decay: v })}
              tooltip="L2 regularization strength. Prevents weights from growing too large. 0.1 is standard for AdamW. Higher values = stronger regularization." />
            <Slider label="warmup_steps" value={trainConfig.warmup_steps}
              min={0} max={500} step={10} onChange={v => setTrainConfig({ warmup_steps: v })}
              tooltip="Gradually increase learning rate from 0 for this many steps. Prevents early training instability when loss gradients are large. 100 is fine for small models." />
            <Slider label="grad_clip" value={trainConfig.grad_clip}
              min={0} max={5} step={0.1} onChange={v => setTrainConfig({ grad_clip: v })}
              tooltip="Cap gradient magnitude to prevent exploding gradients. 1.0 is standard. Set to 0 to disable clipping (not recommended)." />

            <button className="config-apply-btn" onClick={handleUpdateTraining}>
              ✓ Apply Training Config
            </button>
          </>
        )}

        {/* ── Data ── */}
        {section === 'data' && (
          <>
            <div className="config-section-label">Select Dataset</div>
            <div className="config-dataset-list">
              {datasets.map((d: any) => (
                <button key={d.name}
                  className={`config-dataset-item ${d.name === datasetName ? 'active' : ''}`}
                  onClick={() => handleSelectDataset(d.name)}>
                  <span className="dataset-name">{d.name}</span>
                  <span className="dataset-size">{d.size_kb ? `${d.size_kb.toFixed(0)} KB` : ''}</span>
                </button>
              ))}
              {datasets.length === 0 && (
                <p className="config-note">Loading datasets…</p>
              )}
            </div>

            <div className="config-section-label" style={{ marginTop: 16 }}>
              Upload Custom Dataset
            </div>

            {/* Drag-and-drop zone */}
            <div
              className={`config-dropzone ${dragOver ? 'drag-over' : ''} ${uploading ? 'uploading' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => {
                const el = document.getElementById('config-file-input');
                if (el) el.click();
              }}
            >
              <input
                id="config-file-input"
                type="file"
                accept=".txt,.md,.csv,.text"
                onChange={handleUpload}
                style={{ display: 'none' }}
              />
              {uploading ? (
                <div className="dropzone-content">
                  <span className="dropzone-icon">⏳</span>
                  <span>Uploading…</span>
                </div>
              ) : (
                <div className="dropzone-content">
                  <span className="dropzone-icon">📁</span>
                  <span>Drag & drop a file here</span>
                  <span className="dropzone-hint">or click to browse (.txt, .md, .csv)</span>
                </div>
              )}
            </div>

            {/* Paste text area */}
            <div className="config-section-label" style={{ marginTop: 12 }}>
              Or Paste Text
            </div>
            <textarea
              className="config-textarea"
              placeholder="Paste your training text here (min 100 characters)…"
              value={pasteText}
              onChange={e => setPasteText(e.target.value)}
              rows={4}
            />
            <div className="config-textarea-footer">
              <span className="config-char-count">
                {pasteText.length} chars{pasteText.length > 0 && pasteText.length < 100 ? ' (need ≥ 100)' : ''}
              </span>
              <button
                className="config-apply-btn"
                onClick={handleUploadText}
                disabled={uploading || pasteText.trim().length < 100}
                style={{ marginTop: 0, padding: '4px 12px', fontSize: '0.85rem' }}
              >
                {uploading ? '⏳' : '📤'} Upload Text
              </button>
            </div>

            {/* Upload result feedback */}
            {uploadResult && (
              <div className="config-upload-result">
                <div className="upload-result-header">
                  ✅ Uploaded successfully — {uploadResult.char_count.toLocaleString()} characters
                </div>
                <div className="upload-result-preview">
                  {uploadResult.preview}…
                </div>
              </div>
            )}
          </>
        )}

        {/* ── Snapshots ── */}
        {section === 'snapshots' && (
          <>
            <div className="config-section-label">Save Current Model</div>
            <input type="text" placeholder="Snapshot name" value={snapshotName}
              onChange={e => setSnapshotName(e.target.value)}
              className="config-input" />
            <input type="text" placeholder="Description (optional)" value={snapshotDesc}
              onChange={e => setSnapshotDesc(e.target.value)}
              className="config-input" style={{ marginTop: 6 }} />
            <button className="config-apply-btn" onClick={handleSaveSnapshot}
              disabled={saving || !snapshotName.trim()} style={{ marginTop: 8 }}>
              {saving ? '⏳ Saving…' : '💾 Save Snapshot'}
            </button>

            <div className="config-section-label" style={{ marginTop: 20 }}>Saved Snapshots</div>
            {snapshots.length === 0 ? (
              <p className="config-note">No snapshots yet.</p>
            ) : (
              <div className="config-snapshot-list">
                {snapshots.map((s: any) => (
                  <div key={s.name} className="config-snapshot-item">
                    <div>
                      <div className="snapshot-name">{s.name}</div>
                      {s.description && <div className="snapshot-desc">{s.description}</div>}
                      <div className="snapshot-meta">
                        {s.timestamp && new Date(s.timestamp).toLocaleDateString()}
                      </div>
                    </div>
                    <button className="secondary" onClick={() => handleLoadSnapshot(s.name)}>
                      Load
                    </button>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

      </div>
    </div>
  );
}
