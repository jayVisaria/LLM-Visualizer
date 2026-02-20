import { useEffect, useRef } from 'react';
import { useAppStore } from '../../store/useAppStore';
import { getConfig, getStatus } from '../../api/client';

export default function StatusBar() {
  const {
    modelConfig, datasetName, phase, currentStep, currentLoss,
    modelParams, lastError,
    loadServerConfig, setTrainingProgress, setPhase, setError, setModelReady,
  } = useAppStore();

  const pollRef = useRef<number | null>(null);

  // On mount: fetch server config to sync store
  useEffect(() => {
    let cancelled = false;
    getConfig()
      .then(data => {
        if (!cancelled) {
          loadServerConfig(data);
          setModelReady(true);
        }
      })
      .catch(() => {
        if (!cancelled) setModelReady(false);
      });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll training status every 3s when training
  useEffect(() => {
    if (phase === 'training') {
      const poll = () => {
        getStatus()
          .then(data => {
            if (data.current_step != null) {
              setTrainingProgress(data.current_step, data.current_loss ?? 0);
            }
            if (!data.is_training) setPhase('idle');
          })
          .catch(() => {});
      };
      poll();
      pollRef.current = window.setInterval(poll, 3000);
      return () => { if (pollRef.current) window.clearInterval(pollRef.current); };
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  const phaseLabel: Record<string, string> = {
    idle: '● Ready',
    training: '◉ Training',
    generating: '◉ Generating',
    rebuilding: '◉ Rebuilding',
    'loading-snapshot': '◉ Loading',
  };

  const phaseColor: Record<string, string> = {
    idle: 'var(--green)',
    training: 'var(--yellow)',
    generating: 'var(--accent)',
    rebuilding: 'var(--orange)',
    'loading-snapshot': 'var(--purple)',
  };

  return (
    <div className="status-bar">
      <div className="status-bar-left">
        <span className="status-indicator" style={{ color: phaseColor[phase] || 'var(--green)' }}>
          {phaseLabel[phase] || '● Ready'}
        </span>
        <span className="status-divider">|</span>
        <span className="status-item">
          <span className="status-label">Model</span>
          <span className="status-value">
            {modelConfig.d_model}d / {modelConfig.n_heads}h / {modelConfig.n_layers}L
          </span>
        </span>
        {modelParams != null && Number.isFinite(modelParams) && modelParams > 0 && (
          <>
            <span className="status-divider">|</span>
            <span className="status-item">
              <span className="status-label">Params</span>
              <span className="status-value">
                {modelParams > 1e6 ? `${(modelParams / 1e6).toFixed(1)}M` : `${(modelParams / 1e3).toFixed(0)}K`}
              </span>
            </span>
          </>
        )}
        <span className="status-divider">|</span>
        <span className="status-item">
          <span className="status-label">Data</span>
          <span className="status-value">{datasetName}</span>
        </span>
      </div>

      <div className="status-bar-right">
        {phase === 'training' && (
          <span className="status-training-info">
            Step {currentStep} {currentLoss != null && `• Loss ${currentLoss.toFixed(4)}`}
          </span>
        )}
        {lastError && (
          <span className="status-error" onClick={() => setError(null)} title="Click to dismiss">
            ⚠ {lastError}
          </span>
        )}
      </div>
    </div>
  );
}
