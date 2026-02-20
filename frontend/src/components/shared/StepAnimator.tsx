import { useState, useCallback, useEffect } from 'react';

interface StepAnimatorProps {
  /** Array of step data — each step will be revealed one at a time */
  steps: any[];
  /** Render function for the current visible steps */
  renderSteps: (visibleSteps: any[], currentIndex: number) => React.ReactNode;
  /** Label for the step counter (e.g., "Token", "Merge", "Layer") */
  stepLabel?: string;
  /** Auto-play speed in ms (0 = no autoplay) */
  autoPlaySpeed?: number;
  /** Callback when step changes */
  onStepChange?: (index: number) => void;
  /** Show controls at top or bottom */
  controlPosition?: 'top' | 'bottom';
}

export default function StepAnimator({
  steps,
  renderSteps,
  stepLabel = 'Step',
  autoPlaySpeed = 0,
  onStepChange,
  controlPosition = 'top',
}: StepAnimatorProps) {
  const [current, setCurrent] = useState(0);
  const [playing, setPlaying] = useState(false);
  const total = steps.length;

  const goTo = useCallback((i: number) => {
    const clamped = Math.max(0, Math.min(i, total - 1));
    setCurrent(clamped);
    onStepChange?.(clamped);
  }, [total, onStepChange]);

  const next = () => goTo(current + 1);
  const prev = () => goTo(current - 1);
  const reset = () => { goTo(0); setPlaying(false); };
  const showAll = () => { goTo(total - 1); setPlaying(false); };

  // Auto-play
  useEffect(() => {
    if (!playing || current >= total - 1) {
      if (current >= total - 1) setPlaying(false);
      return;
    }
    const timer = setTimeout(() => goTo(current + 1), autoPlaySpeed || 400);
    return () => clearTimeout(timer);
  }, [playing, current, total, autoPlaySpeed, goTo]);

  // Reset when steps change
  useEffect(() => {
    setCurrent(0);
    setPlaying(false);
  }, [steps.length]);

  if (total === 0) return null;

  const visibleSteps = steps.slice(0, current + 1);

  const controls = (
    <div className="step-animator-controls">
      <div className="step-animator-buttons">
        <button className="secondary" onClick={reset} disabled={current === 0} title="Reset">
          ⏮
        </button>
        <button className="secondary" onClick={prev} disabled={current === 0} title="Previous">
          ◀
        </button>
        <button className="secondary" onClick={() => setPlaying(!playing)} title={playing ? 'Pause' : 'Play'}>
          {playing ? '⏸' : '▶'}
        </button>
        <button className="secondary" onClick={next} disabled={current >= total - 1} title="Next">
          ▶
        </button>
        <button className="secondary" onClick={showAll} disabled={current >= total - 1} title="Show All">
          ⏭
        </button>
      </div>
      <div className="step-animator-info">
        <span className="step-animator-counter">
          {stepLabel} {current + 1} / {total}
        </span>
        <input type="range" min={0} max={total - 1} value={current}
          onChange={e => goTo(Number(e.target.value))}
          className="step-animator-scrubber" />
      </div>
    </div>
  );

  return (
    <div className="step-animator">
      {controlPosition === 'top' && controls}
      <div className="step-animator-content">
        {renderSteps(visibleSteps, current)}
      </div>
      {controlPosition === 'bottom' && controls}
    </div>
  );
}
