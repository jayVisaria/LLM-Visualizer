import { NavLink } from 'react-router-dom';

export interface KeyTakeawayData {
  bullets: string[];
  nextStep?: { label: string; path: string; teaser: string };
  prevStep?: { label: string; path: string };
  tryThis?: string;
}

export default function KeyTakeaway({ bullets, nextStep, prevStep, tryThis }: KeyTakeawayData) {
  return (
    <div className="key-takeaway">
      <h3 className="key-takeaway-title">🎯 Key Takeaway</h3>

      <ul className="takeaway-bullets">
        {bullets.map((b, i) => (
          <li key={i}>{b}</li>
        ))}
      </ul>

      {tryThis && (
        <div className="try-this">
          <strong>🧪 Try This:</strong> {tryThis}
        </div>
      )}

      <div className="page-nav-buttons">
        {prevStep ? (
          <NavLink to={prevStep.path} className="page-nav-btn prev">
            ← {prevStep.label}
          </NavLink>
        ) : <span />}
        {nextStep && (
          <NavLink to={nextStep.path} className="next-step-link">
            <span>Next: {nextStep.label} →</span>
            <span className="next-step-teaser">{nextStep.teaser}</span>
          </NavLink>
        )}
      </div>
    </div>
  );
}
