import { useState } from 'react';
import { NavLink } from 'react-router-dom';

export interface LearningContextData {
  prerequisites?: { label: string; path: string }[];
  plainEnglish: string;
  realWorld: string;
}

export default function LearningContext({ prerequisites, plainEnglish, realWorld }: LearningContextData) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className={`learning-context ${collapsed ? 'collapsed' : ''}`}>
      <div className="learning-context-header" onClick={() => setCollapsed(!collapsed)}>
        <span className="learning-context-title">📖 Learning Context</span>
        <span className="learning-context-toggle">{collapsed ? '▸' : '▾'}</span>
      </div>

      {!collapsed && (
        <div className="learning-context-body">
          {/* Prerequisites */}
          {prerequisites && prerequisites.length > 0 && (
            <div className="learning-context-section lc-prereqs">
              <div className="learning-context-icon">📋</div>
              <div>
                <div className="learning-context-label">Prerequisites</div>
                <div className="learning-context-text">
                  {prerequisites.map((p, i) => (
                    <span key={p.path}>
                      {i > 0 && ' · '}
                      <NavLink to={p.path} className="lc-link">{p.label}</NavLink>
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* In Plain English */}
          <div className="learning-context-section lc-plain">
            <div className="learning-context-icon">💡</div>
            <div>
              <div className="learning-context-label">In Plain English</div>
              <div className="learning-context-text">{plainEnglish}</div>
            </div>
          </div>

          {/* In the Real World */}
          <div className="learning-context-section lc-world">
            <div className="learning-context-icon">🌍</div>
            <div>
              <div className="learning-context-label">In the Real World</div>
              <div className="learning-context-text">{realWorld}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
