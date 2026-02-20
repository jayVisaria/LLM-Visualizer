import { NavLink } from 'react-router-dom';
import { useAppStore } from '../../store/useAppStore';

const pages = [
  { path: '/dashboard', label: 'Dashboard', step: 0, icon: '🏠' },
  { path: '/tokenization', label: 'Tokenization', step: 1, icon: '✂️' },
  { path: '/embeddings', label: 'Embeddings', step: 2, icon: '🔢' },
  { path: '/positional-encoding', label: 'Positional Encoding', step: 3, icon: '📍' },
  { path: '/self-attention', label: 'Self-Attention', step: 4, icon: '🔍' },
  { path: '/multi-head-attention', label: 'Multi-Head Attention', step: 5, icon: '🧩' },
  { path: '/transformer-block', label: 'Transformer Block', step: 6, icon: '🧱' },
  { path: '/full-model', label: 'Full Model', step: 7, icon: '🤖' },
  { path: '/training', label: 'Training', step: 8, icon: '📈' },
  { path: '/backpropagation', label: 'Backpropagation', step: 9, icon: '🔄' },
  { path: '/generation', label: 'Text Generation', step: 10, icon: '✨' },
  { path: '/modern-architectures', label: 'Modern Architectures', step: 11, icon: '⚡' },
];

export default function Sidebar() {
  const { toggleConfigPanel, configPanelOpen } = useAppStore();

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <span className="sidebar-brand-icon">🧠</span>
          <div>
            <h1>LLM Visualizer</h1>
            <p>Build a transformer from scratch</p>
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="sidebar-nav-section-label">Learning Path</div>
        {pages.map((page) => (
          <NavLink
            key={page.path}
            to={page.path}
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            <span className="nav-step">{page.icon}</span>
            <span className="nav-label">{page.label}</span>
            {page.step > 0 && <span className="nav-step-num">{page.step}</span>}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <button
          className={`sidebar-config-btn ${configPanelOpen ? 'active' : ''}`}
          onClick={toggleConfigPanel}
          title="Toggle Configuration Panel"
        >
          <span className="sidebar-config-icon">⚙</span>
          <span>Configure Model</span>
          <span className="sidebar-config-arrow">{configPanelOpen ? '›' : '‹'}</span>
        </button>
      </div>
    </aside>
  );
}
