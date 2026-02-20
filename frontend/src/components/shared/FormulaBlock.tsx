import { useState, useEffect, useRef } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

export interface FormulaTerm {
  symbol: string;   // KaTeX string for the symbol
  name: string;     // short name
  meaning: string;  // plain-English explanation
}

export interface FormulaStep {
  label: string;
  math?: string;    // KaTeX
  text: string;
}

export interface FormulaBlockProps {
  title: string;
  formula: string;            // KaTeX string — the main formula
  source?: string;            // e.g. "Vaswani et al. 2017, §3.2"
  sourceUrl?: string;
  intuition: string;          // one plain-English sentence
  terms: FormulaTerm[];       // every symbol defined
  steps?: FormulaStep[];      // optional derivation steps
}

function KatexSpan({ math, display = false }: { math: string; display?: boolean }) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (ref.current) {
      try {
        katex.render(math, ref.current, { displayMode: display, throwOnError: false, trust: true });
      } catch {
        if (ref.current) ref.current.textContent = math;
      }
    }
  }, [math, display]);
  return <span ref={ref} />;
}

function KatexDiv({ math }: { math: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) {
      try {
        katex.render(math, ref.current, { displayMode: true, throwOnError: false, trust: true });
      } catch {
        if (ref.current) ref.current.textContent = math;
      }
    }
  }, [math]);
  return <div ref={ref} />;
}

export default function FormulaBlock({
  title, formula, source, sourceUrl, intuition, terms, steps,
}: FormulaBlockProps) {
  const [open, setOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<'terms' | 'steps'>('terms');

  return (
    <div className={`formula-block ${open ? 'formula-open' : ''}`}>
      {/* ── Header row ── */}
      <div className="formula-header">
        <div className="formula-header-left">
          <span className="formula-tag">∑ Formula</span>
          <span className="formula-title">{title}</span>
        </div>
        {source && (
          <a
            className="formula-source"
            href={sourceUrl}
            target="_blank"
            rel="noreferrer"
            onClick={e => e.stopPropagation()}
          >
            📄 {source}
          </a>
        )}
      </div>

      {/* ── The formula itself ── */}
      <div className="formula-display">
        <KatexDiv math={formula} />
      </div>

      {/* ── Intuition bar ── */}
      <div className="formula-intuition">
        <span className="formula-intuition-icon">💡</span>
        <span>{intuition}</span>
      </div>

      {/* ── Expand toggle ── */}
      <button
        className="formula-expand-btn"
        onClick={() => setOpen(o => !o)}
      >
        <span className={`formula-chevron ${open ? 'open' : ''}`}>›</span>
        {open ? 'Hide breakdown' : 'Break it down — explain every symbol'}
      </button>

      {/* ── Expanded detail ── */}
      {open && (
        <div className="formula-detail">
          {steps && steps.length > 0 && (
            <div className="formula-tabs">
              <button
                className={`formula-tab ${activeTab === 'terms' ? 'active' : ''}`}
                onClick={() => setActiveTab('terms')}
              >
                🔤 Symbols defined
              </button>
              <button
                className={`formula-tab ${activeTab === 'steps' ? 'active' : ''}`}
                onClick={() => setActiveTab('steps')}
              >
                🪜 Step-by-step
              </button>
            </div>
          )}

          {/* Terms table */}
          {(activeTab === 'terms' || !steps?.length) && (
            <table className="formula-terms-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Name</th>
                  <th>What it means</th>
                </tr>
              </thead>
              <tbody>
                {terms.map((t, i) => (
                  <tr key={i}>
                    <td className="formula-symbol-cell">
                      <KatexSpan math={t.symbol} />
                    </td>
                    <td className="formula-name-cell">{t.name}</td>
                    <td className="formula-meaning-cell">{t.meaning}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {/* Steps */}
          {activeTab === 'steps' && steps && (
            <ol className="formula-steps">
              {steps.map((s, i) => (
                <li key={i} className="formula-step">
                  <div className="formula-step-label">{s.label}</div>
                  {s.math && (
                    <div className="formula-step-math">
                      <KatexDiv math={s.math} />
                    </div>
                  )}
                  <div className="formula-step-text">{s.text}</div>
                </li>
              ))}
            </ol>
          )}
        </div>
      )}
    </div>
  );
}
