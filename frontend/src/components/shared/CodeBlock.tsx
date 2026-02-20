import { useState, useCallback } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const LANG_META: Record<string, { label: string; color: string }> = {
  python: { label: 'Python', color: '#3572A5' },
  javascript: { label: 'JavaScript', color: '#f1e05a' },
  typescript: { label: 'TypeScript', color: '#3178c6' },
  bash: { label: 'Shell', color: '#89e051' },
};

interface Props {
  code: string;
  language?: string;
  title?: string;
  collapsible?: boolean;
}

export default function CodeBlock({ code, language = 'python', title, collapsible = false }: Props) {
  const [open, setOpen] = useState(!collapsible);
  const [copied, setCopied] = useState(false);

  const meta = LANG_META[language] ?? { label: language, color: '#8b949e' };
  const lineCount = code.trim().split('\n').length;

  const handleCopy = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(code.trim()).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [code]);

  return (
    <div className={`code-container ${open ? 'code-open' : 'code-closed'}`}>
      <div
        className="code-header"
        onClick={() => collapsible && setOpen(!open)}
        style={{ cursor: collapsible ? 'pointer' : 'default' }}
      >
        <div className="code-header-left">
          {collapsible && (
            <span className={`code-chevron ${open ? 'open' : ''}`}>›</span>
          )}
          <span className="code-lang-dot" style={{ background: meta.color }} />
          <span className="code-lang-badge" style={{ color: meta.color }}>{meta.label}</span>
          {title && <span className="code-title">{title}</span>}
        </div>
        <div className="code-header-right">
          <span className="code-line-count">{lineCount} lines</span>
          <button className="code-copy-btn" onClick={handleCopy} title="Copy code">
            {copied ? '✓ Copied' : '⎘ Copy'}
          </button>
        </div>
      </div>
      {open && (
        <div className="code-body">
          <SyntaxHighlighter
            language={language}
            style={vscDarkPlus}
            customStyle={{
              margin: 0,
              padding: '18px 16px',
              fontSize: '13px',
              lineHeight: '1.65',
              background: '#0d1117',
              borderRadius: 0,
              border: 'none',
            }}
            showLineNumbers
            lineNumberStyle={{ color: '#3d444d', fontSize: '12px', paddingRight: '16px', userSelect: 'none' }}
          >
            {code.trim()}
          </SyntaxHighlighter>
        </div>
      )}
    </div>
  );
}
