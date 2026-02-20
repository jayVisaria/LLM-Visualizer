import { useState, useCallback } from 'react';
import Plot from 'react-plotly.js';
import PageLayout from '../components/layout/PageLayout';
import FormulaBlock from '../components/shared/FormulaBlock';
import CodeBlock from '../components/shared/CodeBlock';
import { tokenize, getBpeMerges, getVocabStats } from '../api/client';
import { useAppStore } from '../store/useAppStore';
import StepAnimator from '../components/shared/StepAnimator';

export default function TokenizationPage() {
  const sharedText = useAppStore(s => s.sharedText);
  const setSharedText = useAppStore(s => s.setSharedText);
  const [text, setText] = useState(sharedText);
  const handleTextChange = (v: string) => { setText(v); setSharedText(v); };
  const [result, setResult] = useState<any>(null);
  const [merges, setMerges] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [tokResult, mergeResult, statsResult] = await Promise.all([
        tokenize(text),
        getBpeMerges(50),
        getVocabStats(),
      ]);
      setResult(tokResult);
      setMerges(mergeResult);
      setStats(statsResult);
    } catch (e: any) {
      console.error('Tokenization error:', e);
      setError(e?.response?.data?.detail || e?.message || 'API call failed');
    }
    setLoading(false);
  }, [text]);

  return (
    <PageLayout
      title="Tokenization"
      subtitle="Transform raw text into a sequence of integer token IDs using Byte Pair Encoding (BPE) — the same algorithm used by GPT-2, GPT-3, and GPT-4."
      step={1}
      pipelineStep={1}
      learningContext={{
        plainEnglish: "Tokenization chops text into small pieces and gives each piece a number. Think of it like a dictionary where 'the' → 42, 'cat' → 187. The model only sees these numbers, never the raw text.",
        realWorld: "GPT-4 uses a BPE tokenizer with ~100,000 tokens. Our tiny model uses ~756. The same algorithm — just more merges and more training data. Tokenizer quality directly affects model performance: a poor tokenizer wastes capacity encoding common words as multiple tokens.",
      }}
      takeaway={{
        bullets: [
          "Text is converted to bytes, then BPE iteratively merges the most frequent pair of adjacent tokens",
          "The compression ratio tells you how efficient the tokenizer is — higher means fewer tokens per character",
          "The vocabulary size (number of unique tokens) is a key hyperparameter: too small = long sequences, too large = sparse embeddings",
          "Token IDs are the model's only input — everything downstream operates on these integers",
        ],
        nextStep: { label: 'Embeddings', path: '/embeddings', teaser: 'Convert token IDs into meaningful vector representations' },
        tryThis: "Paste text in different languages (try Chinese, Arabic, or emoji). Notice how non-English text produces more tokens per word — that's because BPE was trained on English-heavy data. This is a real limitation of many LLM tokenizers.",
      }}
    >
      {/* Explanation */}
      <div className="card">
        <h3>What is Tokenization?</h3>
        <p className="card-subtitle">
          Language models don't process raw text — they work with numbers. Tokenization is the first step:
          splitting text into subword units (tokens) and mapping each to an integer ID.
        </p>
        <div className="explanation">
          <strong>BPE Algorithm:</strong><br />
          1. Start with individual characters (bytes 0-255) as the vocabulary<br />
          2. Count all adjacent character pairs in the corpus<br />
          3. Merge the most frequent pair into a single new token<br />
          4. Repeat for N iterations (we use 500 merges → vocab of ~756 tokens)<br /><br />
          This produces subword tokens: common words like "the" become single tokens,
          while rare words are split into smaller pieces.
        </div>
        <FormulaBlock
          title="BPE Compression Ratio"
          formula={String.raw`\text{Compression Ratio} = \dfrac{\text{Original bytes}}{\text{Number of tokens}}`}
          source="Sennrich et al. 2016, arXiv:1508.07909"
          sourceUrl="https://arxiv.org/abs/1508.07909"
          intuition="Measures how efficiently BPE packed raw text into tokens — a ratio of 3× means 3 bytes become 1 token on average."
          terms={[
            { symbol: String.raw`\text{Original bytes}`, name: 'byte count', meaning: 'Number of UTF-8 bytes in the raw input string' },
            { symbol: String.raw`\text{Number of tokens}`, name: 'token count', meaning: 'Number of subword tokens produced by the BPE tokenizer' },
          ]}
          steps={[
            { label: 'Encode to bytes', text: 'Convert the input string to its UTF-8 byte sequence (every ASCII character is 1 byte; Unicode chars can be 2-4 bytes)' },
            { label: 'Tokenize', text: 'Run BPE to merge frequent byte pairs iteratively until no more merges apply — each merge reduces the token count by 1' },
            { label: 'Divide', math: String.raw`\text{ratio} = \dfrac{\text{bytes}}{\text{tokens}}`, text: 'A ratio > 1 means compression occurred; a ratio < 1 would mean the tokenizer expanded the text (rare)' },
          ]}
        />
      </div>

      {/* Interactive input */}
      <div className="card">
        <h3>Try It</h3>
        <div className="controls">
          <div className="control-group" style={{ flex: 1 }}>
            <label>Input Text</label>
            <textarea
              value={text}
              onChange={(e) => handleTextChange(e.target.value)}
              rows={4}
            />
          </div>
        </div>
        <button onClick={run} disabled={loading}>
          {loading ? 'Processing…' : 'Tokenize'}
        </button>
        {error && <div style={{ marginTop: 12, padding: '10px 14px', background: '#3d1f1f', border: '1px solid #f85149', borderRadius: 6, color: '#f85149', fontSize: 13 }}>Error: {error}</div>}
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Stats */}
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{result.num_tokens}</div>
              <div className="stat-label">Tokens</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{result.input_bytes}</div>
              <div className="stat-label">Input Bytes</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{result.compression_ratio}×</div>
              <div className="stat-label">Compression</div>
            </div>
            {stats && (
              <div className="stat-card">
                <div className="stat-value">{stats.vocab_size}</div>
                <div className="stat-label">Vocab Size</div>
              </div>
            )}
          </div>

          {/* Token display with step-through animation */}
          <div className="card">
            <h3>Tokenized Output</h3>
            <p className="card-subtitle">Step through tokens one at a time, or show all at once. Hover for token IDs.</p>
            <StepAnimator
              steps={result.tokens}
              stepLabel="Token"
              autoPlaySpeed={200}
              renderSteps={(visible) => (
                <div className="token-display">
                  {visible.map((tok: any, i: number) => (
                    <span
                      key={i}
                      className={`token token-${i % 6} step-animated-item`}
                      title={`ID: ${tok.id} | Bytes: ${tok.byte_length}`}
                    >
                      {tok.text.replace(/ /g, '·').replace(/\n/g, '↵\n')}
                    </span>
                  ))}
                </div>
              )}
            />
          </div>

          {/* Token IDs */}
          <div className="card">
            <h3>Token IDs</h3>
            <p className="card-subtitle">The integer sequence fed into the model.</p>
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '13px',
              padding: '12px',
              background: 'var(--bg-primary)',
              borderRadius: '6px',
              wordBreak: 'break-all',
            }}>
              [{result.token_ids.join(', ')}]
            </div>
          </div>
        </>
      )}

      {/* BPE Merge History with step-through */}
      {merges && merges.merges.length > 0 && (
        <div className="card">
          <h3>BPE Merge History (first {merges.merges.length} of {merges.total_merges})</h3>
          <p className="card-subtitle">
            Step through merge operations to see how BPE builds vocabulary incrementally.
          </p>
          <StepAnimator
            steps={merges.merges}
            stepLabel="Merge"
            autoPlaySpeed={300}
            renderSteps={(visible) => (
              <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                <table className="merge-table">
                  <thead>
                    <tr>
                      <th>Step</th>
                      <th>Pair</th>
                      <th>→</th>
                      <th>New Token</th>
                      <th>Frequency</th>
                      <th>Vocab Size</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visible.map((m: any) => (
                      <tr key={m.step} className="step-animated-item">
                        <td>{m.step}</td>
                        <td>'{m.pair_text[0]}' + '{m.pair_text[1]}'</td>
                        <td>→</td>
                        <td>'{m.new_text}' (id: {m.new_id})</td>
                        <td>{m.frequency.toLocaleString()}</td>
                        <td>{m.vocab_size}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          />
        </div>
      )}

      {/* Token frequency chart */}
      {stats?.token_frequencies && (
        <div className="card">
          <h3>Token Frequency Distribution</h3>
          <div className="plot-container">
            <Plot
              data={[
                {
                  x: stats.token_frequencies.map(
                    (t: any) => `'${t.text.replace(/\n/g, '↵')}' (${t.id})`
                  ),
                  y: stats.token_frequencies.map((t: any) => t.count),
                  type: 'bar',
                  marker: { color: '#58a6ff' },
                },
              ]}
              layout={{
                height: 350,
                margin: { t: 20, b: 100, l: 60, r: 20 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#8b949e', size: 11 },
                xaxis: { tickangle: -45, title: 'Token' },
                yaxis: { title: 'Count', gridcolor: '#21262d' },
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      )}

      {/* Code */}
      <CodeBlock
        title="BPE Tokenizer — Python Implementation"
        collapsible
        code={`class BPETokenizer:
    def __init__(self):
        # Base vocabulary: single bytes 0-255
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []

    def train(self, text: str, num_merges: int = 500):
        tokens = list(text.encode("utf-8"))
        for i in range(num_merges):
            # Count adjacent pairs
            pair_counts = Counter()
            for j in range(len(tokens) - 1):
                pair_counts[(tokens[j], tokens[j+1])] += 1

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            new_id = 256 + i

            # Merge all occurrences
            merged = []
            j = 0
            while j < len(tokens):
                if j < len(tokens)-1 and (tokens[j], tokens[j+1]) == best_pair:
                    merged.append(new_id)
                    j += 2
                else:
                    merged.append(tokens[j])
                    j += 1
            tokens = merged

            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges.append((best_pair, new_id))

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        for (p0, p1), new_id in self.merges:
            tokens = self._merge_pair(tokens, (p0, p1), new_id)
        return tokens

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8")`}
      />
    </PageLayout>
  );
}
