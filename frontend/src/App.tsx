import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/layout/Sidebar';
import StatusBar from './components/layout/StatusBar';
import ConfigPanel from './components/layout/ConfigPanel';
import ErrorBoundary from './components/shared/ErrorBoundary';

import TokenizationPage from './pages/01_Tokenization';
import EmbeddingsPage from './pages/02_Embeddings';
import PositionalEncodingPage from './pages/03_PositionalEncoding';
import SelfAttentionPage from './pages/04_SelfAttention';
import MultiHeadAttentionPage from './pages/05_MultiHeadAttention';
import TransformerBlockPage from './pages/06_TransformerBlock';
import FullModelPage from './pages/07_FullModel';
import TrainingPage from './pages/08_Training';
import BackpropagationPage from './pages/09_Backpropagation';
import GenerationPage from './pages/10_Generation';
import DashboardPage from './pages/00_Dashboard';
import ModernArchitecturesPage from './pages/11_ModernArchitectures';

export default function App() {
  return (
    <HashRouter>
      <div className="app-layout">
        <Sidebar />
        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <main style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <ErrorBoundary>
              <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<DashboardPage />} />
                <Route path="/tokenization" element={<TokenizationPage />} />
                <Route path="/embeddings" element={<EmbeddingsPage />} />
                <Route path="/positional-encoding" element={<PositionalEncodingPage />} />
                <Route path="/self-attention" element={<SelfAttentionPage />} />
                <Route path="/multi-head-attention" element={<MultiHeadAttentionPage />} />
                <Route path="/transformer-block" element={<TransformerBlockPage />} />
                <Route path="/full-model" element={<FullModelPage />} />
                <Route path="/training" element={<TrainingPage />} />
                <Route path="/backpropagation" element={<BackpropagationPage />} />
                <Route path="/generation" element={<GenerationPage />} />
              <Route path="/modern-architectures" element={<ModernArchitecturesPage />} />
              </Routes>
            </ErrorBoundary>
          </main>
          <StatusBar />
        </div>
        <ConfigPanel />
      </div>
    </HashRouter>
  );
}
