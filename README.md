# LLM Visualizer

An interactive educational app for building a Transformer from scratch — tokenization through modern architectures, with live visualizations and a real PyTorch model running in the backend.

## Getting Started

**Requirements:** Python 3.10+, Node.js 18+

```bash
# Backend
cd backend
pip install torch numpy scikit-learn fastapi uvicorn sse-starlette python-multipart pydantic
python3 -m uvicorn main:app --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Stack

| | |
|---|---|
| Frontend | React 19, TypeScript, Vite |
| Backend | FastAPI, PyTorch, NumPy |
| Viz | Plotly.js, KaTeX |
| State | Zustand |

## Structure

```
llm-scratch/
├── backend/
│   ├── core/          # model, tokenizer, trainer, attention
│   └── routers/       # API endpoints per topic
└── frontend/
    └── src/
        ├── pages/     # 00_Dashboard → 11_ModernArchitectures
        ├── components/
        └── store/
```

## License

MIT
