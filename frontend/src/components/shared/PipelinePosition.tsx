import { NavLink } from 'react-router-dom';

const STEPS = [
  { num: 1, label: 'Tokenize', path: '/tokenization' },
  { num: 2, label: 'Embed', path: '/embeddings' },
  { num: 3, label: 'Position', path: '/positional-encoding' },
  { num: 4, label: 'Attention', path: '/self-attention' },
  { num: 5, label: 'Multi-Head', path: '/multi-head-attention' },
  { num: 6, label: 'Block', path: '/transformer-block' },
  { num: 7, label: 'Model', path: '/full-model' },
  { num: 8, label: 'Train', path: '/training' },
  { num: 9, label: 'Backprop', path: '/backpropagation' },
  { num: 10, label: 'Generate', path: '/generation' },
];

interface Props {
  currentStep: number; // 1-10
}

export default function PipelinePosition({ currentStep }: Props) {
  return (
    <div className="pipeline-breadcrumb">
      {STEPS.map((s, i) => (
        <div key={s.num} className="pipeline-bc-item">
          <NavLink
            to={s.path}
            className={`pipeline-dot ${
              s.num === currentStep ? 'current' :
              s.num < currentStep ? 'completed' : 'upcoming'
            }`}
            title={`Step ${s.num}: ${s.label}`}
          >
            {s.num}
          </NavLink>
          {i < STEPS.length - 1 && <span className="pipeline-bc-line" />}
        </div>
      ))}
    </div>
  );
}
