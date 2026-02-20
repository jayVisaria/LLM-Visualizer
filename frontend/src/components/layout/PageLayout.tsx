import type { ReactNode } from 'react';
import LearningContext from '../shared/LearningContext';
import type { LearningContextData } from '../shared/LearningContext';
import KeyTakeaway from '../shared/KeyTakeaway';
import type { KeyTakeawayData } from '../shared/KeyTakeaway';
import PipelinePosition from '../shared/PipelinePosition';

interface Props {
  title: string;
  subtitle: string;
  step: number;
  children: ReactNode;
  pipelineStep?: number;        // 1-10 to show breadcrumb
  learningContext?: LearningContextData;
  takeaway?: KeyTakeawayData;
}

export default function PageLayout({ title, subtitle, step, children, pipelineStep, learningContext, takeaway }: Props) {
  return (
    <div className="main-content">
      <div className="page-header">
        <div className="step-indicator">Step {step} of 10</div>
        {pipelineStep && <PipelinePosition currentStep={pipelineStep} />}
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
      {learningContext && <LearningContext {...learningContext} />}
      {children}
      {takeaway && <KeyTakeaway {...takeaway} />}
    </div>
  );
}
