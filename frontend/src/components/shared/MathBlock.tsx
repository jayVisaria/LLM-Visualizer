import { useEffect, useRef } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface Props {
  math: string;
  display?: boolean;
}

export default function MathBlock({ math, display = true }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) {
      try {
        katex.render(math, ref.current, {
          displayMode: display,
          throwOnError: false,
          trust: true,
        });
      } catch {
        ref.current.textContent = math;
      }
    }
  }, [math, display]);

  return <div ref={ref} className={display ? 'math-block' : 'math-inline'} />;
}
