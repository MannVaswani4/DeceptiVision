import { useEffect, useRef, useState } from 'react';

interface BlurTextProps {
  text: string;
  delay?: number;
  animateBy?: 'words' | 'letters';
  direction?: 'top' | 'bottom' | 'left' | 'right';
  onAnimationComplete?: () => void;
  className?: string;
}

export default function BlurText({
  text,
  delay = 100,
  animateBy = 'words',
  direction = 'top',
  onAnimationComplete,
  className = '',
}: BlurTextProps) {
  const [visibleIndices, setVisibleIndices] = useState<Set<number>>(new Set());
  const units = animateBy === 'words' ? text.split(' ') : text.split('');
  const completedRef = useRef(false);

  const getInitialTransform = () => {
    switch (direction) {
      case 'top': return 'translateY(-16px)';
      case 'bottom': return 'translateY(16px)';
      case 'left': return 'translateX(-16px)';
      case 'right': return 'translateX(16px)';
    }
  };

  useEffect(() => {
    completedRef.current = false;
    setVisibleIndices(new Set());

    const timers: ReturnType<typeof setTimeout>[] = [];
    units.forEach((_, i) => {
      const t = setTimeout(() => {
        setVisibleIndices(prev => new Set([...prev, i]));
        if (i === units.length - 1 && !completedRef.current) {
          completedRef.current = true;
          // slight extra delay so the last word finishes its animation
          setTimeout(() => onAnimationComplete?.(), 400);
        }
      }, i * delay);
      timers.push(t);
    });

    return () => timers.forEach(clearTimeout);
  }, [text, delay, animateBy]);

  return (
    <span className={`inline ${className}`} style={{ display: 'flex', flexWrap: 'wrap', gap: animateBy === 'words' ? '0.3em' : '0' }}>
      {units.map((unit, i) => (
        <span
          key={i}
          style={{
            display: 'inline-block',
            opacity: visibleIndices.has(i) ? 1 : 0,
            filter: visibleIndices.has(i) ? 'blur(0px)' : 'blur(10px)',
            transform: visibleIndices.has(i) ? 'translateY(0) translateX(0)' : getInitialTransform(),
            transition: 'opacity 0.5s ease, filter 0.5s ease, transform 0.5s ease',
          }}
        >
          {unit}
          {animateBy === 'letters' && unit === ' ' ? '\u00A0' : ''}
        </span>
      ))}
    </span>
  );
}
