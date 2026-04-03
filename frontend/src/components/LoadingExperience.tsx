import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Scan, Activity, Mic } from 'lucide-react';

const STEPS = [
  { icon: <Scan className="w-4 h-4" />, label: 'Analyzing facial expressions…', duration: 2800 },
  { icon: <Activity className="w-4 h-4" />, label: 'Tracking body pose with YOLOv8…', duration: 2500 },
  { icon: <Mic className="w-4 h-4" />, label: 'Processing audio stress signals…', duration: 2200 },
];

interface LoadingExperienceProps {
  onComplete?: () => void;
}

export default function LoadingExperience({ onComplete }: LoadingExperienceProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [doneSteps, setDoneSteps] = useState<Set<number>>(new Set());

  useEffect(() => {
    // Smoothly increment progress to 100 over ~8s
    const totalDuration = STEPS.reduce((a, s) => a + s.duration, 0) + 500;
    const interval = 50;
    const increment = (100 / totalDuration) * interval;
    let current = 0;

    const timer = setInterval(() => {
      current += increment;
      if (current >= 100) {
        current = 100;
        clearInterval(timer);
        setTimeout(() => onComplete?.(), 300);
      }
      setProgress(Math.min(100, current));
    }, interval);

    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    // Advance steps
    let elapsed = 0;
    const timers: ReturnType<typeof setTimeout>[] = [];
    STEPS.forEach((step, i) => {
      const t = setTimeout(() => {
        setCurrentStep(i);
        setDoneSteps(prev => {
          const next = new Set(prev);
          if (i > 0) next.add(i - 1);
          return next;
        });
      }, elapsed);
      timers.push(t);
      elapsed += step.duration;
    });

    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <div className="w-full max-w-md mx-auto py-8">
      {/* Animated logo pulse */}
      <div className="flex justify-center mb-8">
        <div className="relative">
          <div className="w-16 h-16 bg-indigo-500 rounded-2xl flex items-center justify-center">
            <motion.div
              animate={{ scale: [1, 1.15, 1] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
            >
              <Scan className="w-7 h-7 text-white" />
            </motion.div>
          </div>
          {/* Ping rings */}
          {[1, 2].map(ring => (
            <motion.div
              key={ring}
              className="absolute inset-0 rounded-2xl border-2 border-indigo-300"
              animate={{ scale: [1, 1.8 + ring * 0.4], opacity: [0.6, 0] }}
              transition={{ duration: 1.8, delay: ring * 0.4, repeat: Infinity, ease: 'easeOut' }}
            />
          ))}
        </div>
      </div>

      {/* Steps */}
      <div className="space-y-3 mb-7">
        {STEPS.map((step, i) => {
          const done = doneSteps.has(i);
          const active = currentStep === i && !done;
          return (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 ${
                active ? 'bg-indigo-50 border border-indigo-100' : done ? 'opacity-40' : 'opacity-25'
              }`}
            >
              <div className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 ${
                done ? 'bg-green-100 text-green-500' : active ? 'bg-indigo-100 text-indigo-500' : 'bg-gray-100 text-gray-400'
              }`}>
                {step.icon}
              </div>
              <span className={`text-sm font-medium ${active ? 'text-indigo-700' : 'text-gray-500'}`}>
                {step.label}
              </span>
              {active && (
                <div className="ml-auto flex gap-1">
                  {[0, 1, 2].map(d => (
                    <motion.div
                      key={d}
                      className="w-1 h-1 bg-indigo-400 rounded-full"
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ duration: 0.8, delay: d * 0.2, repeat: Infinity }}
                    />
                  ))}
                </div>
              )}
            </motion.div>
          );
        })}
      </div>

      {/* Progress bar */}
      <div className="space-y-2">
        <div className="flex justify-between text-xs text-gray-400">
          <span>Processing…</span>
          <span>{Math.round(progress)}%</span>
        </div>
        <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-indigo-400 to-indigo-600 rounded-full"
            style={{ width: `${progress}%` }}
            transition={{ ease: 'linear' }}
          />
        </div>
      </div>
    </div>
  );
}
