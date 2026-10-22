import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Scan, Activity, Cpu, CheckCircle2 } from 'lucide-react';

// Steps mirror the real backend pipeline.
// minDelay = earliest this step can finish (ms).
// null on the last step = waits for the real API to finish.
const STEPS = [
  {
    icon: <Scan className="w-4 h-4" />,
    label: 'Extracting frames from video…',
    detail: 'Sampling at 1 fps, up to 20 frames',
    minDelay: 3000,
  },
  {
    icon: <Activity className="w-4 h-4" />,
    label: 'Running facial emotion CNN…',
    detail: 'Detecting micro-expressions per frame',
    minDelay: 4000,
  },
  {
    icon: <Activity className="w-4 h-4" />,
    label: 'Running YOLOv8 pose estimation…',
    detail: 'Tracking 17 body keypoints per frame',
    minDelay: 4000,
  },
  {
    icon: <Cpu className="w-4 h-4" />,
    label: 'Running deception classifier…',
    detail: 'Random Forest across 30 combined features',
    minDelay: null, // only completes when API resolves
  },
];

interface LoadingExperienceProps {
  apiPromise: Promise<unknown>;
}

export default function LoadingExperience({ apiPromise }: LoadingExperienceProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [doneSteps, setDoneSteps] = useState<Set<number>>(new Set());
  const [elapsed, setElapsed] = useState(0);
  const apiDoneRef = useRef(false);

  // Elapsed-time counter
  useEffect(() => {
    const t = setInterval(() => setElapsed(s => s + 1), 1000);
    return () => clearInterval(t);
  }, []);

  // Mark API as done when it resolves or rejects
  useEffect(() => {
    apiPromise.finally(() => {
      apiDoneRef.current = true;
    });
  }, [apiPromise]);

  // Step sequencer
  useEffect(() => {
    let cancelled = false;

    const run = async () => {
      for (let i = 0; i < STEPS.length; i++) {
        if (cancelled) return;
        setCurrentStep(i);

        const { minDelay } = STEPS[i];

        if (minDelay !== null) {
          // Wait the minimum realistic time for this stage
          await new Promise(res => setTimeout(res, minDelay));
        } else {
          // Last step: poll until the real API has finished
          await new Promise<void>(res => {
            const check = setInterval(() => {
              if (apiDoneRef.current || cancelled) {
                clearInterval(check);
                res();
              }
            }, 200);
          });
        }

        if (cancelled) return;
        setDoneSteps(prev => new Set([...prev, i]));
      }
    };

    run();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fmt = (s: number) =>
    s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;

  const isLastActive =
    currentStep === STEPS.length - 1 && !doneSteps.has(STEPS.length - 1);

  return (
    <div className="w-full max-w-md mx-auto py-8">
      {/* Pulsing icon */}
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
      <div className="space-y-2.5 mb-7">
        {STEPS.map((step, i) => {
          const done = doneSteps.has(i);
          const active = currentStep === i && !done;
          const pending = !done && !active;

          return (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: pending ? 0.25 : 1, x: 0 }}
              transition={{ delay: i * 0.08 }}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl border transition-all duration-300 ${
                active
                  ? 'bg-indigo-50 border-indigo-100'
                  : done
                  ? 'bg-green-50 border-green-100'
                  : 'border-transparent'
              }`}
            >
              {/* Step icon */}
              <div className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 ${
                done
                  ? 'bg-green-100 text-green-500'
                  : active
                  ? 'bg-indigo-100 text-indigo-500'
                  : 'bg-gray-100 text-gray-400'
              }`}>
                {done ? <CheckCircle2 className="w-4 h-4" /> : step.icon}
              </div>

              {/* Labels */}
              <div className="flex-1 min-w-0">
                <p className={`text-sm font-medium leading-tight ${
                  active ? 'text-indigo-700' : done ? 'text-green-700' : 'text-gray-400'
                }`}>
                  {step.label}
                </p>
                {(active || done) && (
                  <p className={`text-xs mt-0.5 ${
                    done ? 'text-green-500' : 'text-indigo-400'
                  }`}>
                    {step.detail}
                  </p>
                )}
              </div>

              {/* Spinner dots while active */}
              {active && (
                <div className="ml-auto flex gap-1 shrink-0">
                  {[0, 1, 2].map(d => (
                    <motion.div
                      key={d}
                      className="w-1.5 h-1.5 bg-indigo-400 rounded-full"
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

      {/* Elapsed + honest note */}
      <div className="text-center space-y-1">
        <p className="text-xs text-gray-400">
          Time elapsed:{' '}
          <span className="font-semibold text-gray-500">{fmt(elapsed)}</span>
          {isLastActive && elapsed > 20 && (
            <span className="ml-2 text-indigo-400">· still running on server…</span>
          )}
        </p>
        {isLastActive && (
          <p className="text-xs text-gray-400">
            Processing on a shared server — this can take up to a minute.
          </p>
        )}
      </div>
    </div>
  );
}
