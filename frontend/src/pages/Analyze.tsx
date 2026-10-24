import { useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, AlertCircle } from 'lucide-react';
import { Link } from 'react-router-dom';
import UploadZone from '../components/UploadZone';
import LoadingExperience from '../components/LoadingExperience';
import ResultsDashboard, { ResultsData } from '../components/ResultsDashboard';

type Stage = 'upload' | 'loading' | 'results' | 'error';

export default function Analyze() {
  const [stage, setStage] = useState<Stage>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<ResultsData | null>(null);
  const [errorMsg, setErrorMsg] = useState('');
  const apiPromiseRef = useRef<Promise<unknown>>(Promise.resolve());

  const handleFile = (f: File) => {
    setFile(f);
  };

  const handleAnalyze = () => {
    if (!file) return;
    setErrorMsg('');

    const formData = new FormData();
    formData.append('file', file);

    const apiBase = import.meta.env.VITE_API_URL ?? '';

    // Store the real fetch promise BEFORE switching to loading stage
    // so LoadingExperience can watch it from the very first render.
    const promise = fetch(`${apiBase}/api/analyze`, {
      method: 'POST',
      body: formData,
    })
      .then(async resp => {
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({ detail: 'Analysis failed.' }));
          throw new Error(err.detail || 'Analysis failed.');
        }
        return resp.json() as Promise<ResultsData>;
      })
      .then(data => {
        setResult(data);
        setStage('results');
      })
      .catch((err: unknown) => {
        console.error(err);
        setErrorMsg(err instanceof Error ? err.message : 'Something went wrong.');
        setStage('error');
      });

    apiPromiseRef.current = promise;
    setStage('loading');
  };

  const reset = () => {
    setStage('upload');
    setFile(null);
    setResult(null);
    setErrorMsg('');
  };

  return (
    <div className=" mt-16 bg-gray-50">
      {/* Top bar */}
      <div className="bg-white border-b border-gray-100 sticky top-0 z-40">
        <div className="max-w-5xl mx-auto px-6 h-14 flex items-center gap-4">
          <Link to="/" className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-800 transition-colors">
            <ArrowLeft className="w-4 h-4" />
            Back
          </Link>
          <div className="h-5 w-px bg-gray-200" />

          {/* Stage breadcrumb */}
          <div className="ml-auto flex items-center gap-1.5 text-xs text-gray-400">
            {(['upload', 'loading', 'results'] as Stage[]).map((s, i) => (
              <span key={s} className="flex items-center gap-1.5">
                {i > 0 && <span>›</span>}
                <span className={stage === s ? 'text-indigo-500 font-semibold' : ''}>
                  {s.charAt(0).toUpperCase() + s.slice(1)}
                </span>
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="max-w-5xl mx-auto px-6 py-12">
        <AnimatePresence mode="wait">

          {/* UPLOAD */}
          {stage === 'upload' && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
              className="max-w-xl mx-auto"
            >
              <div className="text-center mb-8">
                <h1 className="text-2xl font-bold text-gray-900 mb-2">Upload a Video</h1>
                <p className="text-gray-500 text-sm">
                  We'll analyze facial micro-expressions and body language dynamics to detect deceptive cues.
                </p>
              </div>

              <UploadZone onFileSelected={handleFile} />

              <div className="mt-6 flex justify-center">
                <button
                  onClick={handleAnalyze}
                  disabled={!file}
                  className={`btn-primary text-sm px-8 py-3 ${!file ? 'opacity-40 cursor-not-allowed' : ''}`}
                >
                  Run Analysis
                </button>
              </div>

              {/* Tips */}
              <div className="mt-8 card text-xs text-gray-400 space-y-2">
                <p className="font-medium text-gray-500 text-xs uppercase tracking-wider mb-3">Tips for best results</p>
                <p>✓ Video should have a clearly visible face throughout</p>
                <p>✓ Minimum 15–30 seconds for reliable temporal analysis</p>
                <p>✓ Good lighting and stable camera angle improve pose & face tracking</p>
                <p>✓ Subject should be positioned facing the camera</p>
              </div>
            </motion.div>
          )}

          {/* LOADING */}
          {stage === 'loading' && (
            <motion.div
              key="loading"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
              className="text-center"
            >
              <h1 className="text-2xl font-bold text-gray-900 mb-2">Analyzing…</h1>
              <p className="text-gray-400 text-sm mb-8">This usually takes 15–30 seconds depending on video length.</p>
              <LoadingExperience apiPromise={apiPromiseRef.current} />
            </motion.div>
          )}

          {/* RESULTS */}
          {stage === 'results' && result && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
            >
              <div className="flex items-center justify-between mb-8">
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">Analysis Results</h1>
                  {file && (
                    <p className="text-sm text-gray-400 mt-1">{file.name}</p>
                  )}
                </div>
                <button onClick={reset} className="btn-secondary text-sm">
                  Analyze Another
                </button>
              </div>
              <ResultsDashboard data={result} />
            </motion.div>
          )}

          {/* ERROR */}
          {stage === 'error' && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
              className="max-w-md mx-auto text-center"
            >
              <div className="card border-red-100 bg-red-50">
                <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <AlertCircle className="w-6 h-6 text-red-500" />
                </div>
                <h2 className="font-semibold text-gray-900 mb-2">Analysis Failed</h2>
                <p className="text-sm text-gray-500 mb-6">{errorMsg || 'Could not process the video. Please try again.'}</p>
                <button onClick={reset} className="btn-primary">
                  Try Again
                </button>
              </div>
            </motion.div>
          )}

        </AnimatePresence>
      </div>
    </div>
  );
}
