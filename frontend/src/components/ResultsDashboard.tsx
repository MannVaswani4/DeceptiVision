import { motion } from 'framer-motion';
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';
import { AlertTriangle, CheckCircle, Info, Activity, Eye, ShieldCheck, Film } from 'lucide-react';

export interface ModalityScores {
  facial: number;
  body: number;
}

export interface InsightItem {
  text: string;
  severity: 'high' | 'medium' | 'low';
}

export interface TimelinePoint {
  t: number;
  value: number;
}

export interface Diagnostics {
  frames_analyzed: number;
  dominant_emotion: string;
  dominant_emotion_pct: number;
  transitions_count: number;
  hand_face_proximity: string;
  posture_stability: string;
}

export interface ResultsData {
  prediction: 'deceptive' | 'truthful';
  confidence: number;
  scores: ModalityScores;
  insights: InsightItem[];
  emotion_timeline: TimelinePoint[];
  movement_timeline: TimelinePoint[];
  diagnostics?: Diagnostics;
}

const SEVERITY_STYLES: Record<string, string> = {
  high: 'bg-red-50 text-red-700 border border-red-100',
  medium: 'bg-amber-50 text-amber-700 border border-amber-100',
  low: 'bg-green-50 text-green-700 border border-green-100',
};

const SEVERITY_ICONS: Record<string, JSX.Element> = {
  high: <AlertTriangle className="w-4 h-4 shrink-0" />,
  medium: <Info className="w-4 h-4 shrink-0" />,
  low: <CheckCircle className="w-4 h-4 shrink-0" />,
};

function ScoreRing({ value, label, subtext, color }: { value: number; label: string; subtext: string; color: string }) {
  const r = 32;
  const circ = 2 * Math.PI * r;
  const dash = (value / 100) * circ;

  return (
    <div className="flex flex-col items-center text-center p-3">
      <svg width="84" height="84" viewBox="0 0 84 84">
        <circle cx="42" cy="42" r={r} stroke="#F3F4F6" strokeWidth="7" fill="none" />
        <motion.circle
          cx="42" cy="42" r={r}
          stroke={color}
          strokeWidth="7"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={`${circ}`}
          initial={{ strokeDashoffset: circ }}
          animate={{ strokeDashoffset: circ - dash }}
          transition={{ duration: 1.2, ease: 'easeOut', delay: 0.3 }}
          transform="rotate(-90 42 42)"
        />
        <text x="42" y="47" textAnchor="middle" fontSize="15" fontWeight="700" fill="#111827">
          {Math.round(value)}%
        </text>
      </svg>
      <span className="text-sm font-semibold text-gray-800 mt-1">{label}</span>
      <span className="text-xs text-gray-400">{subtext}</span>
    </div>
  );
}

export default function ResultsDashboard({ data }: { data: ResultsData }) {
  const isDeceptive = data.prediction === 'deceptive';

  const verdictColor = isDeceptive ? '#EF4444' : '#10B981';
  const verdictBg = isDeceptive ? 'bg-red-50 border-red-100' : 'bg-green-50 border-green-100';
  const verdictText = isDeceptive ? 'text-red-700' : 'text-green-700';
  const verdictLabel = isDeceptive ? 'Likely Deceptive' : 'Likely Truthful';
  const VerdictIcon = isDeceptive ? AlertTriangle : CheckCircle;

  const chartData = data.emotion_timeline.map((p, i) => ({
    frame: `Frame ${p.t + 1}`,
    emotion: p.value,
    movement: data.movement_timeline[i]?.value ?? 0,
  }));

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="w-full max-w-4xl mx-auto space-y-6"
    >
      {/* ── VERDICT CARD ── */}
      <div className={`rounded-2xl border p-7 ${verdictBg}`}>
        <div className="flex items-start justify-between gap-6 flex-wrap">
          <div className="flex items-center gap-4">
            <div className={`w-14 h-14 rounded-2xl flex items-center justify-center ${isDeceptive ? 'bg-red-100' : 'bg-green-100'}`}>
              <VerdictIcon className={`w-7 h-7 ${verdictText}`} />
            </div>
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1">
                Classification Complete
              </div>
              <h2 className={`text-3xl font-bold ${verdictText}`}>{verdictLabel}</h2>
              <p className="text-xs text-gray-500 mt-1">
                Evaluated by 300-tree Random Forest classifier across 30 multimodal features
              </p>
            </div>
          </div>

          {/* Model Confidence */}
          <div className="text-right">
            <div className="text-xs text-gray-400 mb-1 font-medium">Model Confidence</div>
            <div className={`text-4xl font-bold ${verdictText}`}>{data.confidence}%</div>
            <div className="mt-2 h-1.5 w-32 bg-gray-200 rounded-full overflow-hidden ml-auto">
              <motion.div
                className="h-full rounded-full"
                style={{ backgroundColor: verdictColor }}
                initial={{ width: 0 }}
                animate={{ width: `${data.confidence}%` }}
                transition={{ duration: 1.2, ease: 'easeOut', delay: 0.2 }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* ── DUAL MODALITY GAUGES ── */}
      <div className="card">
        <div className="flex items-center justify-between mb-4 border-b border-gray-100 pb-3">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-indigo-500" />
            <h3 className="font-semibold text-gray-900 text-sm">Active Modalities (Vision Pipeline)</h3>
          </div>
          <span className="text-xs text-gray-400 font-medium">2 Streams · 30 Features</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 divide-y sm:divide-y-0 sm:divide-x divide-gray-100 pt-2">
          <div className="flex items-center justify-center gap-4">
            <ScoreRing
              value={data.scores.facial}
              label="Facial Dynamics"
              subtext="PyTorch FaceCNN (23 Features)"
              color={verdictColor}
            />
          </div>
          <div className="flex items-center justify-center gap-4 pt-4 sm:pt-0">
            <ScoreRing
              value={data.scores.body}
              label="Kinesics & Pose"
              subtext="YOLOv8-Pose 17 Keypoints (7 Features)"
              color={verdictColor}
            />
          </div>
        </div>
      </div>

      {/* ── REAL FEATURE TIMELINES ── */}
      <div className="grid md:grid-cols-2 gap-5">
        {/* Emotion Confidence / Volatility */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Eye className="w-4 h-4 text-indigo-500" />
              <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wider">
                Facial Affect Intensity
              </h4>
            </div>
            <span className="text-[11px] text-gray-400">Frame-by-Frame</span>
          </div>
          <ResponsiveContainer width="100%" height={140}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
              <XAxis dataKey="frame" hide />
              <YAxis domain={[0, 100]} hide />
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #E5E7EB' }}
                formatter={(v: unknown) => [`${(v as number).toFixed(1)}%`, 'Intensity']}
              />
              <Line type="monotone" dataKey="emotion" stroke="#6366F1" strokeWidth={2} dot={{ r: 2 }} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-[11px] text-gray-400 mt-3 text-center">
            Per-frame dominant emotion probability from 48×48 FaceCNN
          </p>
        </div>

        {/* Movement Intensity */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-indigo-500" />
              <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wider">
                Keypoint Displacement
              </h4>
            </div>
            <span className="text-[11px] text-gray-400">YOLOv8 Pose</span>
          </div>
          <ResponsiveContainer width="100%" height={140}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
              <XAxis dataKey="frame" hide />
              <YAxis domain={[0, 100]} hide />
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #E5E7EB' }}
                formatter={(v: unknown) => [`${(v as number).toFixed(1)}%`, 'Displacement']}
              />
              <Bar dataKey="movement" fill="#6366F1" radius={[3, 3, 0, 0]} opacity={0.8} />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-[11px] text-gray-400 mt-3 text-center">
            Frame-to-frame Euclidean shift of 17 body keypoints
          </p>
        </div>
      </div>

      {/* ── VIDEO DIAGNOSTICS (NEW: 100% EVIDENCE BACKED) ── */}
      {data.diagnostics && (
        <div className="card">
          <div className="flex items-center gap-2 mb-4 border-b border-gray-100 pb-3">
            <Film className="w-4 h-4 text-indigo-500" />
            <h3 className="font-semibold text-gray-900 text-sm">Extracted Video Diagnostics</h3>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
            <div className="bg-gray-50 rounded-xl p-3 text-center">
              <div className="text-[11px] uppercase tracking-wider text-gray-400 font-medium">Frames Sampled</div>
              <div className="text-lg font-bold text-gray-900 mt-1">{data.diagnostics.frames_analyzed}</div>
            </div>
            <div className="bg-gray-50 rounded-xl p-3 text-center">
              <div className="text-[11px] uppercase tracking-wider text-gray-400 font-medium">Dominant Emotion</div>
              <div className="text-lg font-bold text-indigo-600 mt-1">
                {data.diagnostics.dominant_emotion}
              </div>
              <div className="text-[10px] text-gray-400">{data.diagnostics.dominant_emotion_pct}% presence</div>
            </div>
            <div className="bg-gray-50 rounded-xl p-3 text-center">
              <div className="text-[11px] uppercase tracking-wider text-gray-400 font-medium">Emotion Switches</div>
              <div className="text-lg font-bold text-gray-900 mt-1">{data.diagnostics.transitions_count}</div>
            </div>
            <div className="bg-gray-50 rounded-xl p-3 text-center">
              <div className="text-[11px] uppercase tracking-wider text-gray-400 font-medium">Hand-to-Face</div>
              <div className="text-sm font-bold text-gray-900 mt-2">{data.diagnostics.hand_face_proximity}</div>
            </div>
            <div className="bg-gray-50 rounded-xl p-3 text-center col-span-2 sm:col-span-1">
              <div className="text-[11px] uppercase tracking-wider text-gray-400 font-medium">Posture Stability</div>
              <div className="text-sm font-bold text-gray-900 mt-2">{data.diagnostics.posture_stability}</div>
            </div>
          </div>
        </div>
      )}

      {/* ── BEHAVIORAL INSIGHTS ── */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 text-sm mb-4">Computed Behavioral Observations</h3>
        <div className="space-y-2.5">
          {data.insights.map((insight, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 + i * 0.1 }}
              className={`flex items-start gap-3 px-4 py-3 rounded-xl text-sm ${SEVERITY_STYLES[insight.severity]}`}
            >
              {SEVERITY_ICONS[insight.severity]}
              <span>{insight.text}</span>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}
