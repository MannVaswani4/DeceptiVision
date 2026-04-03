import { motion } from 'framer-motion';
import {
  LineChart, Line, BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';
import { AlertTriangle, CheckCircle, Info, TrendingUp } from 'lucide-react';

interface ModalityScores {
  facial: number;
  body: number;
  audio: number;
}

interface InsightItem {
  text: string;
  severity: 'high' | 'medium' | 'low';
}

interface TimelinePoint {
  t: number;
  value: number;
}

interface ResultsData {
  prediction: 'deceptive' | 'truthful';
  confidence: number;
  scores: ModalityScores;
  insights: InsightItem[];
  emotion_timeline: TimelinePoint[];
  movement_timeline: TimelinePoint[];
  audio_timeline: TimelinePoint[];
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

function ScoreRing({ value, label, color }: { value: number; label: string; color: string }) {
  const r = 28;
  const circ = 2 * Math.PI * r;
  const dash = (value / 100) * circ;

  return (
    <div className="flex flex-col items-center gap-2">
      <svg width="72" height="72" viewBox="0 0 72 72">
        <circle cx="36" cy="36" r={r} stroke="#F3F4F6" strokeWidth="6" fill="none" />
        <motion.circle
          cx="36" cy="36" r={r}
          stroke={color}
          strokeWidth="6"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={`${circ}`}
          initial={{ strokeDashoffset: circ }}
          animate={{ strokeDashoffset: circ - dash }}
          transition={{ duration: 1.2, ease: 'easeOut', delay: 0.3 }}
          transform="rotate(-90 36 36)"
        />
        <text x="36" y="41" textAnchor="middle" fontSize="13" fontWeight="700" fill="#111827">
          {Math.round(value)}%
        </text>
      </svg>
      <span className="text-xs font-medium text-gray-500">{label}</span>
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
    t: p.t,
    emotion: p.value,
    movement: data.movement_timeline[i]?.value ?? 0,
    audio: data.audio_timeline[i]?.value ?? 0,
  }));

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="w-full max-w-4xl mx-auto space-y-5"
    >
      {/* VERDICT CARD */}
      <div className={`rounded-2xl border p-7 ${verdictBg}`}>
        <div className="flex items-start justify-between gap-6 flex-wrap">
          <div className="flex items-center gap-4">
            <div className={`w-14 h-14 rounded-2xl flex items-center justify-center ${isDeceptive ? 'bg-red-100' : 'bg-green-100'}`}>
              <VerdictIcon className={`w-7 h-7 ${verdictText}`} />
            </div>
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1">Analysis Complete</div>
              <h2 className={`text-3xl font-bold ${verdictText}`}>{verdictLabel}</h2>
            </div>
          </div>

          {/* Confidence */}
          <div className="text-right">
            <div className="text-xs text-gray-400 mb-1 font-medium">Confidence</div>
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

      {/* MODALITY SCORES */}
      <div className="card">
        <div className="flex items-center gap-2 mb-6">
          <TrendingUp className="w-4 h-4 text-indigo-500" />
          <h3 className="font-semibold text-gray-900 text-sm">Modality Scores</h3>
        </div>
        <div className="flex justify-around">
          <ScoreRing value={data.scores.facial} label="Facial" color={verdictColor} />
          <ScoreRing value={data.scores.body} label="Body" color={verdictColor} />
          <ScoreRing value={data.scores.audio} label="Audio" color={verdictColor} />
        </div>
      </div>

      {/* CHARTS */}
      <div className="grid md:grid-cols-3 gap-4">
        {/* Emotion timeline */}
        <div className="card">
          <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">Emotion Volatility</h4>
          <ResponsiveContainer width="100%" height={100}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
              <XAxis dataKey="t" hide />
              <YAxis domain={[0, 100]} hide />
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #E5E7EB' }}
                formatter={(v: unknown) => [`${(v as number).toFixed(1)}%`, 'Score']}
              />
              <Line type="monotone" dataKey="emotion" stroke="#6366F1" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Movement intensity */}
        <div className="card">
          <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">Movement Intensity</h4>
          <ResponsiveContainer width="100%" height={100}>
            <BarChart data={chartData.filter((_, i) => i % 2 === 0)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
              <XAxis dataKey="t" hide />
              <YAxis domain={[0, 100]} hide />
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #E5E7EB' }}
                formatter={(v: unknown) => [`${(v as number).toFixed(1)}%`, 'Intensity']}
              />
              <Bar dataKey="movement" fill="#6366F1" radius={[3, 3, 0, 0]} opacity={0.75} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Audio variation */}
        <div className="card">
          <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">Audio Variation</h4>
          <ResponsiveContainer width="100%" height={100}>
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
              <XAxis dataKey="t" hide />
              <YAxis domain={[0, 100]} hide />
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #E5E7EB' }}
                formatter={(v: unknown) => [`${(v as number).toFixed(1)}%`, 'Variation']}
              />
              <defs>
                <linearGradient id="audioGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366F1" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#6366F1" stopOpacity={0} />
                </linearGradient>
              </defs>
              <Area type="monotone" dataKey="audio" stroke="#6366F1" strokeWidth={2} fill="url(#audioGrad)" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* BEHAVIORAL INSIGHTS */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 text-sm mb-4">Behavioral Insights</h3>
        <div className="space-y-2.5">
          {data.insights.map((insight, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.6 + i * 0.1 }}
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
