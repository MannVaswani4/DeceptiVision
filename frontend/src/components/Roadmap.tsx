import { motion } from 'framer-motion';
import { Radio, Brain, Target } from 'lucide-react';

const milestones = [
  {
    icon: <Radio className="w-5 h-5" />,
    title: 'Real-Time Detection',
    description: 'Live webcam stream analysis with frame-by-frame inference under 100ms latency.',
    status: 'Planned',
    statusColor: 'bg-amber-50 text-amber-600',
  },
  {
    icon: <Brain className="w-5 h-5" />,
    title: 'Transformer-Based Models',
    description: 'Replace Random Forest with fine-tuned ViT + wav2vec2 multimodal transformer for deeper context understanding.',
    status: 'Research',
    statusColor: 'bg-purple-50 text-purple-600',
  },
  {
    icon: <Target className="w-5 h-5" />,
    title: '80%+ Accuracy Goal',
    description: 'Larger training datasets, speaker diarization, and cultural normalization to push accuracy well above human baseline.',
    status: 'Goal',
    statusColor: 'bg-indigo-50 text-indigo-600',
  },
];

export default function Roadmap() {
  return (
    <section id="roadmap" className="py-20 px-6 bg-white">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-14">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="badge mb-4 mx-auto"
          >
            Roadmap
          </motion.div>
          <motion.h2
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-3xl font-bold text-gray-900 mb-4"
          >
            What's Next
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="text-gray-500 max-w-xl mx-auto text-base"
          >
            DeceptiVision is an active research project. Here's where it's going.
          </motion.p>
        </div>

        {/* Milestone timeline */}
        <div className="relative max-w-2xl mx-auto">
          {/* Vertical line */}
          <div className="absolute left-8 top-0 bottom-0 w-px bg-gradient-to-b from-indigo-200 via-indigo-100 to-transparent" />

          <div className="space-y-6">
            {milestones.map((m, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: '-40px' }}
                transition={{ duration: 0.5, delay: i * 0.12 }}
                className="relative flex gap-6"
              >
                {/* Icon node on line */}
                <div className="relative z-10 w-16 h-16 bg-white border border-gray-100 shadow-sm rounded-2xl flex items-center justify-center text-indigo-500 shrink-0">
                  {m.icon}
                </div>

                {/* Card */}
                <div className="card flex-1 mb-0">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <h3 className="font-semibold text-gray-900 text-sm mb-2">{m.title}</h3>
                      <p className="text-xs text-gray-500 leading-relaxed">{m.description}</p>
                    </div>
                    <span className={`text-xs font-semibold px-2.5 py-1 rounded-full whitespace-nowrap ${m.statusColor}`}>
                      {m.status}
                    </span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
