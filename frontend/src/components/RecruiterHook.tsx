import { motion } from 'framer-motion';
import { Users, Brain, ShieldCheck } from 'lucide-react';

const stats = [
  {
    icon: <Users className="w-5 h-5" />,
    value: '~55–60%',
    label: 'Human accuracy',
    description: 'Unaided human judges detect lies at barely above chance level',
    highlight: false,
  },
  {
    icon: <Brain className="w-5 h-5" />,
    value: '~62.5%',
    label: 'DeceptiVision accuracy',
    description: 'Consistent improvement over human baseline using multimodal fusion',
    highlight: true,
  },
  {
    icon: <ShieldCheck className="w-5 h-5" />,
    value: '0 wires',
    label: 'Non-intrusive',
    description: 'No physical sensors. Works entirely from video — unlike a polygraph',
    highlight: false,
  },
];

export default function RecruiterHook() {
  return (
    <section id="why-it-matters" className="py-20 px-6 bg-white">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-14">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="badge mb-4 mx-auto"
          >
            Why It Matters
          </motion.div>
          <motion.h2
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-3xl font-bold text-gray-900 mb-4"
          >
            Beyond Human Intuition
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="text-gray-500 max-w-xl mx-auto text-base"
          >
            Research shows humans are barely better than chance at detecting deception. AI changes that.
          </motion.p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {stats.map((s, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-40px' }}
              transition={{ duration: 0.5, delay: i * 0.12 }}
              className={`rounded-2xl p-7 border ${
                s.highlight
                  ? 'bg-indigo-500 border-indigo-500 text-white'
                  : 'bg-white border-gray-100 shadow-sm'
              }`}
            >
              <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-5 ${
                s.highlight ? 'bg-white/20 text-white' : 'bg-indigo-50 text-indigo-500'
              }`}>
                {s.icon}
              </div>
              <div className={`text-4xl font-bold mb-1 ${s.highlight ? 'text-white' : 'text-gray-900'}`}>
                {s.value}
              </div>
              <div className={`text-sm font-semibold mb-3 ${s.highlight ? 'text-indigo-100' : 'text-gray-700'}`}>
                {s.label}
              </div>
              <p className={`text-sm leading-relaxed ${s.highlight ? 'text-indigo-100' : 'text-gray-500'}`}>
                {s.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
