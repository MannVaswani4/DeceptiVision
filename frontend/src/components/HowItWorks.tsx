import { motion } from 'framer-motion';
import { Upload, Cpu, BarChart2 } from 'lucide-react';

const steps = [
  {
    number: '01',
    icon: <Upload className="w-5 h-5" />,
    title: 'Upload Video',
    description: 'Upload any MP4, MOV, or AVI video of the subject being analyzed. The system works with recordings as short as 15 seconds.',
  },
  {
    number: '02',
    icon: <Cpu className="w-5 h-5" />,
    title: 'AI Extracts Behavioral Signals',
    description: 'Our multimodal pipeline simultaneously analyzes micro-expressions using computer vision, body pose via YOLOv8, and audio stress features.',
  },
  {
    number: '03',
    icon: <BarChart2 className="w-5 h-5" />,
    title: 'Model Predicts Deception',
    description: 'A Random Forest classifier fuses all signals and outputs a truth/deception verdict with confidence score and behavioral insights.',
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="py-20 px-6 bg-gray-50">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-14">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="badge mb-4 mx-auto"
          >
            Process
          </motion.div>
          <motion.h2
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-3xl font-bold text-gray-900 mb-4"
          >
            How It Works
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="text-gray-500 max-w-xl mx-auto text-base"
          >
            From raw video to actionable insights — three steps, seconds of processing.
          </motion.p>
        </div>

        {/* Steps */}
        <div className="relative">
          {/* Connector line */}
          <div className="hidden md:block absolute top-10 left-1/6 right-1/6 h-px bg-gradient-to-r from-transparent via-indigo-200 to-transparent" />

          <div className="grid md:grid-cols-3 gap-8">
            {steps.map((step, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-40px' }}
                transition={{ duration: 0.5, delay: i * 0.15 }}
                className="relative text-center"
              >
                {/* Step number circle */}
                <div className="relative inline-flex mb-6">
                  <div className="w-20 h-20 bg-white rounded-2xl shadow-sm border border-gray-100 flex items-center justify-center text-indigo-500">
                    {step.icon}
                  </div>
                  <span className="absolute -top-2 -right-2 w-6 h-6 bg-indigo-500 text-white text-xs font-bold rounded-full flex items-center justify-center">
                    {i + 1}
                  </span>
                </div>
                <h3 className="font-semibold text-gray-900 text-base mb-3">{step.title}</h3>
                <p className="text-sm text-gray-500 leading-relaxed max-w-xs mx-auto">{step.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
