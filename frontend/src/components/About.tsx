import { motion } from 'framer-motion';
import { Layers, Zap, GitBranch } from 'lucide-react';

const modalities = [
  { icon: <Zap className="w-4 h-4" />, label: 'Facial Micro-expressions', detail: 'FER / OpenCV pipeline' },
  { icon: <GitBranch className="w-4 h-4" />, label: 'Body Pose Analysis', detail: 'YOLOv8 keypoints' },
  { icon: <Layers className="w-4 h-4" />, label: 'Audio Stress Features', detail: 'Spectral & prosody' },
];

export default function About() {
  return (
    <section id="about" className="py-20 px-6 bg-gray-50">
      <div className="max-w-6xl mx-auto">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          {/* Text */}
          <motion.div
            initial={{ opacity: 0, x: -24 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <div className="badge mb-4">About</div>
            <h2 className="text-3xl font-bold text-gray-900 mb-5">
              Multimodal AI.<br />
              <span className="text-indigo-500">One verdict.</span>
            </h2>
            <p className="text-gray-500 text-base leading-relaxed mb-6">
              DeceptiVision is a research-grade multimodal deception detection system. Instead of relying on a single signal channel, it fuses three independent streams of behavioral data — giving it a richer, more robust view of human behavior than any single-modality approach.
            </p>
            <p className="text-gray-500 text-base leading-relaxed">
              The feature vectors from each modality are late-fused and passed to a trained Random Forest classifier, which has been validated on publicly available deception datasets.
            </p>
          </motion.div>

          {/* Modality cards */}
          <motion.div
            initial={{ opacity: 0, x: 24 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="space-y-4"
          >
            {modalities.map((m, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.15 + i * 0.1 }}
                className="card flex items-center gap-4"
              >
                <div className="w-9 h-9 bg-indigo-50 rounded-lg flex items-center justify-center text-indigo-500 shrink-0">
                  {m.icon}
                </div>
                <div>
                  <div className="font-medium text-gray-900 text-sm">{m.label}</div>
                  <div className="text-xs text-gray-400 mt-0.5">{m.detail}</div>
                </div>
                <div className="ml-auto w-2 h-2 bg-green-400 rounded-full animate-pulse-slow" />
              </motion.div>
            ))}

            {/* Fusion box */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.5 }}
              className="mt-2 p-4 rounded-2xl bg-indigo-500 text-white text-sm flex items-center gap-3"
            >
              <div className="w-9 h-9 bg-white/20 rounded-lg flex items-center justify-center shrink-0">
                <Layers className="w-4 h-4" />
              </div>
              <div>
                <div className="font-semibold">Late Fusion → Random Forest</div>
                <div className="text-indigo-200 text-xs mt-0.5">Feature vectors combined at decision level</div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
