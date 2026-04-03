import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { ArrowRight, Eye, Activity, Mic } from 'lucide-react';
import BlurText from '../components/BlurText';
import Cubes from '../components/Cubes';
import FeatureCard from '../components/FeatureCard';
import HowItWorks from '../components/HowItWorks';
import RecruiterHook from '../components/RecruiterHook';
import About from '../components/About';
import Roadmap from '../components/Roadmap';

const features = [
  {
    icon: <Eye className="w-5 h-5" />,
    title: 'Facial Intelligence',
    tag: 'Computer Vision',
    description: 'Detects micro-expressions in 68 facial landmarks at 30fps. Identifies suppressed emotions invisible to the human eye using FER models trained on thousands of faces.',
  },
  {
    icon: <Activity className="w-5 h-5" />,
    title: 'Body Language Analysis',
    tag: 'YOLOv8 Pose',
    description: 'YOLOv8-Pose extracts 17 body keypoints per frame. Tracks hand-to-face gestures, postural shifts, and self-grooming behaviors associated with deceptive states.',
  },
  {
    icon: <Mic className="w-5 h-5" />,
    title: 'Audio Stress Detection',
    tag: 'Signal Processing',
    description: 'Analyzes vocal pitch, MFCC features, and speech rate variation. Elevated vocal stress and irregular prosody are strong independent indicators of deception.',
  },
];

export default function Landing() {
  const handleBlurComplete = () => {
    console.log('Hero animation complete.');
  };

  return (
    <div className="min-h-screen bg-white">
      {/* ── HERO ── */}
      <section className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden">
        {/* Cubes background */}
        <div className="absolute inset-0 opacity-60 pointer-events-none" style={{ pointerEvents: 'auto' }}>
          <Cubes
            gridSize={8}
            maxAngle={25}
            radius={3}
            borderStyle="1px solid #E5E7EB"
            faceColor="#ffffff"
            rippleColor="#6366F1"
            rippleSpeed={1}
            autoAnimate
            rippleOnClick
          />
        </div>

        {/* Hero content */}
        <div className="relative z-10 text-center px-6 max-w-4xl mx-auto pt-24 pb-20">
          {/* Eyebrow tag */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="flex justify-center mb-6"
          >
            <span className="badge">
              <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-pulse" />
              Multimodal AI · Deception Detection
            </span>
          </motion.div>

          {/* BlurText heading */}
          <div className="mb-6 flex justify-center">
            <BlurText
              text="AI-Powered Deception Detection"
              delay={120}
              animateBy="words"
              direction="top"
              onAnimationComplete={handleBlurComplete}
              className="text-5xl md:text-6xl font-bold flex justify-center text-gray-900 leading-tight"
            />
          </div>

          {/* Subheading */}
          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9, duration: 0.6 }}
            className="text-lg text-gray-500 max-w-2xl mx-auto leading-relaxed mb-10"
          >
            Analyze facial expressions, body language, and voice to detect hidden behavioral patterns.
            A multimodal fusion system that sees what humans miss.
          </motion.p>

          {/* CTAs */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.1, duration: 0.5 }}
            className="flex flex-wrap gap-3 justify-center"
          >
            <Link to="/analyze" className="btn-primary text-base px-8 py-3.5">
              Get Started <ArrowRight className="w-4 h-4" />
            </Link>
            <a href="#how-it-works" className="btn-secondary text-base px-8 py-3.5">
              See How It Works
            </a>
          </motion.div>

          {/* Social proof strip */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.5 }}
            className="mt-14 flex flex-wrap justify-center gap-8 text-xs text-gray-400"
          >
            {['Facial · Body · Audio Fusion', 'Random Forest Classifier', '62.5% Accuracy', 'Non-Intrusive Analysis'].map(item => (
              <span key={item} className="flex items-center gap-2">
                <span className="w-1 h-1 bg-indigo-300 rounded-full" />
                {item}
              </span>
            ))}
          </motion.div>
        </div>

        {/* Bottom fade */}
        <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-white to-transparent pointer-events-none" />
      </section>

      {/* ── FEATURES ── */}
      <section id="features" className="py-20 px-6 bg-white">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-14">
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="badge mb-4 mx-auto"
            >
              Capabilities
            </motion.div>
            <motion.h2
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 }}
              className="text-3xl font-bold text-gray-900 mb-4"
            >
              Three Signals. One Truth.
            </motion.h2>
            <motion.p
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="text-gray-500 max-w-xl mx-auto text-base"
            >
              Each modality contributes a unique behavioral lens. Combined, they form a more complete picture than any single approach.
            </motion.p>
          </div>
          <div className="grid md:grid-cols-3 gap-5">
            {features.map((f, i) => (
              <FeatureCard key={i} {...f} delay={i * 0.1} />
            ))}
          </div>
        </div>
      </section>

      {/* ── HOW IT WORKS ── */}
      <HowItWorks />

      {/* ── RECRUITER HOOK ── */}
      <RecruiterHook />

      {/* ── ABOUT ── */}
      <About />

      {/* ── ROADMAP ── */}
      <Roadmap />

      {/* ── FOOTER CTA ── */}
      <section className="py-20 px-6 bg-indigo-500">
        <div className="max-w-3xl mx-auto text-center">
          <motion.h2
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-3xl font-bold text-white mb-4"
          >
            Ready to see it in action?
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-indigo-100 mb-8 text-base"
          >
            Upload a video and get a full behavioral analysis in seconds.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
          >
            <Link to="/analyze" className="inline-flex items-center gap-2 bg-white text-indigo-600 px-8 py-3.5 rounded-xl font-semibold text-sm hover:bg-indigo-50 transition-colors shadow-sm">
              Analyze a Video <ArrowRight className="w-4 h-4" />
            </Link>
          </motion.div>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer className="py-8 px-6 bg-white border-t border-gray-100">
        <div className="max-w-6xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <div className="w-6 h-6 bg-indigo-500 rounded-md flex items-center justify-center">
              <Eye className="w-3 h-3 text-white" />
            </div>
            <span>DeceptiVision</span>
            <span className="text-gray-300">·</span>
            <span>Multimodal AI Deception Detection</span>
          </div>
          <span className="text-xs text-gray-400">Built with React · FastAPI · YOLOv8 · Random Forest</span>
        </div>
      </footer>
    </div>
  );
}
