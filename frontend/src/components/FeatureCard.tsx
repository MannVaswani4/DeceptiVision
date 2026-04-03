import { motion } from 'framer-motion';
import { ReactNode } from 'react';

interface FeatureCardProps {
  icon: ReactNode;
  title: string;
  description: string;
  tag: string;
  delay?: number;
}

export default function FeatureCard({ icon, title, description, tag, delay = 0 }: FeatureCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 0.5, delay }}
      className="card-hover group"
    >
      <div className="flex items-start gap-4">
        <div className="w-11 h-11 bg-indigo-50 rounded-xl flex items-center justify-center text-indigo-500 shrink-0 group-hover:bg-indigo-100 transition-colors">
          {icon}
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <h3 className="font-semibold text-gray-900 text-base">{title}</h3>
            <span className="tag">{tag}</span>
          </div>
          <p className="text-sm text-gray-500 leading-relaxed">{description}</p>
        </div>
      </div>

      {/* Decorative bottom bar */}
      <div className="mt-5 h-0.5 bg-gradient-to-r from-indigo-100 to-transparent rounded-full group-hover:from-indigo-300 transition-colors duration-300" />
    </motion.div>
  );
}
