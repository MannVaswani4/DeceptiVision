import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Film, X, CheckCircle } from 'lucide-react';

interface UploadZoneProps {
  onFileSelected: (file: File) => void;
}

export default function UploadZone({ onFileSelected }: UploadZoneProps) {
  const [selected, setSelected] = useState<File | null>(null);

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted[0]) {
      setSelected(accepted[0]);
      onFileSelected(accepted[0]);
    }
  }, [onFileSelected]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'video/*': ['.mp4', '.mov', '.avi'] },
    multiple: false,
  });

  const clear = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelected(null);
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  return (
    <div className="w-full max-w-xl mx-auto">
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-200
          ${isDragActive
            ? 'border-indigo-400 bg-indigo-50'
            : selected
              ? 'border-green-300 bg-green-50'
              : 'border-gray-200 bg-white hover:border-indigo-300 hover:bg-indigo-50/40'
          }
        `}
      >
        <input {...getInputProps()} />

        <AnimatePresence mode="wait">
          {selected ? (
            <motion.div
              key="selected"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="flex flex-col items-center gap-3"
            >
              <div className="w-14 h-14 bg-green-100 rounded-2xl flex items-center justify-center">
                <CheckCircle className="w-7 h-7 text-green-500" />
              </div>
              <div>
                <div className="font-medium text-gray-900 text-sm">{selected.name}</div>
                <div className="text-xs text-gray-400 mt-1">{formatSize(selected.size)}</div>
              </div>
              <button
                onClick={clear}
                className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-red-500 transition-colors mt-1"
              >
                <X className="w-3.5 h-3.5" /> Remove
              </button>
            </motion.div>
          ) : isDragActive ? (
            <motion.div
              key="dragging"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-3"
            >
              <div className="w-14 h-14 bg-indigo-100 rounded-2xl flex items-center justify-center">
                <Upload className="w-7 h-7 text-indigo-500 animate-bounce" />
              </div>
              <p className="text-indigo-600 font-medium text-sm">Drop to upload</p>
            </motion.div>
          ) : (
            <motion.div
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-3"
            >
              <div className="w-14 h-14 bg-gray-100 rounded-2xl flex items-center justify-center">
                <Film className="w-7 h-7 text-gray-400" />
              </div>
              <div>
                <p className="font-medium text-gray-700 text-sm">
                  Drop your video here, or <span className="text-indigo-500">browse</span>
                </p>
                <p className="text-xs text-gray-400 mt-1">MP4, MOV, AVI — up to 500 MB</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
