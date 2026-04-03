import { useEffect, useRef, useCallback } from 'react';

interface CubesProps {
  gridSize?: number;
  maxAngle?: number;
  radius?: number;
  borderStyle?: string;
  faceColor?: string;
  rippleColor?: string;
  rippleSpeed?: number;
  autoAnimate?: boolean;
  rippleOnClick?: boolean;
  style?: React.CSSProperties;
  className?: string;
}

interface RippleWave {
  x: number;
  y: number;
  t: number;
  id: number;
}

export default function Cubes({
  gridSize = 6,
  maxAngle = 25,
  radius = 2,
  borderStyle = '1px solid #E5E7EB',
  faceColor = '#ffffff',
  rippleColor = '#6366F1',
  rippleSpeed = 1,
  autoAnimate = true,
  rippleOnClick = true,
  style,
  className = '',
}: CubesProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const ripplesRef = useRef<RippleWave[]>([]);
  const rippleIdRef = useRef(0);
  const mouseRef = useRef({ x: -1000, y: -1000 });
  const timeRef = useRef(0);

  const hexToRgb = (hex: string) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? { r: parseInt(result[1], 16), g: parseInt(result[2], 16), b: parseInt(result[3], 16) }
      : { r: 99, g: 102, b: 241 };
  };

  const parseBorder = (b: string) => {
    const parts = b.split(' ');
    return { width: parseFloat(parts[0] || '1'), color: parts[2] || '#E5E7EB' };
  };

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const cellW = W / gridSize;
    const cellH = H / gridSize;
    const border = parseBorder(borderStyle);
    const rgb = hexToRgb(rippleColor);
    const t = timeRef.current;

    for (let row = 0; row < gridSize; row++) {
      for (let col = 0; col < gridSize; col++) {
        const cx = col * cellW + cellW / 2;
        const cy = row * cellH + cellH / 2;

        // Auto-animate wave
        let angle = 0;
        if (autoAnimate) {
          angle = Math.sin(t * 0.8 + col * 0.5 + row * 0.3) * (maxAngle * 0.3);
        }

        // Mouse proximity influence
        const dx = cx - mouseRef.current.x;
        const dy = cy - mouseRef.current.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const influence = Math.max(0, 1 - dist / (cellW * radius * 2));
        const mouseAngle = influence * maxAngle;
        angle += mouseAngle;
        angle = Math.min(maxAngle, angle);

        // Ripple influence
        let rippleInfluence = 0;
        for (const wave of ripplesRef.current) {
          const wdx = cx - wave.x;
          const wdy = cy - wave.y;
          const wdist = Math.sqrt(wdx * wdx + wdy * wdy);
          const waveRadius = wave.t * rippleSpeed * 120;
          const waveFalloff = 60;
          const diff = Math.abs(wdist - waveRadius);
          if (diff < waveFalloff) {
            rippleInfluence += (1 - diff / waveFalloff) * Math.exp(-wave.t * 0.5);
          }
        }

        // Draw cube face
        const depth = (angle / maxAngle) * (cellW * 0.12);
        const pad = 4;
        const x0 = col * cellW + pad;
        const y0 = row * cellH + pad;
        const fw = cellW - pad * 2;
        const fh = cellH - pad * 2;

        // Face fill
        const ripAlpha = Math.min(1, rippleInfluence) * 0.15;
        ctx.fillStyle = faceColor;
        ctx.beginPath();
        ctx.roundRect(x0, y0, fw, fh, radius);
        ctx.fill();

        // Ripple color overlay on face
        if (ripAlpha > 0.01) {
          ctx.fillStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},${ripAlpha})`;
          ctx.beginPath();
          ctx.roundRect(x0, y0, fw, fh, radius);
          ctx.fill();
        }

        // Border
        ctx.strokeStyle = border.color;
        ctx.lineWidth = border.width;
        ctx.beginPath();
        ctx.roundRect(x0, y0, fw, fh, radius);
        ctx.stroke();

        // Top face (3D effect) — only draw if angle > 0
        if (depth > 0.5) {
          const topAlpha = Math.min(0.08, (angle / maxAngle) * 0.08);
          ctx.fillStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},${topAlpha + ripAlpha * 0.5})`;
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(x0 + depth, y0 - depth);
          ctx.lineTo(x0 + fw + depth, y0 - depth);
          ctx.lineTo(x0 + fw, y0);
          ctx.closePath();
          ctx.fill();
          ctx.strokeStyle = border.color;
          ctx.lineWidth = border.width;
          ctx.stroke();

          // Right face
          ctx.fillStyle = `rgba(0,0,0,${topAlpha * 0.5})`;
          ctx.beginPath();
          ctx.moveTo(x0 + fw, y0);
          ctx.lineTo(x0 + fw + depth, y0 - depth);
          ctx.lineTo(x0 + fw + depth, y0 + fh - depth);
          ctx.lineTo(x0 + fw, y0 + fh);
          ctx.closePath();
          ctx.fill();
          ctx.strokeStyle = border.color;
          ctx.lineWidth = border.width;
          ctx.stroke();
        }
      }
    }

    // Advance ripples
    ripplesRef.current = ripplesRef.current
      .map(w => ({ ...w, t: w.t + 0.016 * rippleSpeed }))
      .filter(w => w.t < 4);

    timeRef.current += 0.016;
    animRef.current = requestAnimationFrame(draw);
  }, [gridSize, maxAngle, radius, borderStyle, faceColor, rippleColor, rippleSpeed, autoAnimate]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const parent = canvas.parentElement;
      if (parent) {
        canvas.width = parent.clientWidth;
        canvas.height = parent.clientHeight;
      }
    };
    resize();
    window.addEventListener('resize', resize);
    animRef.current = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener('resize', resize);
    };
  }, [draw]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    mouseRef.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const handleMouseLeave = () => {
    mouseRef.current = { x: -1000, y: -1000 };
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!rippleOnClick) return;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    ripplesRef.current.push({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      t: 0,
      id: rippleIdRef.current++,
    });
  };

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ display: 'block', width: '100%', height: '100%', cursor: 'crosshair', ...style }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
    />
  );
}
