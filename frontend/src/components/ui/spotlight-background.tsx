'use client';
import React, { useState, useCallback, useEffect } from 'react';
import { motion, useSpring, useTransform } from 'framer-motion';
import { cn } from '@/lib/utils';

type SpotlightBackgroundProps = {
  children: React.ReactNode;
  className?: string;
  spotlightSize?: number;
  spotlightClassName?: string;
};

export function SpotlightBackground({
  children,
  className,
  spotlightSize = 400,
  spotlightClassName,
}: SpotlightBackgroundProps) {
  const [isHovered, setIsHovered] = useState(false);

  const mouseX = useSpring(0, { bounce: 0, duration: 0.1 });
  const mouseY = useSpring(0, { bounce: 0, duration: 0.1 });

  const spotlightLeft = useTransform(mouseX, (x) => `${x - spotlightSize / 2}px`);
  const spotlightTop = useTransform(mouseY, (y) => `${y - spotlightSize / 2}px`);

  const handleMouseMove = useCallback(
    (event: MouseEvent) => {
      mouseX.set(event. clientX);
      mouseY. set(event.clientY);
    },
    [mouseX, mouseY]
  );

  useEffect(() => {
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseenter', () => setIsHovered(true));
    window.addEventListener('mouseleave', () => setIsHovered(false));

    setIsHovered(true);

    return () => {
      window. removeEventListener('mousemove', handleMouseMove);
      window. removeEventListener('mouseenter', () => setIsHovered(true));
      window.removeEventListener('mouseleave', () => setIsHovered(false));
    };
  }, [handleMouseMove]);

  return (
    <div className={cn('relative min-h-screen', className)}>
      {/* Spotlight effect - renders BEHIND content with z-0 */}
      <motion. div
        className={cn(
          'pointer-events-none fixed rounded-full bg-[radial-gradient(circle_at_center,var(--tw-gradient-stops),transparent_70%)] blur-3xl transition-opacity duration-300',
          // Aurora-themed gradient colors
          'from-aurora-purple/30 via-aurora-blue/20 to-transparent',
          isHovered ? 'opacity-100' : 'opacity-0',
          'z-0', // ← BEHIND everything
          spotlightClassName
        )}
        style={{
          width: spotlightSize,
          height: spotlightSize,
          left: spotlightLeft,
          top: spotlightTop,
        }}
      />

      {/* Page content - renders ON TOP with z-10 */}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
}