"use client";

import { Sidebar } from "@/components/Sidebar";
import { NeuralConsole } from "@/components/NeuralConsole";
import { InteractionZone } from "@/components/InteractionZone";
import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import { useEffect, useState, useRef } from "react";

// Animated Orb Component with Enhanced Motion
function AnimatedOrb({ 
  className, 
  delay = 0,
  color = "primary"
}: { 
  className?: string; 
  delay?: number;
  color?: "primary" | "pink" | "blue" | "purple";
}) {
  const colorMap = {
    primary: "from-emerald-500/20 to-cyan-500/10",
    pink: "from-pink-500/15 to-rose-500/10",
    blue: "from-blue-500/15 to-indigo-500/10", 
    purple: "from-purple-500/15 to-violet-500/10"
  };

  return (
    <motion.div
      className={`absolute rounded-full blur-3xl pointer-events-none bg-gradient-radial ${colorMap[color]} ${className}`}
      animate={{
        scale: [1, 1.3, 1, 1.2, 1],
        opacity: [0.2, 0.5, 0.3, 0.6, 0.2],
        x: [0, 50, -30, 40, 0],
        y: [0, -40, 30, -20, 0],
        rotate: [0, 90, 180, 270, 360],
      }}
      transition={{
        duration: 20 + delay * 2,
        repeat: Infinity,
        ease: "easeInOut",
        delay: delay,
      }}
    />
  );
}

// Enhanced Floating Particles with Trails
function FloatingParticles() {
  const [particles, setParticles] = useState<Array<{
    id: number;
    x: number;
    y: number;
    size: number;
    delay: number;
    duration: number;
    color: string;
  }>>([]);

  useEffect(() => {
    const colors = [
      'rgba(34, 197, 94, 0.6)',
      'rgba(59, 130, 246, 0.5)',
      'rgba(168, 85, 247, 0.5)',
      'rgba(236, 72, 153, 0.4)',
      'rgba(34, 197, 94, 0.8)',
    ];
    
    const newParticles = Array.from({ length: 50 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 4 + 1,
      delay: Math.random() * 8,
      duration: Math.random() * 15 + 10,
      color: colors[Math.floor(Math.random() * colors.length)],
    }));
    setParticles(newParticles);
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
      {particles.map((particle) => (
        <motion.div
          key={particle.id}
          className="absolute rounded-full"
          style={{
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            width: particle.size,
            height: particle.size,
            backgroundColor: particle.color,
            boxShadow: `0 0 ${particle.size * 3}px ${particle.color}`,
          }}
          animate={{
            y: [0, -150, 0],
            x: [0, Math.random() * 100 - 50, 0],
            opacity: [0, 1, 0.8, 1, 0],
            scale: [0.5, 1.5, 1, 1.2, 0.5],
          }}
          transition={{
            duration: particle.duration,
            repeat: Infinity,
            delay: particle.delay,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}

// Data Stream Animation (Matrix-like)
function DataStream() {
  const [streams, setStreams] = useState<Array<{
    id: number;
    x: number;
    chars: string[];
    speed: number;
    opacity: number;
  }>>([]);

  useEffect(() => {
    const chars = '01アイウエオカキクケコサシスセソ';
    const newStreams = Array.from({ length: 15 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      chars: Array.from({ length: 20 }, () => chars[Math.floor(Math.random() * chars.length)]),
      speed: Math.random() * 10 + 5,
      opacity: Math.random() * 0.3 + 0.1,
    }));
    setStreams(newStreams);
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden z-0 opacity-30">
      {streams.map((stream) => (
        <motion.div
          key={stream.id}
          className="absolute text-primary font-mono text-xs leading-tight"
          style={{ left: `${stream.x}%`, opacity: stream.opacity }}
          animate={{ y: ['-100%', '100vh'] }}
          transition={{
            duration: stream.speed,
            repeat: Infinity,
            ease: "linear",
          }}
        >
          {stream.chars.map((char, i) => (
            <motion.div 
              key={i}
              animate={{ opacity: [0.3, 1, 0.3] }}
              transition={{ duration: 0.5, repeat: Infinity, delay: i * 0.05 }}
            >
              {char}
            </motion.div>
          ))}
        </motion.div>
      ))}
    </div>
  );
}

// Animated Circuit Lines
function CircuitLines() {
  return (
    <svg className="fixed inset-0 w-full h-full pointer-events-none z-0 opacity-20">
      <defs>
        <linearGradient id="circuit-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="rgba(34, 197, 94, 0)" />
          <stop offset="50%" stopColor="rgba(34, 197, 94, 0.8)" />
          <stop offset="100%" stopColor="rgba(34, 197, 94, 0)" />
        </linearGradient>
      </defs>
      
      {/* Horizontal Lines */}
      {[20, 40, 60, 80].map((y) => (
        <motion.line
          key={`h-${y}`}
          x1="0%"
          y1={`${y}%`}
          x2="100%"
          y2={`${y}%`}
          stroke="url(#circuit-gradient)"
          strokeWidth="1"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ 
            pathLength: [0, 1, 0],
            opacity: [0, 0.5, 0],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            delay: y / 20,
            ease: "easeInOut",
          }}
        />
      ))}
      
      {/* Vertical Lines */}
      {[25, 50, 75].map((x) => (
        <motion.line
          key={`v-${x}`}
          x1={`${x}%`}
          y1="0%"
          x2={`${x}%`}
          y2="100%"
          stroke="url(#circuit-gradient)"
          strokeWidth="1"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ 
            pathLength: [0, 1, 0],
            opacity: [0, 0.3, 0],
          }}
          transition={{
            duration: 5,
            repeat: Infinity,
            delay: x / 25,
            ease: "easeInOut",
          }}
        />
      ))}
    </svg>
  );
}

// Pulse Rings Effect
function PulseRings() {
  return (
    <div className="fixed inset-0 pointer-events-none z-0 flex items-center justify-center">
      {[1, 2, 3, 4].map((i) => (
        <motion.div
          key={i}
          className="absolute rounded-full border border-primary/20"
          style={{ width: `${i * 200}px`, height: `${i * 200}px` }}
          animate={{
            scale: [1, 1.5, 1],
            opacity: [0.1, 0.3, 0.1],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            delay: i * 0.5,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}

// Mouse Follower Glow
function MouseGlow() {
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);
  
  const springConfig = { damping: 25, stiffness: 150 };
  const x = useSpring(mouseX, springConfig);
  const y = useSpring(mouseY, springConfig);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mouseX.set(e.clientX - 200);
      mouseY.set(e.clientY - 200);
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [mouseX, mouseY]);

  return (
    <motion.div
      className="fixed w-[400px] h-[400px] rounded-full pointer-events-none z-0"
      style={{
        x,
        y,
        background: 'radial-gradient(circle, rgba(34, 197, 94, 0.08) 0%, transparent 70%)',
      }}
    />
  );
}

// Scan Line Effect (Enhanced)
function ScanLine() {
  return (
    <>
      {/* Horizontal Scan */}
      <motion.div
        className="fixed left-0 right-0 h-[2px] pointer-events-none z-50 opacity-20"
        style={{
          background: 'linear-gradient(90deg, transparent, rgba(34, 197, 94, 0.8), transparent)',
          boxShadow: '0 0 20px rgba(34, 197, 94, 0.5)',
        }}
        animate={{ y: ["0vh", "100vh"] }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: "linear",
        }}
      />
      
      {/* Vertical Scan */}
      <motion.div
        className="fixed top-0 bottom-0 w-[2px] pointer-events-none z-50 opacity-10"
        style={{
          background: 'linear-gradient(180deg, transparent, rgba(59, 130, 246, 0.6), transparent)',
          boxShadow: '0 0 15px rgba(59, 130, 246, 0.4)',
        }}
        animate={{ x: ["0vw", "100vw"] }}
        transition={{
          duration: 12,
          repeat: Infinity,
          ease: "linear",
        }}
      />
    </>
  );
}

// Grid Overlay with Animated Dots
function GridOverlay() {
  return (
    <div className="fixed inset-0 pointer-events-none z-0">
      <div 
        className="absolute inset-0"
        style={{
          backgroundImage: `
            linear-gradient(rgba(34, 197, 94, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(34, 197, 94, 0.02) 1px, transparent 1px)
          `,
          backgroundSize: '60px 60px',
        }}
      />
      
      {/* Animated intersection dots */}
      <svg className="absolute inset-0 w-full h-full">
        {Array.from({ length: 10 }).map((_, row) =>
          Array.from({ length: 15 }).map((_, col) => (
            <motion.circle
              key={`${row}-${col}`}
              cx={`${(col + 1) * 6.67}%`}
              cy={`${(row + 1) * 10}%`}
              r="1.5"
              fill="rgba(34, 197, 94, 0.3)"
              animate={{
                opacity: [0.1, 0.5, 0.1],
                scale: [0.8, 1.2, 0.8],
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                delay: (row + col) * 0.1,
              }}
            />
          ))
        )}
      </svg>
    </div>
  );
}

// HUD Corners with Animation
function HUDCorners() {
  return (
    <>
      {/* Top Left */}
      <motion.div 
        className="fixed top-4 left-4 w-16 h-16 pointer-events-none z-50"
        initial={{ opacity: 0, scale: 0.5 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.5 }}
      >
        <motion.div 
          className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-primary to-transparent"
          animate={{ scaleX: [0.5, 1, 0.5], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
        <motion.div 
          className="absolute top-0 left-0 h-full w-[2px] bg-gradient-to-b from-primary to-transparent"
          animate={{ scaleY: [0.5, 1, 0.5], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
        />
      </motion.div>
      
      {/* Top Right */}
      <motion.div 
        className="fixed top-4 right-4 w-16 h-16 pointer-events-none z-50"
        initial={{ opacity: 0, scale: 0.5 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.6 }}
      >
        <motion.div 
          className="absolute top-0 right-0 w-full h-[2px] bg-gradient-to-l from-primary to-transparent"
          animate={{ scaleX: [0.5, 1, 0.5], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
        <motion.div 
          className="absolute top-0 right-0 h-full w-[2px] bg-gradient-to-b from-primary to-transparent"
          animate={{ scaleY: [0.5, 1, 0.5], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
        />
      </motion.div>
      
      {/* Bottom Left */}
      <motion.div 
        className="fixed bottom-4 left-4 w-16 h-16 pointer-events-none z-50"
        initial={{ opacity: 0, scale: 0.5 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.7 }}
      >
        <motion.div 
          className="absolute bottom-0 left-0 w-full h-[2px] bg-gradient-to-r from-primary to-transparent"
          animate={{ scaleX: [0.5, 1, 0.5], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
        <motion.div 
          className="absolute bottom-0 left-0 h-full w-[2px] bg-gradient-to-t from-primary to-transparent"
          animate={{ scaleY: [0.5, 1, 0.5], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
        />
      </motion.div>
      
      {/* Bottom Right */}
      <motion.div 
        className="fixed bottom-4 right-4 w-16 h-16 pointer-events-none z-50"
        initial={{ opacity: 0, scale: 0.5 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.8 }}
      >
        <motion.div 
          className="absolute bottom-0 right-0 w-full h-[2px] bg-gradient-to-l from-primary to-transparent"
          animate={{ scaleX: [0.5, 1, 0.5], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
        <motion.div 
          className="absolute bottom-0 right-0 h-full w-[2px] bg-gradient-to-t from-primary to-transparent"
          animate={{ scaleY: [0.5, 1, 0.5], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
        />
      </motion.div>
    </>
  );
}

// Floating Tech Icons
function FloatingIcons() {
  const icons = ['⚡', '🧠', '💎', '🔮', '⚛️', '🌐'];
  
  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
      {icons.map((icon, i) => (
        <motion.div
          key={i}
          className="absolute text-2xl opacity-10"
          style={{
            left: `${10 + i * 15}%`,
            top: `${20 + (i % 3) * 25}%`,
          }}
          animate={{
            y: [0, -30, 0],
            x: [0, 20, 0],
            rotate: [0, 360],
            opacity: [0.05, 0.15, 0.05],
          }}
          transition={{
            duration: 10 + i * 2,
            repeat: Infinity,
            delay: i * 0.5,
          }}
        >
          {icon}
        </motion.div>
      ))}
    </div>
  );
}

export default function Home() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="flex h-screen w-full bg-[#030305] overflow-hidden font-sans selection:bg-primary/30 selection:text-primary relative">
      {/* Background Effects */}
      <div className="fixed inset-0 bg-gradient-to-br from-black via-[#030308] to-[#050510] z-0" />
      
      {/* Animated Gradient Mesh */}
      <motion.div 
        className="fixed inset-0 gradient-mesh z-0"
        animate={{ opacity: [0.4, 0.7, 0.4] }}
        transition={{ duration: 8, repeat: Infinity }}
      />
      
      {/* Animated Orbs */}
      <AnimatedOrb className="w-[700px] h-[700px] -top-60 -left-60" color="primary" delay={0} />
      <AnimatedOrb className="w-[600px] h-[600px] top-1/3 right-0 translate-x-1/2" color="pink" delay={2} />
      <AnimatedOrb className="w-[500px] h-[500px] bottom-0 left-1/3" color="blue" delay={4} />
      <AnimatedOrb className="w-[400px] h-[400px] top-1/4 right-1/3" color="purple" delay={1} />
      <AnimatedOrb className="w-[300px] h-[300px] bottom-1/4 left-1/4" color="primary" delay={3} />
      
      {/* Circuit Lines */}
      <CircuitLines />
      
      {/* Grid Overlay with Dots */}
      <GridOverlay />
      
      {/* Data Stream (Matrix Effect) */}
      {mounted && <DataStream />}
      
      {/* Floating Particles */}
      {mounted && <FloatingParticles />}
      
      {/* Pulse Rings */}
      <PulseRings />
      
      {/* Floating Icons */}
      <FloatingIcons />
      
      {/* Mouse Glow */}
      {mounted && <MouseGlow />}
      
      {/* Scan Lines */}
      <ScanLine />
      
      {/* HUD Corners */}
      <HUDCorners />

      {/* Main Content */}
      <motion.div 
        className="flex h-screen w-full relative z-10"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
      >
        {/* Column A: Left Sidebar */}
        <motion.div
          initial={{ x: -100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2, type: "spring" }}
        >
          <Sidebar />
        </motion.div>

        {/* Column B: Center Stage */}
        <motion.div
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.4, type: "spring" }}
          className="flex-1"
        >
          <InteractionZone />
        </motion.div>

        {/* Column C: Right Sidebar */}
        <motion.div
          initial={{ x: 100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.6, type: "spring" }}
        >
          <NeuralConsole />
        </motion.div>
      </motion.div>

      {/* Vignette Effect */}
      <div 
        className="fixed inset-0 pointer-events-none z-40"
        style={{
          background: 'radial-gradient(ellipse at center, transparent 0%, rgba(0,0,0,0.5) 100%)',
        }}
      />
      
      {/* Noise Texture Overlay */}
      <div 
        className="fixed inset-0 pointer-events-none z-30 opacity-[0.015]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`,
        }}
      />
    </div>
  );
}
