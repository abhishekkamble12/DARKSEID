"use client";

import React from 'react';
import { useAgentStore, AgentRole } from "@/store/useAgentStore";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { motion, AnimatePresence } from "framer-motion";
import { Terminal, Activity, Zap, Search, AlertTriangle, CheckCircle2, Cpu, Radio, Waves } from "lucide-react";
import { cn } from "@/lib/utils";

const agentConfig: Record<AgentRole, { 
    color: string; 
    bgColor: string;
    glowColor: string;
    icon: any; 
    label: string 
}> = {
    Supervisor: { 
        color: "text-blue-400", 
        bgColor: "bg-blue-500/10 border-blue-500/30",
        glowColor: "rgba(59, 130, 246, 0.5)",
        icon: Zap, 
        label: "Supervisor" 
    },
    Research: { 
        color: "text-cyan-400", 
        bgColor: "bg-cyan-500/10 border-cyan-500/30",
        glowColor: "rgba(6, 182, 212, 0.5)",
        icon: Search, 
        label: "Research" 
    },
    Examiner: { 
        color: "text-purple-400", 
        bgColor: "bg-purple-500/10 border-purple-500/30",
        glowColor: "rgba(168, 85, 247, 0.5)",
        icon: CheckCircle2, 
        label: "Examiner" 
    },
    Chat: { 
        color: "text-emerald-400", 
        bgColor: "bg-emerald-500/10 border-emerald-500/30",
        glowColor: "rgba(16, 185, 129, 0.5)",
        icon: Activity, 
        label: "Chat" 
    },
    RAG: { 
        color: "text-amber-400", 
        bgColor: "bg-amber-500/10 border-amber-500/30",
        glowColor: "rgba(245, 158, 11, 0.5)",
        icon: Search, 
        label: "RAG Engine" 
    },
    LearningArchitect: { 
        color: "text-pink-400", 
        bgColor: "bg-pink-500/10 border-pink-500/30",
        glowColor: "rgba(236, 72, 153, 0.5)",
        icon: Cpu, 
        label: "Architect" 
    },
    Voice: { 
        color: "text-indigo-400", 
        bgColor: "bg-indigo-500/10 border-indigo-500/30",
        glowColor: "rgba(99, 102, 241, 0.5)",
        icon: Waves, 
        label: "Voice" 
    },
};

export function NeuralConsole() {
    const logs = useAgentStore((state) => state.logs);
    const isThinking = useAgentStore((state) => state.isThinking);

    return (
        <aside className="w-80 border-l border-white/[0.06] glass-panel flex flex-col h-screen overflow-hidden relative">
            {/* Decorative Elements */}
            <div className="absolute top-0 left-0 w-32 h-32 bg-gradient-radial from-primary/10 to-transparent blur-3xl pointer-events-none" />
            <div className="absolute bottom-20 right-0 w-24 h-24 bg-gradient-radial from-purple-500/10 to-transparent blur-2xl pointer-events-none" />

            {/* Header */}
            <motion.div 
                className="p-5 border-b border-white/[0.06] relative"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
            >
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <motion.div 
                            className="relative"
                            animate={isThinking ? { rotate: 360 } : {}}
                            transition={{ duration: 3, repeat: isThinking ? Infinity : 0, ease: "linear" }}
                        >
                            <div className={cn(
                                "p-2 rounded-lg transition-all duration-300",
                                isThinking 
                                    ? "bg-primary/20 border border-primary/40" 
                                    : "bg-white/5 border border-white/10"
                            )}>
                                <Terminal className={cn(
                                    "w-5 h-5 transition-colors",
                                    isThinking ? "text-primary" : "text-muted-foreground"
                                )} />
                            </div>
                            {isThinking && (
                                <motion.div 
                                    className="absolute inset-0 rounded-lg border border-primary"
                                    animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0, 0.5] }}
                                    transition={{ duration: 1.5, repeat: Infinity }}
                                />
                            )}
                        </motion.div>
                        <div>
                            <h2 className="text-sm font-bold uppercase tracking-[0.1em] bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
                                Neural Console
                            </h2>
                            <p className="text-[9px] text-muted-foreground/60 font-mono uppercase tracking-wider">
                                Agent Activity Log
                            </p>
                        </div>
                    </div>
                    
                    {/* Status Indicator */}
                    <div className={cn(
                        "flex items-center gap-2 px-2.5 py-1 rounded-full border transition-all",
                        isThinking 
                            ? "bg-primary/10 border-primary/30" 
                            : "bg-white/[0.02] border-white/[0.06]"
                    )}>
                        <motion.div 
                            className={cn(
                                "h-2 w-2 rounded-full",
                                isThinking ? "bg-primary" : "bg-muted-foreground/30"
                            )}
                            animate={isThinking ? { 
                                scale: [1, 1.3, 1],
                                boxShadow: [
                                    '0 0 5px rgba(34,197,94,0.5)',
                                    '0 0 15px rgba(34,197,94,0.8)',
                                    '0 0 5px rgba(34,197,94,0.5)'
                                ]
                            } : {}}
                            transition={{ duration: 1, repeat: isThinking ? Infinity : 0 }}
                        />
                        <span className={cn(
                            "text-[9px] font-bold uppercase tracking-wider",
                            isThinking ? "text-primary" : "text-muted-foreground/60"
                        )}>
                            {isThinking ? "Active" : "Idle"}
                        </span>
                    </div>
                </div>
            </motion.div>

            {/* Logs Area */}
            <ScrollArea className="flex-1 p-4">
                <div className="space-y-4">
                    <AnimatePresence initial={false} mode="popLayout">
                        {logs.map((log, index) => {
                            const config = agentConfig[log.agent];
                            const Icon = config.icon;

                            return (
                                <motion.div
                                    key={log.id}
                                    layout
                                    initial={{ opacity: 0, x: 30, scale: 0.9 }}
                                    animate={{ opacity: 1, x: 0, scale: 1 }}
                                    exit={{ opacity: 0, x: -30, scale: 0.9 }}
                                    transition={{
                                        type: "spring",
                                        stiffness: 200,
                                        damping: 25,
                                        delay: index === 0 ? 0.1 : 0
                                    }}
                                    className="relative group"
                                >
                                    {/* Glow on active */}
                                    {log.status === 'active' && (
                                        <motion.div 
                                            className="absolute -inset-1 rounded-xl opacity-30 blur-lg"
                                            style={{ backgroundColor: config.glowColor }}
                                            animate={{ opacity: [0.2, 0.4, 0.2] }}
                                            transition={{ duration: 1.5, repeat: Infinity }}
                                        />
                                    )}
                                    
                                    <div className={cn(
                                        "relative p-3 rounded-xl border transition-all duration-300",
                                        "bg-white/[0.02] border-white/[0.06]",
                                        "hover:bg-white/[0.04] hover:border-white/[0.1]",
                                        log.status === 'active' && "border-l-2",
                                        log.status === 'active' && config.color.replace('text-', 'border-l-')
                                    )}>
                                        {/* Header */}
                                        <div className="flex items-center justify-between mb-2">
                                            <Badge 
                                                variant="outline" 
                                                className={cn(
                                                    "text-[9px] py-0.5 px-2 font-bold uppercase border rounded-md gap-1.5",
                                                    config.bgColor,
                                                    config.color
                                                )}
                                            >
                                                <Icon className="w-3 h-3" />
                                                {config.label}
                                            </Badge>
                                            <span suppressHydrationWarning className="text-[9px] text-muted-foreground/40 font-mono">
                                                {log.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                            </span>
                                        </div>
                                        
                                        {/* Message */}
                                        <div className="relative">
                                            <div className="absolute left-0 top-0 bottom-0 w-0.5 bg-gradient-to-b from-primary/40 to-transparent rounded-full" />
                                            <p className="text-[11px] text-foreground/70 leading-relaxed pl-3 font-mono">
                                                <span className={cn("font-bold mr-1.5", config.color)}>{">"}</span>
                                                {log.message}
                                            </p>
                                        </div>

                                        {/* Active Progress Bar */}
                                        {log.status === 'active' && (
                                            <motion.div
                                                className="mt-3 h-1 bg-white/[0.03] rounded-full overflow-hidden"
                                                initial={{ opacity: 0 }}
                                                animate={{ opacity: 1 }}
                                            >
                                                <motion.div
                                                    className={cn(
                                                        "h-full rounded-full",
                                                        "bg-gradient-to-r from-primary via-emerald-400 to-primary"
                                                    )}
                                                    animate={{ x: ["-100%", "100%"] }}
                                                    transition={{ 
                                                        repeat: Infinity, 
                                                        duration: 1.5, 
                                                        ease: "linear" 
                                                    }}
                                                    style={{ width: "50%" }}
                                                />
                                            </motion.div>
                                        )}
                                    </div>
                                </motion.div>
                            );
                        })}
                    </AnimatePresence>
                </div>
            </ScrollArea>

            {/* Enhanced Signal Pipeline Footer */}
            <motion.div 
                className="p-4 border-t border-white/[0.06] relative overflow-hidden"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
            >
                {/* Animated background glow */}
                <motion.div
                    className="absolute inset-0 bg-gradient-to-t from-primary/5 to-transparent"
                    animate={{ opacity: isThinking ? [0.3, 0.6, 0.3] : 0.1 }}
                    transition={{ duration: 2, repeat: Infinity }}
                />
                
                <div className="relative z-10 flex items-center gap-4">
                    {/* Enhanced Status Dot */}
                    <div className="relative">
                        <motion.div 
                            className={cn(
                                "w-4 h-4 rounded-full",
                                isThinking ? "bg-primary" : "bg-muted-foreground/30"
                            )}
                            animate={isThinking ? {
                                scale: [1, 1.3, 1],
                                boxShadow: [
                                    '0 0 15px rgba(34,197,94,0.5)',
                                    '0 0 35px rgba(34,197,94,0.8)',
                                    '0 0 15px rgba(34,197,94,0.5)'
                                ]
                            } : {}}
                            transition={{ duration: 1, repeat: Infinity }}
                        />
                        {isThinking && (
                            <>
                                <motion.div 
                                    className="absolute inset-0 rounded-full border-2 border-primary"
                                    animate={{ scale: [1, 2.5], opacity: [0.8, 0] }}
                                    transition={{ duration: 1.5, repeat: Infinity }}
                                />
                                <motion.div 
                                    className="absolute inset-0 rounded-full border border-primary/50"
                                    animate={{ scale: [1, 3], opacity: [0.5, 0] }}
                                    transition={{ duration: 1.5, repeat: Infinity, delay: 0.3 }}
                                />
                            </>
                        )}
                    </div>

                    {/* Signal Bars with Audio Visualizer Effect */}
                    <div className="flex-1">
                        <div className="text-[8px] uppercase font-bold text-muted-foreground/50 mb-2 tracking-[0.2em] flex items-center gap-2">
                            <motion.div
                                animate={{ rotate: isThinking ? 360 : 0 }}
                                transition={{ duration: 3, repeat: isThinking ? Infinity : 0, ease: "linear" }}
                            >
                                <Waves className="w-3 h-3" />
                            </motion.div>
                            Signal Pipeline
                            {isThinking && (
                                <motion.span
                                    className="text-primary"
                                    animate={{ opacity: [0.5, 1, 0.5] }}
                                    transition={{ duration: 0.5, repeat: Infinity }}
                                >
                                    ●
                                </motion.span>
                            )}
                        </div>
                        <div className="flex gap-1">
                            {Array.from({ length: 12 }).map((_, i) => (
                                <motion.div 
                                    key={i} 
                                    className="h-6 flex-1 rounded-sm overflow-hidden bg-white/[0.02] border border-white/[0.04]"
                                >
                                    <motion.div
                                        className={cn(
                                            "h-full w-full rounded-sm",
                                            isThinking 
                                                ? "bg-gradient-to-t from-primary/80 via-emerald-400/60 to-cyan-400/40" 
                                                : "bg-white/5"
                                        )}
                                        animate={isThinking ? {
                                            scaleY: [0.2, 0.9, 0.4, 1, 0.3, 0.7, 0.2],
                                            opacity: [0.4, 1, 0.6, 1, 0.5, 0.8, 0.4]
                                        } : { scaleY: 0.15, opacity: 0.2 }}
                                        transition={{ 
                                            repeat: Infinity, 
                                            duration: 0.6 + Math.random() * 0.4, 
                                            delay: i * 0.05,
                                            ease: "easeInOut"
                                        }}
                                        style={{ 
                                            originY: 1,
                                            boxShadow: isThinking ? '0 0 8px rgba(34,197,94,0.6)' : 'none'
                                        }}
                                    />
                                </motion.div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Enhanced Bottom Stats */}
                <motion.div 
                    className="mt-3 flex items-center justify-between"
                    animate={{ opacity: [0.6, 1, 0.6] }}
                    transition={{ duration: 3, repeat: Infinity }}
                >
                    <motion.div 
                        className="flex items-center gap-1.5 text-[8px] text-muted-foreground/50 font-mono uppercase px-2 py-1 rounded-md bg-white/[0.02]"
                        whileHover={{ scale: 1.05, backgroundColor: 'rgba(255,255,255,0.05)' }}
                    >
                        <motion.div
                            animate={{ rotate: [0, 360] }}
                            transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                        >
                            <Cpu className="w-3 h-3 text-primary/60" />
                        </motion.div>
                        <span>v2.0.1</span>
                    </motion.div>
                    
                    <motion.div 
                        className="flex items-center gap-1.5 text-[8px] text-muted-foreground/50 font-mono uppercase px-2 py-1 rounded-md bg-white/[0.02]"
                        whileHover={{ scale: 1.05, backgroundColor: 'rgba(255,255,255,0.05)' }}
                    >
                        <motion.div
                            animate={{ scale: [1, 1.2, 1] }}
                            transition={{ duration: 1, repeat: Infinity }}
                        >
                            <Radio className="w-3 h-3 text-blue-400/60" />
                        </motion.div>
                        <span>{logs.length} events</span>
                    </motion.div>
                </motion.div>
            </motion.div>
        </aside>
    );
}
