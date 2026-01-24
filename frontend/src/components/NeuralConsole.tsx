"use client";

import React from 'react';
import { useAgentStore, AgentRole } from "@/store/useAgentStore";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { motion, AnimatePresence } from "framer-motion";
import { Terminal, Activity, Zap, Search, AlertTriangle, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

const agentConfig: Record<AgentRole, { color: string; icon: any; label: string }> = {
    Supervisor: { color: "text-blue-400 border-blue-400/30 bg-blue-400/10", icon: Zap, label: "Supervisor" },
    RAG: { color: "text-amber-400 border-amber-400/30 bg-amber-400/10", icon: Search, label: "RAG Engine" },
    Diagnostician: { color: "text-rose-500 border-rose-500/30 bg-rose-500/10", icon: AlertTriangle, label: "Diagnostician" },
    Fixer: { color: "text-emerald-400 border-emerald-400/30 bg-emerald-400/10", icon: CheckCircle2, label: "Fixer" },
};

export function NeuralConsole() {
    const logs = useAgentStore((state) => state.logs);
    const isThinking = useAgentStore((state) => state.isThinking);

    return (
        <aside className="w-80 border-l border-white/10 bg-black/40 backdrop-blur-xl flex flex-col h-screen overflow-hidden">
            <div className="p-6 border-b border-white/10 flex items-center justify-between bg-black/20">
                <div className="flex items-center gap-2">
                    <Terminal className="w-5 h-5 text-primary" />
                    <h2 className="text-sm font-bold uppercase tracking-widest">Neural Console</h2>
                </div>
                <div className="flex items-center gap-2">
                    <div className={cn(
                        "h-2 w-2 rounded-full",
                        isThinking ? "bg-primary animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.8)]" : "bg-muted-foreground/30"
                    )} />
                    <span className="text-[10px] text-muted-foreground font-mono uppercase tracking-tighter">
                        {isThinking ? "Processing" : "Idle"}
                    </span>
                </div>
            </div>

            <ScrollArea className="flex-1 p-4">
                <div className="space-y-6">
                    <AnimatePresence initial={false} mode="popLayout">
                        {logs.map((log, index) => {
                            const config = agentConfig[log.agent];
                            const Icon = config.icon;

                            return (
                                <motion.div
                                    key={log.id}
                                    initial={{ opacity: 0, x: 20, filter: "blur(10px)" }}
                                    animate={{ opacity: 1, x: 0, filter: "blur(0px)" }}
                                    exit={{ opacity: 0, x: -20 }}
                                    transition={{
                                        type: "spring",
                                        stiffness: 150,
                                        damping: 20,
                                        delay: index === 0 ? 0 : 0.1
                                    }}
                                    className="space-y-2 relative"
                                >
                                    <div className="flex items-center justify-between">
                                        <Badge variant="outline" className={cn("text-[9px] py-0 px-1.5 font-bold uppercase border h-5", config.color)}>
                                            <Icon className="w-3 h-3 mr-1" />
                                            {config.label}
                                        </Badge>
                                        <span suppressHydrationWarning className="text-[9px] text-muted-foreground/60 font-mono italic">
                                            {log.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                        </span>
                                    </div>
                                    <div className="pl-3 border-l border-primary/20 py-1 transition-colors hover:border-primary/50">
                                        <p className="text-[11px] text-foreground/80 leading-relaxed font-mono">
                                            <span className="text-primary/60 font-bold mr-2">{">"}</span>
                                            {log.message}
                                        </p>
                                    </div>
                                    {log.status === 'active' && (
                                        <motion.div
                                            className="h-[1px] bg-primary/20 w-full overflow-hidden mt-2"
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                        >
                                            <motion.div
                                                className="h-full bg-primary"
                                                animate={{ x: ["-100%", "100%"] }}
                                                transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                                                style={{ width: "40%" }}
                                            />
                                        </motion.div>
                                    )}
                                </motion.div>
                            );
                        })}
                    </AnimatePresence>
                </div>
            </ScrollArea>

            <div className="p-4 border-t border-white/10 bg-black/60">
                <div className="flex items-center gap-4">
                    <div className="relative flex h-2 w-2">
                        <span className={cn(
                            "absolute inline-flex h-full w-full rounded-full opacity-75",
                            isThinking ? "animate-ping bg-primary" : "bg-muted-foreground/20"
                        )}></span>
                        <span className={cn(
                            "relative inline-flex rounded-full h-2 w-2",
                            isThinking ? "bg-primary" : "bg-muted-foreground/40"
                        )}></span>
                    </div>
                    <div className="flex-1">
                        <div className="text-[9px] uppercase font-bold text-muted-foreground mb-1.5 tracking-[0.2em]">Signal Pipeline</div>
                        <div className="flex gap-1.5">
                            {[1, 2, 3, 4, 5, 6].map((i) => (
                                <div key={i} className="h-1 flex-1 bg-white/5 rounded-full overflow-hidden border border-white/5">
                                    <motion.div
                                        className="h-full bg-primary/50 shadow-[0_0_8px_rgba(34,197,94,0.4)]"
                                        animate={{
                                            opacity: isThinking ? [0.2, 1, 0.2] : 0.2,
                                            scaleY: isThinking ? [1, 1.5, 1] : 1
                                        }}
                                        transition={{ repeat: Infinity, duration: 1.5, delay: i * 0.15 }}
                                    />
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </aside>
    );
}
