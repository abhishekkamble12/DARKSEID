"use client";

import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ChatInterface } from "./ChatInterface";
import { useAgentStore } from "@/store/useAgentStore";
import { MessageSquare, Mic, Map as MapIcon, Sparkles, Radio, Cpu, Activity } from "lucide-react";
import { VoiceAvatar } from "./VoiceAvatar";
import { LearningArchitect } from "./LearningArchitect";
import { motion, AnimatePresence } from "framer-motion";

export function InteractionZone() {
    const { activeTab, setActiveTab, isThinking } = useAgentStore();

    return (
        <main className="flex-1 flex flex-col min-w-0 relative overflow-hidden">
            {/* Subtle gradient overlay */}
            <div className="absolute inset-0 bg-gradient-to-b from-black/20 via-transparent to-black/40 pointer-events-none z-0" />
            
            {/* Header Bar */}
            <motion.div 
                className="h-16 border-b border-white/[0.06] flex items-center px-6 justify-between glass-panel relative z-10"
                initial={{ y: -20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.2 }}
            >
                {/* Tab Navigation */}
                <Tabs
                    value={activeTab}
                    onValueChange={(v) => setActiveTab(v as any)}
                    className="w-full max-w-2xl"
                >
                    <TabsList className="bg-white/[0.03] border border-white/[0.08] p-1.5 rounded-xl">
                        <TabsTrigger 
                            value="text" 
                            className="gap-2 rounded-lg data-[state=active]:bg-gradient-to-r data-[state=active]:from-primary/20 data-[state=active]:to-emerald-500/10 data-[state=active]:text-primary data-[state=active]:shadow-lg data-[state=active]:border data-[state=active]:border-primary/30 transition-all duration-300"
                        >
                            <MessageSquare className="w-4 h-4" />
                            <span className="hidden sm:inline">Text Diagnosis</span>
                        </TabsTrigger>
                        <TabsTrigger 
                            value="voice" 
                            className="gap-2 rounded-lg data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-500/20 data-[state=active]:to-cyan-500/10 data-[state=active]:text-blue-400 data-[state=active]:shadow-lg data-[state=active]:border data-[state=active]:border-blue-500/30 transition-all duration-300"
                        >
                            <Mic className="w-4 h-4" />
                            <span className="hidden sm:inline">Voice Tutor</span>
                        </TabsTrigger>
                        <TabsTrigger 
                            value="architect" 
                            className="gap-2 rounded-lg data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-500/20 data-[state=active]:to-pink-500/10 data-[state=active]:text-purple-400 data-[state=active]:shadow-lg data-[state=active]:border data-[state=active]:border-purple-500/30 transition-all duration-300"
                        >
                            <MapIcon className="w-4 h-4" />
                            <span className="hidden sm:inline">Learning Architect</span>
                        </TabsTrigger>
                    </TabsList>
                </Tabs>

                {/* Status Indicators */}
                <div className="flex items-center gap-4">
                    {/* AI Status */}
                    <motion.div 
                        className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.03] border border-white/[0.08]"
                        animate={isThinking ? { 
                            borderColor: ['rgba(34,197,94,0.3)', 'rgba(34,197,94,0.6)', 'rgba(34,197,94,0.3)']
                        } : {}}
                        transition={{ duration: 1.5, repeat: Infinity }}
                    >
                        <motion.div 
                            className={`h-2 w-2 rounded-full ${isThinking ? 'bg-primary' : 'bg-emerald-500'}`}
                            animate={isThinking ? { 
                                scale: [1, 1.5, 1],
                                opacity: [1, 0.5, 1]
                            } : {
                                scale: 1
                            }}
                            transition={{ duration: 1, repeat: isThinking ? Infinity : 0 }}
                            style={{
                                boxShadow: isThinking 
                                    ? '0 0 15px rgba(34, 197, 94, 0.8)' 
                                    : '0 0 10px rgba(16, 185, 129, 0.6)'
                            }}
                        />
                        <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                            {isThinking ? (
                                <span className="text-primary">Processing</span>
                            ) : (
                                'Neural Link Stable'
                            )}
                        </span>
                    </motion.div>

                    {/* Activity Indicator */}
                    <motion.div
                        className="hidden md:flex items-center gap-1.5 px-2 py-1 rounded-md bg-primary/5 border border-primary/20"
                        animate={{ opacity: [0.5, 1, 0.5] }}
                        transition={{ duration: 3, repeat: Infinity }}
                    >
                        <Activity className="w-3 h-3 text-primary" />
                        <span className="text-[9px] font-mono text-primary/80">LIVE</span>
                    </motion.div>
                </div>
            </motion.div>

            {/* Content Area */}
            <div className="flex-1 overflow-hidden relative z-10">
                <AnimatePresence mode="wait">
                    <motion.div
                        key={activeTab}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.2 }}
                        className="h-full"
                    >
                        {activeTab === 'text' && <ChatInterface />}
                        {activeTab === 'voice' && <VoiceAvatar />}
                        {activeTab === 'architect' && <LearningArchitect />}
                    </motion.div>
                </AnimatePresence>
            </div>
        </main>
    );
}
