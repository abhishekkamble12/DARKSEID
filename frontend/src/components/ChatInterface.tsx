"use client";

import React, { useState, useRef, useEffect } from 'react';
import { useAgentStore } from "@/store/useAgentStore";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, Sparkles, Mic, BrainCircuit, Globe, Loader2, Trash2, Bot, User, Zap, Wand2, Activity, Radio, Cpu } from "lucide-react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion, AnimatePresence, useMotionValue, useSpring, useTransform } from "framer-motion";
import { QuizCard } from "./QuizCard";
import { MindmapRenderer } from "./MindmapRenderer";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

// Animated Wave Background
function WaveBackground() {
    return (
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <svg className="absolute bottom-0 w-full h-48 opacity-10" viewBox="0 0 1440 320">
                <motion.path
                    fill="rgba(34, 197, 94, 0.3)"
                    animate={{
                        d: [
                            "M0,192L48,197.3C96,203,192,213,288,229.3C384,245,480,267,576,250.7C672,235,768,181,864,181.3C960,181,1056,235,1152,234.7C1248,235,1344,181,1392,154.7L1440,128L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z",
                            "M0,128L48,154.7C96,181,192,235,288,234.7C384,235,480,181,576,181.3C672,181,768,235,864,250.7C960,267,1056,245,1152,229.3C1248,213,1344,203,1392,197.3L1440,192L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z",
                        ]
                    }}
                    transition={{ duration: 10, repeat: Infinity, repeatType: "reverse" }}
                />
            </svg>
        </div>
    );
}

// Typing Indicator with Character Animation
function TypingIndicator() {
    const text = "Analyzing Neural Path";
    return (
        <div className="flex items-center gap-1">
            {text.split('').map((char, i) => (
                <motion.span
                    key={i}
                    className="text-primary font-bold text-xs uppercase tracking-wider"
                    animate={{ opacity: [0.3, 1, 0.3] }}
                    transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.05 }}
                >
                    {char === ' ' ? '\u00A0' : char}
                </motion.span>
            ))}
        </div>
    );
}

export function ChatInterface() {
    const { chatHistory, addChatMessage, addLog, isThinking, setThinking, clearChat, sessionId, setSessionId } = useAgentStore();
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const [isFocused, setIsFocused] = useState(false);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [chatHistory, isThinking]);

    const handleSend = async () => {
        const trimmedInput = input.trim();
        if (!trimmedInput || isThinking) return;

        addChatMessage({
            role: 'user',
            content: trimmedInput,
            type: 'text'
        });

        addLog({
            agent: 'Supervisor',
            message: `Routing request: "${trimmedInput.substring(0, 30)}..."`,
            status: 'active'
        });

        setThinking(true);
        setInput('');

        try {
            const response = await api.chat(trimmedInput, sessionId || undefined);

            if (response.session_id && response.session_id !== sessionId) {
                setSessionId(response.session_id);
            }

            addChatMessage({
                role: 'assistant',
                content: response.response,
                type: response.type,
                data: response.data
            });

            addLog({
                agent: 'Supervisor',
                message: `Response received. Status: Success.`,
                status: 'completed'
            });

        } catch (error) {
            console.error(error);
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            
            addLog({
                agent: 'Diagnostician',
                message: `System Error: ${errorMsg}`,
                status: 'completed'
            });
            
            let userMessage = "⚠️ **System Error**: ";
            if (errorMsg.includes("Network error") || errorMsg.includes("Unable to connect")) {
                userMessage += "I'm having trouble connecting to the AI service. Please check your internet connection.";
            } else if (errorMsg.includes("Authentication") || errorMsg.includes("API key")) {
                userMessage += "There's an issue with the AI service configuration.";
            } else if (errorMsg.includes("Rate limit") || errorMsg.includes("quota")) {
                userMessage += "We've hit the rate limit. Please wait a moment.";
            } else if (errorMsg.includes("Failed to fetch") || errorMsg.includes("fetch")) {
                userMessage += "Lost connection to neural core. Please ensure the backend is running.";
            } else {
                userMessage += `${errorMsg}`;
            }
            
            addChatMessage({
                role: 'assistant',
                content: userMessage,
                type: 'text'
            });
        } finally {
            setThinking(false);
        }
    };

    return (
        <div className="flex flex-col h-full relative overflow-hidden">
            {/* Background gradient effect */}
            <div className="absolute inset-0 pointer-events-none">
                <motion.div 
                    className="absolute top-0 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-[120px]"
                    animate={{ 
                        x: [0, 50, 0],
                        y: [0, 30, 0],
                        scale: [1, 1.2, 1],
                    }}
                    transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
                />
                <motion.div 
                    className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/5 rounded-full blur-[120px]"
                    animate={{ 
                        x: [0, -40, 0],
                        y: [0, -50, 0],
                        scale: [1, 1.3, 1],
                    }}
                    transition={{ duration: 18, repeat: Infinity, ease: "easeInOut", delay: 2 }}
                />
                <motion.div 
                    className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-blue-500/3 rounded-full blur-[150px]"
                    animate={{ 
                        scale: [1, 1.5, 1],
                        opacity: [0.3, 0.6, 0.3],
                    }}
                    transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
                />
            </div>
            
            {/* Wave Background */}
            <WaveBackground />

            {/* Clear Chat Button */}
            <motion.div 
                className="absolute top-4 right-6 z-20"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
            >
                <Button
                    variant="ghost"
                    size="sm"
                    onClick={clearChat}
                    className="text-muted-foreground hover:text-rose-400 transition-all gap-2 glass border-white/[0.06] hover:border-rose-500/30 hover:bg-rose-500/10"
                >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">Clear</span>
                </Button>
            </motion.div>

            {/* Messages Area */}
            <ScrollArea className="flex-1 p-6 h-full relative z-10">
                <div className="max-w-3xl mx-auto space-y-6 pb-10">
                    <AnimatePresence initial={false}>
                        {chatHistory.map((msg, index) => (
                            <motion.div
                                key={msg.id}
                                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                                animate={{ opacity: 1, y: 0, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.95 }}
                                transition={{ 
                                    type: "spring",
                                    stiffness: 200,
                                    damping: 20,
                                    delay: index === chatHistory.length - 1 ? 0.1 : 0
                                }}
                                className={cn(
                                    "flex gap-4",
                                    msg.role === 'user' ? "flex-row-reverse" : "flex-row"
                                )}
                            >
                                {/* Avatar */}
                                <motion.div 
                                    className={cn(
                                        "w-10 h-10 rounded-xl flex items-center justify-center shrink-0 border-2 shadow-xl",
                                        msg.role === 'user'
                                            ? "bg-gradient-to-br from-primary/30 to-emerald-600/20 border-primary/50 text-primary"
                                            : "bg-gradient-to-br from-white/10 to-white/5 border-white/20 text-white/80"
                                    )}
                                    whileHover={{ scale: 1.1, rotate: 5 }}
                                >
                                    {msg.role === 'user' ? (
                                        <User className="w-5 h-5" />
                                    ) : (
                                        <Bot className="w-5 h-5" />
                                    )}
                                </motion.div>

                                {/* Message Content */}
                                <div className={cn(
                                    "flex flex-col gap-2 max-w-[80%]",
                                    msg.role === 'user' ? "items-end" : "items-start"
                                )}>
                                    <motion.div 
                                        className={cn(
                                            "px-5 py-4 rounded-2xl text-sm leading-relaxed relative overflow-hidden",
                                            msg.role === 'user'
                                                ? "bg-gradient-to-br from-primary to-emerald-600 text-white font-medium shadow-lg shadow-primary/20"
                                                : "glass border-white/[0.08] text-foreground/90"
                                        )}
                                        whileHover={{ scale: 1.01 }}
                                    >
                                        {/* Shimmer on user message */}
                                        {msg.role === 'user' && (
                                            <div className="absolute inset-0 opacity-30">
                                                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent shimmer" />
                                            </div>
                                        )}
                                        
                                        <div className={cn(
                                            "relative",
                                            msg.role === 'assistant' && "prose prose-invert prose-sm prose-p:my-2 prose-headings:my-3 prose-ul:my-2 prose-li:my-1"
                                        )}>
                                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                                {msg.content}
                                            </ReactMarkdown>
                                        </div>
                                    </motion.div>

                                    {/* Quiz or Mindmap */}
                                    {msg.type === 'quiz' && msg.data && (
                                        <motion.div
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: 0.3 }}
                                        >
                                            <QuizCard items={msg.data} />
                                        </motion.div>
                                    )}

                                    {msg.type === 'mindmap' && msg.data && (
                                        <motion.div
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: 0.3 }}
                                        >
                                            <MindmapRenderer chart={msg.data} />
                                        </motion.div>
                                    )}

                                    {/* Timestamp */}
                                    <span suppressHydrationWarning className="text-[9px] text-muted-foreground/50 px-1 font-mono uppercase tracking-wider">
                                        {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                    </span>
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {/* Enhanced Thinking Indicator */}
                    <AnimatePresence>
                        {isThinking && (
                            <motion.div
                                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                                animate={{ opacity: 1, y: 0, scale: 1 }}
                                exit={{ opacity: 0, y: -20, scale: 0.95 }}
                                className="flex gap-4"
                            >
                                {/* Animated Avatar */}
                                <motion.div 
                                    className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary/30 to-emerald-500/20 border-2 border-primary/50 flex items-center justify-center shrink-0 relative overflow-hidden"
                                    animate={{ 
                                        boxShadow: [
                                            '0 0 20px rgba(34,197,94,0.3)',
                                            '0 0 50px rgba(34,197,94,0.6)',
                                            '0 0 20px rgba(34,197,94,0.3)'
                                        ]
                                    }}
                                    transition={{ duration: 1.5, repeat: Infinity }}
                                >
                                    {/* Rotating ring */}
                                    <motion.div
                                        className="absolute inset-1 rounded-lg border-2 border-transparent border-t-primary/60"
                                        animate={{ rotate: 360 }}
                                        transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                                    />
                                    {/* Pulse effect */}
                                    <motion.div
                                        className="absolute inset-0 bg-primary/20 rounded-xl"
                                        animate={{ scale: [1, 1.5, 1], opacity: [0.3, 0, 0.3] }}
                                        transition={{ duration: 1, repeat: Infinity }}
                                    />
                                    <BrainCircuit className="w-6 h-6 text-primary relative z-10" />
                                </motion.div>
                                
                                {/* Thinking Card */}
                                <motion.div 
                                    className="glass border-primary/30 px-6 py-4 rounded-2xl relative overflow-hidden"
                                    animate={{
                                        borderColor: ['rgba(34,197,94,0.3)', 'rgba(34,197,94,0.6)', 'rgba(34,197,94,0.3)']
                                    }}
                                    transition={{ duration: 2, repeat: Infinity }}
                                >
                                    {/* Shimmer effect */}
                                    <motion.div
                                        className="absolute inset-0 bg-gradient-to-r from-transparent via-primary/10 to-transparent"
                                        animate={{ x: ['-100%', '100%'] }}
                                        transition={{ duration: 2, repeat: Infinity }}
                                    />
                                    
                                    <div className="relative z-10 flex items-center gap-4">
                                        <div className="flex items-center gap-3">
                                            <motion.div
                                                animate={{ rotate: [0, 360] }}
                                                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                                            >
                                                <Cpu className="w-4 h-4 text-primary" />
                                            </motion.div>
                                            <TypingIndicator />
                                        </div>
                                        
                                        {/* Animated bars */}
                                        <div className="flex gap-1 items-end h-4">
                                            {[0, 1, 2, 3, 4].map((i) => (
                                                <motion.div
                                                    key={i}
                                                    className="w-1 bg-primary rounded-full"
                                                    animate={{ 
                                                        height: ['4px', '16px', '8px', '14px', '4px'],
                                                    }}
                                                    transition={{ 
                                                        repeat: Infinity, 
                                                        duration: 1, 
                                                        delay: i * 0.1,
                                                        ease: "easeInOut"
                                                    }}
                                                    style={{
                                                        boxShadow: '0 0 6px rgba(34,197,94,0.8)'
                                                    }}
                                                />
                                            ))}
                                        </div>
                                    </div>
                                    
                                    {/* Progress bar */}
                                    <motion.div 
                                        className="mt-3 h-1 bg-white/5 rounded-full overflow-hidden"
                                    >
                                        <motion.div
                                            className="h-full bg-gradient-to-r from-primary via-emerald-400 to-primary rounded-full"
                                            animate={{ x: ['-100%', '100%'] }}
                                            transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                                            style={{ width: '50%' }}
                                        />
                                    </motion.div>
                                </motion.div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                    
                    <div ref={messagesEndRef} />
                </div>
            </ScrollArea>

            {/* Input Area */}
            <motion.div 
                className="p-6 pt-4 relative z-10"
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.4 }}
            >
                {/* Gradient fade at top */}
                <div className="absolute inset-x-0 top-0 h-20 bg-gradient-to-t from-black/80 to-transparent pointer-events-none -translate-y-full" />
                
                <div className="max-w-3xl mx-auto">
                    {/* Input Container */}
                    <div className="relative group">
                        {/* Glow Effect */}
                        <motion.div 
                            className="absolute -inset-1 bg-gradient-to-r from-primary/40 via-emerald-500/40 to-cyan-500/40 rounded-2xl blur-xl opacity-0 group-focus-within:opacity-100 transition-opacity duration-500"
                            animate={isFocused ? {
                                opacity: [0.3, 0.5, 0.3],
                            } : { opacity: 0 }}
                            transition={{ duration: 2, repeat: Infinity }}
                        />
                        
                        {/* Input Field */}
                        <div className={cn(
                            "relative flex items-center gap-3 rounded-2xl p-2 pl-5 transition-all duration-300",
                            "glass border-white/[0.08]",
                            isFocused && "border-primary/40 bg-white/[0.04]"
                        )}>
                            <motion.div
                                animate={{ rotate: isThinking ? 360 : 0 }}
                                transition={{ duration: 2, repeat: isThinking ? Infinity : 0, ease: "linear" }}
                            >
                                <Wand2 className={cn(
                                    "w-5 h-5 shrink-0 transition-colors",
                                    isFocused ? "text-primary" : "text-muted-foreground/50"
                                )} />
                            </motion.div>
                            
                            <Input
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                                onFocus={() => setIsFocused(true)}
                                onBlur={() => setIsFocused(false)}
                                placeholder={isThinking ? "Darksied is thinking..." : "Ask Darksied anything..."}
                                disabled={isThinking}
                                className="bg-transparent border-none focus-visible:ring-0 text-sm h-11 disabled:opacity-50 placeholder:text-muted-foreground/40"
                            />
                            
                            <div className="flex items-center gap-2 pr-1">
                                <Button 
                                    variant="ghost" 
                                    size="icon" 
                                    className="h-10 w-10 text-muted-foreground hover:text-blue-400 hover:bg-blue-500/10 rounded-xl transition-all"
                                >
                                    <Mic className="w-5 h-5" />
                                </Button>
                                
                                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                                    <Button
                                        onClick={handleSend}
                                        disabled={!input.trim() || isThinking}
                                        className={cn(
                                            "h-10 w-10 rounded-xl p-0 transition-all duration-300",
                                            "bg-gradient-to-br from-primary to-emerald-600 hover:from-primary hover:to-emerald-500",
                                            "shadow-lg shadow-primary/25 hover:shadow-primary/40",
                                            "disabled:opacity-30 disabled:shadow-none"
                                        )}
                                    >
                                        {isThinking ? (
                                            <Loader2 className="w-4 h-4 animate-spin" />
                                        ) : (
                                            <Send className="w-4 h-4" />
                                        )}
                                    </Button>
                                </motion.div>
                            </div>
                        </div>
                    </div>

                    {/* Enhanced Status Footer */}
                    <motion.div 
                        className="mt-4 flex justify-center items-center gap-4"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.6 }}
                    >
                        {/* Status Pills */}
                        {[
                            { label: 'Multi-Agent System', color: 'emerald', icon: Activity },
                            { label: 'RAG Enhanced', color: 'blue', icon: Radio },
                            { label: 'Socratic Method', color: 'purple', icon: Sparkles },
                        ].map((status, i) => (
                            <motion.div
                                key={status.label}
                                className={`flex items-center gap-2 px-3 py-1.5 rounded-full bg-${status.color}-500/5 border border-${status.color}-500/20`}
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ delay: 0.7 + i * 0.1 }}
                                whileHover={{ scale: 1.05, borderColor: `rgba(var(--${status.color}-500), 0.5)` }}
                            >
                                <motion.div className="relative">
                                    <motion.span 
                                        className={`w-2 h-2 rounded-full bg-${status.color}-500 block`}
                                        animate={{ 
                                            scale: [1, 1.3, 1],
                                            opacity: [0.7, 1, 0.7],
                                        }}
                                        transition={{ duration: 2, repeat: Infinity, delay: i * 0.3 }}
                                        style={{
                                            boxShadow: status.color === 'emerald' 
                                                ? '0 0 10px rgba(16,185,129,0.8)'
                                                : status.color === 'blue'
                                                ? '0 0 10px rgba(59,130,246,0.8)'
                                                : '0 0 10px rgba(168,85,247,0.8)'
                                        }}
                                    />
                                    <motion.span 
                                        className={`absolute inset-0 rounded-full bg-${status.color}-500`}
                                        animate={{ scale: [1, 2], opacity: [0.5, 0] }}
                                        transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.3 }}
                                    />
                                </motion.div>
                                <span className={`text-[9px] text-${status.color}-400/80 uppercase tracking-[0.15em] font-bold`}>
                                    {status.label}
                                </span>
                            </motion.div>
                        ))}
                    </motion.div>
                    
                    {/* Animated bottom line */}
                    <motion.div 
                        className="mt-4 h-[1px] bg-gradient-to-r from-transparent via-primary/20 to-transparent"
                        animate={{ opacity: [0.3, 0.6, 0.3] }}
                        transition={{ duration: 3, repeat: Infinity }}
                    />
                </div>
            </motion.div>
        </div>
    );
}
