"use client";

import React, { useState, useRef, useEffect } from 'react';
import { useAgentStore } from "@/store/useAgentStore";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, Sparkles, Mic, BrainCircuit, Globe, Loader2, Trash2 } from "lucide-react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion, AnimatePresence } from "framer-motion";
import { QuizCard } from "./QuizCard";
import { MindmapRenderer } from "./MindmapRenderer";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

export function ChatInterface() {
    const { chatHistory, addChatMessage, addLog, isThinking, setThinking, clearChat, sessionId, setSessionId } = useAgentStore();
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [chatHistory, isThinking]);

    const handleSend = async () => {
        const trimmedInput = input.trim();
        if (!trimmedInput || isThinking) return;

        // Add user message
        addChatMessage({
            role: 'user',
            content: trimmedInput,
            type: 'text'
        });

        // Supervisor Log - User Intent
        addLog({
            agent: 'Supervisor',
            message: `Routing request: "${trimmedInput.substring(0, 30)}..."`,
            status: 'active'
        });

        setThinking(true);
        setInput('');

        try {
            // Call Backend API
            const response = await api.chat(trimmedInput, sessionId || undefined);

            // Update Session ID if new
            if (response.session_id && response.session_id !== sessionId) {
                setSessionId(response.session_id);
            }

            // Add Assistant Message
            addChatMessage({
                role: 'assistant',
                content: response.response,
                type: response.type,
                data: response.data
            });

            // Diagnosis Log
            addLog({
                agent: 'Supervisor',
                message: `Response received from agent. Status: Success.`,
                status: 'completed'
            });

        } catch (error) {
            console.error(error);
            addLog({
                agent: 'Diagnostician',
                message: `System Error: ${error}`,
                status: 'completed'
            });
            addChatMessage({
                role: 'assistant',
                content: "⚠️ **System Error**: I lost connection to the neural core. Please ensure the backend is running.",
                type: 'text'
            });
        } finally {
            setThinking(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-black/20 relative">
            <div className="absolute top-4 right-6 z-20">
                <Button
                    variant="ghost"
                    size="sm"
                    onClick={clearChat}
                    className="text-muted-foreground hover:text-rose-500 transition-colors gap-2 bg-black/40 backdrop-blur-md border border-white/5"
                >
                    <Trash2 className="w-3.5 h-3.5" />
                    Clear Terminal
                </Button>
            </div>

            <ScrollArea className="flex-1 p-6 h-full">
                <div className="max-w-3xl mx-auto space-y-8 pb-10">
                    <AnimatePresence initial={false}>
                        {chatHistory.map((msg) => (
                            <motion.div
                                key={msg.id}
                                initial={{ opacity: 0, y: 10, filter: "blur(5px)" }}
                                animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                                className={cn(
                                    "flex gap-4",
                                    msg.role === 'user' ? "flex-row-reverse" : "flex-row"
                                )}
                            >
                                <div className={cn(
                                    "w-9 h-9 rounded-xl flex items-center justify-center shrink-0 border shadow-lg",
                                    msg.role === 'user'
                                        ? "bg-primary/20 border-primary/40 text-primary"
                                        : "bg-white/5 border-white/10 text-muted-foreground"
                                )}>
                                    {msg.role === 'user' ? <BrainCircuit className="w-5 h-5" /> : <Globe className="w-5 h-5" />}
                                </div>

                                <div className={cn(
                                    "flex flex-col gap-2 max-w-[85%]",
                                    msg.role === 'user' ? "items-end" : "items-start"
                                )}>
                                    <div className={cn(
                                        "px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm",
                                        msg.role === 'user'
                                            ? "bg-primary text-primary-foreground font-semibold"
                                            : "glass text-foreground/90 prose prose-invert prose-sm"
                                    )}>
                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                            {msg.content}
                                        </ReactMarkdown>
                                    </div>

                                    {msg.type === 'quiz' && msg.data && (
                                        <QuizCard items={msg.data} />
                                    )}

                                    {msg.type === 'mindmap' && msg.data && (
                                        <MindmapRenderer chart={msg.data} />
                                    )}

                                    <span suppressHydrationWarning className="text-[10px] text-muted-foreground/60 px-1 font-mono uppercase tracking-tighter">
                                        {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                    </span>
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {isThinking && (
                        <motion.div
                            initial={{ opacity: 0, y: 5 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="flex gap-4"
                        >
                            <div className="w-9 h-9 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center shrink-0">
                                <Loader2 className="w-5 h-5 text-primary animate-spin" />
                            </div>
                            <div className="glass px-4 py-3 rounded-2xl flex items-center gap-3">
                                <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-primary animate-pulse">Analyzing Neural Path</span>
                                <div className="flex gap-1">
                                    {[0, 1, 2].map((i) => (
                                        <motion.div
                                            key={i}
                                            animate={{ scale: [1, 1.5, 1], opacity: [0.3, 1, 0.3] }}
                                            transition={{ repeat: Infinity, duration: 1, delay: i * 0.2 }}
                                            className="w-1 h-1 rounded-full bg-primary"
                                        />
                                    ))}
                                </div>
                            </div>
                        </motion.div>
                    )}
                    <div ref={messagesEndRef} />
                </div>
            </ScrollArea>

            <div className="p-6 bg-gradient-to-t from-black to-transparent border-t border-white/5">
                <div className="max-w-3xl mx-auto relative group">
                    <div className="absolute -inset-1 bg-gradient-to-r from-primary/30 to-accent/30 rounded-2xl blur-lg opacity-0 group-focus-within:opacity-100 transition duration-500"></div>
                    <div className="relative flex items-center gap-2 bg-black/60 backdrop-blur-3xl border border-white/10 rounded-2xl p-2 pl-4 focus-within:border-primary/50 transition-colors">
                        <Sparkles className="w-4 h-4 text-primary shrink-0" />
                        <Input
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                            placeholder={isThinking ? "Darksied is processing..." : "Ask Darksied anything..."}
                            disabled={isThinking}
                            className="bg-transparent border-none focus-visible:ring-0 text-sm h-10 disabled:opacity-50"
                        />
                        <div className="flex items-center gap-1 pr-1">
                            <Button variant="ghost" size="icon" className="h-9 w-9 text-muted-foreground hover:text-primary rounded-xl">
                                <Mic className="w-5 h-5" />
                            </Button>
                            <Button
                                onClick={handleSend}
                                disabled={!input.trim() || isThinking}
                                className="h-9 w-9 rounded-xl bg-primary hover:bg-primary/80 text-primary-foreground p-0 disabled:opacity-30 transition-all"
                            >
                                {isThinking ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                            </Button>
                        </div>
                    </div>
                    <div className="mt-3 flex justify-center gap-6">
                        <div className="text-[9px] text-muted-foreground uppercase tracking-[0.3em] flex items-center gap-2 font-bold">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]"></span>
                            LiveKit Ready
                        </div>
                        <div className="text-[9px] text-muted-foreground uppercase tracking-[0.3em] flex items-center gap-2 font-bold">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]"></span>
                            Neural Network Active
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
