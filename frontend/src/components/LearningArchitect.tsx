"use client";

import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MindmapRenderer } from "./MindmapRenderer";
import { QuizCard } from "./QuizCard";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { 
    BrainCircuit, 
    Lightbulb, 
    Map as MapIcon, 
    RotateCcw, 
    Sparkles,
    FileText,
    Upload,
    Loader2,
    AlertCircle,
    CheckCircle2,
    BookOpen,
    Zap
} from "lucide-react";
import { useAgentStore } from "@/store/useAgentStore";
import { api } from "@/lib/api";

export function LearningArchitect() {
    const [topic, setTopic] = useState('');
    const [error, setError] = useState<string | null>(null);
    
    const { 
        sessionId, 
        learningMaterials, 
        setLearningMaterials,
        clearLearningMaterials,
        addLog,
        documents 
    } = useAgentStore();
    
    // Filter documents for current session
    const sessionDocuments = documents.filter(d => d.sessionId === sessionId);
    const hasDocuments = sessionDocuments.length > 0;
    
    const generateMaterials = useCallback(async (regenerate = false) => {
        if (!topic.trim() && !regenerate) {
            setError('Please enter a topic or upload a document first');
            return;
        }
        
        const topicToUse = regenerate && learningMaterials.topic 
            ? learningMaterials.topic 
            : topic.trim() || 'the uploaded document';
        
        setError(null);
        setLearningMaterials({ isGenerating: true });
        
        addLog({
            agent: 'LearningArchitect',
            message: `Generating learning materials for: "${topicToUse}"`,
            status: 'active'
        });
        
        try {
            const result = await api.generateLearning(topicToUse, sessionId);
            
            setLearningMaterials({
                mindmap: result.mindmap,
                quiz: result.quiz?.cards || null,
                topic: topicToUse,
                generatedAt: new Date(),
                isGenerating: false,
            });
            
            addLog({
                agent: 'LearningArchitect',
                message: `Successfully generated mindmap and ${result.quiz?.cards?.length || 0} quiz cards`,
                status: 'completed'
            });
            
        } catch (err) {
            const errorMsg = err instanceof Error ? err.message : 'Failed to generate materials';
            setError(errorMsg);
            setLearningMaterials({ isGenerating: false });
            
            addLog({
                agent: 'LearningArchitect',
                message: `Error: ${errorMsg}`,
                status: 'error'
            });
        }
    }, [topic, sessionId, learningMaterials.topic, setLearningMaterials, addLog]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            generateMaterials();
        }
    };

    return (
        <div className="h-full flex flex-col bg-black/20">
            {/* Header with Topic Input */}
            <div className="border-b border-white/10 bg-black/40 backdrop-blur-sm">
                <div className="p-4">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-2 rounded-lg bg-primary/10 border border-primary/30">
                            <BrainCircuit className="w-5 h-5 text-primary" />
                        </div>
                        <div>
                            <h2 className="text-sm font-bold">Learning Architect</h2>
                            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
                                Generate Mindmaps & Quiz Cards from Documents
                            </p>
                        </div>
                    </div>
                    
                    {/* Document Status */}
                    <div className="flex items-center gap-2 mb-3">
                        {hasDocuments ? (
                            <motion.div 
                                initial={{ opacity: 0, scale: 0.9 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/30"
                            >
                                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                                <span className="text-xs text-emerald-400">
                                    {sessionDocuments.length} document{sessionDocuments.length > 1 ? 's' : ''} ready
                                </span>
                            </motion.div>
                        ) : (
                            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-500/10 border border-amber-500/30">
                                <Upload className="w-3.5 h-3.5 text-amber-400" />
                                <span className="text-xs text-amber-400">
                                    Upload a document to get started
                                </span>
                            </div>
                        )}
                    </div>
                    
                    {/* Topic Input */}
                    <div className="flex gap-2">
                        <div className="flex-1 relative">
                            <BookOpen className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                            <Input
                                value={topic}
                                onChange={(e) => setTopic(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder={hasDocuments 
                                    ? "Enter topic or leave blank to analyze entire document..." 
                                    : "Upload a document first, then enter a topic..."
                                }
                                className="pl-10 bg-black/40 border-white/10 focus:border-primary/50"
                                disabled={learningMaterials.isGenerating}
                            />
                        </div>
                        <Button
                            onClick={() => generateMaterials()}
                            disabled={learningMaterials.isGenerating || (!hasDocuments && !topic.trim())}
                            className="gap-2 bg-primary/10 border border-primary/30 hover:bg-primary/20 text-primary"
                        >
                            {learningMaterials.isGenerating ? (
                                <>
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    Generating...
                                </>
                            ) : (
                                <>
                                    <Zap className="w-4 h-4" />
                                    Generate
                                </>
                            )}
                        </Button>
                    </div>
                    
                    {/* Error Display */}
                    <AnimatePresence>
                        {error && (
                            <motion.div
                                initial={{ opacity: 0, y: -10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -10 }}
                                className="mt-3 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 flex items-center gap-2"
                            >
                                <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
                                <span className="text-xs text-rose-400">{error}</span>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 flex gap-0 overflow-hidden">
                {/* Left Side: Mindmap */}
                <div className="flex-[1.5] border-r border-white/10 flex flex-col">
                    <div className="p-4 border-b border-white/10 flex items-center justify-between bg-black/20">
                        <div className="flex items-center gap-2">
                            <MapIcon className="w-4 h-4 text-primary" />
                            <span className="text-xs font-bold uppercase tracking-wider">Neural Map Visualization</span>
                        </div>
                        {learningMaterials.mindmap && (
                            <Button 
                                variant="ghost" 
                                size="sm" 
                                className="h-8 text-[10px] uppercase font-bold text-muted-foreground gap-2"
                                onClick={() => generateMaterials(true)}
                                disabled={learningMaterials.isGenerating}
                            >
                                <RotateCcw className="w-3 h-3" />
                                Regenerate
                            </Button>
                        )}
                    </div>
                    <div className="flex-1 p-4 overflow-auto relative">
                        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(34,197,94,0.05)_0%,transparent_70%)] pointer-events-none" />
                        
                        <AnimatePresence mode="wait">
                            {learningMaterials.isGenerating ? (
                                <motion.div
                                    key="loading"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    className="h-full flex flex-col items-center justify-center"
                                >
                                    <motion.div
                                        animate={{ rotate: 360 }}
                                        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                                        className="w-16 h-16 rounded-full border-2 border-primary/30 border-t-primary mb-4"
                                    />
                                    <p className="text-sm text-muted-foreground">Generating mindmap...</p>
                                    <p className="text-xs text-muted-foreground/60 mt-1">Analyzing document content</p>
                                </motion.div>
                            ) : learningMaterials.mindmap ? (
                                <motion.div
                                    key="mindmap"
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    exit={{ opacity: 0, scale: 0.95 }}
                                    className="h-full"
                                >
                                    <MindmapRenderer chart={learningMaterials.mindmap} />
                                </motion.div>
                            ) : (
                                <motion.div
                                    key="empty"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    className="h-full flex flex-col items-center justify-center text-center p-8"
                                >
                                    <div className="w-20 h-20 rounded-full bg-primary/5 border border-primary/20 flex items-center justify-center mb-4">
                                        <MapIcon className="w-8 h-8 text-primary/40" />
                                    </div>
                                    <h3 className="text-sm font-medium mb-2">No Mindmap Generated</h3>
                                    <p className="text-xs text-muted-foreground max-w-xs">
                                        {hasDocuments 
                                            ? "Enter a topic above and click Generate to create a visual mindmap from your documents."
                                            : "Upload a document first, then generate a mindmap to visualize key concepts."
                                        }
                                    </p>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>

                {/* Right Side: Quiz Cards */}
                <div className="flex-1 flex flex-col bg-black/40">
                    <div className="p-4 border-b border-white/10 flex items-center gap-2 bg-black/20">
                        <Lightbulb className="w-4 h-4 text-primary" />
                        <span className="text-xs font-bold uppercase tracking-wider">Diagnostic Probes</span>
                        {learningMaterials.quiz && (
                            <span className="ml-auto text-[10px] text-muted-foreground">
                                {learningMaterials.quiz.length} questions
                            </span>
                        )}
                    </div>
                    <ScrollArea className="flex-1">
                        <div className="p-4">
                            <AnimatePresence mode="wait">
                                {learningMaterials.isGenerating ? (
                                    <motion.div
                                        key="loading"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                        className="flex flex-col items-center justify-center py-20"
                                    >
                                        <Loader2 className="w-8 h-8 text-primary animate-spin mb-4" />
                                        <p className="text-sm text-muted-foreground">Creating quiz cards...</p>
                                    </motion.div>
                                ) : learningMaterials.quiz && learningMaterials.quiz.length > 0 ? (
                                    <motion.div
                                        key="quiz"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                    >
                                        <QuizCard items={learningMaterials.quiz} />
                                        
                                        {/* Architect Insight */}
                                        <div className="mt-8 p-4 rounded-lg bg-primary/5 border border-primary/20">
                                            <div className="flex items-center gap-2 text-primary mb-3">
                                                <Sparkles className="w-4 h-4" />
                                                <span className="text-xs font-bold uppercase tracking-widest">Architect Insight</span>
                                            </div>
                                            <p className="text-xs text-muted-foreground leading-relaxed">
                                                These diagnostic probes test your understanding of <strong className="text-white">"{learningMaterials.topic}"</strong>. 
                                                Focus on "why" and "how" questions to build deep comprehension.
                                            </p>
                                            {learningMaterials.generatedAt && (
                                                <p className="text-[10px] text-muted-foreground/60 mt-2">
                                                    Generated: {learningMaterials.generatedAt.toLocaleString()}
                                                </p>
                                            )}
                                        </div>
                                    </motion.div>
                                ) : (
                                    <motion.div
                                        key="empty"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                        className="flex flex-col items-center justify-center py-20 text-center"
                                    >
                                        <div className="w-16 h-16 rounded-full bg-primary/5 border border-primary/20 flex items-center justify-center mb-4">
                                            <Lightbulb className="w-6 h-6 text-primary/40" />
                                        </div>
                                        <h3 className="text-sm font-medium mb-2">No Quiz Cards Yet</h3>
                                        <p className="text-xs text-muted-foreground max-w-xs">
                                            Quiz cards will appear here after you generate learning materials.
                                        </p>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    </ScrollArea>
                </div>
            </div>

            {/* Status Bar */}
            {learningMaterials.topic && (
                <motion.div 
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="border-t border-white/10 bg-black/40 px-4 py-2 flex items-center justify-between"
                >
                    <div className="flex items-center gap-4 text-[10px] text-muted-foreground">
                        <span>Topic: <strong className="text-white">{learningMaterials.topic}</strong></span>
                        {learningMaterials.mindmap && <span>• Mindmap ready</span>}
                        {learningMaterials.quiz && <span>• {learningMaterials.quiz.length} quiz cards</span>}
                    </div>
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={clearLearningMaterials}
                        className="h-6 text-[10px] text-muted-foreground hover:text-rose-400"
                    >
                        Clear All
                    </Button>
                </motion.div>
            )}
        </div>
    );
}
