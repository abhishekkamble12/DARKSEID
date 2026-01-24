"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight, RotateCcw, Volume2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface QuizItem {
    question: string;
    answer: string;
    hint?: string;
}

interface QuizCardProps {
    items: QuizItem[];
}

export function QuizCard({ items }: QuizCardProps) {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [isFlipped, setIsFlipped] = useState(false);

    const next = () => {
        setIsFlipped(false);
        setCurrentIndex((prev) => (prev + 1) % items.length);
    };

    const prev = () => {
        setIsFlipped(false);
        setCurrentIndex((prev) => (prev - 1 + items.length) % items.length);
    };

    return (
        <div className="w-full max-w-md mx-auto my-6 space-y-4">
            <div className="relative h-64 perspective-1000">
                <AnimatePresence mode="wait">
                    <motion.div
                        key={currentIndex + (isFlipped ? '-back' : '-front')}
                        initial={{ opacity: 0, rotateY: isFlipped ? -90 : 90 }}
                        animate={{ opacity: 1, rotateY: 0 }}
                        exit={{ opacity: 0, rotateY: isFlipped ? 90 : -90 }}
                        transition={{ duration: 0.4 }}
                        className="w-full h-full"
                    >
                        <div
                            onClick={() => setIsFlipped(!isFlipped)}
                            className="w-full h-full rounded-2xl p-8 flex flex-col items-center justify-center text-center cursor-pointer glass border-primary/20 hover:border-primary/50 transition-all shadow-[0_0_20px_rgba(34,197,94,0.1)]"
                        >
                            <div className="absolute top-4 right-4 flex gap-2">
                                <Button variant="ghost" size="icon" className="h-8 w-8 text-primary">
                                    <Volume2 className="h-4 w-4" />
                                </Button>
                                <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground">
                                    <RotateCcw className="h-4 w-4" />
                                </Button>
                            </div>

                            <div className="text-[10px] uppercase tracking-widest text-primary font-bold mb-4">
                                {isFlipped ? "Diagnosis Result" : `Knowledge Probe ${currentIndex + 1}/${items.length}`}
                            </div>

                            <h3 className="text-lg font-medium leading-tight">
                                {isFlipped ? items[currentIndex].answer : items[currentIndex].question}
                            </h3>

                            {!isFlipped && items[currentIndex].hint && (
                                <p className="mt-4 text-xs text-muted-foreground italic">
                                    Hint: {items[currentIndex].hint}
                                </p>
                            )}

                            <div className="absolute bottom-4 text-[10px] text-muted-foreground uppercase tracking-widest">
                                Click to flip
                            </div>
                        </div>
                    </motion.div>
                </AnimatePresence>
            </div>

            <div className="flex items-center justify-center gap-4">
                <Button variant="outline" size="icon" onClick={prev} className="rounded-full border-white/10 hover:bg-primary/20">
                    <ChevronLeft className="h-4 w-4" />
                </Button>
                <div className="flex gap-1">
                    {items.map((_, i) => (
                        <div
                            key={i}
                            className={cn(
                                "h-1 rounded-full transition-all duration-300",
                                i === currentIndex ? "w-4 bg-primary" : "w-1 bg-white/20"
                            )}
                        />
                    ))}
                </div>
                <Button variant="outline" size="icon" onClick={next} className="rounded-full border-white/10 hover:bg-primary/20">
                    <ChevronRight className="h-4 w-4" />
                </Button>
            </div>
        </div>
    );
}
