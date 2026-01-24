"use client";

import React from 'react';
import { MindmapRenderer } from "./MindmapRenderer";
import { QuizCard } from "./QuizCard";
import { ScrollArea } from "@/components/ui/scroll-area";
import { BrainCircuit, Lightbulb, Map as MapIcon, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";

export function LearningArchitect() {
    const mockMindmap = `mindmap
  root((Quantum Physics))
    Wave-Particle Duality
      De Broglie Hypothesis
      Double Slit Experiment
    Uncertainty Principle
      Heisenberg
      Energy-Time
    Atomic Structure
      Bohr Model
      Schrodinger Equation`;

    const mockQuiz = [
        {
            question: "What does the Double Slit Experiment prove about light?",
            answer: "Light behaves as both a wave and a particle, demonstrating wave-particle duality.",
            hint: "Think about interference patterns."
        },
        {
            question: "State the Heisenberg Uncertainty Principle.",
            answer: "It is impossible to simultaneously know the exact position and momentum of a particle.",
            hint: "It limits our precision at the quantum level."
        },
        {
            question: "What is the primary contribution of the Bohr Model?",
            answer: "Electrons orbit the nucleus in specific, quantized energy levels.",
            hint: "Discrete orbits."
        },
    ];

    return (
        <div className="h-full flex flex-col">
            <div className="flex-1 flex gap-0">
                {/* Left Side: Mindmap */}
                <div className="flex-[1.5] border-r border-white/10 flex flex-col">
                    <div className="p-4 border-b border-white/10 flex items-center justify-between bg-black/20">
                        <div className="flex items-center gap-2">
                            <MapIcon className="w-4 h-4 text-primary" />
                            <span className="text-xs font-bold uppercase tracking-wider">Neural Map Visualization</span>
                        </div>
                        <Button variant="ghost" size="sm" className="h-8 text-[10px] uppercase font-bold text-muted-foreground gap-2">
                            <RotateCcw className="w-3 h-3" />
                            Regenerate
                        </Button>
                    </div>
                    <div className="flex-1 p-8 overflow-hidden relative">
                        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(34,197,94,0.05)_0%,transparent_70%)] pointer-events-none"></div>
                        <MindmapRenderer chart={mockMindmap} />
                    </div>
                </div>

                {/* Right Side: Quiz Cards */}
                <div className="flex-1 flex flex-col bg-black/40">
                    <div className="p-4 border-b border-white/10 flex items-center gap-2 bg-black/20">
                        <Lightbulb className="w-4 h-4 text-primary" />
                        <span className="text-xs font-bold uppercase tracking-wider">Diagnostic Probes</span>
                    </div>
                    <ScrollArea className="flex-1">
                        <div className="p-8 flex flex-col items-center justify-center min-h-[500px]">
                            <QuizCard items={mockQuiz} />

                            <div className="mt-12 max-w-sm text-center">
                                <div className="flex items-center justify-center gap-2 text-primary mb-4">
                                    <BrainCircuit className="w-5 h-5" />
                                    <span className="text-sm font-bold uppercase tracking-widest">Architect Insight</span>
                                </div>
                                <p className="text-xs text-muted-foreground leading-relaxed">
                                    These probes are designed to stress-test your "Wave-Particle Duality" mental model.
                                    Success here will unlock the "Quantum Phase" module.
                                </p>
                            </div>
                        </div>
                    </ScrollArea>
                </div>
            </div>
        </div>
    );
}
