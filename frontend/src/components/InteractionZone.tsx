"use client";

import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ChatInterface } from "./ChatInterface";
import { useAgentStore } from "@/store/useAgentStore";
import { MessageSquare, Mic, Map as MapIcon, Sparkles } from "lucide-react";
import { VoiceAvatar } from "./VoiceAvatar"; // We'll create this next
import { LearningArchitect } from "./LearningArchitect"; // We'll create this next

export function InteractionZone() {
    const { activeTab, setActiveTab } = useAgentStore();

    return (
        <main className="flex-1 flex flex-col min-w-0 bg-black">
            <div className="h-16 border-b border-white/10 flex items-center px-6 justify-between bg-black/40 backdrop-blur-xl z-10">
                <Tabs
                    value={activeTab}
                    onValueChange={(v) => setActiveTab(v as any)}
                    className="w-full max-w-2xl"
                >
                    <TabsList className="bg-white/5 border border-white/10 p-1">
                        <TabsTrigger value="text" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                            <MessageSquare className="w-4 h-4" />
                            Text Diagnosis
                        </TabsTrigger>
                        <TabsTrigger value="voice" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                            <Mic className="w-4 h-4" />
                            Voice Tutor
                        </TabsTrigger>
                        <TabsTrigger value="architect" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                            <MapIcon className="w-4 h-4" />
                            Learning Architect
                        </TabsTrigger>
                    </TabsList>
                </Tabs>

                <div className="flex items-center gap-3">
                    <div className="h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]"></div>
                    <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Neural Link Stable</span>
                </div>
            </div>

            <div className="flex-1 overflow-hidden">
                {activeTab === 'text' && <ChatInterface />}
                {activeTab === 'voice' && <VoiceAvatar />}
                {activeTab === 'architect' && <LearningArchitect />}
            </div>
        </main>
    );
}
