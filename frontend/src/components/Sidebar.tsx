"use client";

import React, { useRef, useState } from 'react';
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
    Plus,
    History,
    Settings,
    Shield,
    FileUp,
    BrainCircuit,
    CheckCircle2,
    AlertCircle
} from "lucide-react";
import { useAgentStore } from "@/store/useAgentStore";
import { Progress } from "@/components/ui/progress";
import { api } from "@/lib/api";

export function Sidebar() {
    const { isVectorizing, sessionId, setVectorizing, setUploadProgress, setSessionId, addLog } = useAgentStore();
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');
    const [uploadMessage, setUploadMessage] = useState('');
    const [localProgress, setLocalProgress] = useState(0);

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        // Validate file type
        const allowedTypes = ['.pdf', '.txt', '.md', '.csv', '.doc', '.docx'];
        const fileExt = '.' + file.name.split('.').pop()?.toLowerCase();
        if (!allowedTypes.includes(fileExt)) {
            setUploadStatus('error');
            setUploadMessage(`Invalid file type. Allowed: ${allowedTypes.join(', ')}`);
            setTimeout(() => setUploadStatus('idle'), 3000);
            return;
        }

        // Generate session ID if not exists
        let currentSessionId = sessionId;
        if (!currentSessionId) {
            currentSessionId = Math.random().toString(36).substring(2, 15);
            setSessionId(currentSessionId);
        }

        setVectorizing(true);
        setLocalProgress(10);
        setUploadProgress(10);
        setUploadStatus('idle');
        
        addLog({
            agent: 'RAG',
            message: `Uploading document: ${file.name}`,
            status: 'active'
        });

        try {
            // Simulate progress updates
            let currentProgress = 10;
            const progressInterval = setInterval(() => {
                currentProgress = Math.min(currentProgress + 15, 85);
                setLocalProgress(currentProgress);
                setUploadProgress(currentProgress);
            }, 500);

            const result = await api.uploadFile(file, currentSessionId);
            
            clearInterval(progressInterval);
            setLocalProgress(100);
            setUploadProgress(100);
            
            const totalChunks = (result.text_chunks || 0) + (result.table_chunks || 0) + (result.image_chunks || 0);
            addLog({
                agent: 'RAG',
                message: `Document indexed: ${result.filename} (${totalChunks} chunks: ${result.text_chunks || 0} text, ${result.table_chunks || 0} tables, ${result.image_chunks || 0} images)`,
                status: 'completed'
            });

            setUploadStatus('success');
            setUploadMessage(`${file.name} indexed successfully!`);
            
            setTimeout(() => {
                setVectorizing(false);
                setLocalProgress(0);
                setUploadProgress(0);
                setTimeout(() => setUploadStatus('idle'), 2000);
            }, 1000);

        } catch (error) {
            console.error('Upload error:', error);
            setVectorizing(false);
            setLocalProgress(0);
            setUploadProgress(0);
            setUploadStatus('error');
            setUploadMessage('Upload failed. Is the backend running?');
            
            addLog({
                agent: 'RAG',
                message: `Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
                status: 'completed'
            });
            
            setTimeout(() => setUploadStatus('idle'), 3000);
        }

        // Reset file input
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const sessions = [
        { id: '1', title: 'Quantum Physics Gap', date: '2h ago' },
        { id: '2', title: 'Calculus Mental Model', date: '5h ago' },
        { id: '3', title: 'Python Loop Logic', date: 'Yesterday' },
    ];

    return (
        <aside className="w-80 border-r border-white/10 bg-black/40 backdrop-blur-xl flex flex-col h-screen overflow-hidden">
            <div className="p-6 flex items-center gap-3 border-b border-white/10">
                <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center border border-primary/50">
                    <BrainCircuit className="text-primary w-6 h-6" />
                </div>
                <div>
                    <h1 className="text-xl font-bold tracking-tighter neon-text">Darksied 🩺</h1>
                    <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Diagnostic Engine</p>
                </div>
            </div>

            <div className="p-4">
                <Button className="w-full justify-start gap-2 bg-primary/10 hover:bg-primary/20 border border-primary/30 text-primary">
                    <Plus className="w-4 h-4" />
                    New Diagnostic Session
                </Button>
            </div>

            <div className="flex-1 overflow-hidden flex flex-col px-4">
                <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-4 px-2 uppercase tracking-wider">
                    <History className="w-3 h-3" />
                    Session History
                </div>
                <ScrollArea className="flex-1 -mx-2 px-2">
                    <div className="space-y-1">
                        {sessions.map((session) => (
                            <button
                                key={session.id}
                                className="w-full text-left px-3 py-3 rounded-lg hover:bg-white/5 transition-colors group relative"
                            >
                                <div className="text-sm font-medium text-foreground/90 group-hover:text-primary transition-colors">
                                    {session.title}
                                </div>
                                <div className="text-[10px] text-muted-foreground mt-1">
                                    {session.date}
                                </div>
                            </button>
                        ))}
                    </div>
                </ScrollArea>
            </div>

            <div className="p-4 mt-auto border-t border-white/10 space-y-4">
                {/* Hidden file input with id for label */}
                <input
                    ref={fileInputRef}
                    id="file-upload-input"
                    type="file"
                    accept=".pdf,.txt,.md,.csv,.doc,.docx"
                    onChange={handleFileChange}
                    className="hidden"
                />
                
                {/* Use label for reliable file input trigger */}
                <label 
                    htmlFor="file-upload-input"
                    className="block w-full p-4 rounded-xl bg-white/5 border border-white/10 group cursor-pointer hover:border-primary/50 transition-all text-left"
                >
                    <div className="flex items-center gap-3 mb-3">
                        <div className={`p-2 rounded-lg ${
                            uploadStatus === 'success' ? 'bg-green-500/20 text-green-400' :
                            uploadStatus === 'error' ? 'bg-red-500/20 text-red-400' :
                            'bg-primary/10 text-primary'
                        }`}>
                            {uploadStatus === 'success' ? <CheckCircle2 className="w-5 h-5" /> :
                             uploadStatus === 'error' ? <AlertCircle className="w-5 h-5" /> :
                             <FileUp className="w-5 h-5" />}
                        </div>
                        <div>
                            <div className="text-sm font-semibold">
                                {uploadStatus === 'success' ? 'Upload Complete!' :
                                 uploadStatus === 'error' ? 'Upload Failed' :
                                 'Upload Knowledge'}
                            </div>
                            <div className="text-[10px] text-muted-foreground">
                                {uploadStatus !== 'idle' ? uploadMessage : 'PDFs, Docs, Links'}
                            </div>
                        </div>
                    </div>

                    {isVectorizing && (
                        <div className="space-y-2">
                            <div className="flex justify-between text-[10px] uppercase font-bold text-primary animate-pulse">
                                <span>Vectorizing...</span>
                                <span>{localProgress}%</span>
                            </div>
                            <Progress value={localProgress} className="h-1 bg-white/10" />
                        </div>
                    )}
                </label>

                <div className="flex items-center justify-between px-2">
                    <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-foreground">
                        <Settings className="w-5 h-5" />
                    </Button>
                    <div className="flex items-center gap-2">
                        <Shield className="w-4 h-4 text-primary" />
                        <span className="text-[10px] font-bold uppercase tracking-widest text-primary">Core Secure</span>
                    </div>
                </div>
            </div>
        </aside>
    );
}
