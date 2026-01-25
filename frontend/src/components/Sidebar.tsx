"use client";

import React, { useCallback, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { 
    Settings, 
    PlusCircle, 
    MessageSquare, 
    FileUp, 
    Shield, 
    Clock,
    Zap,
    ChevronRight,
    Trash2,
    FileText,
    FileSpreadsheet,
    File,
    Loader2,
    CheckCircle2,
    AlertCircle,
    X,
    FolderOpen
} from "lucide-react";
import { useAgentStore } from "@/store/useAgentStore";
import { api, ACCEPTED_FILE_EXTENSIONS, SUPPORTED_FILE_TYPES } from "@/lib/api";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";

// File type icon mapping
function getFileIcon(filename: string) {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
        case 'pdf':
            return <FileText className="w-4 h-4 text-rose-400" />;
        case 'csv':
            return <FileSpreadsheet className="w-4 h-4 text-emerald-400" />;
        case 'doc':
        case 'docx':
            return <FileText className="w-4 h-4 text-blue-400" />;
        case 'md':
            return <FileText className="w-4 h-4 text-purple-400" />;
        default:
            return <File className="w-4 h-4 text-muted-foreground" />;
    }
}

// Format relative time
function formatRelativeTime(date: Date): string {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days === 1) return 'Yesterday';
    return `${days}d ago`;
}

export function Sidebar() {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
    const [uploadMessage, setUploadMessage] = useState('');
    const [showDocuments, setShowDocuments] = useState(false);
    
    const { 
        sessionId, 
        sessions,
        currentSessionName,
        createNewSession,
        switchSession,
        deleteSession,
        addLog, 
        addDocument,
        removeDocument,
        clearSessionDocuments,
        setUploading,
        setUploadProgress,
        setUploadError,
        isUploading,
        documents
    } = useAgentStore();
    
    // Filter documents for current session
    const sessionDocuments = documents.filter(d => d.sessionId === sessionId);

    const handleFileChange = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        // Validate file type
        const fileExt = '.' + file.name.split('.').pop()?.toLowerCase();
        const acceptedExts = ACCEPTED_FILE_EXTENSIONS.split(',');
        if (!acceptedExts.includes(fileExt)) {
            setUploadStatus('error');
            setUploadMessage(`Unsupported file type. Accepted: ${ACCEPTED_FILE_EXTENSIONS}`);
            setTimeout(() => setUploadStatus('idle'), 5000);
            return;
        }

        setUploadStatus('uploading');
        setUploading(true);
        setUploadProgress(0);
        setUploadError(null);
        
        addLog({
            agent: 'RAG',
            message: `Uploading: ${file.name}`,
            status: 'active'
        });

        try {
            const result = await api.uploadFile(file, sessionId);
            
            addDocument({
                filename: result.filename,
                textChunks: result.text_chunks,
                tableChunks: result.table_chunks,
                imageChunks: result.image_chunks,
                sessionId: sessionId,
            });
            
            addLog({
                agent: 'RAG',
                message: `Indexed: ${result.filename} (${result.text_chunks} chunks)`,
                status: 'completed'
            });
            
            setUploadStatus('success');
            setUploadMessage(`${result.filename} indexed successfully!`);
            
        } catch (err) {
            const errorMsg = err instanceof Error ? err.message : 'Upload failed';
            setUploadStatus('error');
            setUploadMessage(errorMsg);
            setUploadError(errorMsg);
            
            addLog({
                agent: 'RAG',
                message: `Upload error: ${errorMsg}`,
                status: 'error'
            });
        } finally {
            setUploading(false);
            setUploadProgress(100);
            
            // Reset status after delay
            setTimeout(() => {
                setUploadStatus('idle');
                setUploadMessage('');
            }, 4000);
            
            // Clear input
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    }, [sessionId, addLog, addDocument, setUploading, setUploadProgress, setUploadError]);

    const handleNewSession = () => {
        createNewSession();
        addLog({
            agent: 'Supervisor',
            message: 'New diagnostic session initialized',
            status: 'completed'
        });
    };

    const handleClearDocuments = async () => {
        try {
            await api.clearSession(sessionId);
            clearSessionDocuments();
            addLog({
                agent: 'RAG',
                message: 'Session documents cleared',
                status: 'completed'
            });
        } catch (err) {
            // Clear locally even if backend fails
            clearSessionDocuments();
        }
    };

    return (
        <aside className="w-72 flex-shrink-0 flex flex-col border-r border-white/10 bg-black/60 backdrop-blur-xl relative overflow-hidden">
            {/* Animated background gradient */}
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-blue-500/5 pointer-events-none" />
            <div className="absolute top-0 left-0 w-40 h-40 bg-primary/10 rounded-full blur-[100px] pointer-events-none" />
            
            <div className="relative z-10 flex flex-col h-full">
                {/* Header */}
                <div className="p-4 border-b border-white/10">
                    <motion.div 
                        className="flex items-center gap-3 group cursor-pointer"
                        whileHover={{ x: 2 }}
                    >
                        <div className="relative">
                            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-primary/20 to-blue-500/20 flex items-center justify-center border border-white/10 overflow-hidden group-hover:border-primary/50 transition-colors">
                                <Image src="/logo.svg" alt="Darksied" width={28} height={28} className="opacity-90" />
                            </div>
                            <motion.div 
                                className="absolute -top-0.5 -right-0.5 w-3 h-3 rounded-full bg-emerald-500 border-2 border-black"
                                animate={{ scale: [1, 1.2, 1] }}
                                transition={{ duration: 2, repeat: Infinity }}
                            />
                        </div>
                        <div className="flex-1">
                            <h1 className="font-bold text-lg tracking-tight">Darksied</h1>
                            <p className="text-[10px] text-muted-foreground flex items-center gap-1">
                                <Zap className="w-3 h-3 text-primary" />
                                Neural Engine v2.0
                            </p>
                        </div>
                    </motion.div>
                </div>

                {/* New Session Button */}
                <div className="p-3">
                    <motion.button
                        onClick={handleNewSession}
                        className="w-full p-3 rounded-xl bg-gradient-to-r from-primary/10 to-blue-500/10 border border-primary/30 hover:border-primary/50 flex items-center gap-3 group transition-all"
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                    >
                        <div className="p-1.5 rounded-lg bg-primary/20">
                            <PlusCircle className="w-4 h-4 text-primary" />
                        </div>
                        <span className="text-sm font-medium">New Diagnostic Session</span>
                        <ChevronRight className="w-4 h-4 text-muted-foreground ml-auto group-hover:translate-x-1 transition-transform" />
                    </motion.button>
                </div>

                {/* Session History */}
                <div className="flex-1 flex flex-col min-h-0">
                    <div className="px-4 py-2 flex items-center gap-2">
                        <Clock className="w-3.5 h-3.5 text-muted-foreground" />
                        <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                            Session History
                        </span>
                    </div>
                    
                    <ScrollArea className="flex-1 px-3">
                        <div className="space-y-1 pb-3">
                            {sessions.slice(0, 10).map((session, index) => (
                                <motion.button
                                    key={session.id}
                                    initial={{ opacity: 0, x: -10 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: index * 0.05 }}
                                    onClick={() => switchSession(session.id)}
                                    className={`w-full p-2.5 rounded-lg text-left transition-all group relative ${
                                        session.id === sessionId 
                                            ? 'bg-primary/10 border border-primary/30' 
                                            : 'hover:bg-white/5 border border-transparent'
                                    }`}
                                >
                                    <div className="flex items-start gap-2.5">
                                        <div className={`mt-0.5 p-1 rounded ${
                                            session.id === sessionId ? 'bg-primary/20' : 'bg-white/5'
                                        }`}>
                                            <MessageSquare className={`w-3 h-3 ${
                                                session.id === sessionId ? 'text-primary' : 'text-muted-foreground'
                                            }`} />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-1.5">
                                                <span className="text-xs font-medium truncate">
                                                    {session.name}
                                                </span>
                                                {session.id === sessionId && (
                                                    <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                                                )}
                                            </div>
                                            <div className="flex items-center gap-2 text-[10px] text-muted-foreground mt-0.5">
                                                <span suppressHydrationWarning>{formatRelativeTime(new Date(session.lastActiveAt))}</span>
                                                {session.documentCount > 0 && (
                                                    <span className="flex items-center gap-0.5">
                                                        <FileText className="w-2.5 h-2.5" />
                                                        {session.documentCount}
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                        {session.id !== sessionId && (
                                            <div
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    deleteSession(session.id);
                                                }}
                                                onKeyDown={(e) => {
                                                    if (e.key === 'Enter' || e.key === ' ') {
                                                        e.preventDefault();
                                                        e.stopPropagation();
                                                        deleteSession(session.id);
                                                    }
                                                }}
                                                role="button"
                                                tabIndex={0}
                                                className="opacity-0 group-hover:opacity-100 p-1 hover:bg-rose-500/20 rounded transition-all cursor-pointer focus:outline-none focus:ring-2 focus:ring-rose-500/50"
                                            >
                                                <Trash2 className="w-3 h-3 text-rose-400" />
                                            </div>
                                        )}
                                    </div>
                                </motion.button>
                            ))}
                        </div>
                    </ScrollArea>
                </div>

                {/* Documents Section */}
                {sessionDocuments.length > 0 && (
                    <div className="border-t border-white/10">
                        <button
                            onClick={() => setShowDocuments(!showDocuments)}
                            className="w-full px-4 py-2 flex items-center gap-2 hover:bg-white/5 transition-colors"
                        >
                            <FolderOpen className="w-3.5 h-3.5 text-primary" />
                            <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                                Documents ({sessionDocuments.length})
                            </span>
                            <ChevronRight className={`w-3.5 h-3.5 text-muted-foreground ml-auto transition-transform ${showDocuments ? 'rotate-90' : ''}`} />
                        </button>
                        
                        <AnimatePresence>
                            {showDocuments && (
                                <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: 'auto', opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    className="overflow-hidden"
                                >
                                    <div className="px-3 pb-2 space-y-1">
                                        {sessionDocuments.map(doc => (
                                            <div 
                                                key={doc.id}
                                                className="flex items-center gap-2 p-2 rounded-lg bg-white/5 group"
                                            >
                                                {getFileIcon(doc.filename)}
                                                <div className="flex-1 min-w-0">
                                                    <p className="text-xs truncate">{doc.filename}</p>
                                                    <p className="text-[10px] text-muted-foreground">
                                                        {doc.textChunks} chunks
                                                    </p>
                                                </div>
                                                <button
                                                    onClick={() => removeDocument(doc.id)}
                                                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-rose-500/20 rounded transition-all"
                                                >
                                                    <X className="w-3 h-3 text-rose-400" />
                                                </button>
                                            </div>
                                        ))}
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            onClick={handleClearDocuments}
                                            className="w-full h-7 text-[10px] text-rose-400 hover:text-rose-300 hover:bg-rose-500/10"
                                        >
                                            Clear All Documents
                                        </Button>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                )}

                {/* Upload Section */}
                <div className="p-3 border-t border-white/10">
                    <input
                        ref={fileInputRef}
                        type="file"
                        id="file-upload-input"
                        accept={ACCEPTED_FILE_EXTENSIONS}
                        onChange={handleFileChange}
                        className="hidden"
                    />
                    
                    <label 
                        htmlFor="file-upload-input"
                        className={`block w-full p-3 rounded-xl border border-dashed transition-all cursor-pointer ${
                            uploadStatus === 'uploading' 
                                ? 'border-primary/50 bg-primary/5' 
                                : uploadStatus === 'success'
                                    ? 'border-emerald-500/50 bg-emerald-500/5'
                                    : uploadStatus === 'error'
                                        ? 'border-rose-500/50 bg-rose-500/5'
                                        : 'border-white/20 hover:border-primary/40 hover:bg-primary/5'
                        }`}
                    >
                        <div className="flex items-center gap-3">
                            <div className={`p-2 rounded-lg ${
                                uploadStatus === 'uploading' ? 'bg-primary/20' :
                                uploadStatus === 'success' ? 'bg-emerald-500/20' :
                                uploadStatus === 'error' ? 'bg-rose-500/20' :
                                'bg-white/5'
                            }`}>
                                {uploadStatus === 'uploading' ? (
                                    <Loader2 className="w-4 h-4 text-primary animate-spin" />
                                ) : uploadStatus === 'success' ? (
                                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                                ) : uploadStatus === 'error' ? (
                                    <AlertCircle className="w-4 h-4 text-rose-400" />
                                ) : (
                                    <FileUp className="w-4 h-4 text-muted-foreground" />
                                )}
                            </div>
                            <div className="flex-1">
                                <p className={`text-xs font-medium ${
                                    uploadStatus === 'success' ? 'text-emerald-400' :
                                    uploadStatus === 'error' ? 'text-rose-400' :
                                    ''
                                }`}>
                                    {uploadStatus === 'uploading' ? 'Uploading...' :
                                     uploadStatus === 'success' ? 'Upload Complete' :
                                     uploadStatus === 'error' ? 'Upload Failed' :
                                     'Upload Knowledge'}
                                </p>
                                <p className="text-[10px] text-muted-foreground">
                                    {uploadMessage || 'PDFs, Docs, TXT, CSV, MD'}
                                </p>
                            </div>
                        </div>
                    </label>
                </div>

                {/* Footer */}
                <div className="p-3 border-t border-white/10 flex items-center justify-between">
                    <button className="p-2 rounded-lg hover:bg-white/5 transition-colors">
                        <Settings className="w-4 h-4 text-muted-foreground" />
                    </button>
                    <div className="flex items-center gap-2 text-[10px]">
                        <Shield className="w-3.5 h-3.5 text-emerald-400" />
                        <span className="text-emerald-400 font-medium">Core Secure</span>
                    </div>
                </div>
            </div>
        </aside>
    );
}
