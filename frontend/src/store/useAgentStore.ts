/**
 * Darksied Global State Management
 * Production-ready Zustand store for multi-agent system
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { generateSessionId } from '@/lib/api';

// ============================================================================
// Types
// ============================================================================

export type AgentRole = 
    | 'Supervisor' 
    | 'Research' 
    | 'Examiner' 
    | 'Chat' 
    | 'RAG' 
    | 'LearningArchitect'
    | 'Voice';

export type AgentStatus = 'pending' | 'active' | 'completed' | 'error';

export interface ThoughtLog {
    id: string;
    agent: AgentRole;
    message: string;
    timestamp: Date;
    status: AgentStatus;
}

export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
    type?: 'text' | 'quiz' | 'mindmap';
    data?: unknown;
    routedAgent?: AgentRole;
}

export interface UploadedDocument {
    id: string;
    filename: string;
    uploadedAt: Date;
    textChunks: number;
    tableChunks: number;
    imageChunks: number;
    sessionId: string;
}

export interface SessionInfo {
    id: string;
    name: string;
    createdAt: Date;
    lastActiveAt: Date;
    messageCount: number;
    documentCount: number;
}

export interface HealthStatus {
    isHealthy: boolean;
    lastChecked: Date | null;
    backend: 'connected' | 'disconnected' | 'checking';
    qdrant: 'connected' | 'disconnected' | 'checking';
    postgres: 'connected' | 'disconnected' | 'checking';
}

export interface LearningMaterials {
    mindmap: string | null;
    quiz: Array<{
        question: string;
        options?: string[];
        answer: string;
        explanation?: string;
        hint?: string;
    }> | null;
    topic: string;
    generatedAt: Date | null;
    isGenerating: boolean;
}

// ============================================================================
// Store Interface
// ============================================================================

interface AgentState {
    // Core State
    logs: ThoughtLog[];
    chatHistory: ChatMessage[];
    activeTab: 'text' | 'voice' | 'architect';
    
    // Session State
    sessionId: string;
    sessions: SessionInfo[];
    currentSessionName: string;
    
    // Document State
    documents: UploadedDocument[];
    isUploading: boolean;
    uploadProgress: number;
    uploadError: string | null;
    
    // Processing State
    isThinking: boolean;
    currentAgent: AgentRole | null;
    
    // Health State
    health: HealthStatus;
    
    // Learning Architect State
    learningMaterials: LearningMaterials;
    
    // Voice State
    isSpeaking: boolean;
    isListening: boolean;
    voiceTranscript: string;
    
    // Actions - Logs
    addLog: (log: Omit<ThoughtLog, 'id' | 'timestamp'>) => void;
    updateLogStatus: (id: string, status: AgentStatus) => void;
    clearLogs: () => void;
    
    // Actions - Chat
    addChatMessage: (msg: Omit<ChatMessage, 'id' | 'timestamp'>) => void;
    clearChat: () => void;
    
    // Actions - Tab
    setActiveTab: (tab: 'text' | 'voice' | 'architect') => void;
    
    // Actions - Session
    setSessionId: (id: string) => void;
    createNewSession: (name?: string) => string;
    switchSession: (id: string) => void;
    deleteSession: (id: string) => void;
    updateSessionActivity: () => void;
    
    // Actions - Documents
    addDocument: (doc: Omit<UploadedDocument, 'id' | 'uploadedAt'>) => void;
    removeDocument: (id: string) => void;
    clearSessionDocuments: () => void;
    setUploading: (isUploading: boolean) => void;
    setUploadProgress: (progress: number) => void;
    setUploadError: (error: string | null) => void;
    
    // Actions - Processing
    setThinking: (isThinking: boolean) => void;
    setCurrentAgent: (agent: AgentRole | null) => void;
    
    // Actions - Health
    setHealth: (health: Partial<HealthStatus>) => void;
    
    // Actions - Learning
    setLearningMaterials: (materials: Partial<LearningMaterials>) => void;
    clearLearningMaterials: () => void;
    
    // Actions - Voice
    setIsSpeaking: (speaking: boolean) => void;
    setIsListening: (listening: boolean) => void;
    setVoiceTranscript: (transcript: string) => void;
    
    // Actions - Reset
    resetStore: () => void;
}

// ============================================================================
// Initial State
// ============================================================================

const initialSessionId = generateSessionId();

const initialState = {
    logs: [
        {
            id: '1',
            agent: 'Supervisor' as AgentRole,
            message: 'System Initialized. Waiting for diagnostic request...',
            timestamp: new Date(),
            status: 'completed' as AgentStatus,
        }
    ],
    chatHistory: [
        {
            id: '1',
            role: 'assistant' as const,
            content: "Hello, I am **Darksied** - your multi-agent learning companion. I can help you with:\n\n• **Research** - Web search, LeetCode problems, DSA explanations\n• **Quizzes** - Generate MCQ questions on any topic\n• **Documents** - Upload PDFs/docs and ask questions (RAG)\n• **Learning** - Create mindmaps and study materials\n• **Voice** - Talk to me for Socratic tutoring\n\nHow can I assist your learning journey today?",
            timestamp: new Date(),
        }
    ],
    activeTab: 'text' as const,
    sessionId: initialSessionId,
    sessions: [
        {
            id: initialSessionId,
            name: 'New Session',
            createdAt: new Date(),
            lastActiveAt: new Date(),
            messageCount: 1,
            documentCount: 0,
        }
    ],
    currentSessionName: 'New Session',
    documents: [],
    isUploading: false,
    uploadProgress: 0,
    uploadError: null,
    isThinking: false,
    currentAgent: null,
    health: {
        isHealthy: true,
        lastChecked: null,
        backend: 'checking' as const,
        qdrant: 'checking' as const,
        postgres: 'checking' as const,
    },
    learningMaterials: {
        mindmap: null,
        quiz: null,
        topic: '',
        generatedAt: null,
        isGenerating: false,
    },
    isSpeaking: false,
    isListening: false,
    voiceTranscript: '',
};

// ============================================================================
// Store Implementation
// ============================================================================

export const useAgentStore = create<AgentState>()(
    persist(
        (set, get) => ({
            ...initialState,

            // -----------------------------------------------------------------
            // Log Actions
            // -----------------------------------------------------------------
            
            addLog: (log) => set((state) => ({
                logs: [
                    {
                        ...log,
                        id: Math.random().toString(36).substring(2, 9),
                        timestamp: new Date(),
                    },
                    ...state.logs,
                ].slice(0, 100), // Keep last 100 logs
            })),

            updateLogStatus: (id, status) => set((state) => ({
                logs: state.logs.map(log => 
                    log.id === id ? { ...log, status } : log
                ),
            })),

            clearLogs: () => set({ 
                logs: [{
                    id: '1',
                    agent: 'Supervisor',
                    message: 'Logs cleared. System ready.',
                    timestamp: new Date(),
                    status: 'completed',
                }] 
            }),

            // -----------------------------------------------------------------
            // Chat Actions
            // -----------------------------------------------------------------
            
            addChatMessage: (msg) => set((state) => {
                const newMessage = {
                    ...msg,
                    id: Math.random().toString(36).substring(2, 9),
                    timestamp: new Date(),
                };
                
                // Update session message count
                const updatedSessions = state.sessions.map(s => 
                    s.id === state.sessionId 
                        ? { ...s, messageCount: s.messageCount + 1, lastActiveAt: new Date() }
                        : s
                );
                
                return {
                    chatHistory: [...state.chatHistory, newMessage],
                    sessions: updatedSessions,
                };
            }),

            clearChat: () => set((state) => ({
                chatHistory: [{
                    id: Math.random().toString(36).substring(2, 9),
                    role: 'assistant',
                    content: "Chat cleared. How can I help you?",
                    timestamp: new Date(),
                }],
                sessions: state.sessions.map(s => 
                    s.id === state.sessionId 
                        ? { ...s, messageCount: 1 }
                        : s
                ),
            })),

            // -----------------------------------------------------------------
            // Tab Actions
            // -----------------------------------------------------------------
            
            setActiveTab: (tab) => set({ activeTab: tab }),

            // -----------------------------------------------------------------
            // Session Actions
            // -----------------------------------------------------------------
            
            setSessionId: (sessionId) => set({ sessionId }),

            createNewSession: (name) => {
                const newId = generateSessionId();
                const sessionName = name || `Session ${get().sessions.length + 1}`;
                
                set((state) => ({
                    sessionId: newId,
                    currentSessionName: sessionName,
                    sessions: [
                        {
                            id: newId,
                            name: sessionName,
                            createdAt: new Date(),
                            lastActiveAt: new Date(),
                            messageCount: 1,
                            documentCount: 0,
                        },
                        ...state.sessions,
                    ],
                    chatHistory: [{
                        id: '1',
                        role: 'assistant',
                        content: `Started new session: **${sessionName}**. How can I help you today?`,
                        timestamp: new Date(),
                    }],
                    documents: [],
                    learningMaterials: {
                        mindmap: null,
                        quiz: null,
                        topic: '',
                        generatedAt: null,
                        isGenerating: false,
                    },
                }));
                
                return newId;
            },

            switchSession: (id) => {
                const session = get().sessions.find(s => s.id === id);
                if (session) {
                    set({
                        sessionId: id,
                        currentSessionName: session.name,
                        // Note: In production, you'd fetch chat history from backend
                    });
                }
            },

            deleteSession: (id) => set((state) => {
                const filteredSessions = state.sessions.filter(s => s.id !== id);
                
                // If deleting current session, switch to first available or create new
                if (state.sessionId === id) {
                    if (filteredSessions.length > 0) {
                        return {
                            sessions: filteredSessions,
                            sessionId: filteredSessions[0].id,
                            currentSessionName: filteredSessions[0].name,
                        };
                    } else {
                        const newId = generateSessionId();
                        return {
                            sessions: [{
                                id: newId,
                                name: 'New Session',
                                createdAt: new Date(),
                                lastActiveAt: new Date(),
                                messageCount: 1,
                                documentCount: 0,
                            }],
                            sessionId: newId,
                            currentSessionName: 'New Session',
                            chatHistory: [{
                                id: '1',
                                role: 'assistant',
                                content: "Session deleted. Starting fresh. How can I help?",
                                timestamp: new Date(),
                            }],
                        };
                    }
                }
                
                return { sessions: filteredSessions };
            }),

            updateSessionActivity: () => set((state) => ({
                sessions: state.sessions.map(s => 
                    s.id === state.sessionId 
                        ? { ...s, lastActiveAt: new Date() }
                        : s
                ),
            })),

            // -----------------------------------------------------------------
            // Document Actions
            // -----------------------------------------------------------------
            
            addDocument: (doc) => set((state) => ({
                documents: [
                    {
                        ...doc,
                        id: Math.random().toString(36).substring(2, 9),
                        uploadedAt: new Date(),
                    },
                    ...state.documents,
                ],
                sessions: state.sessions.map(s => 
                    s.id === state.sessionId 
                        ? { ...s, documentCount: s.documentCount + 1 }
                        : s
                ),
            })),

            removeDocument: (id) => set((state) => ({
                documents: state.documents.filter(d => d.id !== id),
                sessions: state.sessions.map(s => 
                    s.id === state.sessionId 
                        ? { ...s, documentCount: Math.max(0, s.documentCount - 1) }
                        : s
                ),
            })),

            clearSessionDocuments: () => set((state) => ({
                documents: state.documents.filter(d => d.sessionId !== state.sessionId),
                sessions: state.sessions.map(s => 
                    s.id === state.sessionId 
                        ? { ...s, documentCount: 0 }
                        : s
                ),
            })),

            setUploading: (isUploading) => set({ isUploading }),
            setUploadProgress: (uploadProgress) => set({ uploadProgress }),
            setUploadError: (uploadError) => set({ uploadError }),

            // -----------------------------------------------------------------
            // Processing Actions
            // -----------------------------------------------------------------
            
            setThinking: (isThinking) => set({ isThinking }),
            setCurrentAgent: (currentAgent) => set({ currentAgent }),

            // -----------------------------------------------------------------
            // Health Actions
            // -----------------------------------------------------------------
            
            setHealth: (health) => set((state) => ({
                health: { ...state.health, ...health, lastChecked: new Date() },
            })),

            // -----------------------------------------------------------------
            // Learning Actions
            // -----------------------------------------------------------------
            
            setLearningMaterials: (materials) => set((state) => ({
                learningMaterials: { ...state.learningMaterials, ...materials },
            })),

            clearLearningMaterials: () => set({
                learningMaterials: {
                    mindmap: null,
                    quiz: null,
                    topic: '',
                    generatedAt: null,
                    isGenerating: false,
                },
            }),

            // -----------------------------------------------------------------
            // Voice Actions
            // -----------------------------------------------------------------
            
            setIsSpeaking: (isSpeaking) => set({ isSpeaking }),
            setIsListening: (isListening) => set({ isListening }),
            setVoiceTranscript: (voiceTranscript) => set({ voiceTranscript }),

            // -----------------------------------------------------------------
            // Reset Action
            // -----------------------------------------------------------------
            
            resetStore: () => {
                const newId = generateSessionId();
                set({
                    ...initialState,
                    sessionId: newId,
                    sessions: [{
                        id: newId,
                        name: 'New Session',
                        createdAt: new Date(),
                        lastActiveAt: new Date(),
                        messageCount: 1,
                        documentCount: 0,
                    }],
                });
            },
        }),
        {
            name: 'darksied-store',
            storage: createJSONStorage(() => localStorage),
            partialize: (state) => ({
                // Only persist these fields
                sessions: state.sessions,
                sessionId: state.sessionId,
                currentSessionName: state.currentSessionName,
            }),
        }
    )
);

// ============================================================================
// Selectors (for performance optimization)
// ============================================================================

export const selectCurrentSession = (state: AgentState) => 
    state.sessions.find(s => s.id === state.sessionId);

export const selectSessionDocuments = (state: AgentState) =>
    state.documents.filter(d => d.sessionId === state.sessionId);

export const selectRecentLogs = (state: AgentState, count = 10) =>
    state.logs.slice(0, count);

export const selectIsProcessing = (state: AgentState) =>
    state.isThinking || state.isUploading || state.learningMaterials.isGenerating;
