import { create } from 'zustand';

export type AgentRole = 'Supervisor' | 'RAG' | 'Diagnostician' | 'Fixer';

export interface ThoughtLog {
    id: string;
    agent: AgentRole;
    message: string;
    timestamp: Date;
    status: 'pending' | 'active' | 'completed';
}

export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    type?: 'text' | 'quiz' | 'mindmap';
    data?: any;
}

interface AgentState {
    logs: ThoughtLog[];
    chatHistory: ChatMessage[];
    activeTab: 'text' | 'voice' | 'architect';
    isVectorizing: boolean;
    uploadProgress: number;
    isThinking: boolean;
    sessionId: string | null;

    // Actions
    addLog: (log: Omit<ThoughtLog, 'id' | 'timestamp'>) => void;
    addChatMessage: (msg: Omit<ChatMessage, 'id' | 'timestamp'>) => void;
    setActiveTab: (tab: 'text' | 'voice' | 'architect') => void;
    setVectorizing: (isVectorizing: boolean) => void;
    setUploadProgress: (progress: number) => void;
    setThinking: (isThinking: boolean) => void;
    setSessionId: (id: string) => void;
    clearLogs: () => void;
    clearChat: () => void;
}

export const useAgentStore = create<AgentState>((set) => ({
    logs: [
        {
            id: '1',
            agent: 'Supervisor',
            message: 'System Initialized. Waiting for diagnostic request...',
            timestamp: new Date(),
            status: 'completed',
        }
    ],
    chatHistory: [
        {
            id: '1',
            role: 'assistant',
            content: "Hello, I am Darksied. I'm here to diagnose your learning gaps and build a personalized neural path for you. How can I help today?",
            timestamp: new Date(),
        }
    ],
    activeTab: 'text',
    isVectorizing: false,
    uploadProgress: 0,
    isThinking: false,
    sessionId: null,

    addLog: (log) => set((state) => ({
        logs: [
            {
                ...log,
                id: Math.random().toString(36).substring(7),
                timestamp: new Date(),
            },
            ...state.logs,
        ].slice(0, 50), // Keep last 50 logs
    })),

    addChatMessage: (msg) => set((state) => ({
        chatHistory: [
            ...state.chatHistory,
            {
                ...msg,
                id: Math.random().toString(36).substring(7),
                timestamp: new Date(),
            },
        ],
    })),

    setActiveTab: (tab) => set({ activeTab: tab }),
    setVectorizing: (isVectorizing) => set({ isVectorizing }),
    setUploadProgress: (uploadProgress) => set({ uploadProgress }),
    setThinking: (isThinking) => set({ isThinking }),
    setSessionId: (sessionId) => set({ sessionId }),
    clearLogs: () => set({ logs: [] }),
    clearChat: () => set({ chatHistory: [] }),
}));
