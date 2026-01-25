"use client";

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
    Mic, 
    MicOff, 
    Settings, 
    Sparkles, 
    X, 
    Volume2, 
    VolumeX,
    Phone,
    PhoneOff,
    Bot,
    Waves,
    AlertCircle,
    Loader2,
    RefreshCw,
    Wifi,
    Radio,
    Zap,
    HelpCircle,
    CheckCircle2,
    PlayCircle,
    ArrowRight,
    Info
} from "lucide-react";
import { useAgentStore } from "@/store/useAgentStore";
import { api, VoiceStatus } from "@/lib/api";

// =============================================================================
// TYPES
// =============================================================================

type AvatarState = 'idle' | 'connecting' | 'listening' | 'thinking' | 'speaking' | 'error';
type VoiceMode = 'livekit' | 'browser' | 'checking';

interface SpeechRecognitionEvent extends Event {
    results: SpeechRecognitionResultList;
    resultIndex: number;
}

interface SpeechRecognitionErrorEvent extends Event {
    error: string;
    message: string;
}

// =============================================================================
// BROWSER API DETECTION
// =============================================================================

const isBrowser = typeof window !== 'undefined';
const SpeechRecognition = isBrowser ? 
    (window.SpeechRecognition || (window as any).webkitSpeechRecognition) : null;
const speechSynthesis = isBrowser ? window.speechSynthesis : null;

// =============================================================================
// COMPONENT
// =============================================================================

export function VoiceAvatar() {
    // ----- Core State -----
    const [voiceMode, setVoiceMode] = useState<VoiceMode>('checking');
    const [voiceStatus, setVoiceStatus] = useState<VoiceStatus | null>(null);
    const [isMuted, setIsMuted] = useState(true);
    const [isSpeakerOn, setIsSpeakerOn] = useState(true);
    const [isConnected, setIsConnected] = useState(false);
    const [avatarState, setAvatarState] = useState<AvatarState>('idle');
    const [currentSubtitle, setCurrentSubtitle] = useState("Initializing voice system...");
    const [audioLevel, setAudioLevel] = useState<number[]>(Array(12).fill(20));
    const [showHelp, setShowHelp] = useState(false);
    const [hasSeenHelp, setHasSeenHelp] = useState(false);
    
    // ----- Browser Speech State -----
    const [transcript, setTranscript] = useState('');
    const [interimTranscript, setInterimTranscript] = useState('');
    const [errorMessage, setErrorMessage] = useState('');
    const [isSupported, setIsSupported] = useState(true);
    const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
    const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null);
    const [showSettings, setShowSettings] = useState(false);
    const [speechRate, setSpeechRate] = useState(1.0);
    const [speechPitch, setSpeechPitch] = useState(1.0);
    
    // ----- Refs -----
    const recognitionRef = useRef<any>(null);
    const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const animationFrameRef = useRef<number>(0);
    const streamRef = useRef<MediaStream | null>(null);
    
    // ----- Store -----
    const { addLog, sessionId, setSessionId, addChatMessage } = useAgentStore();

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    // Check voice service status on mount
    useEffect(() => {
        const checkVoiceService = async () => {
            try {
                const status = await api.getVoiceStatus();
                setVoiceStatus(status);
                
                if (status.available && status.mode === 'livekit') {
                    setVoiceMode('livekit');
                    setCurrentSubtitle("Real-time voice ready! Click 'Start Voice Session' to begin.");
                } else {
                    setVoiceMode('browser');
                    setCurrentSubtitle("Voice ready! Click 'Start Voice Session' to begin talking.");
                }
                
                // Show help on first visit
                const helpSeen = localStorage.getItem('darksied-voice-help-seen');
                if (!helpSeen && !isConnected) {
                    setTimeout(() => setShowHelp(true), 2000);
                }
            } catch {
                setVoiceMode('browser');
                setCurrentSubtitle("Voice ready! Click 'Start Voice Session' to begin talking.");
            }
        };
        
        checkVoiceService();
    }, []);

    // Check browser support
    useEffect(() => {
        if (!SpeechRecognition && voiceMode === 'browser') {
            setIsSupported(false);
            setErrorMessage('Speech recognition not supported. Please use Chrome, Edge, or Safari.');
        }
        
        // Load voices
        if (speechSynthesis) {
            const loadVoices = () => {
                const availableVoices = speechSynthesis.getVoices();
                if (availableVoices.length > 0) {
                    setVoices(availableVoices);
                    const preferredVoice = availableVoices.find(v => 
                        v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Microsoft'))
                    ) || availableVoices.find(v => 
                        v.lang.startsWith('en') && v.localService
                    ) || availableVoices.find(v => 
                        v.lang.startsWith('en')
                    ) || availableVoices[0];
                    
                    if (preferredVoice && !selectedVoice) {
                        setSelectedVoice(preferredVoice);
                    }
                }
            };
            
            loadVoices();
            speechSynthesis.onvoiceschanged = loadVoices;
            [100, 500, 1000, 2000].forEach(delay => setTimeout(loadVoices, delay));
        }
    }, [voiceMode, selectedVoice]);

    // =========================================================================
    // CLEANUP
    // =========================================================================

    const cleanup = useCallback(() => {
        if (recognitionRef.current) {
            try { recognitionRef.current.abort(); } catch {}
        }
        if (speechSynthesis) {
            speechSynthesis.cancel();
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
        }
        if (audioContextRef.current) {
            audioContextRef.current.close();
        }
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
        }
    }, []);

    useEffect(() => {
        return () => cleanup();
    }, [cleanup]);

    // =========================================================================
    // AUDIO VISUALIZATION
    // =========================================================================

    const initAudioAnalyzer = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;
            
            audioContextRef.current = new AudioContext();
            analyserRef.current = audioContextRef.current.createAnalyser();
            analyserRef.current.fftSize = 32;
            
            const source = audioContextRef.current.createMediaStreamSource(stream);
            source.connect(analyserRef.current);
            
            return true;
        } catch (err) {
            console.warn('Microphone access denied:', err);
            return false;
        }
    }, []);

    const updateAudioVisualization = useCallback(() => {
        if (!analyserRef.current || avatarState === 'idle') {
            setAudioLevel(Array(12).fill(20));
            return;
        }
        
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);
        
        const levels = Array.from({ length: 12 }, (_, i) => {
            const index = Math.floor(i * dataArray.length / 12);
            const value = dataArray[index] || 0;
            return Math.max(10, (value / 255) * 80 + 10);
        });
        
        setAudioLevel(levels);
        animationFrameRef.current = requestAnimationFrame(updateAudioVisualization);
    }, [avatarState]);

    useEffect(() => {
        if (avatarState === 'listening' || avatarState === 'speaking') {
            updateAudioVisualization();
        } else {
            if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
            setAudioLevel(Array(12).fill(20));
        }
        
        return () => {
            if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
        };
    }, [avatarState, updateAudioVisualization]);

    // Simulated levels when analyzer not available
    useEffect(() => {
        if ((avatarState === 'speaking' || avatarState === 'listening') && !analyserRef.current) {
            const interval = setInterval(() => {
                setAudioLevel(prev => prev.map(() => 
                    avatarState === 'speaking' 
                        ? Math.random() * 60 + 20 
                        : avatarState === 'listening'
                            ? Math.random() * 40 + 10
                            : 20
                ));
            }, 100);
            return () => clearInterval(interval);
        }
    }, [avatarState]);

    // =========================================================================
    // SPEECH RECOGNITION (Browser Mode)
    // =========================================================================

    const initRecognition = useCallback(() => {
        if (!SpeechRecognition) return null;
        
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';
        recognition.maxAlternatives = 1;
        
        recognition.onstart = () => {
            setAvatarState('listening');
            setCurrentSubtitle("🎤 I'm listening... speak your question naturally.");
            setErrorMessage('');
        };
        
        recognition.onresult = (event: SpeechRecognitionEvent) => {
            let finalTranscript = '';
            let interimText = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const result = event.results[i];
                if (result.isFinal) {
                    finalTranscript += result[0].transcript;
                } else {
                    interimText += result[0].transcript;
                }
            }
            
            if (finalTranscript) setTranscript(prev => prev + finalTranscript);
            setInterimTranscript(interimText);
            if (finalTranscript || interimText) {
                setCurrentSubtitle(finalTranscript || interimText);
            }
        };
        
        recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
            switch (event.error) {
                case 'not-allowed':
                case 'permission-denied':
                    setErrorMessage('⚠️ Microphone permission denied. Please allow microphone access in your browser settings.');
                    setAvatarState('error');
                    break;
                case 'no-speech':
                    if (isConnected && !isMuted) {
                        setTimeout(() => { try { recognition.start(); } catch {} }, 100);
                    }
                    break;
                case 'network':
                    setErrorMessage('⚠️ Network error. Please check your internet connection.');
                    setAvatarState('error');
                    break;
                default:
                    if (isConnected && !isMuted) {
                        setTimeout(() => { try { recognition.start(); } catch {} }, 100);
                    }
            }
        };
        
        recognition.onend = () => {
            if (transcript && isConnected && !isMuted) {
                processUserInput(transcript);
                setTranscript('');
                setInterimTranscript('');
            }
            
            if (isConnected && !isMuted && avatarState === 'listening') {
                setTimeout(() => { try { recognition.start(); } catch {} }, 100);
            }
        };
        
        return recognition;
    }, [isConnected, isMuted, avatarState, transcript]);

    // =========================================================================
    // PROCESS USER INPUT
    // =========================================================================

    const processUserInput = async (userText: string) => {
        if (!userText.trim()) return;
        
        setAvatarState('thinking');
        setCurrentSubtitle("🧠 Processing your question...");
        
        addLog({ agent: 'Voice', message: `Input: "${userText.substring(0, 50)}..."`, status: 'active' });
        addChatMessage({ role: 'user', content: `🎤 ${userText}`, type: 'text' });

        try {
            let currentSessionId = sessionId;
            if (!currentSessionId) {
                currentSessionId = Math.random().toString(36).substring(2, 15);
                setSessionId(currentSessionId);
            }

            const response = await api.chat(userText, currentSessionId);
            
            addChatMessage({
                role: 'assistant',
                content: response.response,
                type: response.type,
                data: response.data
            });

            addLog({ agent: 'Voice', message: 'Response generated', status: 'completed' });

            if (isSpeakerOn) {
                await speakText(response.response);
            } else {
                setCurrentSubtitle(response.response);
                setAvatarState('idle');
            }

        } catch (error) {
            console.error('API Error:', error);
            setAvatarState('error');
            const errorMsg = error instanceof Error ? error.message : 'Failed to get response';
            setCurrentSubtitle(`❌ Error: ${errorMsg}`);
            setErrorMessage(errorMsg);
            
            setTimeout(() => {
                if (isConnected) {
                    setAvatarState('idle');
                    setErrorMessage('');
                    setCurrentSubtitle("Ready for your next question.");
                }
            }, 3000);
        }
    };

    // =========================================================================
    // TEXT-TO-SPEECH (Browser Mode)
    // =========================================================================

    const speakText = useCallback((text: string): Promise<void> => {
        return new Promise((resolve) => {
            if (!speechSynthesis || !text || !isSpeakerOn) {
                resolve();
                return;
            }

            speechSynthesis.cancel();

            const cleanText = text
                .replace(/\*\*/g, '')
                .replace(/\*/g, '')
                .replace(/```[\s\S]*?```/g, 'code block')
                .replace(/`([^`]+)`/g, '$1')
                .replace(/#{1,6}\s/g, '')
                .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
                .replace(/[⚠️🎤📝🧠❌✅🚀💡]/g, '')
                .replace(/\s+/g, ' ')
                .trim();

            if (!cleanText || cleanText.length < 2) {
                resolve();
                return;
            }

            const maxLength = 200;
            const textToSpeak = cleanText.length > maxLength 
                ? cleanText.substring(0, maxLength) + '...'
                : cleanText;

            try {
                const utterance = new SpeechSynthesisUtterance(textToSpeak);
                utteranceRef.current = utterance;
                
                if (selectedVoice) utterance.voice = selectedVoice;
                utterance.rate = Math.max(0.5, Math.min(2, speechRate));
                utterance.pitch = Math.max(0.5, Math.min(2, speechPitch));
                utterance.volume = 1;

                let hasStarted = false;
                let timeoutId: NodeJS.Timeout | null = null;
                let resumeInterval: NodeJS.Timeout | null = null;

                const cleanupSpeech = () => {
                    if (timeoutId) clearTimeout(timeoutId);
                    if (resumeInterval) clearInterval(resumeInterval);
                };

                utterance.onstart = () => {
                    hasStarted = true;
                    setAvatarState('speaking');
                    setCurrentSubtitle(text);
                    
                    resumeInterval = setInterval(() => {
                        if (speechSynthesis.speaking && !speechSynthesis.paused) {
                            speechSynthesis.pause();
                            speechSynthesis.resume();
                        }
                    }, 10000);
                };

                utterance.onend = () => {
                    cleanupSpeech();
                    setAvatarState('idle');
                    setCurrentSubtitle("✅ Ready! Click the microphone to ask another question.");
                    resolve();
                };

                utterance.onerror = () => {
                    cleanupSpeech();
                    setAvatarState('idle');
                    setCurrentSubtitle(text);
                    resolve();
                };

                timeoutId = setTimeout(() => {
                    if (!hasStarted) {
                        setAvatarState('idle');
                        setCurrentSubtitle(text);
                        resolve();
                    }
                }, 5000);

                speechSynthesis.speak(utterance);

                setTimeout(() => {
                    if (!speechSynthesis.speaking && !hasStarted) {
                        speechSynthesis.cancel();
                        speechSynthesis.speak(utterance);
                    }
                }, 100);

            } catch {
                setAvatarState('idle');
                setCurrentSubtitle(text);
                resolve();
            }
        });
    }, [selectedVoice, speechRate, speechPitch, isSpeakerOn]);

    // =========================================================================
    // CONNECTION HANDLERS
    // =========================================================================

    const handleConnect = async () => {
        if (isConnected) {
            // Disconnect
            cleanup();
            setIsConnected(false);
            setAvatarState('idle');
            setIsMuted(true);
            setTranscript('');
            setInterimTranscript('');
            setCurrentSubtitle("Session ended. Click 'Start Voice Session' to begin again.");
            setErrorMessage('');
            addLog({ agent: 'Voice', message: 'Disconnected', status: 'completed' });
        } else {
            // Connect
            setAvatarState('connecting');
            setCurrentSubtitle("🔌 Connecting... Please allow microphone access when prompted.");
            setErrorMessage('');
            
            try {
                // Initialize audio
                const audioInitialized = await Promise.race([
                    initAudioAnalyzer(),
                    new Promise<boolean>(r => setTimeout(() => r(false), 10000))
                ]);
                
                if (!audioInitialized) {
                    console.warn('Microphone not granted, continuing without visualization');
                }

                // For browser mode, initialize recognition
                if (voiceMode === 'browser') {
                    recognitionRef.current = initRecognition();
                }
                
                setIsConnected(true);
                addLog({ agent: 'Voice', message: `Connected (${voiceMode} mode)`, status: 'active' });

                const greeting = `Hello! I'm Darksied, your AI learning companion. ${
                    voiceMode === 'livekit' 
                        ? "Real-time voice is active." 
                        : "I'm ready to help you learn. Click the microphone button below to start speaking!"
                }`;
                
                setCurrentSubtitle(greeting);
                setAvatarState('speaking');
                
                await new Promise(r => setTimeout(r, 100));
                
                if (isSpeakerOn && voices.length > 0) {
                    await speakText(greeting);
                } else {
                    await new Promise(r => setTimeout(r, 2000));
                }
                
                setAvatarState('idle');
                setCurrentSubtitle("✅ Connected! Click the microphone button to start speaking.");
                
            } catch (err) {
                console.error('Connection error:', err);
                setErrorMessage('❌ Failed to initialize voice. Please check your microphone permissions.');
                setAvatarState('error');
                setIsConnected(false);
            }
        }
    };

    const handleMicToggle = async () => {
        if (!isConnected) {
            setCurrentSubtitle("⚠️ Please connect first by clicking 'Start Voice Session'.");
            return;
        }
        
        const newMutedState = !isMuted;
        setIsMuted(newMutedState);
        
        if (!newMutedState) {
            if (!recognitionRef.current) {
                recognitionRef.current = initRecognition();
            }
            
            try {
                recognitionRef.current?.start();
                setAvatarState('listening');
                setCurrentSubtitle("🎤 Listening... Speak your question now!");
                addLog({ agent: 'Voice', message: 'Listening', status: 'active' });
            } catch (err) {
                setErrorMessage('❌ Failed to start listening. Please check microphone permissions.');
            }
        } else {
            try { recognitionRef.current?.stop(); } catch {}
            
            if (transcript) {
                processUserInput(transcript);
                setTranscript('');
                setInterimTranscript('');
            } else {
                setAvatarState('idle');
                setCurrentSubtitle("✅ Microphone muted. Click again to unmute and speak.");
            }
            
            addLog({ agent: 'Voice', message: 'Muted', status: 'completed' });
        }
    };

    const stopSpeaking = () => {
        if (speechSynthesis) speechSynthesis.cancel();
        setAvatarState('idle');
    };

    // =========================================================================
    // VISUAL HELPERS
    // =========================================================================

    const getAvatarColor = () => {
        switch (avatarState) {
            case 'connecting': return 'from-cyan-500/30 to-blue-500/30';
            case 'listening': return 'from-blue-500/30 to-cyan-500/30';
            case 'thinking': return 'from-amber-500/30 to-orange-500/30';
            case 'speaking': return 'from-emerald-500/30 to-green-500/30';
            case 'error': return 'from-rose-500/30 to-red-500/30';
            default: return 'from-primary/20 to-primary/10';
        }
    };

    const getBorderColor = () => {
        switch (avatarState) {
            case 'connecting': return 'border-cyan-500/50';
            case 'listening': return 'border-blue-500/50';
            case 'thinking': return 'border-amber-500/50';
            case 'speaking': return 'border-emerald-500/50';
            case 'error': return 'border-rose-500/50';
            default: return 'border-primary/50';
        }
    };

    const getStatusText = () => {
        switch (avatarState) {
            case 'connecting': return 'Connecting...';
            case 'listening': return 'Listening...';
            case 'thinking': return 'Processing...';
            case 'speaking': return 'Speaking...';
            case 'error': return 'Error';
            default: return isConnected ? 'Ready' : 'Not Connected';
        }
    };

    const getStatusColor = () => {
        switch (avatarState) {
            case 'connecting': return 'text-cyan-400';
            case 'listening': return 'text-blue-400';
            case 'thinking': return 'text-amber-400';
            case 'speaking': return 'text-emerald-400';
            case 'error': return 'text-rose-400';
            default: return isConnected ? 'text-primary' : 'text-muted-foreground';
        }
    };

    // =========================================================================
    // RENDER - UNSUPPORTED BROWSER
    // =========================================================================

    if (!isSupported && voiceMode === 'browser') {
        return (
            <div className="h-full flex flex-col items-center justify-center p-6 bg-black">
                <AlertCircle className="w-16 h-16 text-rose-500 mb-4" />
                <h2 className="text-xl font-bold text-white mb-2">Browser Not Supported</h2>
                <p className="text-muted-foreground text-center max-w-md mb-4">
                    Voice features require Speech Recognition support.
                    Please use <strong>Chrome</strong>, <strong>Edge</strong>, or <strong>Safari</strong>.
                </p>
                <p className="text-sm text-muted-foreground text-center max-w-md">
                    You can still use the <strong>Text Diagnosis</strong> tab to chat with Darksied.
                </p>
            </div>
        );
    }

    // =========================================================================
    // RENDER - MAIN COMPONENT
    // =========================================================================

    return (
        <div className="h-full flex flex-col items-center justify-center p-6 bg-black relative overflow-hidden">
            {/* Background Effects */}
            <motion.div 
                animate={{
                    scale: [1, 1.3, 1],
                    opacity: [0.1, 0.25, 0.1],
                    rotate: [0, 180, 360],
                }}
                transition={{ repeat: Infinity, duration: 15, ease: "easeInOut" }}
                className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[900px] bg-gradient-radial ${getAvatarColor()} rounded-full blur-[180px] pointer-events-none`}
            />
            
            {/* Grid Pattern */}
            <div 
                className="absolute inset-0 pointer-events-none opacity-[0.02]"
                style={{
                    backgroundImage: `linear-gradient(rgba(34, 197, 94, 0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(34, 197, 94, 0.5) 1px, transparent 1px)`,
                    backgroundSize: '50px 50px',
                }}
            />

            {/* Floating Particles */}
            {isConnected && (
                <div className="absolute inset-0 pointer-events-none overflow-hidden">
                    {Array.from({ length: 20 }).map((_, i) => (
                        <motion.div
                            key={i}
                            className="absolute w-1 h-1 rounded-full bg-primary/60"
                            style={{ left: `${Math.random() * 100}%`, top: `${Math.random() * 100}%` }}
                            animate={{ y: [0, -100, 0], opacity: [0, 1, 0], scale: [0.5, 1.5, 0.5] }}
                            transition={{ duration: 5 + Math.random() * 5, repeat: Infinity, delay: Math.random() * 5 }}
                        />
                    ))}
                </div>
            )}

            {/* Error Banner */}
            <AnimatePresence>
                {errorMessage && (
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="absolute top-4 left-1/2 -translate-x-1/2 z-50 max-w-md"
                    >
                        <div className="flex items-center gap-2 px-4 py-3 rounded-lg bg-rose-500/20 border border-rose-500/50 text-rose-400">
                            <AlertCircle className="w-5 h-5 flex-shrink-0" />
                            <span className="text-sm">{errorMessage}</span>
                            <button onClick={() => setErrorMessage('')} className="ml-2 hover:text-white flex-shrink-0">
                                <X className="w-4 h-4" />
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Help Button */}
            <button
                onClick={() => {
                    setShowHelp(true);
                    setHasSeenHelp(true);
                    localStorage.setItem('darksied-voice-help-seen', 'true');
                }}
                className="absolute top-4 right-4 z-50 p-2 rounded-full bg-white/5 border border-white/10 hover:bg-white/10 transition-all group"
                title="How to use Voice Tutor"
            >
                <HelpCircle className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
            </button>

            {/* Help Panel */}
            <AnimatePresence>
                {showHelp && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="absolute inset-0 z-50 flex items-center justify-center p-6 bg-black/80 backdrop-blur-xl"
                    >
                        <motion.div
                            initial={{ y: 20 }}
                            animate={{ y: 0 }}
                            className="bg-black/90 border border-white/20 rounded-2xl p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto"
                        >
                            <div className="flex items-center justify-between mb-6">
                                <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                                    <HelpCircle className="w-6 h-6 text-primary" />
                                    How to Use Voice Tutor
                                </h2>
                                <button
                                    onClick={() => setShowHelp(false)}
                                    className="p-2 rounded-full hover:bg-white/10 transition-colors"
                                >
                                    <X className="w-5 h-5 text-muted-foreground" />
                                </button>
                            </div>

                            <div className="space-y-6">
                                {/* Step 1 */}
                                <div className="flex gap-4">
                                    <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary/20 border border-primary/50 flex items-center justify-center">
                                        <span className="text-primary font-bold">1</span>
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="text-lg font-semibold text-white mb-2">Start Voice Session</h3>
                                        <p className="text-muted-foreground">
                                            Click the large green <strong className="text-primary">"Start Voice Session"</strong> button to connect.
                                            You'll be asked to allow microphone access - click <strong>"Allow"</strong>.
                                        </p>
                                    </div>
                                </div>

                                {/* Step 2 */}
                                <div className="flex gap-4">
                                    <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary/20 border border-primary/50 flex items-center justify-center">
                                        <span className="text-primary font-bold">2</span>
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="text-lg font-semibold text-white mb-2">Enable Microphone</h3>
                                        <p className="text-muted-foreground">
                                            Click the <strong className="text-blue-400">microphone button</strong> (large button in the center) to unmute.
                                            When active, it will glow blue and pulse.
                                        </p>
                                    </div>
                                </div>

                                {/* Step 3 */}
                                <div className="flex gap-4">
                                    <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary/20 border border-primary/50 flex items-center justify-center">
                                        <span className="text-primary font-bold">3</span>
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="text-lg font-semibold text-white mb-2">Speak Your Question</h3>
                                        <p className="text-muted-foreground">
                                            Speak naturally! The AI will listen and process your question.
                                            You'll see your words appear as you speak.
                                        </p>
                                    </div>
                                </div>

                                {/* Step 4 */}
                                <div className="flex gap-4">
                                    <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary/20 border border-primary/50 flex items-center justify-center">
                                        <span className="text-primary font-bold">4</span>
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="text-lg font-semibold text-white mb-2">Listen to Response</h3>
                                        <p className="text-muted-foreground">
                                            The AI will process your question and speak the answer back to you.
                                            You can ask follow-up questions by clicking the microphone again.
                                        </p>
                                    </div>
                                </div>

                                {/* Tips */}
                                <div className="pt-4 border-t border-white/10">
                                    <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                                        <Sparkles className="w-5 h-5 text-primary" />
                                        Tips
                                    </h3>
                                    <ul className="space-y-2 text-muted-foreground">
                                        <li className="flex items-start gap-2">
                                            <CheckCircle2 className="w-4 h-4 text-primary mt-0.5 flex-shrink-0" />
                                            <span>Speak clearly and at a normal pace</span>
                                        </li>
                                        <li className="flex items-start gap-2">
                                            <CheckCircle2 className="w-4 h-4 text-primary mt-0.5 flex-shrink-0" />
                                            <span>Click the speaker icon to toggle voice responses on/off</span>
                                        </li>
                                        <li className="flex items-start gap-2">
                                            <CheckCircle2 className="w-4 h-4 text-primary mt-0.5 flex-shrink-0" />
                                            <span>Use the settings icon to adjust voice speed and pitch</span>
                                        </li>
                                        <li className="flex items-start gap-2">
                                            <CheckCircle2 className="w-4 h-4 text-primary mt-0.5 flex-shrink-0" />
                                            <span>You can ask questions about uploaded documents</span>
                                        </li>
                                    </ul>
                                </div>
                            </div>

                            <button
                                onClick={() => setShowHelp(false)}
                                className="mt-6 w-full px-6 py-3 bg-primary/10 border border-primary/30 text-primary rounded-lg font-semibold hover:bg-primary/20 transition-colors"
                            >
                                Got it! Let's start
                            </button>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Status Badge */}
            <motion.div 
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="absolute top-8 left-1/2 -translate-x-1/2 flex items-center gap-3 z-10"
            >
                {/* Mode Indicator */}
                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${
                    voiceMode === 'livekit' 
                        ? 'bg-purple-500/20 border border-purple-500/30' 
                        : 'bg-blue-500/20 border border-blue-500/30'
                }`}>
                    {voiceMode === 'livekit' ? (
                        <>
                            <Radio className="w-3 h-3 text-purple-400" />
                            <span className="text-[10px] font-bold uppercase tracking-wider text-purple-400">LiveKit</span>
                        </>
                    ) : voiceMode === 'checking' ? (
                        <>
                            <Loader2 className="w-3 h-3 text-blue-400 animate-spin" />
                            <span className="text-[10px] font-bold uppercase tracking-wider text-blue-400">Checking...</span>
                        </>
                    ) : (
                        <>
                            <Zap className="w-3 h-3 text-blue-400" />
                            <span className="text-[10px] font-bold uppercase tracking-wider text-blue-400">Browser</span>
                        </>
                    )}
                </div>

                {/* Status Badge */}
                <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-black/50 border border-white/10 backdrop-blur-sm">
                    <motion.div 
                        animate={{ 
                            scale: avatarState !== 'idle' ? [1, 1.2, 1] : 1,
                            opacity: isConnected ? 1 : 0.5
                        }}
                        transition={{ repeat: avatarState !== 'idle' ? Infinity : 0, duration: 1 }}
                        className={`w-2 h-2 rounded-full ${
                            avatarState === 'connecting' ? 'bg-cyan-500' :
                            avatarState === 'listening' ? 'bg-blue-500' :
                            avatarState === 'thinking' ? 'bg-amber-500' :
                            avatarState === 'speaking' ? 'bg-emerald-500' :
                            avatarState === 'error' ? 'bg-rose-500' :
                            isConnected ? 'bg-primary' : 'bg-muted-foreground'
                        }`}
                    />
                    <span className={`text-xs font-semibold uppercase tracking-wider ${getStatusColor()}`}>
                        {getStatusText()}
                    </span>
                </div>

                {/* Interim Transcript */}
                {interimTranscript && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="px-3 py-1.5 rounded-full bg-blue-500/20 border border-blue-500/30"
                    >
                        <span className="text-xs text-blue-400 italic">
                            {interimTranscript.substring(0, 30)}...
                        </span>
                    </motion.div>
                )}
            </motion.div>

            {/* Main Avatar Orb */}
            <div className="relative w-72 h-72 mb-8">
                {/* Rings */}
                {isConnected && [1, 2, 3, 4].map((ring) => (
                    <motion.div
                        key={ring}
                        className={`absolute inset-0 border-2 rounded-full ${
                            avatarState === 'listening' ? 'border-blue-500/30' :
                            avatarState === 'speaking' ? 'border-emerald-500/30' :
                            avatarState === 'thinking' ? 'border-amber-500/30' :
                            avatarState === 'connecting' ? 'border-cyan-500/30' :
                            'border-primary/20'
                        }`}
                        animate={{ scale: [1, 2], opacity: [0.5, 0] }}
                        transition={{ repeat: Infinity, duration: 2, delay: ring * 0.4, ease: "easeOut" }}
                    />
                ))}

                {/* Rotating Rings */}
                {isConnected && (
                    <>
                        <motion.div
                            className="absolute inset-[-10px] border-2 border-transparent border-t-primary/40 border-r-primary/20 rounded-full"
                            animate={{ rotate: 360 }}
                            transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
                        />
                        <motion.div
                            className="absolute inset-[-20px] border border-transparent border-b-blue-500/30 border-l-blue-500/20 rounded-full"
                            animate={{ rotate: -360 }}
                            transition={{ repeat: Infinity, duration: 5, ease: "linear" }}
                        />
                    </>
                )}
                
                {/* Glow Effects */}
                <motion.div
                    animate={{ scale: [1, 1.2, 1], opacity: [0.15, 0.4, 0.15] }}
                    transition={{ repeat: Infinity, duration: 2.5, ease: "easeInOut" }}
                    className={`absolute inset-[-20px] bg-gradient-to-br ${getAvatarColor()} rounded-full blur-2xl`}
                />
                <motion.div
                    animate={{ scale: [1.15, 1, 1.15], opacity: [0.1, 0.35, 0.1] }}
                    transition={{ repeat: Infinity, duration: 3.5, ease: "easeInOut", delay: 0.5 }}
                    className={`absolute inset-[-40px] bg-gradient-to-br ${getAvatarColor()} rounded-full blur-3xl`}
                />

                {/* Main Orb */}
                <motion.div
                    animate={{ scale: avatarState === 'speaking' ? [1, 1.02, 1] : 1 }}
                    transition={{ repeat: Infinity, duration: 0.5, ease: "easeInOut" }}
                    className={`relative w-full h-full rounded-full bg-gradient-to-br from-black to-gray-900 border-2 ${getBorderColor()} flex items-center justify-center overflow-hidden shadow-2xl`}
                >
                    <div className={`absolute inset-0 bg-gradient-to-br ${getAvatarColor()} opacity-50`} />
                    
                    <AnimatePresence mode="wait">
                        {avatarState === 'idle' && !isConnected ? (
                            <motion.div
                                key="bot"
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.8 }}
                                className="relative z-10"
                            >
                                <Bot className="w-24 h-24 text-primary/60" />
                            </motion.div>
                        ) : avatarState === 'error' ? (
                            <motion.div
                                key="error"
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.8 }}
                                className="relative z-10"
                            >
                                <AlertCircle className="w-24 h-24 text-rose-400/60" />
                            </motion.div>
                        ) : avatarState === 'thinking' || avatarState === 'connecting' ? (
                            <motion.div
                                key="thinking"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="relative z-10"
                            >
                                <Loader2 className={`w-20 h-20 animate-spin ${
                                    avatarState === 'connecting' ? 'text-cyan-400' : 'text-amber-400'
                                }`} />
                            </motion.div>
                        ) : (
                            <motion.div 
                                key="waveform"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="flex items-center gap-1 h-32 relative z-10"
                            >
                                {audioLevel.map((height, i) => (
                                    <motion.div
                                        key={i}
                                        animate={{ height }}
                                        transition={{ duration: 0.1 }}
                                        className={`w-2 rounded-full ${
                                            avatarState === 'listening' ? 'bg-blue-400/80' :
                                            avatarState === 'speaking' ? 'bg-emerald-400/80' :
                                            'bg-primary/60'
                                        }`}
                                        style={{ minHeight: '8px' }}
                                    />
                                ))}
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {(avatarState === 'thinking' || avatarState === 'connecting') && (
                        <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                            className={`absolute inset-4 border-2 border-transparent rounded-full ${
                                avatarState === 'connecting' ? 'border-t-cyan-500/50' : 'border-t-amber-500/50'
                            }`}
                        />
                    )}
                </motion.div>
            </div>

            {/* Subtitle */}
            <div className="max-w-2xl text-center space-y-4 mb-8 z-10 px-4">
                <motion.p 
                    key={currentSubtitle.substring(0, 50)}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className="text-lg font-medium leading-relaxed text-white/90 max-h-32 overflow-y-auto"
                >
                    {currentSubtitle.length > 200 ? currentSubtitle.substring(0, 200) + '...' : currentSubtitle}
                </motion.p>
            </div>

            {/* Main Controls - Large and Clear */}
            <div className="flex flex-col items-center gap-4 z-10 relative w-full max-w-md px-6">
                {/* Primary Action Button */}
                {!isConnected ? (
                    <motion.button
                        type="button"
                        onClick={handleConnect}
                        disabled={avatarState === 'thinking' || avatarState === 'connecting'}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        className={`w-full py-4 px-6 rounded-xl flex items-center justify-center gap-3 font-semibold text-lg transition-all ${
                            (avatarState === 'thinking' || avatarState === 'connecting')
                                ? 'opacity-50 cursor-not-allowed bg-white/5 border-2 border-white/10'
                                : 'bg-gradient-to-r from-emerald-500 to-green-500 hover:from-emerald-400 hover:to-green-400 text-white shadow-lg shadow-emerald-500/30 border-2 border-emerald-400/50'
                        }`}
                    >
                        <PlayCircle className="w-6 h-6" />
                        <span>Start Voice Session</span>
                    </motion.button>
                ) : (
                    <motion.button
                    type="button"
                    onClick={handleConnect}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="w-full py-3 px-6 rounded-xl flex items-center justify-center gap-3 font-semibold bg-rose-500/20 hover:bg-rose-500/30 text-rose-400 border-2 border-rose-500/50 transition-all"
                    >
                        <PhoneOff className="w-5 h-5" />
                        <span>End Session</span>
                    </motion.button>
                )}

                {/* Secondary Controls Row */}
                {isConnected && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="flex items-center gap-4 w-full"
                    >
                        {/* Microphone Button - Large and Prominent */}
                        <motion.button
                            type="button"
                            onClick={handleMicToggle}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            className={`flex-1 py-4 px-6 rounded-xl flex flex-col items-center justify-center gap-2 transition-all ${
                                isMuted
                                    ? 'bg-white/10 border-2 border-white/20 text-white hover:bg-white/20'
                                    : 'bg-blue-500/20 border-2 border-blue-500/50 text-blue-400 shadow-lg shadow-blue-500/30 animate-pulse'
                            }`}
                        >
                            {isMuted ? (
                                <>
                                    <MicOff className="w-8 h-8" />
                                    <span className="text-sm font-semibold">Click to Speak</span>
                                </>
                            ) : (
                                <>
                                    <Mic className="w-8 h-8" />
                                    <span className="text-sm font-semibold">Listening...</span>
                                </>
                            )}
                        </motion.button>

                        {/* Speaker Toggle */}
                        <motion.button
                            type="button"
                            onClick={() => { setIsSpeakerOn(!isSpeakerOn); if (isSpeakerOn) stopSpeaking(); }}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            className={`w-16 h-16 rounded-xl flex items-center justify-center border-2 transition-all ${
                                isSpeakerOn 
                                    ? 'bg-white/10 border-white/20 text-white hover:bg-white/20' 
                                    : 'bg-rose-500/20 border-rose-500/50 text-rose-400'
                            }`}
                            title={isSpeakerOn ? "Turn off voice responses" : "Turn on voice responses"}
                        >
                            {isSpeakerOn ? <Volume2 className="w-6 h-6" /> : <VolumeX className="w-6 h-6" />}
                        </motion.button>

                        {/* Settings */}
                        <motion.button
                            type="button"
                            onClick={() => setShowSettings(!showSettings)}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            className={`w-16 h-16 rounded-xl flex items-center justify-center border-2 transition-all ${
                                showSettings 
                                    ? 'border-primary/50 bg-primary/10 text-primary' 
                                    : 'border-white/10 bg-white/5 text-muted-foreground hover:text-white hover:bg-white/10'
                            }`}
                            title="Voice settings"
                        >
                            <Settings className="w-6 h-6" />
                        </motion.button>
                    </motion.div>
                )}

                {/* Quick Instructions */}
                {!isConnected && !hasSeenHelp && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 1 }}
                        className="mt-4 p-4 rounded-lg bg-white/5 border border-white/10 w-full"
                    >
                        <div className="flex items-start gap-3">
                            <Info className="w-5 h-5 text-primary mt-0.5 flex-shrink-0" />
                            <div className="flex-1">
                                <p className="text-sm text-white/80 mb-2">
                                    <strong>Quick Start:</strong> Click "Start Voice Session" above, then allow microphone access when prompted.
                                </p>
                                <button
                                    onClick={() => setShowHelp(true)}
                                    className="text-xs text-primary hover:text-primary/80 underline"
                                >
                                    Need more help? Click here
                                </button>
                            </div>
                        </div>
                    </motion.div>
                )}
            </div>

            {/* Settings Panel */}
            <AnimatePresence>
                {showSettings && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 20 }}
                        className="absolute bottom-32 bg-black/90 backdrop-blur-xl border border-white/20 rounded-xl p-5 w-80 z-50"
                    >
                        <h3 className="text-base font-bold text-white mb-4 flex items-center gap-2">
                            <Settings className="w-4 h-4" />
                            Voice Settings
                        </h3>
                        
                        {/* Voice Mode Info */}
                        <div className="mb-4 p-3 rounded-lg bg-white/5 border border-white/10">
                            <div className="flex items-center gap-2 mb-1">
                                {voiceMode === 'livekit' ? (
                                    <Radio className="w-4 h-4 text-purple-400" />
                                ) : (
                                    <Zap className="w-4 h-4 text-blue-400" />
                                )}
                                <span className="text-xs font-semibold text-white">
                                    {voiceMode === 'livekit' ? 'LiveKit Mode' : 'Browser Mode'}
                                </span>
                            </div>
                            <p className="text-[10px] text-muted-foreground">
                                {voiceMode === 'livekit' 
                                    ? 'Connected to real-time voice server'
                                    : 'Using browser speech APIs'
                                }
                            </p>
                        </div>
                        
                        <div className="mb-4">
                            <label className="text-xs text-muted-foreground mb-2 block">Voice</label>
                            <select
                                value={selectedVoice?.name || ''}
                                onChange={(e) => {
                                    const voice = voices.find(v => v.name === e.target.value);
                                    if (voice) setSelectedVoice(voice);
                                }}
                                className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary/50"
                            >
                                {voices.map((voice) => (
                                    <option key={voice.name} value={voice.name}>
                                        {voice.name} ({voice.lang})
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div className="mb-4">
                            <label className="text-xs text-muted-foreground mb-2 block">Speed: {speechRate.toFixed(1)}x</label>
                            <input
                                type="range"
                                min="0.5"
                                max="2"
                                step="0.1"
                                value={speechRate}
                                onChange={(e) => setSpeechRate(parseFloat(e.target.value))}
                                className="w-full accent-primary"
                            />
                        </div>

                        <div className="mb-4">
                            <label className="text-xs text-muted-foreground mb-2 block">Pitch: {speechPitch.toFixed(1)}</label>
                            <input
                                type="range"
                                min="0.5"
                                max="2"
                                step="0.1"
                                value={speechPitch}
                                onChange={(e) => setSpeechPitch(parseFloat(e.target.value))}
                                className="w-full accent-primary"
                            />
                        </div>

                        <button
                            onClick={() => speakText("This is a test of the voice settings.")}
                            className="w-full px-4 py-2 bg-primary/10 border border-primary/30 text-primary rounded-lg text-sm hover:bg-primary/20 transition-colors font-semibold"
                        >
                            Test Voice
                        </button>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Bottom Status */}
            <motion.div 
                className="absolute bottom-4 flex items-center gap-3 text-xs"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
            >
                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${
                    isConnected 
                        ? 'bg-emerald-500/10 border border-emerald-500/30' 
                        : 'bg-white/5 border border-white/10'
                }`}>
                    <motion.div 
                        className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-500' : 'bg-muted-foreground/50'}`}
                        animate={isConnected ? {
                            scale: [1, 1.3, 1],
                            boxShadow: ['0 0 5px rgba(16,185,129,0.5)', '0 0 15px rgba(16,185,129,0.8)', '0 0 5px rgba(16,185,129,0.5)']
                        } : {}}
                        transition={{ duration: 1.5, repeat: Infinity }}
                    />
                    <span className={`text-[10px] uppercase tracking-wider font-semibold ${isConnected ? 'text-emerald-400' : 'text-muted-foreground'}`}>
                        {isConnected ? 'Connected' : 'Not Connected'}
                    </span>
                </div>
            </motion.div>
        </div>
    );
}
