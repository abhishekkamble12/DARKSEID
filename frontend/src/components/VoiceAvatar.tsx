"use client";

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from "@/components/ui/button";
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
    Waves
} from "lucide-react";
import { useAgentStore } from "@/store/useAgentStore";

// Avatar states
type AvatarState = 'idle' | 'listening' | 'thinking' | 'speaking';

export function VoiceAvatar() {
    const [isMuted, setIsMuted] = useState(true);
    const [isSpeakerOn, setIsSpeakerOn] = useState(true);
    const [isConnected, setIsConnected] = useState(false);
    const [avatarState, setAvatarState] = useState<AvatarState>('idle');
    const [currentSubtitle, setCurrentSubtitle] = useState("Click 'Connect' to start your voice session with Darksied AI Tutor");
    const [audioLevel, setAudioLevel] = useState<number[]>(Array(12).fill(20));
    const { addLog } = useAgentStore();
    
    // Simulate audio visualization
    useEffect(() => {
        if (avatarState === 'speaking' || avatarState === 'listening') {
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
        } else {
            setAudioLevel(Array(12).fill(20));
        }
    }, [avatarState]);

    const handleConnect = () => {
        if (isConnected) {
            // Disconnect
            setIsConnected(false);
            setAvatarState('idle');
            setIsMuted(true);
            setCurrentSubtitle("Session ended. Click 'Connect' to start a new session.");
            addLog({
                agent: 'Supervisor',
                message: 'Voice session disconnected',
                status: 'completed'
            });
        } else {
            // Connect
            setIsConnected(true);
            setAvatarState('thinking');
            setCurrentSubtitle("Connecting to Darksied Neural Voice Link...");
            
            addLog({
                agent: 'Supervisor',
                message: 'Initiating voice session with AI Tutor',
                status: 'active'
            });

            // Simulate connection and greeting
            setTimeout(() => {
                setAvatarState('speaking');
                setCurrentSubtitle("Hello! I'm Darksied, your AI learning companion. I use the Socratic method to help you truly understand concepts. What would you like to explore today?");
                
                addLog({
                    agent: 'Supervisor',
                    message: 'Voice session connected - AI Tutor ready',
                    status: 'completed'
                });

                // After speaking, go to listening
                setTimeout(() => {
                    setAvatarState('idle');
                    setCurrentSubtitle("I'm ready to listen. Unmute your microphone to speak, or type in the chat.");
                }, 4000);
            }, 2000);
        }
    };

    const handleMicToggle = () => {
        if (!isConnected) return;
        
        const newMutedState = !isMuted;
        setIsMuted(newMutedState);
        
        if (!newMutedState) {
            // Started listening
            setAvatarState('listening');
            setCurrentSubtitle("I'm listening... speak your question or concept you'd like to understand.");
            
            // Simulate receiving speech and responding
            setTimeout(() => {
                setAvatarState('thinking');
                setCurrentSubtitle("Analyzing your question and formulating a Socratic response...");
                
                setTimeout(() => {
                    setAvatarState('speaking');
                    setCurrentSubtitle("That's an interesting question! Let me ask you this: What do you think happens when we apply this concept to a real-world scenario? Consider how it might affect the outcome...");
                    
                    setTimeout(() => {
                        setAvatarState('idle');
                        setCurrentSubtitle("What are your thoughts on that? Feel free to continue our discussion.");
                    }, 5000);
                }, 2000);
            }, 5000);
        } else {
            setAvatarState('idle');
            setCurrentSubtitle("Microphone muted. Unmute to continue speaking.");
        }
    };

    const getAvatarColor = () => {
        switch (avatarState) {
            case 'listening': return 'from-blue-500/30 to-cyan-500/30';
            case 'thinking': return 'from-amber-500/30 to-orange-500/30';
            case 'speaking': return 'from-emerald-500/30 to-green-500/30';
            default: return 'from-primary/20 to-primary/10';
        }
    };

    const getBorderColor = () => {
        switch (avatarState) {
            case 'listening': return 'border-blue-500/50';
            case 'thinking': return 'border-amber-500/50';
            case 'speaking': return 'border-emerald-500/50';
            default: return 'border-primary/50';
        }
    };

    const getStatusText = () => {
        switch (avatarState) {
            case 'listening': return 'Listening...';
            case 'thinking': return 'Processing...';
            case 'speaking': return 'Speaking...';
            default: return isConnected ? 'Ready' : 'Disconnected';
        }
    };

    const getStatusColor = () => {
        switch (avatarState) {
            case 'listening': return 'text-blue-400';
            case 'thinking': return 'text-amber-400';
            case 'speaking': return 'text-emerald-400';
            default: return isConnected ? 'text-primary' : 'text-muted-foreground';
        }
    };

    return (
        <div className="h-full flex flex-col items-center justify-center p-6 bg-black relative overflow-hidden">
            {/* Animated Background Gradients */}
            <motion.div 
                animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.1, 0.2, 0.1],
                }}
                transition={{ repeat: Infinity, duration: 8, ease: "easeInOut" }}
                className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-radial ${getAvatarColor()} rounded-full blur-[150px] pointer-events-none`}
            />

            {/* Status Badge */}
            <motion.div 
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="absolute top-8 flex items-center gap-3"
            >
                <div className={`flex items-center gap-2 px-4 py-2 rounded-full bg-black/50 border border-white/10 backdrop-blur-sm`}>
                    <motion.div 
                        animate={{ 
                            scale: avatarState !== 'idle' ? [1, 1.2, 1] : 1,
                            opacity: isConnected ? 1 : 0.5
                        }}
                        transition={{ repeat: avatarState !== 'idle' ? Infinity : 0, duration: 1 }}
                        className={`w-2 h-2 rounded-full ${
                            avatarState === 'listening' ? 'bg-blue-500' :
                            avatarState === 'thinking' ? 'bg-amber-500' :
                            avatarState === 'speaking' ? 'bg-emerald-500' :
                            isConnected ? 'bg-primary' : 'bg-muted-foreground'
                        }`}
                    />
                    <span className={`text-xs font-semibold uppercase tracking-wider ${getStatusColor()}`}>
                        {getStatusText()}
                    </span>
                </div>
            </motion.div>

            {/* Main Avatar Orb */}
            <div className="relative w-72 h-72 mb-12">
                {/* Outer glow rings */}
                <motion.div
                    animate={{
                        scale: [1, 1.15, 1],
                        opacity: [0.2, 0.4, 0.2],
                    }}
                    transition={{ repeat: Infinity, duration: 3, ease: "easeInOut" }}
                    className={`absolute inset-[-20px] bg-gradient-to-br ${getAvatarColor()} rounded-full blur-2xl`}
                />
                <motion.div
                    animate={{
                        scale: [1.1, 1, 1.1],
                        opacity: [0.1, 0.3, 0.1],
                    }}
                    transition={{ repeat: Infinity, duration: 4, ease: "easeInOut", delay: 0.5 }}
                    className={`absolute inset-[-40px] bg-gradient-to-br ${getAvatarColor()} rounded-full blur-3xl`}
                />

                {/* Main orb container */}
                <motion.div
                    animate={{
                        scale: avatarState === 'speaking' ? [1, 1.02, 1] : 1,
                    }}
                    transition={{ repeat: Infinity, duration: 0.5, ease: "easeInOut" }}
                    className={`relative w-full h-full rounded-full bg-gradient-to-br from-black to-gray-900 border-2 ${getBorderColor()} flex items-center justify-center overflow-hidden shadow-2xl`}
                >
                    {/* Inner gradient overlay */}
                    <div className={`absolute inset-0 bg-gradient-to-br ${getAvatarColor()} opacity-50`} />
                    
                    {/* Avatar Icon or Waveform */}
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
                                            avatarState === 'thinking' ? 'bg-amber-400/80' :
                                            avatarState === 'speaking' ? 'bg-emerald-400/80' :
                                            'bg-primary/60'
                                        }`}
                                        style={{ minHeight: '8px' }}
                                    />
                                ))}
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* Thinking spinner overlay */}
                    {avatarState === 'thinking' && (
                        <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                            className="absolute inset-4 border-2 border-transparent border-t-amber-500/50 rounded-full"
                        />
                    )}
                </motion.div>
            </div>

            {/* Subtitle Area */}
            <div className="max-w-2xl text-center space-y-4 mb-16 z-10 px-4">
                <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-[10px] font-bold uppercase tracking-wider mb-3"
                >
                    {avatarState === 'speaking' ? (
                        <>
                            <Waves className="w-3 h-3 text-emerald-400" />
                            <span className="text-emerald-400">Darksied Speaking</span>
                        </>
                    ) : avatarState === 'listening' ? (
                        <>
                            <Mic className="w-3 h-3 text-blue-400" />
                            <span className="text-blue-400">Listening to You</span>
                        </>
                    ) : avatarState === 'thinking' ? (
                        <>
                            <Sparkles className="w-3 h-3 text-amber-400" />
                            <span className="text-amber-400">Processing</span>
                        </>
                    ) : (
                        <>
                            <Sparkles className="w-3 h-3 text-primary" />
                            <span className="text-primary">AI Voice Tutor</span>
                        </>
                    )}
                </motion.div>
                
                <motion.p 
                    key={currentSubtitle}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className="text-xl font-medium leading-relaxed text-white/90"
                >
                    "{currentSubtitle}"
                </motion.p>
            </div>

            {/* Control Buttons */}
            <div className="flex items-center gap-4 z-10">
                {/* Settings */}
                <Button
                    variant="outline"
                    size="icon"
                    className="w-12 h-12 rounded-full border-white/10 bg-white/5 text-muted-foreground hover:text-white hover:bg-white/10"
                >
                    <Settings className="w-5 h-5" />
                </Button>

                {/* Speaker Toggle */}
                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setIsSpeakerOn(!isSpeakerOn)}
                    className={`w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all ${
                        isSpeakerOn 
                            ? 'bg-white/5 border-white/20 text-white' 
                            : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
                    }`}
                >
                    {isSpeakerOn ? <Volume2 className="w-5 h-5" /> : <VolumeX className="w-5 h-5" />}
                </motion.button>

                {/* Main Mic Button */}
                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleMicToggle}
                    disabled={!isConnected}
                    className={`w-20 h-20 rounded-full flex items-center justify-center shadow-lg transition-all border-4 ${
                        !isConnected 
                            ? 'bg-white/5 border-white/10 text-muted-foreground cursor-not-allowed'
                            : isMuted
                                ? 'bg-white/10 border-white/20 text-white hover:bg-white/20'
                                : 'bg-blue-500/20 border-blue-500/50 text-blue-400 shadow-blue-500/20'
                    }`}
                >
                    {isMuted ? <MicOff className="w-8 h-8" /> : <Mic className="w-8 h-8" />}
                </motion.button>

                {/* Connect/Disconnect Button */}
                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleConnect}
                    className={`w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all ${
                        isConnected 
                            ? 'bg-rose-500/10 border-rose-500/30 text-rose-400 hover:bg-rose-500/20' 
                            : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20'
                    }`}
                >
                    {isConnected ? <PhoneOff className="w-5 h-5" /> : <Phone className="w-5 h-5" />}
                </motion.button>

                {/* Close */}
                <Button
                    variant="outline"
                    size="icon"
                    className="w-12 h-12 rounded-full border-white/10 bg-white/5 text-muted-foreground hover:text-white hover:bg-white/10"
                >
                    <X className="w-5 h-5" />
                </Button>
            </div>

            {/* Bottom Status */}
            <div className="absolute bottom-8 flex items-center gap-4">
                <div className={`flex items-center gap-2 text-[10px] uppercase tracking-[0.15em] ${isConnected ? 'text-emerald-400' : 'text-muted-foreground'}`}>
                    <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-emerald-500 animate-pulse' : 'bg-muted-foreground'}`} />
                    {isConnected ? 'Voice Channel Active' : 'Voice Channel Inactive'}
                </div>
                <span className="text-white/20">|</span>
                <span className="text-[10px] text-muted-foreground uppercase tracking-[0.15em]">
                    Powered by Neural Voice Engine
                </span>
            </div>
        </div>
    );
}
