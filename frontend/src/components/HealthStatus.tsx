"use client";

import React, { useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
    Activity, 
    Database, 
    Server, 
    Wifi, 
    WifiOff,
    CheckCircle2,
    AlertCircle,
    Loader2,
    RefreshCw
} from 'lucide-react';
import { useAgentStore } from '@/store/useAgentStore';
import { api } from '@/lib/api';

interface StatusDotProps {
    status: 'connected' | 'disconnected' | 'checking';
    label: string;
}

function StatusDot({ status, label }: StatusDotProps) {
    return (
        <div className="flex items-center gap-2">
            <div className="relative">
                {status === 'checking' ? (
                    <Loader2 className="w-3 h-3 text-amber-400 animate-spin" />
                ) : (
                    <motion.div 
                        className={`w-2 h-2 rounded-full ${
                            status === 'connected' ? 'bg-emerald-500' : 'bg-rose-500'
                        }`}
                        animate={status === 'connected' ? {
                            scale: [1, 1.2, 1],
                            boxShadow: [
                                '0 0 0 0 rgba(16, 185, 129, 0.4)',
                                '0 0 0 4px rgba(16, 185, 129, 0.1)',
                                '0 0 0 0 rgba(16, 185, 129, 0.4)'
                            ]
                        } : {}}
                        transition={{ duration: 2, repeat: Infinity }}
                    />
                )}
            </div>
            <span className={`text-[10px] uppercase tracking-wider ${
                status === 'connected' ? 'text-emerald-400' :
                status === 'checking' ? 'text-amber-400' :
                'text-rose-400'
            }`}>
                {label}
            </span>
        </div>
    );
}

export function HealthStatus() {
    const { health, setHealth } = useAgentStore();

    const checkHealth = useCallback(async () => {
        setHealth({ 
            backend: 'checking',
            qdrant: 'checking',
            postgres: 'checking'
        });

        try {
            const status = await api.checkHealth();
            
            setHealth({
                isHealthy: status.status === 'healthy',
                backend: 'connected',
                qdrant: status.qdrant,
                postgres: status.postgres,
            });
        } catch {
            setHealth({
                isHealthy: false,
                backend: 'disconnected',
                qdrant: 'disconnected',
                postgres: 'disconnected',
            });
        }
    }, [setHealth]);

    // Check health on mount and periodically
    useEffect(() => {
        checkHealth();
        const interval = setInterval(checkHealth, 30000); // Every 30 seconds
        return () => clearInterval(interval);
    }, [checkHealth]);

    return (
        <div className="flex items-center gap-4">
            {/* Main Status Indicator */}
            <motion.div 
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${
                    health.isHealthy 
                        ? 'bg-emerald-500/10 border-emerald-500/30' 
                        : health.backend === 'checking'
                            ? 'bg-amber-500/10 border-amber-500/30'
                            : 'bg-rose-500/10 border-rose-500/30'
                }`}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
            >
                {health.backend === 'checking' ? (
                    <Loader2 className="w-3.5 h-3.5 text-amber-400 animate-spin" />
                ) : health.isHealthy ? (
                    <Wifi className="w-3.5 h-3.5 text-emerald-400" />
                ) : (
                    <WifiOff className="w-3.5 h-3.5 text-rose-400" />
                )}
                <span className={`text-[10px] font-bold uppercase tracking-wider ${
                    health.isHealthy ? 'text-emerald-400' :
                    health.backend === 'checking' ? 'text-amber-400' :
                    'text-rose-400'
                }`}>
                    {health.backend === 'checking' ? 'Checking...' :
                     health.isHealthy ? 'Neural Link Stable' :
                     'Connection Lost'}
                </span>
            </motion.div>

            {/* Refresh Button */}
            <button
                onClick={checkHealth}
                className="p-1.5 rounded-lg hover:bg-white/5 transition-colors"
                title="Refresh health status"
            >
                <RefreshCw className={`w-3.5 h-3.5 text-muted-foreground ${
                    health.backend === 'checking' ? 'animate-spin' : ''
                }`} />
            </button>
        </div>
    );
}

// Expanded health panel for detailed view
export function HealthPanel() {
    const { health, setHealth } = useAgentStore();

    const checkHealth = useCallback(async () => {
        setHealth({ 
            backend: 'checking',
            qdrant: 'checking',
            postgres: 'checking'
        });

        try {
            const status = await api.checkHealth();
            
            setHealth({
                isHealthy: status.status === 'healthy',
                backend: 'connected',
                qdrant: status.qdrant,
                postgres: status.postgres,
            });
        } catch {
            setHealth({
                isHealthy: false,
                backend: 'disconnected',
                qdrant: 'disconnected',
                postgres: 'disconnected',
            });
        }
    }, [setHealth]);

    return (
        <motion.div 
            className="p-4 rounded-xl bg-black/40 border border-white/10"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
        >
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-primary" />
                    <span className="text-xs font-bold uppercase tracking-wider">System Health</span>
                </div>
                <button
                    onClick={checkHealth}
                    className="p-1 rounded hover:bg-white/5 transition-colors"
                >
                    <RefreshCw className={`w-3 h-3 text-muted-foreground ${
                        health.backend === 'checking' ? 'animate-spin' : ''
                    }`} />
                </button>
            </div>

            <div className="space-y-3">
                {/* Backend API */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Server className="w-3.5 h-3.5 text-muted-foreground" />
                        <span className="text-xs">Backend API</span>
                    </div>
                    <StatusDot status={health.backend} label={health.backend} />
                </div>

                {/* Qdrant Vector DB */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Database className="w-3.5 h-3.5 text-muted-foreground" />
                        <span className="text-xs">Qdrant (RAG)</span>
                    </div>
                    <StatusDot status={health.qdrant} label={health.qdrant} />
                </div>

                {/* PostgreSQL */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Database className="w-3.5 h-3.5 text-muted-foreground" />
                        <span className="text-xs">PostgreSQL</span>
                    </div>
                    <StatusDot status={health.postgres} label={health.postgres} />
                </div>
            </div>

            {/* Last Checked */}
            {health.lastChecked && (
                <div className="mt-4 pt-3 border-t border-white/10">
                    <span className="text-[10px] text-muted-foreground">
                        Last checked: {health.lastChecked.toLocaleTimeString()}
                    </span>
                </div>
            )}

            {/* Warning if unhealthy */}
            <AnimatePresence>
                {!health.isHealthy && health.backend !== 'checking' && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-4 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30"
                    >
                        <div className="flex items-start gap-2">
                            <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                            <div>
                                <p className="text-xs text-rose-400 font-medium">Connection Issues Detected</p>
                                <p className="text-[10px] text-muted-foreground mt-1">
                                    Some services may be unavailable. Check that the backend and Docker services are running.
                                </p>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
}
