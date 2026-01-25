"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { 
    BrainCircuit, 
    Mic, 
    FileUp, 
    Map, 
    Sparkles, 
    Zap, 
    Shield, 
    ArrowRight,
    CheckCircle2,
    Play,
    Github,
    Twitter,
    MessageSquare,
    BookOpen,
    Target,
    Lightbulb
} from 'lucide-react';

const features = [
    {
        icon: MessageSquare,
        title: "AI Diagnostic Chat",
        description: "Engage in Socratic dialogue that identifies your knowledge gaps and builds understanding from first principles.",
        color: "from-emerald-500 to-green-600"
    },
    {
        icon: Mic,
        title: "Voice Tutor",
        description: "Natural voice conversations with an AI tutor that adapts to your learning style in real-time.",
        color: "from-blue-500 to-cyan-600"
    },
    {
        icon: Map,
        title: "Learning Architect",
        description: "Auto-generated mindmaps and quiz cards that visualize concepts and test your understanding.",
        color: "from-purple-500 to-pink-600"
    },
    {
        icon: FileUp,
        title: "Knowledge Upload",
        description: "Upload PDFs, docs, and notes. Our RAG system indexes everything for contextual learning.",
        color: "from-amber-500 to-orange-600"
    }
];

const stats = [
    { value: "10x", label: "Faster Learning" },
    { value: "95%", label: "Retention Rate" },
    { value: "24/7", label: "AI Availability" },
    { value: "∞", label: "Patience" }
];

const testimonials = [
    {
        quote: "Darksied identified gaps in my understanding that I didn't even know existed. Game changer.",
        author: "Sarah Chen",
        role: "PhD Student, MIT"
    },
    {
        quote: "The voice tutor feels like having a patient professor available whenever I need help.",
        author: "Marcus Johnson",
        role: "Software Engineer"
    },
    {
        quote: "Finally, an AI that doesn't just give answers but teaches you how to think.",
        author: "Dr. Elena Rodriguez",
        role: "Professor of Physics"
    }
];

export default function LandingPage() {
    const [hoveredFeature, setHoveredFeature] = useState<number | null>(null);

    return (
        <div className="min-h-screen bg-black text-white overflow-x-hidden">
            {/* Animated Background */}
            <div className="fixed inset-0 pointer-events-none">
                <div className="absolute top-1/4 left-1/4 w-[600px] h-[600px] bg-emerald-500/10 rounded-full blur-[150px]" />
                <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-blue-500/10 rounded-full blur-[120px]" />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-purple-500/5 rounded-full blur-[200px]" />
            </div>

            {/* Navigation */}
            <nav className="relative z-50 border-b border-white/10 backdrop-blur-xl">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/50 flex items-center justify-center">
                                <BrainCircuit className="w-6 h-6 text-emerald-400" />
                            </div>
                            <span className="text-xl font-bold tracking-tight">
                                Darksied<span className="text-emerald-400">.</span>
                            </span>
                        </div>
                        <div className="hidden md:flex items-center gap-8">
                            <a href="#features" className="text-sm text-white/70 hover:text-white transition-colors">Features</a>
                            <a href="#how-it-works" className="text-sm text-white/70 hover:text-white transition-colors">How it Works</a>
                            <a href="#testimonials" className="text-sm text-white/70 hover:text-white transition-colors">Testimonials</a>
                        </div>
                        <Link 
                            href="/"
                            className="px-5 py-2.5 rounded-full bg-emerald-500 hover:bg-emerald-400 text-black font-semibold text-sm transition-all hover:scale-105"
                        >
                            Launch App →
                        </Link>
                    </div>
                </div>
            </nav>

            {/* Hero Section */}
            <section className="relative z-10 pt-20 pb-32 px-6">
                <div className="max-w-7xl mx-auto">
                    <motion.div 
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8 }}
                        className="text-center max-w-4xl mx-auto"
                    >
                        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 mb-8">
                            <Sparkles className="w-4 h-4 text-emerald-400" />
                            <span className="text-sm text-white/80">AI-Powered Learning Diagnostics</span>
                        </div>
                        
                        <h1 className="text-5xl md:text-7xl font-bold leading-tight mb-6">
                            Learn Smarter with
                            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-400">
                                Neural Diagnostics
                            </span>
                        </h1>
                        
                        <p className="text-xl text-white/60 mb-10 max-w-2xl mx-auto leading-relaxed">
                            Darksied uses multi-agent AI to diagnose your learning gaps, build personalized knowledge maps, and guide you through mastery with Socratic dialogue.
                        </p>
                        
                        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                            <Link 
                                href="/"
                                className="group flex items-center gap-2 px-8 py-4 rounded-full bg-gradient-to-r from-emerald-500 to-cyan-500 text-black font-bold text-lg transition-all hover:scale-105 hover:shadow-[0_0_40px_rgba(16,185,129,0.4)]"
                            >
                                Start Learning Free
                                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                            </Link>
                            <button className="flex items-center gap-2 px-8 py-4 rounded-full border border-white/20 hover:bg-white/5 font-semibold text-lg transition-all">
                                <Play className="w-5 h-5" />
                                Watch Demo
                            </button>
                        </div>
                    </motion.div>

                    {/* Hero Image/Preview */}
                    <motion.div 
                        initial={{ opacity: 0, y: 40 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, delay: 0.3 }}
                        className="mt-20 relative"
                    >
                        <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent z-10 pointer-events-none" />
                        <div className="rounded-2xl overflow-hidden border border-white/10 shadow-2xl shadow-emerald-500/10">
                            <div className="bg-gradient-to-br from-gray-900 to-black p-2">
                                <div className="flex items-center gap-2 mb-2">
                                    <div className="w-3 h-3 rounded-full bg-red-500" />
                                    <div className="w-3 h-3 rounded-full bg-yellow-500" />
                                    <div className="w-3 h-3 rounded-full bg-green-500" />
                                </div>
                                <div className="aspect-video bg-gradient-to-br from-gray-800/50 to-gray-900/50 rounded-lg flex items-center justify-center">
                                    <div className="text-center">
                                        <BrainCircuit className="w-20 h-20 text-emerald-500/50 mx-auto mb-4" />
                                        <p className="text-white/40 text-lg">Interactive AI Learning Dashboard</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* Stats Section */}
            <section className="relative z-10 py-16 border-y border-white/10 bg-white/[0.02]">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                        {stats.map((stat, i) => (
                            <motion.div 
                                key={i}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.1 }}
                                viewport={{ once: true }}
                                className="text-center"
                            >
                                <div className="text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400 mb-2">
                                    {stat.value}
                                </div>
                                <div className="text-white/60 text-sm uppercase tracking-wider">
                                    {stat.label}
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section id="features" className="relative z-10 py-32 px-6">
                <div className="max-w-7xl mx-auto">
                    <motion.div 
                        initial={{ opacity: 0 }}
                        whileInView={{ opacity: 1 }}
                        viewport={{ once: true }}
                        className="text-center mb-16"
                    >
                        <h2 className="text-4xl md:text-5xl font-bold mb-4">
                            Everything You Need to
                            <span className="text-emerald-400"> Master Anything</span>
                        </h2>
                        <p className="text-white/60 text-lg max-w-2xl mx-auto">
                            Four powerful AI agents working together to accelerate your learning journey.
                        </p>
                    </motion.div>

                    <div className="grid md:grid-cols-2 gap-6">
                        {features.map((feature, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.1 }}
                                viewport={{ once: true }}
                                onMouseEnter={() => setHoveredFeature(i)}
                                onMouseLeave={() => setHoveredFeature(null)}
                                className={`group relative p-8 rounded-2xl border border-white/10 bg-white/[0.02] hover:bg-white/[0.05] transition-all duration-300 cursor-pointer overflow-hidden ${hoveredFeature === i ? 'scale-[1.02]' : ''}`}
                            >
                                <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-5 transition-opacity`} />
                                <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-6`}>
                                    <feature.icon className="w-7 h-7 text-white" />
                                </div>
                                <h3 className="text-2xl font-bold mb-3">{feature.title}</h3>
                                <p className="text-white/60 leading-relaxed">{feature.description}</p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* How It Works */}
            <section id="how-it-works" className="relative z-10 py-32 px-6 bg-gradient-to-b from-transparent via-emerald-500/5 to-transparent">
                <div className="max-w-7xl mx-auto">
                    <motion.div 
                        initial={{ opacity: 0 }}
                        whileInView={{ opacity: 1 }}
                        viewport={{ once: true }}
                        className="text-center mb-16"
                    >
                        <h2 className="text-4xl md:text-5xl font-bold mb-4">
                            How <span className="text-emerald-400">Darksied</span> Works
                        </h2>
                        <p className="text-white/60 text-lg max-w-2xl mx-auto">
                            A simple three-step process to transform your learning.
                        </p>
                    </motion.div>

                    <div className="grid md:grid-cols-3 gap-8">
                        {[
                            { step: "01", icon: Target, title: "Diagnose", desc: "AI analyzes your knowledge and identifies specific gaps and misconceptions." },
                            { step: "02", icon: Lightbulb, title: "Learn", desc: "Engage in Socratic dialogue that builds understanding from first principles." },
                            { step: "03", icon: BookOpen, title: "Master", desc: "Test your knowledge with generated quizzes and visualize concepts with mindmaps." }
                        ].map((item, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.2 }}
                                viewport={{ once: true }}
                                className="relative text-center"
                            >
                                <div className="text-8xl font-bold text-white/[0.03] absolute top-0 left-1/2 -translate-x-1/2 -translate-y-4">
                                    {item.step}
                                </div>
                                <div className="relative">
                                    <div className="w-16 h-16 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center mx-auto mb-6">
                                        <item.icon className="w-8 h-8 text-emerald-400" />
                                    </div>
                                    <h3 className="text-2xl font-bold mb-3">{item.title}</h3>
                                    <p className="text-white/60">{item.desc}</p>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Testimonials */}
            <section id="testimonials" className="relative z-10 py-32 px-6">
                <div className="max-w-7xl mx-auto">
                    <motion.div 
                        initial={{ opacity: 0 }}
                        whileInView={{ opacity: 1 }}
                        viewport={{ once: true }}
                        className="text-center mb-16"
                    >
                        <h2 className="text-4xl md:text-5xl font-bold mb-4">
                            Loved by <span className="text-emerald-400">Learners</span>
                        </h2>
                    </motion.div>

                    <div className="grid md:grid-cols-3 gap-6">
                        {testimonials.map((t, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.1 }}
                                viewport={{ once: true }}
                                className="p-8 rounded-2xl border border-white/10 bg-white/[0.02]"
                            >
                                <p className="text-white/80 text-lg mb-6 leading-relaxed">"{t.quote}"</p>
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-emerald-500 to-cyan-500" />
                                    <div>
                                        <div className="font-semibold">{t.author}</div>
                                        <div className="text-white/50 text-sm">{t.role}</div>
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="relative z-10 py-32 px-6">
                <div className="max-w-4xl mx-auto text-center">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        viewport={{ once: true }}
                        className="p-12 rounded-3xl bg-gradient-to-br from-emerald-500/10 to-cyan-500/10 border border-emerald-500/20"
                    >
                        <h2 className="text-4xl md:text-5xl font-bold mb-4">
                            Ready to Learn Smarter?
                        </h2>
                        <p className="text-white/60 text-lg mb-8 max-w-xl mx-auto">
                            Join thousands of learners who are mastering new skills faster with AI-powered diagnostics.
                        </p>
                        <Link 
                            href="/"
                            className="inline-flex items-center gap-2 px-8 py-4 rounded-full bg-gradient-to-r from-emerald-500 to-cyan-500 text-black font-bold text-lg transition-all hover:scale-105 hover:shadow-[0_0_40px_rgba(16,185,129,0.4)]"
                        >
                            Get Started Free
                            <ArrowRight className="w-5 h-5" />
                        </Link>
                    </motion.div>
                </div>
            </section>

            {/* Footer */}
            <footer className="relative z-10 border-t border-white/10 py-12 px-6">
                <div className="max-w-7xl mx-auto">
                    <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-lg bg-emerald-500/20 border border-emerald-500/50 flex items-center justify-center">
                                <BrainCircuit className="w-5 h-5 text-emerald-400" />
                            </div>
                            <span className="font-bold">Darksied</span>
                        </div>
                        <div className="flex items-center gap-6 text-white/50 text-sm">
                            <a href="#" className="hover:text-white transition-colors">Privacy</a>
                            <a href="#" className="hover:text-white transition-colors">Terms</a>
                            <a href="#" className="hover:text-white transition-colors">Documentation</a>
                        </div>
                        <div className="flex items-center gap-4">
                            <a href="#" className="w-10 h-10 rounded-full bg-white/5 hover:bg-white/10 flex items-center justify-center transition-colors">
                                <Github className="w-5 h-5" />
                            </a>
                            <a href="#" className="w-10 h-10 rounded-full bg-white/5 hover:bg-white/10 flex items-center justify-center transition-colors">
                                <Twitter className="w-5 h-5" />
                            </a>
                        </div>
                    </div>
                    <div className="text-center text-white/30 text-sm mt-8">
                        © 2026 Darksied. Built with Neural Intelligence.
                    </div>
                </div>
            </footer>
        </div>
    );
}
