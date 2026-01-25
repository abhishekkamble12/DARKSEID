"use client";

import React, { useEffect, useRef, useId, useState } from 'react';
import mermaid from 'mermaid';

interface MindmapRendererProps {
    chart: string;
}

function sanitizeMermaidMindmap(mermaidCode: string): string {
    /**
     * Sanitize Mermaid mindmap code to fix parsing errors.
     * Replaces nested parentheses in node labels with brackets.
     */
    if (!mermaidCode) return mermaidCode;
    
    const lines = mermaidCode.split('\n');
    const sanitizedLines: string[] = [];
    
    for (const line of lines) {
        const stripped = line.trim();
        
        // Skip empty lines and diagram declaration
        if (!stripped || stripped === 'mindmap') {
            sanitizedLines.push(line);
            continue;
        }
        
        // Process node lines (lines with indentation that contain node definitions)
        if (line.startsWith(' ') || line.startsWith('\t')) {
            if (line.includes('(')) {
                // Replace nested parentheses with brackets
                let result = '';
                let parenDepth = 0;
                
                for (const char of line) {
                    if (char === '(') {
                        parenDepth++;
                        if (parenDepth > 1) {
                            // Nested parenthesis - replace with bracket
                            result += '[';
                        } else {
                            // Structural parenthesis - keep it
                            result += char;
                        }
                    } else if (char === ')') {
                        if (parenDepth > 1) {
                            // Closing nested parenthesis - replace with bracket
                            result += ']';
                            parenDepth--;
                        } else {
                            // Closing structural parenthesis - keep it
                            result += char;
                            parenDepth--;
                        }
                    } else {
                        result += char;
                    }
                }
                
                sanitizedLines.push(result);
                continue;
            }
        }
        
        sanitizedLines.push(line);
    }
    
    return sanitizedLines.join('\n');
}

export function MindmapRenderer({ chart }: MindmapRendererProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const uniqueId = useId().replace(/:/g, '-');
    const [isClient, setIsClient] = useState(false);

    useEffect(() => {
        setIsClient(true);
    }, []);

    useEffect(() => {
        if (!isClient || !containerRef.current || !chart) return;

        const renderChart = async () => {
            try {
                // Sanitize the mermaid code before rendering
                const sanitizedChart = sanitizeMermaidMindmap(chart);
                
                // Re-initialize mermaid on each render
                mermaid.initialize({
                    startOnLoad: false,
                    theme: 'dark',
                    securityLevel: 'loose',
                    fontFamily: 'ui-sans-serif, system-ui, sans-serif',
                    themeVariables: {
                        primaryColor: '#22c55e',
                        primaryTextColor: '#fff',
                        primaryBorderColor: '#22c55e',
                        lineColor: '#22c55e',
                        secondaryColor: '#1e293b',
                        tertiaryColor: '#0f172a'
                    }
                });

                // Generate unique ID for this render
                const id = `mermaid-${uniqueId}-${Date.now()}`;
                
                const { svg } = await mermaid.render(id, sanitizedChart);
                if (containerRef.current) {
                    containerRef.current.innerHTML = svg;
                }
            } catch (error) {
                console.error('Mermaid render error:', error);
                if (containerRef.current) {
                    containerRef.current.innerHTML = `<div class="text-rose-400 text-sm p-4">Failed to render mindmap: ${error instanceof Error ? error.message : 'Unknown error'}</div>`;
                }
            }
        };

        renderChart();
    }, [chart, uniqueId, isClient]);

    if (!isClient) {
        return (
            <div className="w-full bg-black/20 rounded-xl p-6 border border-white/5 overflow-auto flex justify-center items-center min-h-[300px]">
                <div className="text-muted-foreground text-sm">Loading mindmap...</div>
            </div>
        );
    }

    return (
        <div className="w-full h-full bg-black/20 rounded-xl p-6 border border-white/5 overflow-auto flex justify-center items-center min-h-[300px]">
            <div ref={containerRef} className="mermaid-container [&_svg]:max-w-full" />
        </div>
    );
}
