import React, { useState } from 'react';
import { MarkdownView } from '../ui/MarkdownView';

interface ThinkingBlockProps {
  thinking: string;
  durationSeconds?: number;
  isStreaming?: boolean;
}

export const ThinkingBlock: React.FC<ThinkingBlockProps> = ({
  thinking,
  durationSeconds,
  isStreaming = false,
}) => {
  const [isExpanded, setIsExpanded] = useState(isStreaming);

  if (!thinking && !isStreaming) return null;

  return (
    <div className="rounded-2xl border border-teal-500/30 bg-teal-950/10 dark:bg-teal-950/20 overflow-hidden shadow-xs my-2 text-xs">
      {/* Accordion Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-3.5 py-2.5 flex items-center justify-between bg-teal-500/5 hover:bg-teal-500/10 text-teal-600 dark:text-teal-400 font-medium select-none transition-colors"
      >
        <div className="flex items-center gap-2">
          {isStreaming ? (
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-teal-500" />
            </span>
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
          )}
          <span className="font-semibold">
            {isStreaming
              ? `Thinking Process... (${durationSeconds ? durationSeconds.toFixed(1) : '0.0'}s)`
              : `Thought Process (${durationSeconds ? durationSeconds.toFixed(1) : '0.0'}s)`}
          </span>
        </div>

        <div className="flex items-center gap-1.5 text-[11px] opacity-80">
          <span>{isExpanded ? 'Hide' : 'Show reasoning'}</span>
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className={`transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </div>
      </button>

      {/* Accordion Content */}
      {isExpanded && (
        <div className="p-3.5 border-t border-teal-500/20 text-muted-foreground bg-background/50 font-mono text-[11.5px] leading-relaxed max-h-96 overflow-y-auto select-text">
          <MarkdownView content={thinking || 'Evaluating tokens and constraints...'} />
        </div>
      )}
    </div>
  );
};
