import React, { useState } from 'react';
import { McpToolCall } from '../../types';

interface McpToolCallBlockProps {
  toolCall: McpToolCall;
}

export const McpToolCallBlock: React.FC<McpToolCallBlockProps> = ({ toolCall }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="rounded-2xl border border-border bg-secondary/80 overflow-hidden my-2.5 text-xs shadow-xs">
      {/* Header */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className="px-3.5 py-2.5 flex items-center justify-between cursor-pointer hover:bg-muted/50 transition-colors select-none"
      >
        <div className="flex items-center gap-2">
          <span className="p-1 rounded-lg bg-teal-500/10 text-teal-500 border border-teal-500/20">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
            </svg>
          </span>
          <span className="font-mono font-bold text-foreground">
            tool: {toolCall.name}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {toolCall.status === 'running' && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-500/15 text-amber-500 border border-amber-500/30">
              <span className="h-1.5 w-1.5 rounded-full bg-amber-500 animate-ping" />
              Running...
            </span>
          )}
          {toolCall.status === 'completed' && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30">
              ✓ Executed
            </span>
          )}
          {toolCall.status === 'error' && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-red-500/15 text-red-500 border border-red-500/30">
              ✕ Error
            </span>
          )}

          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className={`text-muted-foreground transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </div>
      </div>

      {/* Expanded Payload / Output View */}
      {isExpanded && (
        <div className="p-3.5 border-t border-border bg-background space-y-2.5 font-mono text-[11px]">
          <div>
            <span className="text-muted-foreground block text-[10px] uppercase tracking-wider mb-1">
              INPUT ARGUMENTS (JSON-RPC)
            </span>
            <pre className="p-2.5 rounded-xl bg-secondary border border-border text-foreground overflow-x-auto select-text">
              {JSON.stringify(toolCall.arguments, null, 2)}
            </pre>
          </div>

          {toolCall.result && (
            <div>
              <span className="text-emerald-600 dark:text-emerald-400 block text-[10px] uppercase tracking-wider mb-1">
                RETURNED MCP PAYLOAD
              </span>
              <pre className="p-2.5 rounded-xl bg-secondary border border-border text-foreground overflow-x-auto select-text">
                {JSON.stringify(toolCall.result, null, 2)}
              </pre>
            </div>
          )}

          {toolCall.error && (
            <div>
              <span className="text-red-500 block text-[10px] uppercase tracking-wider mb-1">
                ERROR
              </span>
              <div className="p-2.5 rounded-xl bg-red-500/10 border border-red-500/30 text-red-500 select-text">
                {toolCall.error}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
