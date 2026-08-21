// Streaming Server-Sent Events (SSE) reader for OpenAI-compatible chat completions
// Extended with Thinking (Chain-of-Thought) and Model Context Protocol (MCP) tool-call streaming.

import { McpToolCall } from '../types';

export interface StreamCallbacks {
  onToken: (token: string, fullText: string, tps: number) => void;
  onThinking?: (thinkingChunk: string, fullThinking: string, elapsedSec: number) => void;
  onToolCall?: (toolCalls: McpToolCall[]) => void;
  onComplete: (fullText: string, totalTokens: number, avgTps: number, thinkingText?: string, toolCalls?: McpToolCall[]) => void;
  onError: (error: Error) => void;
}

export async function streamChatCompletion(
  endpoint: string,
  body: Record<string, unknown>,
  signal: AbortSignal,
  callbacks: StreamCallbacks
): Promise<void> {
  let fullText = '';
  let fullThinking = '';
  let isInsideThinkTag = false;
  let tokenCount = 0;
  let startTime = 0;
  const toolCallsMap: Record<number, { id: string; name: string; arguments: string }> = {};

  try {
    const res = await fetch(`${endpoint}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify({ ...body, stream: true }),
      signal,
    });

    if (!res.ok) {
      const errText = await res.text();
      let parsedMessage = errText;
      try {
        const json = JSON.parse(errText);
        parsedMessage = json.error?.message || errText;
      } catch {
        // use raw text
      }
      throw new Error(`API error (${res.status}): ${parsedMessage}`);
    }

    if (!res.body) {
      throw new Error('ReadableStream not supported by browser or empty response');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      if (startTime === 0) {
        startTime = performance.now();
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith(':')) continue;

        if (trimmed === 'data: [DONE]') {
          const totalSec = (performance.now() - startTime) / 1000;
          const avgTps = totalSec > 0 ? tokenCount / totalSec : 0;
          const finalToolCalls = convertToolCallsMap(toolCallsMap);
          callbacks.onComplete(fullText, tokenCount, avgTps, fullThinking.trim() || undefined, finalToolCalls);
          return;
        }

        if (trimmed.startsWith('data: ')) {
          try {
            const data = JSON.parse(trimmed.slice(6));
            const choice = data.choices?.[0];
            const delta = choice?.delta;

            // 1. Check reasoning_content (DeepSeek-R1 / Qwen-Thinking format)
            if (delta?.reasoning_content) {
              tokenCount++;
              fullThinking += delta.reasoning_content;
              const elapsedSec = (performance.now() - startTime) / 1000;
              callbacks.onThinking?.(delta.reasoning_content, fullThinking, elapsedSec);
            }

            // 2. Check tool_calls delta (MCP tool execution)
            if (delta?.tool_calls && Array.isArray(delta.tool_calls)) {
              for (const tc of delta.tool_calls) {
                const idx = tc.index ?? 0;
                if (!toolCallsMap[idx]) {
                  toolCallsMap[idx] = {
                    id: tc.id || `call_${Date.now()}_${idx}`,
                    name: tc.function?.name || '',
                    arguments: tc.function?.arguments || '',
                  };
                } else {
                  if (tc.function?.name) toolCallsMap[idx].name += tc.function.name;
                  if (tc.function?.arguments) toolCallsMap[idx].arguments += tc.function.arguments;
                }
              }
              const currentTools = convertToolCallsMap(toolCallsMap);
              if (currentTools && currentTools.length > 0) {
                callbacks.onToolCall?.(currentTools);
              }
            }

            // 3. Check regular content delta with <think> tag handling
            const content = delta?.content;
            if (content) {
              tokenCount++;
              const elapsedSec = (performance.now() - startTime) / 1000;
              const currentTps = elapsedSec > 0 ? tokenCount / elapsedSec : 0;

              // Parse <think> and </think> tags if model emits thoughts in plain content
              if (content.includes('<think>')) {
                isInsideThinkTag = true;
                const parts = content.split('<think>');
                if (parts[0]) {
                  fullText += parts[0];
                  callbacks.onToken(parts[0], fullText, currentTps);
                }
                if (parts[1]) {
                  fullThinking += parts[1];
                  callbacks.onThinking?.(parts[1], fullThinking, elapsedSec);
                }
              } else if (content.includes('</think>')) {
                isInsideThinkTag = false;
                const parts = content.split('</think>');
                if (parts[0]) {
                  fullThinking += parts[0];
                  callbacks.onThinking?.(parts[0], fullThinking, elapsedSec);
                }
                if (parts[1]) {
                  fullText += parts[1];
                  callbacks.onToken(parts[1], fullText, currentTps);
                }
              } else if (isInsideThinkTag) {
                fullThinking += content;
                callbacks.onThinking?.(content, fullThinking, elapsedSec);
              } else {
                fullText += content;
                callbacks.onToken(content, fullText, currentTps);
              }
            }
          } catch {
            // ignore unparseable chunk
          }
        }
      }
    }

    const totalSec = (performance.now() - startTime) / 1000;
    const avgTps = totalSec > 0 ? tokenCount / totalSec : 0;
    const finalToolCalls = convertToolCallsMap(toolCallsMap);
    callbacks.onComplete(fullText, tokenCount, avgTps, fullThinking.trim() || undefined, finalToolCalls);
  } catch (err: unknown) {
    if (signal.aborted) {
      const totalSec = startTime > 0 ? (performance.now() - startTime) / 1000 : 0;
      const finalToolCalls = convertToolCallsMap(toolCallsMap);
      callbacks.onComplete(fullText, tokenCount, totalSec > 0 ? tokenCount / totalSec : 0, fullThinking.trim() || undefined, finalToolCalls);
    } else {
      callbacks.onError(err instanceof Error ? err : new Error(String(err)));
    }
  }
}

function convertToolCallsMap(map: Record<number, { id: string; name: string; arguments: string }>): McpToolCall[] | undefined {
  const entries = Object.values(map);
  if (entries.length === 0) return undefined;
  return entries.map((e) => {
    let parsedArgs = {};
    try {
      parsedArgs = JSON.parse(e.arguments || '{}');
    } catch {
      parsedArgs = { raw: e.arguments };
    }
    return {
      id: e.id,
      name: e.name,
      arguments: parsedArgs,
      status: 'completed',
    };
  });
}
