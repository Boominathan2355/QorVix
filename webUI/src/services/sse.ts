// Streaming Server-Sent Events (SSE) reader for OpenAI-compatible chat completions

export interface StreamCallbacks {
  onToken: (token: string, fullText: string, tps: number) => void;
  onComplete: (fullText: string, totalTokens: number, avgTps: number) => void;
  onError: (error: Error) => void;
}

export async function streamChatCompletion(
  endpoint: string,
  body: Record<string, unknown>,
  signal: AbortSignal,
  callbacks: StreamCallbacks
): Promise<void> {
  let fullText = '';
  let tokenCount = 0;
  let startTime = 0;

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
          callbacks.onComplete(fullText, tokenCount, avgTps);
          return;
        }

        if (trimmed.startsWith('data: ')) {
          try {
            const data = JSON.parse(trimmed.slice(6));
            const delta = data.choices?.[0]?.delta?.content;
            if (delta) {
              tokenCount++;
              fullText += delta;
              const elapsedSec = (performance.now() - startTime) / 1000;
              const currentTps = elapsedSec > 0 ? tokenCount / elapsedSec : 0;
              callbacks.onToken(delta, fullText, currentTps);
            }
          } catch {
            // ignore unparseable chunk
          }
        }
      }
    }

    const totalSec = (performance.now() - startTime) / 1000;
    const avgTps = totalSec > 0 ? tokenCount / totalSec : 0;
    callbacks.onComplete(fullText, tokenCount, avgTps);
  } catch (err: unknown) {
    if (signal.aborted) {
      const totalSec = startTime > 0 ? (performance.now() - startTime) / 1000 : 0;
      callbacks.onComplete(fullText, tokenCount, totalSec > 0 ? tokenCount / totalSec : 0);
    } else {
      callbacks.onError(err instanceof Error ? err : new Error(String(err)));
    }
  }
}
