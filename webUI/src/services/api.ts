import { ModelInfo, GeneratedImage, AudioTranscriptionResult, EmbeddingResult } from '../types';

export class QorvixApiClient {
  private baseUrl: string;
  private metricsUrl: string;

  constructor(baseUrl = 'http://localhost:2005', metricsUrl = 'http://localhost:2009') {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.metricsUrl = metricsUrl.replace(/\/+$/, '');
  }

  setBaseUrl(url: string) {
    this.baseUrl = url.replace(/\/+$/, '');
  }

  setMetricsUrl(url: string) {
    this.metricsUrl = url.replace(/\/+$/, '');
  }

  getBaseUrl() {
    return this.baseUrl;
  }

  getMetricsUrl() {
    return this.metricsUrl;
  }

  async checkHealth(): Promise<{ ok: boolean; latencyMs: number }> {
    const start = performance.now();
    try {
      const res = await fetch(`${this.baseUrl}/v1/models`, { method: 'GET', signal: AbortSignal.timeout(3000) });
      const latencyMs = Math.round(performance.now() - start);
      return { ok: res.ok, latencyMs };
    } catch {
      return { ok: false, latencyMs: 0 };
    }
  }

  async getModels(): Promise<ModelInfo[]> {
    try {
      const res = await fetch(`${this.baseUrl}/v1/models`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      return (data.data || []).map((m: ModelInfo) => ({
        ...m,
        is_multimodal: m.id.toLowerCase().includes('llava') || m.id.toLowerCase().includes('vision'),
      }));
    } catch {
      // Fallback default list if server is initializing
      return [
        {
          id: 'qorvix-default-model',
          object: 'model',
          created: Date.now(),
          owned_by: 'qorvix',
          architecture: 'transformer',
          quantization: 'Q4_K_M',
        },
      ];
    }
  }

  async generateImage(params: {
    prompt: string;
    negativePrompt?: string;
    steps?: number;
    guidance?: number;
    seed?: number;
    width?: number;
    height?: number;
  }): Promise<GeneratedImage> {
    const res = await fetch(`${this.baseUrl}/v1/images/generations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: params.prompt,
        negative_prompt: params.negativePrompt,
        steps: params.steps || 20,
        guidance_scale: params.guidance || 7.5,
        seed: params.seed,
        size: `${params.width || 512}x${params.height || 512}`,
        response_format: 'b64_json',
      }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Image Generation Error (${res.status}): ${text}`);
    }

    const data = await res.json();
    const item = data.data?.[0];
    const url = item?.b64_json
      ? `data:image/png;base64,${item.b64_json}`
      : item?.url || '';

    return {
      id: Math.random().toString(36).substring(2, 9),
      url,
      prompt: params.prompt,
      negativePrompt: params.negativePrompt,
      steps: params.steps || 20,
      guidance: params.guidance || 7.5,
      seed: params.seed || Math.floor(Math.random() * 1000000),
      width: params.width || 512,
      height: params.height || 512,
      timestamp: Date.now(),
    };
  }

  async transcribeAudio(file: Blob, language?: string): Promise<AudioTranscriptionResult> {
    const formData = new FormData();
    formData.append('file', file, 'recording.wav');
    formData.append('model', 'whisper');
    if (language) formData.append('language', language);
    formData.append('response_format', 'verbose_json');

    const res = await fetch(`${this.baseUrl}/v1/audio/transcriptions`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Whisper Transcription Error (${res.status}): ${text}`);
    }

    return await res.json();
  }

  async translateAudio(file: Blob): Promise<AudioTranscriptionResult> {
    const formData = new FormData();
    formData.append('file', file, 'recording.wav');
    formData.append('model', 'whisper');
    formData.append('response_format', 'verbose_json');

    const res = await fetch(`${this.baseUrl}/v1/audio/translations`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Whisper Translation Error (${res.status}): ${text}`);
    }

    return await res.json();
  }

  async createEmbeddings(input: string | string[], model = 'bert'): Promise<EmbeddingResult[]> {
    const res = await fetch(`${this.baseUrl}/v1/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input, model }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Embedding Error (${res.status}): ${text}`);
    }

    const data = await res.json();
    const inputs = Array.isArray(input) ? input : [input];

    return (data.data || []).map((d: { embedding: number[] }, idx: number) => {
      const vec = d.embedding;
      let sumSq = 0;
      for (const val of vec) sumSq += val * val;
      const norm = Math.sqrt(sumSq);

      return {
        text: inputs[idx] || `Item ${idx + 1}`,
        vector: vec,
        norm,
        dim: vec.length,
      };
    });
  }

  async fetchRawMetrics(): Promise<string> {
    try {
      const res = await fetch(`${this.metricsUrl}/metrics`, { signal: AbortSignal.timeout(3000) });
      if (!res.ok) return '';
      return await res.text();
    } catch {
      return '';
    }
  }
}

export const api = new QorvixApiClient();
