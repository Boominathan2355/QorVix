// Type definitions for QorVix Studio & Dashboard

export type PageId = 'chat' | 'dashboard' | 'settings' | 'vision' | 'audio' | 'images' | 'embeddings' | 'models' | 'memory' | 'performance' | 'metrics';

export type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

export interface ImageContentPart {
  type: 'image_url';
  image_url: {
    url: string; // base64 data URI or URL
  };
}

export interface AudioContentPart {
  type: 'input_audio';
  input_audio: {
    data: string; // base64 encoded audio
    format: string; // wav, mp3, ogg, flac
  };
}

export interface VideoContentPart {
  type: 'video_url';
  video_url: {
    url: string; // base64 data URI or video URL
  };
}

export interface TextContentPart {
  type: 'text';
  text: string;
}

export type ContentPart = TextContentPart | ImageContentPart | AudioContentPart | VideoContentPart;
export type ChatContent = string | ContentPart[];

// Model Context Protocol (MCP) Schemas
export interface McpTool {
  name: string;
  description: string;
  inputSchema: {
    type?: string;
    properties?: Record<string, any>;
    required?: string[];
  };
  serverId: string;
  enabled?: boolean;
}

export interface McpServer {
  id: string;
  name: string;
  type: 'stdio' | 'sse';
  command?: string;
  args?: string[];
  url?: string;
  status: 'connected' | 'disconnected' | 'connecting' | 'error';
  error?: string;
  tools: McpTool[];
}

export interface McpToolCall {
  id: string;
  name: string;
  arguments: Record<string, any>;
  status: 'pending' | 'running' | 'completed' | 'error';
  result?: any;
  error?: string;
  durationMs?: number;
}

export interface ThinkingState {
  isThinking: boolean;
  text: string;
  durationSeconds: number;
  startTime?: number;
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  thinking?: string; // Chain-of-thought / <think> reasoning tokens
  thinkingDuration?: number; // Thinking duration in seconds
  images?: string[]; // data URLs
  videos?: string[]; // video data URLs
  audioUrl?: string; // audio playback url
  audioData?: string; // base64 audio data
  audioFormat?: string;
  toolCalls?: McpToolCall[]; // MCP tool calls requested by assistant
  toolCallId?: string; // For role="tool", the call ID this answers
  generatedImage?: GeneratedImage;
  timestamp: number;
  tokensPerSec?: number;
  totalTokens?: number;
  latencyMs?: number;
  error?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
  systemPrompt?: string;
  model: string;
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
  repeatPenalty: number;
  enableThinking?: boolean;
  enableMcpTools?: boolean;
}

export interface ModelCapabilities {
  text: boolean;
  image: boolean;
  audio: boolean;
  video: boolean;
  thinking: boolean;
  mcp_tools: boolean;
}

export interface ModelInfo {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  architecture?: string;
  context_length?: number;
  embedding_length?: number;
  vocab_size?: number;
  block_count?: number;
  quantization?: string;
  backend?: string;
  is_multimodal?: boolean;
  capabilities?: ModelCapabilities;
}

export interface AudioTranscriptionResult {
  text: string;
  segments?: {
    id: number;
    start: number;
    end: number;
    text: string;
  }[];
  language?: string;
  duration?: number;
}

export interface GeneratedImage {
  id: string;
  url: string; // base64 or blob URL
  prompt: string;
  negativePrompt?: string;
  steps: number;
  guidance: number;
  seed: number;
  width: number;
  height: number;
  timestamp: number;
}

export interface EmbeddingResult {
  text: string;
  vector: number[];
  norm: number;
  dim: number;
}

export interface MemoryTierStats {
  vramTotalBytes: number;
  vramUsedBytes: number;
  vramAllocatedBytes: number;
  ramTotalBytes: number;
  ramUsedBytes: number;
  spoolTotalBytes: number;
  spoolUsedBytes: number;
  kvCacheTotalPages: number;
  kvCacheUsedPages: number;
  slabFragmentation: number;
}

export interface BenchmarkRun {
  id: string;
  timestamp: number;
  concurrency: number;
  totalRequests: number;
  successfulRequests: number;
  tpsAvg: number;
  ttftMsAvg: number;
  latencyP50Ms: number;
  latencyP95Ms: number;
  latencyP99Ms: number;
  model: string;
}

export interface ServerMetrics {
  uptimeSeconds: number;
  totalRequests: number;
  activeRequests: number;
  promptTokensTotal: number;
  completionTokensTotal: number;
  tpsCurrent: number;
  avgLatencyMs: number;
  status200Count: number;
  status400Count: number;
  status500Count: number;
  metricsRaw: string;
}

export interface AppSettings {
  baseUrl: string;
  metricsUrl: string;
  defaultModel: string;
  theme: 'dark' | 'light' | 'system';
  enableStreaming: boolean;
  defaultSystemPrompt: string;
  autoScroll: boolean;
  soundEffects: boolean;
}
