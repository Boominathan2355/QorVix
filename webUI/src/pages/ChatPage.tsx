import React, { useState, useRef, useEffect } from 'react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Slider } from '../components/ui/Slider';
import { Badge } from '../components/ui/Badge';
import { Switch } from '../components/ui/Switch';
import { Modal } from '../components/ui/Modal';
import { MarkdownView } from '../components/ui/MarkdownView';
import { useToast } from '../components/ui/Toast';
import { ThinkingBlock } from '../components/chat/ThinkingBlock';
import { McpToolCallBlock } from '../components/chat/McpToolCallBlock';
import { McpManagerModal } from '../components/chat/McpManagerModal';
import {
  SendIcon,
  StopIcon,
  SlidersIcon,
  UploadIcon,
  SparklesIcon,
  ZapIcon,
  CopyIcon,
  CheckIcon,
  BotIcon,
  VisionIcon,
  AudioIcon,
  ImageIcon,
  DownloadIcon,
  TrashIcon,
  ChatIcon,
} from '../components/icons/Icons';
import { streamChatCompletion } from '../services/sse';
import { api } from '../services/api';
import { mcpService } from '../services/mcp';
import { ChatMessage, ChatSession, ModelInfo, GeneratedImage, McpToolCall } from '../types';

export type OmniMode = 'chat' | 'vision' | 'audio' | 'image';

interface ChatPageProps {
  models: ModelInfo[];
  selectedModel: string;
  activeSession: ChatSession;
  onUpdateSession: (updater: (s: ChatSession) => ChatSession) => void;
}

export const ChatPage: React.FC<ChatPageProps> = ({
  models,
  selectedModel,
  activeSession,
  onUpdateSession,
}) => {
  const { error: toastError, success: toastSuccess } = useToast();
  const [omniMode, setOmniMode] = useState<OmniMode>('chat');
  const [inputText, setInputText] = useState('');
  const [attachedImages, setAttachedImages] = useState<string[]>([]);
  const [attachedVideos, setAttachedVideos] = useState<string[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentTps, setCurrentTps] = useState(0);
  const [showSettings, setShowSettings] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [selectedLightboxImage, setSelectedLightboxImage] = useState<GeneratedImage | null>(null);
  const [isMcpModalOpen, setIsMcpModalOpen] = useState(false);
  const [mcpToolsCount, setMcpToolsCount] = useState(() => mcpService.getEnabledTools().length);

  // Live Thinking Process states
  const [liveThinking, setLiveThinking] = useState('');
  const [thinkingDuration, setThinkingDuration] = useState(0);
  const [isThinkingStreaming, setIsThinkingStreaming] = useState(false);

  // Native Multimodal Audio states
  const [isRecording, setIsRecording] = useState(false);
  const [recordedAudioBlob, setRecordedAudioBlob] = useState<Blob | null>(null);
  const [recordedAudioUrl, setRecordedAudioUrl] = useState<string | null>(null);
  const [recordedAudioBase64, setRecordedAudioBase64] = useState<string | null>(null);

  // Image Gen options (Stable Diffusion fallback)
  const [sdSteps, setSdSteps] = useState(25);
  const [sdGuidance, setSdGuidance] = useState(7.5);
  const [sdNegative, setSdNegative] = useState('blurry, distorted, low quality');

  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Live Audio visualizer refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

  // Model capabilities
  const currentModelInfo = models.find((m) => m.id === selectedModel);
  const capabilities = currentModelInfo?.capabilities || {
    text: true,
    image: true,
    audio: true,
    video: true,
    thinking: true,
    mcp_tools: true,
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeSession?.messages, isRecording, attachedImages, liveThinking]);

  useEffect(() => {
    return () => {
      if (recordedAudioUrl) URL.revokeObjectURL(recordedAudioUrl);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      if (audioContextRef.current) audioContextRef.current.close();
    };
  }, [recordedAudioUrl]);

  // -------------------------------------------------------------
  // Live Audio Recording & Visualizer
  // -------------------------------------------------------------
  const startWaveformVisualizer = (stream: MediaStream) => {
    const audioCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
    audioContextRef.current = audioCtx;

    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 128;
    analyserRef.current = analyser;

    const source = audioCtx.createMediaStreamSource(stream);
    source.connect(analyser);

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animationFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 2;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height * 0.85;
        ctx.fillStyle = '#14b8a6';
        ctx.fillRect(x, (canvas.height - barHeight) / 2, barWidth - 1, barHeight || 2);
        x += barWidth;
      }
    };

    draw();
  };

  const handleStartRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];

      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setRecordedAudioBlob(blob);
        if (recordedAudioUrl) URL.revokeObjectURL(recordedAudioUrl);
        setRecordedAudioUrl(URL.createObjectURL(blob));

        // Convert blob to base64 for multimodal input_audio
        const reader = new FileReader();
        reader.onloadend = () => {
          const res = reader.result as string;
          const base64 = res.split(',')[1] || '';
          setRecordedAudioBase64(base64);
        };
        reader.readAsDataURL(blob);

        stream.getTracks().forEach((t) => t.stop());
        if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      };

      recorder.start();
      setIsRecording(true);
      startWaveformVisualizer(stream);
    } catch {
      toastError('Microphone access denied. Please grant browser microphone permission.');
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    Array.from(files).forEach((file) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          setAttachedImages((prev) => [...prev, event.target!.result as string]);
        }
      };
      reader.readAsDataURL(file);
    });
  };

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    Array.from(files).forEach((file) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          setAttachedVideos((prev) => [...prev, event.target!.result as string]);
        }
      };
      reader.readAsDataURL(file);
    });
  };

  // -------------------------------------------------------------
  // Multimodal Chat & Thinking Inference Dispatcher
  // -------------------------------------------------------------
  const handleSendChat = async () => {
    const text = inputText.trim();
    if ((!text && attachedImages.length === 0 && attachedVideos.length === 0 && !recordedAudioBase64) || isGenerating) return;

    const userMsgId = `msg-${Date.now()}-u`;
    const assistantMsgId = `msg-${Date.now()}-a`;

    const userMessage: ChatMessage = {
      id: userMsgId,
      role: 'user',
      content: text,
      images: attachedImages.length > 0 ? [...attachedImages] : undefined,
      videos: attachedVideos.length > 0 ? [...attachedVideos] : undefined,
      audioUrl: recordedAudioUrl || undefined,
      audioData: recordedAudioBase64 || undefined,
      audioFormat: recordedAudioBase64 ? 'wav' : undefined,
      timestamp: Date.now(),
    };

    const initialAssistantMessage: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    };

    onUpdateSession((prev) => ({
      ...prev,
      messages: [...prev.messages, userMessage, initialAssistantMessage],
      title: prev.messages.length === 0 ? text.slice(0, 32) || 'Any-to-Any Conversation' : prev.title,
    }));

    setInputText('');
    setAttachedImages([]);
    setAttachedVideos([]);
    setRecordedAudioBlob(null);
    setRecordedAudioUrl(null);
    setRecordedAudioBase64(null);
    setIsGenerating(true);
    setCurrentTps(0);
    setLiveThinking('');
    setThinkingDuration(0);
    setIsThinkingStreaming(false);

    // Build payload messages including multimodal image_url, video_url and input_audio parts
    const payloadMessages = [...activeSession.messages, userMessage].map((m) => {
      const parts: any[] = [];
      if (m.content) parts.push({ type: 'text', text: m.content });
      if (m.images && m.images.length > 0) {
        for (const img of m.images) {
          parts.push({ type: 'image_url', image_url: { url: img } });
        }
      }
      if (m.videos && m.videos.length > 0) {
        for (const vid of m.videos) {
          parts.push({ type: 'video_url', video_url: { url: vid } });
        }
      }
      if (m.audioData) {
        parts.push({ type: 'input_audio', input_audio: { data: m.audioData, format: m.audioFormat || 'wav' } });
      }

      return {
        role: m.role,
        content: parts.length > 1 ? parts : m.content || '',
      };
    });

    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    const reqStart = performance.now();

    // Prepare MCP tools schema if enabled
    const toolsSchema = activeSession.enableMcpTools !== false ? mcpService.formatForOpenAi() : undefined;

    await streamChatCompletion(
      api.getBaseUrl(),
      {
        model: selectedModel || activeSession.model,
        messages: payloadMessages,
        temperature: activeSession.temperature,
        top_p: activeSession.topP,
        max_tokens: activeSession.maxTokens,
        tools: toolsSchema && toolsSchema.length > 0 ? toolsSchema : undefined,
      },
      abortController.signal,
      {
        onToken: (_token, fullText, tps) => {
          setCurrentTps(Math.round(tps * 10) / 10);
          onUpdateSession((prev) => ({
            ...prev,
            messages: prev.messages.map((m) =>
              m.id === assistantMsgId ? { ...m, content: fullText } : m
            ),
          }));
        },
        onThinking: (_chunk, fullThinkingText, elapsedSec) => {
          setIsThinkingStreaming(true);
          setLiveThinking(fullThinkingText);
          setThinkingDuration(elapsedSec);
        },
        onToolCall: (toolCalls) => {
          onUpdateSession((prev) => ({
            ...prev,
            messages: prev.messages.map((m) =>
              m.id === assistantMsgId ? { ...m, toolCalls } : m
            ),
          }));
        },
        onComplete: async (fullText, totalTokens, avgTps, thinkingResult, toolCalls) => {
          setIsGenerating(false);
          setIsThinkingStreaming(false);
          const totalSec = (performance.now() - reqStart) / 1000;
          const latency = Math.round(totalSec * 1000);

          // If assistant called MCP tools, automatically execute them!
          let executedToolCalls = toolCalls;
          if (toolCalls && toolCalls.length > 0) {
            executedToolCalls = await Promise.all(
              toolCalls.map(async (tc) => {
                const res = await mcpService.executeToolCall(tc);
                return {
                  ...tc,
                  status: res.error ? 'error' as const : 'completed' as const,
                  result: res.result,
                  error: res.error,
                };
              })
            );
          }

          onUpdateSession((prev) => ({
            ...prev,
            messages: prev.messages.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    content: fullText,
                    thinking: thinkingResult || liveThinking || undefined,
                    thinkingDuration: thinkingDuration || undefined,
                    toolCalls: executedToolCalls,
                    tokensPerSec: Math.round(avgTps * 10) / 10,
                    totalTokens,
                    latencyMs: latency,
                  }
                : m
            ),
          }));
          setLiveThinking('');
        },
        onError: (err) => {
          setIsGenerating(false);
          setIsThinkingStreaming(false);
          toastError(err.message, 'Inference Engine Error');
          onUpdateSession((prev) => ({
            ...prev,
            messages: prev.messages.map((m) =>
              m.id === assistantMsgId
                ? { ...m, content: `*(Error: ${err.message})*`, error: err.message }
                : m
            ),
          }));
        },
      }
    );
  };

  const handleSendImageGen = async () => {
    const text = inputText.trim();
    if (!text || isGenerating) return;

    const userMsgId = `msg-${Date.now()}-u`;
    const assistantMsgId = `msg-${Date.now()}-a`;

    onUpdateSession((prev) => ({
      ...prev,
      messages: [
        ...prev.messages,
        { id: userMsgId, role: 'user', content: `[🎨 Synthesize Image]: ${text}`, timestamp: Date.now() },
        { id: assistantMsgId, role: 'assistant', content: 'Synthesizing image with Stable Diffusion...', timestamp: Date.now() },
      ],
    }));

    setInputText('');
    setIsGenerating(true);

    try {
      const img = await api.generateImage({
        prompt: text,
        negativePrompt: sdNegative,
        steps: sdSteps,
        guidance: sdGuidance,
      });

      onUpdateSession((prev) => ({
        ...prev,
        messages: prev.messages.map((m) =>
          m.id === assistantMsgId
            ? {
                ...m,
                content: `Generated image for prompt: **"${text}"**`,
                generatedImage: img,
              }
            : m
        ),
      }));
      toastSuccess('Image synthesized successfully!');
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Image synthesis failed');
      onUpdateSession((prev) => ({
        ...prev,
        messages: prev.messages.map((m) =>
          m.id === assistantMsgId
            ? { ...m, content: `*(Failed to generate image: ${err instanceof Error ? err.message : 'Engine error'})*` }
            : m
        ),
      }));
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSend = () => {
    if (omniMode === 'image') {
      handleSendImageGen();
    } else {
      handleSendChat();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleCopyMessage = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleDownload = (img: GeneratedImage) => {
    const a = document.createElement('a');
    a.href = img.url;
    a.download = `qorvix-${img.id}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="flex-1 flex flex-col h-full w-full overflow-hidden relative">
      {/* Omni-Mode Selector Tabs & MCP Tools Strip */}
      <div className="w-full border-b border-border/50 bg-card/20 px-4 md:px-8 py-2.5 shrink-0 z-10">
        <div className="max-w-4xl mx-auto w-full flex items-center justify-between gap-3">
          <div className="flex items-center gap-1.5 bg-secondary/80 p-1 rounded-2xl border border-border">
            <button
              onClick={() => setOmniMode('chat')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                omniMode === 'chat'
                  ? 'bg-background text-teal-500 dark:text-teal-400 shadow-sm border border-border'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <ChatIcon size={14} />
              <span>Chat</span>
            </button>

            {capabilities.image && (
              <button
                onClick={() => setOmniMode('vision')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                  omniMode === 'vision'
                    ? 'bg-background text-sky-500 dark:text-sky-400 shadow-sm border border-border'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                <VisionIcon size={14} />
                <span>Vision</span>
              </button>
            )}

            {capabilities.audio && (
              <button
                onClick={() => setOmniMode('audio')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                  omniMode === 'audio'
                    ? 'bg-background text-purple-500 dark:text-purple-400 shadow-sm border border-border'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                <AudioIcon size={14} />
                <span>Audio</span>
              </button>
            )}

            <button
              onClick={() => setOmniMode('image')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                omniMode === 'image'
                  ? 'bg-background text-pink-500 dark:text-pink-400 shadow-sm border border-border'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <ImageIcon size={14} />
              <span>Stable Diffusion</span>
            </button>
          </div>

          <div className="flex items-center gap-2">
            {/* MCP Tools Status Chip */}
            <button
              onClick={() => setIsMcpModalOpen(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-teal-500/30 bg-teal-500/10 hover:bg-teal-500/20 text-teal-600 dark:text-teal-400 text-xs font-semibold font-mono transition-all shadow-xs"
              title="Manage Model Context Protocol Tools"
            >
              <ZapIcon size={13} />
              <span>{mcpToolsCount} MCP Tools</span>
            </button>

            <button
              onClick={() => setShowSettings(!showSettings)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-border bg-card hover:bg-secondary text-xs font-medium text-muted-foreground hover:text-foreground transition-all shadow-xs"
            >
              <SlidersIcon size={14} />
              <span className="hidden sm:inline">Parameters</span>
            </button>
          </div>
        </div>
      </div>

      {/* Message Feed Canvas */}
      <div className="flex-1 overflow-y-auto w-full px-4 md:px-8 py-6">
        <div className="max-w-4xl mx-auto w-full space-y-6">
          {activeSession.messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center space-y-6 max-w-2xl mx-auto py-8">
              <div className="space-y-3">
                <div className="inline-flex p-3.5 rounded-3xl bg-teal-500/10 text-teal-500 border border-teal-500/20 shadow-md shadow-teal-500/10">
                  <SparklesIcon size={28} />
                </div>
                <h2 className="text-2xl md:text-3xl font-extrabold text-foreground tracking-tight">
                  QorVix Multimodal & Thinking Dashboard
                </h2>
                <p className="text-xs md:text-sm text-muted-foreground max-w-md mx-auto">
                  Unified multimodal inference engine with native Text, Image, Audio, DeepSeek-R1 Thinking, and Model Context Protocol (MCP) tool execution.
                </p>
              </div>

              {/* Quick Modality Starters */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full text-left">
                <div
                  onClick={() => {
                    setOmniMode('chat');
                    setInputText('Analyze the mathematical proof for continuous batching in multi-tenant inference engines.');
                  }}
                  className="p-3.5 rounded-2xl border border-border bg-card/60 hover:bg-secondary/70 hover:border-teal-500/40 cursor-pointer transition-all duration-150 group shadow-xs flex items-start gap-3"
                >
                  <div className="p-2 rounded-xl bg-background border border-border shadow-xs text-teal-500 shrink-0">
                    <ChatIcon size={18} />
                  </div>
                  <div className="space-y-0.5">
                    <h4 className="text-xs font-bold text-foreground group-hover:text-teal-500 transition-colors">
                      Deep Thinking & Reasoning
                    </h4>
                    <p className="text-[11px] text-muted-foreground">
                      Chain-of-thought tokens, mathematical reasoning traces
                    </p>
                  </div>
                </div>

                <div
                  onClick={() => {
                    setOmniMode('vision');
                    document.getElementById('omni-file-upload')?.click();
                  }}
                  className="p-3.5 rounded-2xl border border-border bg-card/60 hover:bg-secondary/70 hover:border-sky-500/40 cursor-pointer transition-all duration-150 group shadow-xs flex items-start gap-3"
                >
                  <div className="p-2 rounded-xl bg-background border border-border shadow-xs text-sky-500 shrink-0">
                    <VisionIcon size={18} />
                  </div>
                  <div className="space-y-0.5">
                    <h4 className="text-xs font-bold text-foreground group-hover:text-sky-500 transition-colors">
                      Multimodal Vision & OCR
                    </h4>
                    <p className="text-[11px] text-muted-foreground">
                      Upload image for 576-patch CLIP ViT inspection
                    </p>
                  </div>
                </div>

                <div
                  onClick={() => {
                    setOmniMode('audio');
                    handleStartRecording();
                  }}
                  className="p-3.5 rounded-2xl border border-border bg-card/60 hover:bg-secondary/70 hover:border-purple-500/40 cursor-pointer transition-all duration-150 group shadow-xs flex items-start gap-3"
                >
                  <div className="p-2 rounded-xl bg-background border border-border shadow-xs text-purple-500 shrink-0">
                    <AudioIcon size={18} />
                  </div>
                  <div className="space-y-0.5">
                    <h4 className="text-xs font-bold text-foreground group-hover:text-purple-500 transition-colors">
                      Native Multimodal Audio
                    </h4>
                    <p className="text-[11px] text-muted-foreground">
                      Attach live voice waveform directly into model context
                    </p>
                  </div>
                </div>

                <div
                  onClick={() => setIsMcpModalOpen(true)}
                  className="p-3.5 rounded-2xl border border-border bg-card/60 hover:bg-secondary/70 hover:border-amber-500/40 cursor-pointer transition-all duration-150 group shadow-xs flex items-start gap-3"
                >
                  <div className="p-2 rounded-xl bg-background border border-border shadow-xs text-amber-500 shrink-0">
                    <ZapIcon size={18} />
                  </div>
                  <div className="space-y-0.5">
                    <h4 className="text-xs font-bold text-foreground group-hover:text-amber-500 transition-colors">
                      Model Context Protocol (MCP)
                    </h4>
                    <p className="text-[11px] text-muted-foreground">
                      Connect external SQLite, filesystem, and API tools
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            activeSession.messages.map((msg) => {
              const isUser = msg.role === 'user';
              return (
                <div
                  key={msg.id}
                  className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'} animate-in fade-in duration-150`}
                >
                  {!isUser && (
                    <div className="h-7 w-7 rounded-xl bg-teal-500/15 border border-teal-500/30 text-teal-500 flex items-center justify-center shrink-0 mt-1 shadow-xs">
                      <BotIcon size={15} />
                    </div>
                  )}

                  <div
                    className={`max-w-2xl rounded-2xl p-4 space-y-2.5 shadow-xs ${
                      isUser
                        ? 'bg-teal-600 text-white font-medium rounded-tr-xs'
                        : 'bg-card border border-border text-foreground rounded-tl-xs'
                    }`}
                  >
                    {/* Multimodal Attached Images */}
                    {msg.images && msg.images.length > 0 && (
                      <div className="flex flex-wrap gap-2 pb-1">
                        {msg.images.map((img, i) => (
                          <img
                            key={i}
                            src={img}
                            alt="attached"
                            className="max-h-48 rounded-xl object-contain border border-white/20 shadow-md bg-black/20"
                          />
                        ))}
                      </div>
                    )}

                    {/* Multimodal Attached Videos */}
                    {msg.videos && msg.videos.length > 0 && (
                      <div className="flex flex-wrap gap-2 pb-1">
                        {msg.videos.map((vid, i) => (
                          <video
                            key={i}
                            src={vid}
                            controls
                            className="max-h-56 rounded-xl object-contain border border-white/20 shadow-md bg-black/40"
                          />
                        ))}
                      </div>
                    )}

                    {/* Multimodal Attached Audio */}
                    {msg.audioUrl && (
                      <div className="p-2.5 rounded-xl bg-secondary/80 border border-border flex items-center gap-3">
                        <AudioIcon size={16} className="text-purple-500 shrink-0" />
                        <audio src={msg.audioUrl} controls className="h-7 w-56 rounded-lg" />
                      </div>
                    )}

                    {/* Chain-of-Thought (Thinking) Collapsible Box */}
                    {msg.thinking && (
                      <ThinkingBlock
                        thinking={msg.thinking}
                        durationSeconds={msg.thinkingDuration}
                      />
                    )}

                    {/* Active Live Thinking stream for latest assistant message */}
                    {!isUser && isThinkingStreaming && liveThinking && !msg.thinking && (
                      <ThinkingBlock
                        thinking={liveThinking}
                        durationSeconds={thinkingDuration}
                        isStreaming={true}
                      />
                    )}

                    {/* MCP Tool Calls Executed by Assistant */}
                    {msg.toolCalls && msg.toolCalls.length > 0 && (
                      <div className="space-y-1.5 pt-1">
                        {msg.toolCalls.map((tc) => (
                          <McpToolCallBlock key={tc.id} toolCall={tc} />
                        ))}
                      </div>
                    )}

                    {/* Final text content */}
                    {isUser ? (
                      <p className="whitespace-pre-wrap text-sm leading-relaxed">{msg.content}</p>
                    ) : (
                      <MarkdownView content={msg.content} />
                    )}

                    {/* Inline Generated Image */}
                    {msg.generatedImage && (
                      <div className="pt-2 space-y-2">
                        <div
                          onClick={() => setSelectedLightboxImage(msg.generatedImage!)}
                          className="relative rounded-xl overflow-hidden border border-border bg-background cursor-pointer group shadow-md max-w-sm"
                        >
                          <img
                            src={msg.generatedImage.url}
                            alt={msg.generatedImage.prompt}
                            className="w-full h-auto object-cover group-hover:scale-102 transition-transform duration-200"
                          />
                          <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-white text-xs font-semibold gap-1.5">
                            <span>Click to Zoom</span>
                          </div>
                        </div>

                        <div className="flex items-center justify-between text-[11px] font-mono text-muted-foreground">
                          <span>{msg.generatedImage.width}×{msg.generatedImage.height} • Seed: {msg.generatedImage.seed}</span>
                          <button
                            onClick={() => handleDownload(msg.generatedImage!)}
                            className="flex items-center gap-1 text-teal-500 hover:text-teal-400 font-semibold"
                          >
                            <DownloadIcon size={12} /> Download PNG
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Telemetry pill */}
                    {!isUser && (msg.tokensPerSec !== undefined || msg.latencyMs !== undefined) && (
                      <div className="flex items-center justify-between pt-2 border-t border-border/80 text-[11px] font-mono text-muted-foreground">
                        <div className="flex items-center gap-3">
                          {msg.tokensPerSec !== undefined && (
                            <span className="flex items-center gap-1 text-teal-500 font-semibold">
                              <ZapIcon size={12} /> {msg.tokensPerSec} tok/s
                            </span>
                          )}
                          {msg.totalTokens !== undefined && <span>{msg.totalTokens} tokens</span>}
                          {msg.latencyMs !== undefined && <span>{msg.latencyMs} ms</span>}
                        </div>
                        <button
                          onClick={() => handleCopyMessage(msg.id, msg.content)}
                          className="hover:text-foreground p-1 transition-colors"
                          title="Copy message"
                        >
                          {copiedId === msg.id ? (
                            <CheckIcon size={14} className="text-emerald-500" />
                          ) : (
                            <CopyIcon size={14} />
                          )}
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Floating Prompt Bar with Native Multimodal & Attachments */}
      <div className="w-full px-4 md:px-8 pb-5 pt-2 bg-gradient-to-t from-background via-background/95 to-transparent shrink-0 z-10">
        <div className="max-w-4xl mx-auto w-full space-y-2">
          {/* Inline Live Microphone Waveform Banner */}
          {isRecording && (
            <div className="p-3 rounded-2xl bg-card border border-teal-500/40 shadow-lg flex items-center justify-between gap-4 animate-in slide-in-from-bottom-2 duration-150">
              <div className="flex items-center gap-3">
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-500 opacity-75" />
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-red-600" />
                </span>
                <span className="text-xs font-semibold text-foreground">Recording Native Multimodal Audio...</span>
              </div>

              <canvas ref={canvasRef} width={200} height={32} className="h-8 w-48 rounded-lg bg-background/80" />

              <Button variant="danger" size="sm" onClick={handleStopRecording}>
                Attach Audio
              </Button>
            </div>
          )}

          {/* Recorded Audio Ready Pill */}
          {recordedAudioBlob && !isRecording && (
            <div className="p-2.5 rounded-2xl bg-secondary border border-border flex items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-2">
                <AudioIcon size={16} className="text-purple-500" />
                <span className="font-semibold text-foreground">Audio Input Attached</span>
                {recordedAudioUrl && <audio src={recordedAudioUrl} controls className="h-6 w-48" />}
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => { setRecordedAudioBlob(null); setRecordedAudioUrl(null); setRecordedAudioBase64(null); }}
                  className="text-muted-foreground hover:text-red-500 p-1"
                  title="Remove Audio"
                >
                  <TrashIcon size={14} />
                </button>
              </div>
            </div>
          )}

          {/* Floating Input Pill */}
          <div className="rounded-3xl border border-border bg-card/95 backdrop-blur-2xl shadow-xl p-2.5 space-y-2 focus-within:border-teal-500/50 focus-within:ring-2 focus-within:ring-teal-500/20 transition-all">
            {/* Attached Image Previews */}
            {attachedImages.length > 0 && (
              <div className="flex gap-2 pb-1 px-1 overflow-x-auto">
                {attachedImages.map((img, idx) => (
                  <div key={idx} className="relative group shrink-0">
                    <img
                      src={img}
                      alt="attachment"
                      className="h-14 w-14 object-cover rounded-xl border border-teal-500/40 shadow-xs"
                    />
                    <button
                      onClick={() => setAttachedImages((prev) => prev.filter((_, i) => i !== idx))}
                      className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-red-500 text-white text-[10px] flex items-center justify-center shadow-xs hover:bg-red-600"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Attached Video Previews */}
            {attachedVideos.length > 0 && (
              <div className="flex gap-2 pb-1 px-1 overflow-x-auto">
                {attachedVideos.map((vid, idx) => (
                  <div key={idx} className="relative group shrink-0">
                    <video
                      src={vid}
                      className="h-14 w-20 object-cover rounded-xl border border-indigo-500/40 shadow-xs bg-black/40"
                    />
                    <button
                      onClick={() => setAttachedVideos((prev) => prev.filter((_, i) => i !== idx))}
                      className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-red-500 text-white text-[10px] flex items-center justify-center shadow-xs hover:bg-red-600"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="flex items-end gap-1.5 px-1">
              {/* Image Attachment Button */}
              {capabilities.image && (
                <label className="p-2 text-muted-foreground hover:text-sky-500 hover:bg-secondary rounded-xl cursor-pointer transition-colors shrink-0" title="Attach image for Multimodal Vision">
                  <UploadIcon size={18} />
                  <input
                    id="omni-file-upload"
                    type="file"
                    accept="image/*"
                    multiple
                    onChange={handleImageUpload}
                    className="hidden"
                  />
                </label>
              )}

              {/* Video Attachment Button */}
              {capabilities.video && (
                <label className="p-2 text-muted-foreground hover:text-indigo-500 hover:bg-secondary rounded-xl cursor-pointer transition-colors shrink-0" title="Attach video for Multimodal Video Analysis">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="23 7 16 12 23 17 23 7" />
                    <rect width="14" height="14" x="1" y="5" rx="2" ry="2" />
                  </svg>
                  <input
                    type="file"
                    accept="video/*"
                    multiple
                    onChange={handleVideoUpload}
                    className="hidden"
                  />
                </label>
              )}

              {/* Multimodal Audio Button */}
              {capabilities.audio && (
                <button
                  onClick={isRecording ? handleStopRecording : handleStartRecording}
                  className={`p-2 rounded-xl transition-colors shrink-0 ${
                    isRecording ? 'text-red-500 bg-red-500/15' : 'text-muted-foreground hover:text-purple-500 hover:bg-secondary'
                  }`}
                  title={isRecording ? 'Stop Recording' : 'Record Native Audio'}
                >
                  <AudioIcon size={18} />
                </button>
              )}

              {/* Textarea */}
              <textarea
                ref={textareaRef}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
                placeholder={
                  omniMode === 'image'
                    ? 'Enter Stable Diffusion image prompt...'
                    : 'Ask anything, attach image, or speak audio...'
                }
                className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none resize-none py-2 max-h-36 font-sans"
              />

              {/* Send or Stop Button */}
              {isGenerating ? (
                <Button
                  variant="danger"
                  size="icon"
                  onClick={() => { abortControllerRef.current?.abort(); setIsGenerating(false); }}
                  className="rounded-2xl shrink-0"
                  title="Stop"
                >
                  <StopIcon size={16} />
                </Button>
              ) : (
                <Button
                  variant="primary"
                  size="icon"
                  onClick={handleSend}
                  disabled={!inputText.trim() && attachedImages.length === 0 && !recordedAudioBase64}
                  className="rounded-2xl shrink-0"
                  title="Send"
                >
                  <SendIcon size={16} />
                </Button>
              )}
            </div>
          </div>

          {/* Live Token Generation Speedometer */}
          {isGenerating && currentTps > 0 && (
            <div className="flex justify-center pt-1">
              <Badge variant="primary" size="sm" pulse>
                Generating at {currentTps} tok/s
              </Badge>
            </div>
          )}
        </div>
      </div>

      {/* Lightbox Modal for Full Image Zoom */}
      {selectedLightboxImage && (
        <Modal
          isOpen={!!selectedLightboxImage}
          onClose={() => setSelectedLightboxImage(null)}
          title="Synthesized Image Preview"
          maxWidth="2xl"
        >
          <div className="space-y-4">
            <div className="rounded-2xl overflow-hidden bg-background border border-border flex items-center justify-center">
              <img
                src={selectedLightboxImage.url}
                alt={selectedLightboxImage.prompt}
                className="max-h-[60vh] w-auto object-contain"
              />
            </div>
            <div className="p-3 rounded-xl bg-secondary border border-border text-xs space-y-1">
              <p className="font-semibold text-foreground">{selectedLightboxImage.prompt}</p>
              <p className="text-[11px] font-mono text-muted-foreground">
                Steps: {selectedLightboxImage.steps} • CFG: {selectedLightboxImage.guidance} • Seed: {selectedLightboxImage.seed} • {selectedLightboxImage.width}×{selectedLightboxImage.height}
              </p>
            </div>
            <div className="flex justify-end">
              <Button variant="primary" size="md" leftIcon={<DownloadIcon size={14} />} onClick={() => handleDownload(selectedLightboxImage)}>
                Download PNG
              </Button>
            </div>
          </div>
        </Modal>
      )}

      {/* MCP Server & Tools Management Modal */}
      <McpManagerModal
        isOpen={isMcpModalOpen}
        onClose={() => setIsMcpModalOpen(false)}
        onToolsUpdated={() => setMcpToolsCount(mcpService.getEnabledTools().length)}
      />

      {/* Sampling & SD Parameter Drawer */}
      {showSettings && (
        <Card
          glass
          className="absolute right-6 top-16 w-80 shadow-2xl p-5 space-y-4 z-40 animate-in slide-in-from-right-5 duration-150"
        >
          <div className="flex items-center justify-between border-b border-border pb-2.5">
            <h4 className="font-bold text-xs text-foreground flex items-center gap-2">
              <SlidersIcon size={15} /> Generation Parameters
            </h4>
            <button onClick={() => setShowSettings(false)} className="text-muted-foreground hover:text-foreground">
              ×
            </button>
          </div>

          <div className="space-y-3">
            <Switch
              checked={activeSession.enableThinking !== false}
              onChange={(val) => onUpdateSession((s) => ({ ...s, enableThinking: val }))}
              label="Chain-of-Thought (Thinking)"
              description="Extract reasoning tokens into expandable trace"
            />

            <Switch
              checked={activeSession.enableMcpTools !== false}
              onChange={(val) => onUpdateSession((s) => ({ ...s, enableMcpTools: val }))}
              label="Enable MCP Tool Calls"
              description="Allow model to execute connected tools"
            />

            <Slider
              label="Temperature (LLM)"
              min={0}
              max={2}
              step={0.05}
              value={activeSession.temperature}
              valueDisplay={activeSession.temperature.toFixed(2)}
              onChange={(e) => onUpdateSession((s) => ({ ...s, temperature: parseFloat(e.target.value) }))}
            />

            <Slider
              label="Top-P (Nucleus)"
              min={0}
              max={1}
              step={0.05}
              value={activeSession.topP}
              valueDisplay={activeSession.topP.toFixed(2)}
              onChange={(e) => onUpdateSession((s) => ({ ...s, topP: parseFloat(e.target.value) }))}
            />

            <Slider
              label="Diffusion Steps (SD)"
              min={5}
              max={50}
              step={1}
              value={sdSteps}
              onChange={(e) => setSdSteps(parseInt(e.target.value))}
            />

            <Slider
              label="CFG Scale (SD)"
              min={1}
              max={20}
              step={0.5}
              value={sdGuidance}
              valueDisplay={sdGuidance.toFixed(1)}
              onChange={(e) => setSdGuidance(parseFloat(e.target.value))}
            />
          </div>
        </Card>
      )}
    </div>
  );
};
