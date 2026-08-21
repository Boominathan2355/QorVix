import React, { useState, useRef, useEffect } from 'react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Slider } from '../components/ui/Slider';
import { Badge } from '../components/ui/Badge';
import { MarkdownView } from '../components/ui/MarkdownView';
import { useToast } from '../components/ui/Toast';
import {
  SendIcon,
  StopIcon,
  PlusIcon,
  TrashIcon,
  SlidersIcon,
  UploadIcon,
  SparklesIcon,
  ZapIcon,
  CopyIcon,
  CheckIcon,
} from '../components/icons/Icons';
import { streamChatCompletion } from '../services/sse';
import { api } from '../services/api';
import { ChatMessage, ChatSession, ModelInfo } from '../types';

interface ChatPageProps {
  models: ModelInfo[];
  selectedModel: string;
}

export const ChatPage: React.FC<ChatPageProps> = ({ models, selectedModel }) => {
  const { error: toastError } = useToast();
  const [sessions, setSessions] = useState<ChatSession[]>(() => {
    const saved = localStorage.getItem('qorvix_chat_sessions');
    if (saved) {
      try { return JSON.parse(saved); } catch { /* ignore */ }
    }
    return [
      {
        id: 'default',
        title: 'New Conversation',
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: [],
        model: selectedModel || 'qorvix-model',
        temperature: 0.7,
        topP: 0.9,
        topK: 40,
        maxTokens: 2048,
        repeatPenalty: 1.1,
      },
    ];
  });

  const [activeSessionId, setActiveSessionId] = useState<string>(sessions[0]?.id || 'default');
  const [inputText, setInputText] = useState('');
  const [attachedImages, setAttachedImages] = useState<string[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentTps, setCurrentTps] = useState(0);
  const [showSettings, setShowSettings] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const activeSession = sessions.find((s) => s.id === activeSessionId) || sessions[0];

  useEffect(() => {
    localStorage.setItem('qorvix_chat_sessions', JSON.stringify(sessions));
  }, [sessions]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeSession?.messages]);

  const updateActiveSession = (updater: (s: ChatSession) => ChatSession) => {
    setSessions((prev) =>
      prev.map((s) => (s.id === activeSessionId ? updater(s) : s))
    );
  };

  const handleNewSession = () => {
    const newSession: ChatSession = {
      id: Math.random().toString(36).substring(2, 9),
      title: 'New Conversation',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
      model: selectedModel || 'qorvix-model',
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      maxTokens: 2048,
      repeatPenalty: 1.1,
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
  };

  const handleDeleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (sessions.length <= 1) {
      handleNewSession();
    }
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeSessionId === id) {
      const remaining = sessions.filter((s) => s.id !== id);
      if (remaining.length > 0) {
        setActiveSessionId(remaining[0].id);
      }
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    for (let i = 0; i < files.length; ++i) {
      const file = files[i];
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          setAttachedImages((prev) => [...prev, event.target!.result as string]);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleCopyMessage = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleSendMessage = async () => {
    if ((!inputText.trim() && attachedImages.length === 0) || isGenerating) return;

    const userMessage: ChatMessage = {
      id: Math.random().toString(36).substring(2, 9),
      role: 'user',
      content: inputText.trim(),
      images: attachedImages.length > 0 ? [...attachedImages] : undefined,
      timestamp: Date.now(),
    };

    const assistantMessageId = Math.random().toString(36).substring(2, 9);
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    };

    const newMessages = [...activeSession.messages, userMessage, assistantMessage];
    const firstUserMsg = activeSession.messages.length === 0;

    updateActiveSession((s) => ({
      ...s,
      title: firstUserMsg ? userMessage.content.slice(0, 30) || 'Image Query' : s.title,
      messages: newMessages,
      updatedAt: Date.now(),
    }));

    setInputText('');
    setAttachedImages([]);
    setIsGenerating(true);
    setCurrentTps(0);

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    // Prepare OpenAI formatted messages payload
    const formattedMessages: Array<{ role: string; content: unknown }> = [];
    if (activeSession.systemPrompt?.trim()) {
      formattedMessages.push({ role: 'system', content: activeSession.systemPrompt.trim() });
    }

    for (const msg of [...activeSession.messages, userMessage]) {
      if (msg.images && msg.images.length > 0) {
        const parts: Array<{ type: string; text?: string; image_url?: { url: string } }> = [];
        if (msg.content) {
          parts.push({ type: 'text', text: msg.content });
        }
        for (const img of msg.images) {
          parts.push({ type: 'image_url', image_url: { url: img } });
        }
        formattedMessages.push({ role: msg.role, content: parts });
      } else {
        formattedMessages.push({ role: msg.role, content: msg.content });
      }
    }

    const startTime = performance.now();

    await streamChatCompletion(
      api.getBaseUrl(),
      {
        model: activeSession.model || selectedModel || 'qorvix-model',
        messages: formattedMessages,
        temperature: activeSession.temperature,
        top_p: activeSession.topP,
        top_k: activeSession.topK,
        max_tokens: activeSession.maxTokens,
        repeat_penalty: activeSession.repeatPenalty,
      },
      abortController.signal,
      {
        onToken: (_token, fullText, tps) => {
          setCurrentTps(Math.round(tps * 10) / 10);
          updateActiveSession((s) => ({
            ...s,
            messages: s.messages.map((m) =>
              m.id === assistantMessageId ? { ...m, content: fullText } : m
            ),
          }));
        },
        onComplete: (fullText, totalTokens, avgTps) => {
          const latencyMs = Math.round(performance.now() - startTime);
          setIsGenerating(false);
          updateActiveSession((s) => ({
            ...s,
            messages: s.messages.map((m) =>
              m.id === assistantMessageId
                ? {
                    ...m,
                    content: fullText,
                    totalTokens,
                    tokensPerSec: Math.round(avgTps * 10) / 10,
                    latencyMs,
                  }
                : m
            ),
          }));
        },
        onError: (err) => {
          setIsGenerating(false);
          toastError(err.message, 'Generation Error');
          updateActiveSession((s) => ({
            ...s,
            messages: s.messages.map((m) =>
              m.id === assistantMessageId
                ? { ...m, error: err.message, content: m.content || '*(Generation failed)*' }
                : m
            ),
          }));
        },
      }
    );
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsGenerating(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex h-[calc(100vh-4rem)] overflow-hidden">
      {/* Session History Sidebar */}
      <div className="w-72 border-r border-slate-800/80 bg-slate-950/60 backdrop-blur-xl flex flex-col shrink-0">
        <div className="p-3 border-b border-slate-800/80 flex items-center justify-between">
          <span className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider">
            Conversations
          </span>
          <Button
            variant="outline"
            size="sm"
            leftIcon={<PlusIcon size={14} />}
            onClick={handleNewSession}
          >
            New Chat
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto p-2.5 space-y-1">
          {sessions.map((s) => {
            const isActive = s.id === activeSessionId;
            return (
              <div
                key={s.id}
                onClick={() => setActiveSessionId(s.id)}
                className={`group flex items-center justify-between p-3 rounded-xl cursor-pointer text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-slate-900 text-teal-300 border border-slate-700/80 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/50'
                }`}
              >
                <div className="flex flex-col truncate pr-2">
                  <span className="truncate font-semibold text-slate-200">{s.title}</span>
                  <span className="text-[10px] text-slate-500 font-mono">
                    {s.messages.length} messages
                  </span>
                </div>
                <button
                  onClick={(e) => handleDeleteSession(s.id, e)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 rounded transition-opacity"
                  title="Delete chat"
                >
                  <TrashIcon size={14} />
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* Main Chat Workspace */}
      <div className="flex-1 flex flex-col bg-slate-900/20 overflow-hidden relative">
        {/* Top Control Bar */}
        <div className="h-12 border-b border-slate-800/60 px-6 flex items-center justify-between bg-slate-950/40 shrink-0">
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono font-bold text-slate-300">
              {activeSession.model || selectedModel}
            </span>
            {isGenerating && (
              <Badge variant="primary" size="sm" pulse>
                {currentTps > 0 ? `${currentTps} tok/s` : 'Generating...'}
              </Badge>
            )}
          </div>
          <Button
            variant="ghost"
            size="sm"
            leftIcon={<SlidersIcon size={14} />}
            onClick={() => setShowSettings(!showSettings)}
          >
            Parameters
          </Button>
        </div>

        {/* Message Stream */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {activeSession.messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto space-y-4">
              <div className="p-4 rounded-3xl bg-teal-500/10 border border-teal-500/20 text-teal-400 shadow-xl shadow-teal-500/5">
                <SparklesIcon size={32} />
              </div>
              <div className="space-y-1.5">
                <h3 className="text-lg font-bold text-slate-100">QorVix Studio Chat</h3>
                <p className="text-xs text-slate-400 leading-relaxed font-sans">
                  Type a prompt below or attach an image for multimodal reasoning. All requests run on your high-performance C++23 native inference engine.
                </p>
              </div>
            </div>
          ) : (
            activeSession.messages.map((msg) => {
              const isUser = msg.role === 'user';
              return (
                <div
                  key={msg.id}
                  className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-3xl rounded-2xl p-4.5 space-y-3 relative group ${
                      isUser
                        ? 'bg-gradient-to-br from-teal-600/90 to-teal-700/90 text-slate-950 font-medium shadow-lg shadow-teal-900/20'
                        : 'bg-slate-900/90 border border-slate-800 text-slate-100 shadow-md'
                    }`}
                  >
                    {/* Attached Images */}
                    {msg.images && msg.images.length > 0 && (
                      <div className="flex flex-wrap gap-2 pt-1">
                        {msg.images.map((img, i) => (
                          <img
                            key={i}
                            src={img}
                            alt="attachment"
                            className="h-36 w-auto object-cover rounded-xl border border-white/20 shadow-md"
                          />
                        ))}
                      </div>
                    )}

                    {/* Text content */}
                    {isUser ? (
                      <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-950 font-sans">
                        {msg.content}
                      </p>
                    ) : (
                      <MarkdownView content={msg.content} />
                    )}

                    {/* Telemetry info for assistant responses */}
                    {!isUser && (msg.tokensPerSec !== undefined || msg.latencyMs !== undefined) && (
                      <div className="flex items-center justify-between pt-2 border-t border-slate-800/80 text-[11px] font-mono text-slate-400">
                        <div className="flex items-center gap-3">
                          {msg.tokensPerSec !== undefined && (
                            <span className="flex items-center gap-1 text-teal-400 font-semibold">
                              <ZapIcon size={12} /> {msg.tokensPerSec} tok/s
                            </span>
                          )}
                          {msg.totalTokens !== undefined && (
                            <span>{msg.totalTokens} tokens</span>
                          )}
                          {msg.latencyMs !== undefined && (
                            <span>{msg.latencyMs} ms</span>
                          )}
                        </div>
                        <button
                          onClick={() => handleCopyMessage(msg.id, msg.content)}
                          className="hover:text-slate-100 p-1 transition-colors"
                          title="Copy response"
                        >
                          {copiedId === msg.id ? (
                            <CheckIcon size={14} className="text-emerald-400" />
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

        {/* Input Bar */}
        <div className="p-4 border-t border-slate-800/80 bg-slate-950/80 backdrop-blur-xl space-y-3">
          {/* Image Previews */}
          {attachedImages.length > 0 && (
            <div className="flex gap-2 pb-1 overflow-x-auto">
              {attachedImages.map((img, idx) => (
                <div key={idx} className="relative group shrink-0">
                  <img
                    src={img}
                    alt="preview"
                    className="h-16 w-16 object-cover rounded-lg border border-teal-500/50 shadow-md"
                  />
                  <button
                    onClick={() => setAttachedImages((prev) => prev.filter((_, i) => i !== idx))}
                    className="absolute -top-1.5 -right-1.5 h-5 w-5 rounded-full bg-red-600 text-white flex items-center justify-center text-xs shadow-md hover:bg-red-500"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}

          <div className="relative flex items-end gap-2 bg-slate-900/90 border border-slate-800 focus-within:border-teal-500/60 focus-within:ring-2 focus-within:ring-teal-500/20 rounded-2xl p-2 transition-all">
            <label className="p-2 text-slate-400 hover:text-teal-400 cursor-pointer rounded-xl hover:bg-slate-800 transition-colors">
              <UploadIcon size={20} />
              <input
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={handleImageUpload}
              />
            </label>

            <textarea
              ref={textareaRef}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Send a prompt... (Enter to send, Shift+Enter for new line)"
              rows={1}
              className="flex-1 bg-transparent text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none resize-none py-2 max-h-36 overflow-y-auto"
            />

            {isGenerating ? (
              <Button
                variant="danger"
                size="icon"
                onClick={handleStop}
                title="Stop generation"
              >
                <StopIcon size={18} />
              </Button>
            ) : (
              <Button
                variant="primary"
                size="icon"
                onClick={handleSendMessage}
                disabled={!inputText.trim() && attachedImages.length === 0}
                title="Send prompt"
              >
                <SendIcon size={18} />
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Sampling Parameter Drawer */}
      {showSettings && (
        <Card
          glass
          className="w-80 border-l border-slate-800 rounded-none bg-slate-950/90 backdrop-blur-2xl p-5 space-y-6 overflow-y-auto shrink-0 animate-in slide-in-from-right-10 duration-200"
        >
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <h4 className="font-bold text-slate-100 text-sm flex items-center gap-2">
              <SlidersIcon size={16} /> Generation Parameters
            </h4>
            <button
              onClick={() => setShowSettings(false)}
              className="text-slate-400 hover:text-slate-100"
            >
              ×
            </button>
          </div>

          <div className="space-y-4">
            <div className="space-y-1.5">
              <label className="block text-xs font-medium text-slate-300">System Prompt</label>
              <textarea
                value={activeSession.systemPrompt || ''}
                onChange={(e) =>
                  updateActiveSession((s) => ({ ...s, systemPrompt: e.target.value }))
                }
                placeholder="You are a helpful, expert AI assistant."
                rows={3}
                className="w-full bg-slate-900 border border-slate-800 rounded-xl p-2.5 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-teal-500/50"
              />
            </div>

            <Slider
              label="Temperature"
              min={0}
              max={2}
              step={0.05}
              value={activeSession.temperature}
              valueDisplay={activeSession.temperature.toFixed(2)}
              onChange={(e) =>
                updateActiveSession((s) => ({ ...s, temperature: parseFloat(e.target.value) }))
              }
            />

            <Slider
              label="Top-P (Nucleus Sampling)"
              min={0}
              max={1}
              step={0.05}
              value={activeSession.topP}
              valueDisplay={activeSession.topP.toFixed(2)}
              onChange={(e) =>
                updateActiveSession((s) => ({ ...s, topP: parseFloat(e.target.value) }))
              }
            />

            <Slider
              label="Top-K"
              min={1}
              max={100}
              step={1}
              value={activeSession.topK}
              onChange={(e) =>
                updateActiveSession((s) => ({ ...s, topK: parseInt(e.target.value) }))
              }
            />

            <Slider
              label="Max Tokens"
              min={128}
              max={8192}
              step={128}
              value={activeSession.maxTokens}
              onChange={(e) =>
                updateActiveSession((s) => ({ ...s, maxTokens: parseInt(e.target.value) }))
              }
            />

            <Slider
              label="Repeat Penalty"
              min={1.0}
              max={2.0}
              step={0.05}
              value={activeSession.repeatPenalty}
              valueDisplay={activeSession.repeatPenalty.toFixed(2)}
              onChange={(e) =>
                updateActiveSession((s) => ({ ...s, repeatPenalty: parseFloat(e.target.value) }))
              }
            />
          </div>
        </Card>
      )}
    </div>
  );
};
