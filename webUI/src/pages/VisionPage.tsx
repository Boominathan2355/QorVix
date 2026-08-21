import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { MarkdownView } from '../components/ui/MarkdownView';
import { useToast } from '../components/ui/Toast';
import {
  VisionIcon,
  UploadIcon,
  SendIcon,
  SparklesIcon,
  ZapIcon,
  LayersIcon,
} from '../components/icons/Icons';
import { streamChatCompletion } from '../services/sse';
import { api } from '../services/api';
import { ModelInfo } from '../types';

interface VisionPageProps {
  models: ModelInfo[];
  selectedModel: string;
}

export const VisionPage: React.FC<VisionPageProps> = ({ selectedModel }) => {
  const { error: toastError } = useToast();
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [telemetry, setTelemetry] = useState<{ tps: number; latencyMs: number; tokens: number } | null>(null);

  const presets = [
    { label: 'Detailed Description', prompt: 'Provide an exhaustive, structured breakdown of everything present in this image.' },
    { label: 'OCR & Text Extraction', prompt: 'Extract and transcribe all text, digits, labels, and diagrams visible in this image verbatim.' },
    { label: 'Visual Question Answering', prompt: 'Identify the primary subject, lighting conditions, spatial layout, and stylistic context.' },
    { label: 'Code / Chart Analysis', prompt: 'Interpret any charts, equations, UI designs, or source code snippets in this image.' },
  ];

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      if (event.target?.result) {
        setImagePreview(event.target.result as string);
        setResponse('');
        setTelemetry(null);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleAnalyze = async () => {
    if (!imagePreview || !prompt.trim() || isAnalyzing) return;

    setIsAnalyzing(true);
    setResponse('');
    setTelemetry(null);

    const abortController = new AbortController();
    const startTime = performance.now();

    await streamChatCompletion(
      api.getBaseUrl(),
      {
        model: selectedModel || 'qorvix-vision-model',
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: prompt.trim() },
              { type: 'image_url', image_url: { url: imagePreview } },
            ],
          },
        ],
        max_tokens: 1536,
        temperature: 0.2,
      },
      abortController.signal,
      {
        onToken: (_token, fullText, tps) => {
          setResponse(fullText);
          setTelemetry({ tps: Math.round(tps * 10) / 10, latencyMs: Math.round(performance.now() - startTime), tokens: 0 });
        },
        onComplete: (fullText, totalTokens, avgTps) => {
          setIsAnalyzing(false);
          setResponse(fullText);
          setTelemetry({
            tps: Math.round(avgTps * 10) / 10,
            latencyMs: Math.round(performance.now() - startTime),
            tokens: totalTokens,
          });
        },
        onError: (err) => {
          setIsAnalyzing(false);
          toastError(err.message, 'Vision Pipeline Error');
          setResponse(`*(Error during vision analysis: ${err.message})*`);
        },
      }
    );
  };

  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="info" size="sm">CLIP ViT-L/14 + LLaVA MLP</Badge>
            <Badge variant="primary" size="sm">336×336 Resolution (576 Patches)</Badge>
          </div>
          <h2 className="text-2xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <VisionIcon size={24} className="text-sky-400" />
            Multimodal Vision Studio
          </h2>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Image Input & Projection Details */}
        <div className="lg:col-span-5 space-y-5">
          <Card glass className="p-6 space-y-4">
            <CardTitle className="text-sm font-semibold text-slate-200 flex items-center justify-between">
              <span>Input Image</span>
              {imagePreview && (
                <button
                  onClick={() => { setImagePreview(null); setResponse(''); }}
                  className="text-xs text-red-400 hover:text-red-300 font-mono"
                >
                  Clear Image
                </button>
              )}
            </CardTitle>

            {!imagePreview ? (
              <label className="flex flex-col items-center justify-center border-2 border-dashed border-slate-700/80 hover:border-teal-500/60 rounded-2xl p-10 cursor-pointer bg-slate-950/40 hover:bg-slate-900/50 transition-all text-center space-y-3 group">
                <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800 text-slate-400 group-hover:text-teal-400 group-hover:scale-105 transition-all">
                  <UploadIcon size={28} />
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-semibold text-slate-200">
                    Click to upload or drag image here
                  </p>
                  <p className="text-xs text-slate-500 font-mono">
                    PNG, JPG, WEBP • Max 20MB
                  </p>
                </div>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            ) : (
              <div className="relative rounded-2xl overflow-hidden border border-slate-700 bg-slate-950 shadow-xl">
                <img
                  src={imagePreview}
                  alt="Uploaded probe"
                  className="w-full h-80 object-contain bg-slate-950/80"
                />
              </div>
            )}

            {/* ViT Architecture Specs */}
            <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2 text-xs font-mono">
              <div className="flex items-center justify-between text-slate-400">
                <span className="flex items-center gap-1.5"><LayersIcon size={14} /> Patch Grid</span>
                <span className="text-teal-300 font-bold">24 × 24 = 576 Tokens</span>
              </div>
              <div className="flex items-center justify-between text-slate-400">
                <span>Embedding Width</span>
                <span className="text-slate-200">1024-d (ViT) → 4096-d (LLM)</span>
              </div>
              <div className="flex items-center justify-between text-slate-400">
                <span>Projector Type</span>
                <span className="text-slate-200">2-Layer GELU MLP</span>
              </div>
            </div>
          </Card>
        </div>

        {/* Right Column: Prompt & Reasoning Output */}
        <div className="lg:col-span-7 space-y-5">
          <Card glass className="p-6 space-y-4">
            <CardTitle className="text-sm font-semibold text-slate-200">
              Prompt & Query
            </CardTitle>

            <div className="space-y-2">
              <div className="flex flex-wrap gap-2">
                {presets.map((preset, idx) => (
                  <button
                    key={idx}
                    onClick={() => setPrompt(preset.prompt)}
                    className="text-xs px-2.5 py-1 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/60 transition-all text-left"
                  >
                    {preset.label}
                  </button>
                ))}
              </div>

              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Ask anything about the uploaded image..."
                rows={3}
                className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-teal-500/60 focus:ring-2 focus:ring-teal-500/20"
              />
            </div>

            <div className="flex items-center justify-between pt-1">
              <div className="text-xs text-slate-500 font-mono">
                {imagePreview ? 'Ready for vision inference' : 'Upload an image first'}
              </div>
              <Button
                variant="primary"
                size="md"
                leftIcon={<SparklesIcon size={16} />}
                loading={isAnalyzing}
                disabled={!imagePreview || !prompt.trim()}
                onClick={handleAnalyze}
              >
                Run Vision Inference
              </Button>
            </div>
          </Card>

          {/* Analysis Output Pane */}
          {(response || isAnalyzing) && (
            <Card glass className="p-6 space-y-4 animate-in fade-in duration-200 border-teal-500/30">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <CardTitle className="text-sm font-bold text-teal-300 flex items-center gap-2">
                  <SparklesIcon size={16} /> Visual Reasoning Response
                </CardTitle>
                {telemetry && (
                  <div className="flex items-center gap-3 text-[11px] font-mono text-slate-400">
                    <span className="text-teal-400 font-semibold">{telemetry.tps} tok/s</span>
                    <span>{telemetry.latencyMs} ms</span>
                  </div>
                )}
              </div>

              <div className="min-h-[140px]">
                {response ? (
                  <MarkdownView content={response} />
                ) : (
                  <div className="flex items-center gap-3 text-slate-400 text-sm py-4">
                    <span className="relative flex h-3 w-3">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal-400 opacity-75" />
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-teal-500" />
                    </span>
                    Encoding patches & running decoder cross-attention...
                  </div>
                )}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};
