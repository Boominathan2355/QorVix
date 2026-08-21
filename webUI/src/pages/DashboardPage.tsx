import React, { useEffect, useState } from 'react';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import {
  ChatIcon,
  VisionIcon,
  AudioIcon,
  ImageIcon,
  EmbeddingsIcon,
  ModelsIcon,
  MemoryIcon,
  PerformanceIcon,
  SparklesIcon,
  ZapIcon,
  CpuIcon,
  LayersIcon,
} from '../components/icons/Icons';
import { api } from '../services/api';
import { ModelInfo, PageId } from '../types';

interface DashboardPageProps {
  onNavigate: (page: PageId) => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({ onNavigate }) => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [online, setOnline] = useState<boolean | null>(null);
  const [latency, setLatency] = useState<number>(0);

  useEffect(() => {
    const load = async () => {
      const { ok, latencyMs } = await api.checkHealth();
      setOnline(ok);
      setLatency(latencyMs);
      if (ok) {
        const m = await api.getModels();
        setModels(m);
      }
    };
    load();
  }, []);

  const featureTiles = [
    {
      id: 'chat' as PageId,
      title: 'Chat & Instruct',
      description: 'Stream low-latency responses with multi-turn context and configurable temperature/sampling.',
      icon: <ChatIcon size={22} className="text-teal-500" />,
      badge: 'Streaming SSE',
      badgeVariant: 'primary' as const,
    },
    {
      id: 'vision' as PageId,
      title: 'Multimodal Vision',
      description: 'Analyze images, perform OCR, and query visual features using CLIP ViT + LLaVA projector.',
      icon: <VisionIcon size={22} className="text-sky-500" />,
      badge: 'CLIP ViT-L/14',
      badgeVariant: 'info' as const,
    },
    {
      id: 'audio' as PageId,
      title: 'Whisper Audio Studio',
      description: 'Transcribe speech with live microphone waveform recording and multilingual translation.',
      icon: <AudioIcon size={22} className="text-purple-500" />,
      badge: 'Log-Mel FFT',
      badgeVariant: 'purple' as const,
    },
    {
      id: 'images' as PageId,
      title: 'Stable Diffusion',
      description: 'Generate high-fidelity imagery with prompt guidance, custom steps, and image history gallery.',
      icon: <ImageIcon size={22} className="text-pink-500" />,
      badge: 'Diffusion UNet',
      badgeVariant: 'warning' as const,
    },
    {
      id: 'embeddings' as PageId,
      title: 'Embeddings Matrix',
      description: 'Extract 384/768-d dense embeddings from BERT encoders and calculate cosine similarities.',
      icon: <EmbeddingsIcon size={22} className="text-emerald-500" />,
      badge: 'BERT Pooled',
      badgeVariant: 'success' as const,
    },
    {
      id: 'memory' as PageId,
      title: '3-Tier Memory Manager',
      description: 'Inspect real-time GPU VRAM, Host RAM, and NVMe Spool allocation with zero-copy transfers.',
      icon: <MemoryIcon size={22} className="text-amber-500" />,
      badge: 'Slab Allocator',
      badgeVariant: 'neutral' as const,
    },
  ];

  return (
    <div className="p-6 md:p-8 space-y-8 max-w-7xl mx-auto">
      {/* Hero Banner */}
      <div className="relative overflow-hidden rounded-3xl border border-border bg-card p-8 shadow-md">
        <div className="relative z-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="space-y-3 max-w-2xl">
            <div className="flex items-center gap-2">
              <Badge variant="primary" size="sm" pulse={online === true}>
                {online ? 'CORE INFERENCE RUNNING' : 'OFFLINE / STANDALONE'}
              </Badge>
              <Badge variant="neutral" size="sm">C++23 Native Engine</Badge>
            </div>
            <h2 className="text-2xl md:text-3xl font-extrabold text-foreground tracking-tight">
              High-Performance Multimodal AI Inference
            </h2>
            <p className="text-sm text-muted-foreground leading-relaxed">
              QorVix brings LLMs, CLIP vision towers, Whisper speech processing, Stable Diffusion image synthesis, and BERT embedding encoders into a single unified high-throughput C++23 runtime.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="primary"
              size="lg"
              leftIcon={<SparklesIcon size={18} />}
              onClick={() => onNavigate('chat')}
            >
              Launch Chat
            </Button>
            <Button
              variant="outline"
              size="lg"
              leftIcon={<PerformanceIcon size={18} />}
              onClick={() => onNavigate('performance')}
            >
              Run Benchmark
            </Button>
          </div>
        </div>
      </div>

      {/* Hardware & Live Status Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card glass className="p-5 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-muted-foreground">
            <span>RUNTIME LATENCY</span>
            <ZapIcon size={16} className="text-teal-500" />
          </div>
          <div className="text-2xl font-bold font-mono text-foreground">
            {online ? `${latency} ms` : '—'}
          </div>
          <div className="text-xs text-muted-foreground flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
            Port 2005 HTTP Engine
          </div>
        </Card>

        <Card glass className="p-5 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-muted-foreground">
            <span>ACTIVE MODELS</span>
            <ModelsIcon size={16} className="text-sky-500" />
          </div>
          <div className="text-2xl font-bold font-mono text-foreground">
            {models.length || 1}
          </div>
          <div className="text-xs text-muted-foreground truncate">
            {models[0]?.id || 'GGUF loaded'}
          </div>
        </Card>

        <Card glass className="p-5 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-muted-foreground">
            <span>BACKEND ACCELERATION</span>
            <CpuIcon size={16} className="text-purple-500" />
          </div>
          <div className="text-2xl font-bold font-mono text-teal-600 dark:text-teal-400">
            CUDA / Vulkan
          </div>
          <div className="text-xs text-muted-foreground">
            Auto-device detection active
          </div>
        </Card>

        <Card glass className="p-5 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-muted-foreground">
            <span>MEMORY TIERING</span>
            <LayersIcon size={16} className="text-amber-500" />
          </div>
          <div className="text-2xl font-bold font-mono text-foreground">
            3-Tier Spool
          </div>
          <div className="text-xs text-muted-foreground">
            VRAM ↔ RAM ↔ NVMe
          </div>
        </Card>
      </div>

      {/* Feature Navigation Tiles */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-bold text-foreground tracking-tight">
            Inference Studios & Diagnostics
          </h3>
          <span className="text-xs font-mono text-muted-foreground">SELECT A WORKSPACE</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {featureTiles.map((tile) => (
            <Card
              key={tile.id}
              glass
              hover
              onClick={() => onNavigate(tile.id)}
              className="cursor-pointer group p-6 space-y-4 transition-all duration-200"
            >
              <div className="flex items-start justify-between">
                <div className="p-3 rounded-2xl bg-secondary border border-border shadow-xs group-hover:scale-105 transition-transform">
                  {tile.icon}
                </div>
                <Badge variant={tile.badgeVariant} size="sm">
                  {tile.badge}
                </Badge>
              </div>

              <div className="space-y-1.5">
                <h4 className="text-base font-bold text-foreground group-hover:text-teal-500 transition-colors flex items-center justify-between">
                  <span>{tile.title}</span>
                  <span className="text-muted-foreground group-hover:text-teal-500 group-hover:translate-x-1 transition-all">→</span>
                </h4>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {tile.description}
                </p>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
};
