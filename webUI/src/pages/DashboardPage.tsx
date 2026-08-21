import React, { useEffect, useState } from 'react';
import { Card, CardContent } from '../components/ui/Card';
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
import { PageId } from '../components/layout/Sidebar';
import { api } from '../services/api';
import { ModelInfo } from '../types';

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
      icon: <ChatIcon size={24} className="text-teal-400" />,
      badge: 'Streaming SSE',
      badgeVariant: 'primary' as const,
      gradient: 'from-teal-500/15 via-teal-500/5 to-transparent',
    },
    {
      id: 'vision' as PageId,
      title: 'Multimodal Vision',
      description: 'Analyze images, perform OCR, and query visual features using CLIP ViT + LLaVA projector.',
      icon: <VisionIcon size={24} className="text-sky-400" />,
      badge: 'CLIP ViT-L/14',
      badgeVariant: 'info' as const,
      gradient: 'from-sky-500/15 via-sky-500/5 to-transparent',
    },
    {
      id: 'audio' as PageId,
      title: 'Whisper Audio Studio',
      description: 'Transcribe speech with live microphone waveform recording and multilingual translation.',
      icon: <AudioIcon size={24} className="text-purple-400" />,
      badge: 'Log-Mel FFT',
      badgeVariant: 'purple' as const,
      gradient: 'from-purple-500/15 via-purple-500/5 to-transparent',
    },
    {
      id: 'images' as PageId,
      title: 'Stable Diffusion',
      description: 'Generate high-fidelity imagery with prompt guidance, custom steps, and image history gallery.',
      icon: <ImageIcon size={24} className="text-pink-400" />,
      badge: 'Diffusion UNet',
      badgeVariant: 'warning' as const,
      gradient: 'from-pink-500/15 via-pink-500/5 to-transparent',
    },
    {
      id: 'embeddings' as PageId,
      title: 'Embeddings Matrix',
      description: 'Extract 384/768-d dense embeddings from BERT encoders and calculate cosine similarities.',
      icon: <EmbeddingsIcon size={24} className="text-emerald-400" />,
      badge: 'BERT Pooled',
      badgeVariant: 'success' as const,
      gradient: 'from-emerald-500/15 via-emerald-500/5 to-transparent',
    },
    {
      id: 'memory' as PageId,
      title: '3-Tier Memory Manager',
      description: 'Inspect real-time GPU VRAM, Host RAM, and NVMe Spool allocation with zero-copy transfers.',
      icon: <MemoryIcon size={24} className="text-amber-400" />,
      badge: 'Slab Allocator',
      badgeVariant: 'neutral' as const,
      gradient: 'from-amber-500/15 via-amber-500/5 to-transparent',
    },
  ];

  return (
    <div className="p-6 md:p-8 space-y-8 max-w-7xl mx-auto">
      {/* Hero Banner */}
      <div className="relative overflow-hidden rounded-3xl border border-teal-500/30 bg-gradient-to-r from-slate-900 via-slate-900/90 to-teal-950/40 p-8 shadow-2xl shadow-teal-500/10">
        <div className="absolute top-0 right-0 -mt-8 -mr-8 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="space-y-3 max-w-2xl">
            <div className="flex items-center gap-2">
              <Badge variant="primary" size="sm" pulse={online === true}>
                {online ? 'CORE INFERENCE RUNNING' : 'OFFLINE / STANDALONE'}
              </Badge>
              <Badge variant="neutral" size="sm">C++23 Native Engine</Badge>
            </div>
            <h2 className="text-2xl md:text-3xl font-extrabold text-slate-100 tracking-tight">
              High-Performance Multimodal AI Inference
            </h2>
            <p className="text-sm text-slate-300 leading-relaxed font-sans">
              QorVix brings LLMs, CLIP vision towers, Whisper speech processing, Stable Diffusion image synthesis, and BERT embedding encoders into a single unified high-throughput C++23 runtime.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="glow"
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
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>RUNTIME LATENCY</span>
            <ZapIcon size={16} className="text-teal-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-slate-100">
            {online ? `${latency} ms` : '—'}
          </div>
          <div className="text-xs text-slate-400 flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            Port 2005 HTTP Engine
          </div>
        </Card>

        <Card glass className="p-5 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>ACTIVE MODELS</span>
            <ModelsIcon size={16} className="text-sky-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-slate-100">
            {models.length || 1}
          </div>
          <div className="text-xs text-slate-400 truncate">
            {models[0]?.id || 'GGUF loaded'}
          </div>
        </Card>

        <Card glass className="p-5 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>BACKEND ACCELERATION</span>
            <CpuIcon size={16} className="text-purple-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-teal-300">
            CUDA / Vulkan
          </div>
          <div className="text-xs text-slate-400">
            Auto-device detection active
          </div>
        </Card>

        <Card glass className="p-5 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>MEMORY TIERING</span>
            <LayersIcon size={16} className="text-amber-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-slate-100">
            3-Tier Spool
          </div>
          <div className="text-xs text-slate-400">
            VRAM ↔ RAM ↔ NVMe
          </div>
        </Card>
      </div>

      {/* Feature Navigation Tiles */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-bold text-slate-100 tracking-tight">
            Inference Studios & Diagnostics
          </h3>
          <span className="text-xs font-mono text-slate-400">SELECT A WORKSPACE</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {featureTiles.map((tile) => (
            <Card
              key={tile.id}
              glass
              hover
              onClick={() => onNavigate(tile.id)}
              className="cursor-pointer group relative overflow-hidden p-6 space-y-4 border-slate-800/80 transition-all duration-200"
            >
              <div className={`absolute inset-0 bg-gradient-to-br ${tile.gradient} opacity-50 group-hover:opacity-100 transition-opacity`} />
              
              <div className="relative z-10 flex items-start justify-between">
                <div className="p-3 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-md group-hover:border-slate-700 transition-colors">
                  {tile.icon}
                </div>
                <Badge variant={tile.badgeVariant} size="sm">
                  {tile.badge}
                </Badge>
              </div>

              <div className="relative z-10 space-y-1.5">
                <h4 className="text-base font-bold text-slate-100 group-hover:text-teal-300 transition-colors flex items-center justify-between">
                  <span>{tile.title}</span>
                  <span className="text-slate-500 group-hover:text-teal-400 group-hover:translate-x-1 transition-all">→</span>
                </h4>
                <p className="text-xs text-slate-400 leading-relaxed">
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
