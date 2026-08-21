import React from 'react';
import { StatusIndicator } from './StatusIndicator';
import { PageId } from './Sidebar';
import { ModelInfo } from '../../types';

interface HeaderProps {
  activePage: PageId;
  models: ModelInfo[];
  selectedModel: string;
  onSelectModel: (model: string) => void;
  onNavigate: (page: PageId) => void;
}

export const Header: React.FC<HeaderProps> = ({
  activePage,
  models,
  selectedModel,
  onSelectModel,
}) => {
  const titles: Record<PageId, { title: string; subtitle: string }> = {
    dashboard: { title: 'Dashboard Overview', subtitle: 'Hardware status, loaded architectures & real-time telemetry' },
    chat: { title: 'Chat & Reasoning Studio', subtitle: 'Low-latency streaming inference, KV-cache sharing & prompt playground' },
    vision: { title: 'Multimodal Vision', subtitle: 'CLIP ViT patch encoding & LLaVA visual question answering' },
    audio: { title: 'Speech & Audio Studio', subtitle: 'Whisper log-mel encoder-decoder transcription & translation' },
    images: { title: 'Stable Diffusion Generator', subtitle: 'On-device text-to-image synthesis & prompt laboratory' },
    embeddings: { title: 'Embeddings & Similarity Matrix', subtitle: 'BERT encoder representations & vector cosine metrics' },
    models: { title: 'Model Registry & GGUF Inspector', subtitle: 'Inspect tensor layouts, quantization types & layer architectures' },
    memory: { title: 'Tiered Memory & VRAM Visualizer', subtitle: '3-tier memory allocation (GPU VRAM, Host RAM, NVMe Spool)' },
    performance: { title: 'Throughput & Benchmarks', subtitle: 'Real-time token speeds (tok/s), latency distributions & stress testing' },
    metrics: { title: 'Prometheus Exporter Metrics', subtitle: 'Raw metrics stream from port 2009 for Grafana dashboards' },
    settings: { title: 'Engine Settings & Ports', subtitle: 'Configure endpoint ports, sampling defaults & theme' },
  };

  const current = titles[activePage] || { title: 'QorVix Studio', subtitle: 'Inference Engine' };

  return (
    <header className="h-16 border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-xl px-6 flex items-center justify-between shrink-0 z-20 select-none">
      <div className="flex flex-col">
        <h1 className="text-base font-bold text-slate-100 tracking-tight flex items-center gap-2">
          {current.title}
        </h1>
        <p className="text-xs text-slate-400 font-mono hidden md:block">
          {current.subtitle}
        </p>
      </div>

      <div className="flex items-center gap-4">
        {/* Model Selector Pill */}
        {models.length > 0 && (
          <div className="relative hidden sm:flex items-center">
            <select
              value={selectedModel}
              onChange={(e) => onSelectModel(e.target.value)}
              className="appearance-none bg-slate-900/90 border border-slate-800 hover:border-slate-700 text-xs font-mono text-teal-300 font-semibold px-3 py-1.5 pr-8 rounded-xl focus:outline-none focus:border-teal-500/50 cursor-pointer shadow-sm transition-colors"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id} className="bg-slate-900 text-slate-100">
                  {m.id}
                </option>
              ))}
            </select>
            <div className="absolute right-2.5 pointer-events-none text-slate-400">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </div>
          </div>
        )}

        {/* Live Server Status Indicator */}
        <StatusIndicator />
      </div>
    </header>
  );
};
