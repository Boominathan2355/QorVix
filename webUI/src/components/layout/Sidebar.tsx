import React from 'react';
import {
  DashboardIcon,
  ChatIcon,
  VisionIcon,
  AudioIcon,
  ImageIcon,
  EmbeddingsIcon,
  ModelsIcon,
  MemoryIcon,
  PerformanceIcon,
  MetricsIcon,
  SettingsIcon,
} from '../icons/Icons';

export type PageId =
  | 'dashboard'
  | 'chat'
  | 'vision'
  | 'audio'
  | 'images'
  | 'embeddings'
  | 'models'
  | 'memory'
  | 'performance'
  | 'metrics'
  | 'settings';

interface SidebarProps {
  activePage: PageId;
  onNavigate: (page: PageId) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activePage,
  onNavigate,
  collapsed,
  onToggleCollapse,
}) => {
  const navItems: { id: PageId; label: string; icon: React.ReactNode; group?: string }[] = [
    { id: 'dashboard', label: 'Overview', icon: <DashboardIcon size={18} />, group: 'Platform' },
    { id: 'chat', label: 'Chat & Instruct', icon: <ChatIcon size={18} />, group: 'Generation' },
    { id: 'vision', label: 'Multimodal Vision', icon: <VisionIcon size={18} />, group: 'Generation' },
    { id: 'audio', label: 'Speech & Whisper', icon: <AudioIcon size={18} />, group: 'Generation' },
    { id: 'images', label: 'Stable Diffusion', icon: <ImageIcon size={18} />, group: 'Generation' },
    { id: 'embeddings', label: 'Embeddings / BERT', icon: <EmbeddingsIcon size={18} />, group: 'Inference' },
    { id: 'models', label: 'Model Registry', icon: <ModelsIcon size={18} />, group: 'Inference' },
    { id: 'memory', label: 'Memory & Tiers', icon: <MemoryIcon size={18} />, group: 'System' },
    { id: 'performance', label: 'Speed & Benchmarks', icon: <PerformanceIcon size={18} />, group: 'System' },
    { id: 'metrics', label: 'Prometheus Metrics', icon: <MetricsIcon size={18} />, group: 'System' },
    { id: 'settings', label: 'Settings', icon: <SettingsIcon size={18} />, group: 'System' },
  ];

  return (
    <aside
      className={`relative flex flex-col h-screen bg-slate-950/90 backdrop-blur-2xl border-r border-slate-800/80 transition-all duration-300 z-30 select-none ${
        collapsed ? 'w-20' : 'w-64'
      }`}
    >
      {/* Brand Header */}
      <div className="flex items-center justify-between px-5 h-16 border-b border-slate-800/80">
        <div className="flex items-center gap-3 overflow-hidden cursor-pointer" onClick={() => onNavigate('dashboard')}>
          <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-teal-400 to-emerald-600 flex items-center justify-center shadow-lg shadow-teal-500/25 shrink-0 border border-teal-300/30">
            <span className="font-extrabold text-slate-950 font-mono text-base tracking-tighter">Q</span>
          </div>
          {!collapsed && (
            <div className="flex flex-col">
              <span className="font-bold text-slate-100 text-sm tracking-tight flex items-center gap-1.5">
                QorVix <span className="text-[10px] bg-teal-950 text-teal-400 border border-teal-800/60 px-1.5 py-0.2 rounded font-mono font-medium">STUDIO</span>
              </span>
              <span className="text-[11px] text-slate-400 font-mono">v1.0 • C++23</span>
            </div>
          )}
        </div>
        <button
          onClick={onToggleCollapse}
          className="text-slate-400 hover:text-slate-100 p-1.5 rounded-lg hover:bg-slate-800/70 transition-colors"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={`transition-transform duration-200 ${collapsed ? 'rotate-180' : ''}`}>
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>
      </div>

      {/* Navigation List */}
      <div className="flex-1 overflow-y-auto py-4 px-3 space-y-1">
        {navItems.map((item, idx) => {
          const isActive = activePage === item.id;
          const showGroup =
            !collapsed && (idx === 0 || navItems[idx - 1].group !== item.group);

          return (
            <React.Fragment key={item.id}>
              {showGroup && item.group && (
                <div className="text-[10px] font-bold font-mono uppercase tracking-wider text-slate-500 px-3 pt-3 pb-1">
                  {item.group}
                </div>
              )}
              <button
                onClick={() => onNavigate(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-150 relative group ${
                  isActive
                    ? 'bg-teal-500/10 text-teal-300 font-semibold border border-teal-500/20 shadow-sm shadow-teal-500/5'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-900/60 border border-transparent'
                }`}
                title={collapsed ? item.label : undefined}
              >
                <div className={`shrink-0 transition-colors ${isActive ? 'text-teal-400' : 'text-slate-400 group-hover:text-slate-200'}`}>
                  {item.icon}
                </div>
                {!collapsed && (
                  <span className="truncate text-left text-xs tracking-wide">
                    {item.label}
                  </span>
                )}
                {isActive && (
                  <div className="absolute left-0 top-2 bottom-2 w-1 bg-teal-400 rounded-r-full shadow-[0_0_8px_rgba(45,212,191,0.8)]" />
                )}
              </button>
            </React.Fragment>
          );
        })}
      </div>

      {/* Bottom Info */}
      <div className="p-3 border-t border-slate-800/80 bg-slate-950/40">
        {!collapsed ? (
          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 flex flex-col gap-1.5">
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-slate-400">Port 2005</span>
              <span className="text-teal-400 font-semibold">Inference</span>
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-slate-400">Port 2007</span>
              <span className="text-teal-400 font-semibold">Dashboard</span>
            </div>
          </div>
        ) : (
          <div className="flex justify-center text-[11px] font-mono text-teal-400 font-bold">
            2005
          </div>
        )}
      </div>
    </aside>
  );
};
