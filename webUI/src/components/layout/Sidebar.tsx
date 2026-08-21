import React from 'react';
import {
  ChatIcon,
  SettingsIcon,
  PlusIcon,
  TrashIcon,
} from '../icons/Icons';
import { ChatSession } from '../../types';

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  sessions: ChatSession[];
  activeSessionId: string;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string, e: React.MouseEvent) => void;
  onOpenSettings: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  collapsed,
  onToggleCollapse,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onOpenSettings,
}) => {
  return (
    <aside
      className={`relative flex flex-col h-screen bg-card border-r border-border transition-all duration-300 z-30 select-none ${
        collapsed ? 'w-16' : 'w-64'
      }`}
    >
      {/* Brand Header */}
      <div className="flex items-center justify-between px-4 h-14 border-b border-border">
        <div
          className="flex items-center gap-2.5 overflow-hidden cursor-pointer"
          onClick={onNewSession}
        >
          <div className="h-8 w-8 rounded-xl bg-gradient-to-br from-teal-400 to-teal-600 flex items-center justify-center shadow-md shadow-teal-500/20 shrink-0 text-slate-950 font-mono font-extrabold text-sm">
            Q
          </div>
          {!collapsed && (
            <span className="font-bold text-foreground text-sm tracking-tight">
              QorVix <span className="text-[10px] text-teal-500 font-mono">STUDIO</span>
            </span>
          )}
        </div>
        <button
          onClick={onToggleCollapse}
          className="text-muted-foreground hover:text-foreground p-1 rounded-lg hover:bg-secondary transition-colors"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className={`transition-transform duration-200 ${collapsed ? 'rotate-180' : ''}`}
          >
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>
      </div>

      {/* New Chat CTA */}
      <div className="p-3">
        <button
          onClick={onNewSession}
          className={`w-full flex items-center justify-center gap-2 py-2 px-3 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-semibold text-xs transition-all shadow-sm shadow-teal-500/20 ${
            collapsed ? 'p-2' : ''
          }`}
          title="New Chat"
        >
          <PlusIcon size={15} />
          {!collapsed && <span>New Chat</span>}
        </button>
      </div>

      {/* Conversations History */}
      <div className="flex-1 overflow-y-auto px-2.5 space-y-1">
        {!collapsed && (
          <div className="text-[10px] font-mono font-bold uppercase tracking-wider text-muted-foreground px-2 pt-1 pb-1">
            Conversations
          </div>
        )}
        <div className="space-y-0.5">
          {sessions.map((s) => {
            const isActive = s.id === activeSessionId;
            return (
              <div
                key={s.id}
                onClick={() => onSelectSession(s.id)}
                className={`group flex items-center justify-between p-2 rounded-xl cursor-pointer text-xs transition-all ${
                  isActive
                    ? 'bg-secondary text-foreground font-semibold border border-border'
                    : 'text-muted-foreground hover:text-foreground hover:bg-secondary/50'
                }`}
                title={collapsed ? s.title : undefined}
              >
                <div className="flex items-center gap-2 truncate pr-1">
                  <ChatIcon size={14} className={isActive ? 'text-teal-500' : 'text-muted-foreground'} />
                  {!collapsed && <span className="truncate">{s.title || 'New Conversation'}</span>}
                </div>
                {!collapsed && (
                  <button
                    onClick={(e) => onDeleteSession(s.id, e)}
                    className="opacity-0 group-hover:opacity-100 hover:text-red-500 p-0.5 rounded transition-opacity"
                    title="Delete"
                  >
                    <TrashIcon size={12} />
                  </button>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Bottom Settings Button */}
      <div className="p-3 border-t border-border bg-card/40">
        <button
          onClick={onOpenSettings}
          className="w-full flex items-center gap-2.5 p-2 rounded-xl text-xs text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
          title="Settings & Ports"
        >
          <SettingsIcon size={16} />
          {!collapsed && <span className="truncate font-medium">Settings & Ports</span>}
        </button>
      </div>
    </aside>
  );
};
