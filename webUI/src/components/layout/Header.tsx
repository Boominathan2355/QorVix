import React from 'react';
import { StatusIndicator } from './StatusIndicator';
import { ModelInfo } from '../../types';
import {
  SunIcon,
  MoonIcon,
  SettingsIcon,
  BotIcon,
} from '../icons/Icons';

interface HeaderProps {
  models: ModelInfo[];
  selectedModel: string;
  onSelectModel: (model: string) => void;
  theme: 'dark' | 'light';
  onToggleTheme: () => void;
  onOpenSettings: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  models,
  selectedModel,
  onSelectModel,
  theme,
  onToggleTheme,
  onOpenSettings,
}) => {
  return (
    <header className="h-14 border-b border-border/80 bg-card/60 backdrop-blur-xl px-4 md:px-6 flex items-center justify-between shrink-0 z-20 select-none transition-colors">
      {/* Left: Model Selector & Server Status */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl border border-border bg-card shadow-xs">
          <BotIcon size={16} className="text-teal-500 shrink-0" />
          <select
            value={selectedModel}
            onChange={(e) => onSelectModel(e.target.value)}
            className="appearance-none bg-transparent text-xs font-semibold font-mono text-foreground focus:outline-none cursor-pointer pr-4"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id} className="bg-popover text-foreground">
                {m.id}
              </option>
            ))}
          </select>
        </div>

        <StatusIndicator />
      </div>

      {/* Right: Theme Toggle & Settings */}
      <div className="flex items-center gap-2">
        <button
          onClick={onToggleTheme}
          className="p-2 rounded-xl border border-border bg-card hover:bg-secondary text-muted-foreground hover:text-foreground transition-all shadow-xs"
          title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
        >
          {theme === 'dark' ? <SunIcon size={16} className="text-amber-400" /> : <MoonIcon size={16} className="text-indigo-500" />}
        </button>

        <button
          onClick={onOpenSettings}
          className="p-2 rounded-xl border border-border bg-card hover:bg-secondary text-muted-foreground hover:text-foreground transition-all shadow-xs"
          title="Settings & Ports"
        >
          <SettingsIcon size={16} />
        </button>
      </div>
    </header>
  );
};
