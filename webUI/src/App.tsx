import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { ToastProvider } from './components/ui/Toast';
import { Modal } from './components/ui/Modal';
import { ChatPage } from './pages/ChatPage';
import { SettingsPage } from './pages/SettingsPage';
import { api } from './services/api';
import { ModelInfo, ChatSession } from './types';

export const App: React.FC = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('google/gemma-4-E2B');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Dark / Light Theme Manager
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const saved = localStorage.getItem('qorvix_theme');
    if (saved === 'light' || saved === 'dark') return saved;
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
      root.classList.remove('light');
    } else {
      root.classList.add('light');
      root.classList.remove('dark');
    }
    localStorage.setItem('qorvix_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  // Chat Sessions state
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
  const activeSession = sessions.find((s) => s.id === activeSessionId) || sessions[0];

  useEffect(() => {
    localStorage.setItem('qorvix_chat_sessions', JSON.stringify(sessions));
  }, [sessions]);

  useEffect(() => {
    const init = async () => {
      const list = await api.getModels();
      setModels(list);
      if (list.length > 0) {
        setSelectedModel(list[0].id);
      }
    };
    init();
  }, []);

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
    const remaining = sessions.filter((s) => s.id !== id);
    if (remaining.length === 0) {
      handleNewSession();
    } else {
      setSessions(remaining);
      if (activeSessionId === id) {
        setActiveSessionId(remaining[0].id);
      }
    }
  };

  const updateActiveSession = (updater: (s: ChatSession) => ChatSession) => {
    setSessions((prev) =>
      prev.map((s) => (s.id === activeSessionId ? updater(s) : s))
    );
  };

  return (
    <ToastProvider>
      <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground font-sans selection:bg-teal-500/30 selection:text-teal-200">
        {/* Sleek Collapsible Conversations Sidebar */}
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelectSession={setActiveSessionId}
          onNewSession={handleNewSession}
          onDeleteSession={handleDeleteSession}
          onOpenSettings={() => setIsSettingsOpen(true)}
        />

        {/* Main Canvas: Unified All-in-One Multimodal Omni-Dashboard */}
        <div className="flex-1 flex flex-col h-full overflow-hidden bg-background">
          <Header
            models={models}
            selectedModel={selectedModel}
            onSelectModel={setSelectedModel}
            theme={theme}
            onToggleTheme={toggleTheme}
            onOpenSettings={() => setIsSettingsOpen(true)}
          />

          <main className="flex-1 overflow-hidden flex flex-col bg-background relative">
            <ChatPage
              models={models}
              selectedModel={selectedModel}
              activeSession={activeSession}
              onUpdateSession={updateActiveSession}
            />
          </main>
        </div>

        {/* Settings & Port Allocations Modal */}
        <Modal
          isOpen={isSettingsOpen}
          onClose={() => setIsSettingsOpen(false)}
          title="Settings & Engine Ports"
          maxWidth="2xl"
        >
          <SettingsPage />
        </Modal>
      </div>
    </ToastProvider>
  );
};
export default App;
