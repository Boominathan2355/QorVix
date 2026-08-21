import React, { useState, useEffect } from 'react';
import { Sidebar, PageId } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { ToastProvider } from './components/ui/Toast';
import { DashboardPage } from './pages/DashboardPage';
import { ChatPage } from './pages/ChatPage';
import { VisionPage } from './pages/VisionPage';
import { AudioPage } from './pages/AudioPage';
import { ImageGenPage } from './pages/ImageGenPage';
import { EmbeddingsPage } from './pages/EmbeddingsPage';
import { ModelsPage } from './pages/ModelsPage';
import { MemoryPage } from './pages/MemoryPage';
import { PerformancePage } from './pages/PerformancePage';
import { MetricsPage } from './pages/MetricsPage';
import { SettingsPage } from './pages/SettingsPage';
import { api } from './services/api';
import { ModelInfo } from './types';

export const App: React.FC = () => {
  const [activePage, setActivePage] = useState<PageId>('dashboard');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('qorvix-default-model');

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

  const renderCurrentPage = () => {
    switch (activePage) {
      case 'dashboard':
        return <DashboardPage onNavigate={setActivePage} />;
      case 'chat':
        return <ChatPage models={models} selectedModel={selectedModel} />;
      case 'vision':
        return <VisionPage models={models} selectedModel={selectedModel} />;
      case 'audio':
        return <AudioPage />;
      case 'images':
        return <ImageGenPage />;
      case 'embeddings':
        return <EmbeddingsPage />;
      case 'models':
        return <ModelsPage />;
      case 'memory':
        return <MemoryPage />;
      case 'performance':
        return <PerformancePage models={models} selectedModel={selectedModel} />;
      case 'metrics':
        return <MetricsPage />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <DashboardPage onNavigate={setActivePage} />;
    }
  };

  return (
    <ToastProvider>
      <div className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100 font-sans">
        {/* Navigation Sidebar */}
        <Sidebar
          activePage={activePage}
          onNavigate={setActivePage}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        />

        {/* Main View Area */}
        <div className="flex-1 flex flex-col h-full overflow-hidden">
          <Header
            activePage={activePage}
            models={models}
            selectedModel={selectedModel}
            onSelectModel={setSelectedModel}
            onNavigate={setActivePage}
          />

          <main className="flex-1 overflow-y-auto bg-gradient-to-b from-slate-950 to-slate-900/40">
            {renderCurrentPage()}
          </main>
        </div>
      </div>
    </ToastProvider>
  );
};
export default App;
