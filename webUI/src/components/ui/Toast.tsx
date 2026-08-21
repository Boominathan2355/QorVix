import React, { createContext, useContext, useState, useCallback } from 'react';

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning';
  title?: string;
  message: string;
  duration?: number;
}

interface ToastContextValue {
  toast: (options: Omit<Toast, 'id'>) => void;
  success: (message: string, title?: string) => void;
  error: (message: string, title?: string) => void;
  info: (message: string, title?: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const toast = useCallback(
    ({ type, title, message, duration = 4000 }: Omit<Toast, 'id'>) => {
      const id = Math.random().toString(36).substring(2, 9);
      setToasts((prev) => [...prev, { id, type, title, message, duration }]);
      if (duration > 0) {
        setTimeout(() => removeToast(id), duration);
      }
    },
    [removeToast]
  );

  const success = useCallback((message: string, title?: string) => toast({ type: 'success', title, message }), [toast]);
  const error = useCallback((message: string, title?: string) => toast({ type: 'error', title, message, duration: 6000 }), [toast]);
  const info = useCallback((message: string, title?: string) => toast({ type: 'info', title, message }), [toast]);

  const borderColors = {
    success: 'border-emerald-500/50 bg-emerald-950/90 text-emerald-100',
    error: 'border-red-500/50 bg-red-950/90 text-red-100',
    warning: 'border-amber-500/50 bg-amber-950/90 text-amber-100',
    info: 'border-sky-500/50 bg-sky-950/90 text-sky-100',
  };

  return (
    <ToastContext.Provider value={{ toast, success, error, info }}>
      {children}
      <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2.5 max-w-sm pointer-events-none">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`pointer-events-auto p-4 rounded-xl border backdrop-blur-xl shadow-2xl transition-all duration-200 animate-in slide-in-from-bottom-5 ${borderColors[t.type]}`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="space-y-1">
                {t.title && <h4 className="text-xs font-bold uppercase tracking-wider">{t.title}</h4>}
                <p className="text-sm font-medium">{t.message}</p>
              </div>
              <button
                onClick={() => removeToast(t.id)}
                className="opacity-70 hover:opacity-100 p-0.5 rounded transition-opacity"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
};

export const useToast = () => {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return ctx;
};
