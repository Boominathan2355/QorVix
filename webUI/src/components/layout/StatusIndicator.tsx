import React, { useEffect, useState } from 'react';
import { api } from '../../services/api';

export const StatusIndicator: React.FC = () => {
  const [online, setOnline] = useState<boolean | null>(null);
  const [latency, setLatency] = useState<number>(0);

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      const { ok, latencyMs } = await api.checkHealth();
      if (mounted) {
        setOnline(ok);
        setLatency(latencyMs);
      }
    };
    check();
    const interval = setInterval(check, 5000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900/90 border border-slate-800 text-xs font-mono select-none">
      <span className="relative flex h-2 w-2">
        {online === true && (
          <>
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
          </>
        )}
        {online === false && (
          <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500" />
        )}
        {online === null && (
          <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500 animate-pulse" />
        )}
      </span>
      <span className={online ? 'text-emerald-400 font-medium' : online === false ? 'text-red-400 font-medium' : 'text-slate-400'}>
        {online === true ? `Online (${latency}ms)` : online === false ? 'Disconnected' : 'Checking...'}
      </span>
    </div>
  );
};
