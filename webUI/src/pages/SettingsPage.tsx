import React, { useState } from 'react';
import { Card, CardTitle } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Switch } from '../components/ui/Switch';
import { Badge } from '../components/ui/Badge';
import { useToast } from '../components/ui/Toast';
import { SettingsIcon, TrashIcon } from '../components/icons/Icons';
import { api } from '../services/api';

export const SettingsPage: React.FC = () => {
  const { success: toastSuccess, error: toastError } = useToast();
  const [baseUrl, setBaseUrl] = useState(() => localStorage.getItem('qorvix_base_url') || 'http://localhost:2005');
  const [metricsUrl, setMetricsUrl] = useState(() => localStorage.getItem('qorvix_metrics_url') || 'http://localhost:2009');
  const [enableStreaming, setEnableStreaming] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);
  const [isTesting, setIsTesting] = useState(false);

  const handleSave = () => {
    localStorage.setItem('qorvix_base_url', baseUrl);
    localStorage.setItem('qorvix_metrics_url', metricsUrl);
    api.setBaseUrl(baseUrl);
    api.setMetricsUrl(metricsUrl);
    toastSuccess('Configuration saved successfully!');
  };

  const handleTestConnection = async () => {
    setIsTesting(true);
    api.setBaseUrl(baseUrl);
    const { ok, latencyMs } = await api.checkHealth();
    setIsTesting(false);
    if (ok) {
      toastSuccess(`Connected to QorVix server! Latency: ${latencyMs}ms`);
    } else {
      toastError(`Failed to reach ${baseUrl}. Ensure 'qorvix serve' is running.`, 'Connection Failed');
    }
  };

  const handleClearData = () => {
    if (window.confirm('Are you sure you want to clear all local chat history and gallery images?')) {
      localStorage.clear();
      toastSuccess('Local storage cache cleared.');
      setTimeout(() => window.location.reload(), 1000);
    }
  };

  const portAllocations = [
    { port: 2005, service: 'QorVix Runtime (Inference)', status: 'Active (qorvix serve)', description: 'OpenAI-compatible HTTP chat, vision, audio & embeddings' },
    { port: 2006, service: 'QorVix Gateway', status: 'Reserved', description: 'Multi-node / model routing, rate limiting & auth proxy' },
    { port: 2007, service: 'QorVix Dashboard (Web UI)', status: 'Active (Vite dev/preview)', description: 'React/TS/Tailwind dashboard interface' },
    { port: 2008, service: 'QorVix Admin API', status: 'Reserved', description: 'Model hot-reload & scheduler control plane' },
    { port: 2009, service: 'QorVix Metrics', status: 'Active (/metrics)', description: 'Prometheus / OpenMetrics scrape target for Grafana' },
    { port: 2010, service: 'QorVix gRPC', status: 'Reserved', description: 'High-speed binary tensor streaming RPC' },
  ];

  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto space-y-6">
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <Badge variant="primary" size="sm">System Configuration</Badge>
          <Badge variant="neutral" size="sm">Single Source of Truth</Badge>
        </div>
        <h2 className="text-2xl font-bold text-foreground tracking-tight flex items-center gap-2">
          <SettingsIcon size={24} className="text-teal-500" />
          Settings & Port Allocations
        </h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
        {/* Connection Settings */}
        <div className="md:col-span-6 space-y-5">
          <Card glass className="p-6 space-y-5">
            <CardTitle className="text-sm font-semibold text-foreground">
              API Connection Endpoints
            </CardTitle>

            <div className="space-y-4">
              <Input
                label="Inference Server URL (Port 2005)"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="http://localhost:2005"
              />

              <Input
                label="Prometheus Metrics URL (Port 2009)"
                value={metricsUrl}
                onChange={(e) => setMetricsUrl(e.target.value)}
                placeholder="http://localhost:2009"
              />

              <div className="flex items-center gap-3 pt-2">
                <Button
                  variant="primary"
                  size="md"
                  onClick={handleSave}
                >
                  Save Settings
                </Button>
                <Button
                  variant="outline"
                  size="md"
                  loading={isTesting}
                  onClick={handleTestConnection}
                >
                  Test Connection
                </Button>
              </div>
            </div>
          </Card>

          {/* Preferences */}
          <Card glass className="p-6 space-y-4">
            <CardTitle className="text-sm font-semibold text-foreground">
              UI Preferences
            </CardTitle>

            <div className="space-y-4">
              <Switch
                checked={enableStreaming}
                onChange={setEnableStreaming}
                label="Enable SSE Streaming"
                description="Stream tokens live as they are synthesized by the engine"
              />
              <Switch
                checked={autoScroll}
                onChange={setAutoScroll}
                label="Auto-scroll Message Stream"
                description="Keep latest tokens in view during generation"
              />
            </div>

            <div className="pt-4 border-t border-border">
              <Button
                variant="danger"
                size="sm"
                leftIcon={<TrashIcon size={14} />}
                onClick={handleClearData}
              >
                Clear Local Chat & Gallery Data
              </Button>
            </div>
          </Card>
        </div>

        {/* Port Allocations Reference */}
        <div className="md:col-span-6 space-y-5">
          <Card glass className="p-6 space-y-4">
            <CardTitle className="text-sm font-semibold text-foreground flex items-center justify-between">
              <span>Port Allocation Registry</span>
              <span className="text-xs font-mono text-teal-600 dark:text-teal-400 font-normal">2005-2010 BLOCK</span>
            </CardTitle>

            <p className="text-xs text-muted-foreground font-sans leading-relaxed">
              QorVix reserves the contiguous 2005–2010 range to avoid firewall and routing conflicts.
            </p>

            <div className="space-y-2.5">
              {portAllocations.map((p) => (
                <div
                  key={p.port}
                  className="p-3 rounded-xl bg-secondary border border-border space-y-1 text-xs font-mono"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-teal-600 dark:text-teal-400">{p.port} • {p.service}</span>
                    <span className="text-[10px] text-muted-foreground bg-card px-2 py-0.5 rounded border border-border">{p.status}</span>
                  </div>
                  <p className="text-[11px] text-muted-foreground font-sans">{p.description}</p>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};
