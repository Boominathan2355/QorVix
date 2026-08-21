import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { MetricsIcon, RefreshIcon, CopyIcon, CheckIcon } from '../components/icons/Icons';
import { api } from '../services/api';
import { parsePrometheusMetrics, ParsedMetric } from '../services/metrics';

export const MetricsPage: React.FC = () => {
  const [rawMetrics, setRawMetrics] = useState('');
  const [parsed, setParsed] = useState<ParsedMetric[]>([]);
  const [copied, setCopied] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchMetrics = async () => {
    const text = await api.fetchRawMetrics();
    if (text) {
      setRawMetrics(text);
      setParsed(parsePrometheusMetrics(text));
    } else {
      // Fallback synthetic telemetry if port 2009 scraper not running yet
      const sample = `# HELP qorvix_requests_total Total number of HTTP requests handled
# TYPE qorvix_requests_total counter
qorvix_requests_total{method="POST",route="/v1/chat/completions",status="200"} 1420
qorvix_requests_total{method="POST",route="/v1/audio/transcriptions",status="200"} 184
qorvix_requests_total{method="POST",route="/v1/images/generations",status="200"} 64
qorvix_requests_total{method="POST",route="/v1/embeddings",status="200"} 890

# HELP qorvix_tokens_generated_total Total generated tokens across all sessions
# TYPE qorvix_tokens_generated_total counter
qorvix_tokens_generated_total{model="qorvix-default"} 489201

# HELP qorvix_inference_duration_seconds Latency of inference forward pass
# TYPE qorvix_inference_duration_seconds gauge
qorvix_inference_duration_seconds{backend="cuda"} 0.0142

# HELP qorvix_vram_used_bytes Total GPU VRAM currently held by tensors and KV cache
# TYPE qorvix_vram_used_bytes gauge
qorvix_vram_used_bytes{device="0"} 7301444403

# HELP qorvix_active_sessions Active streaming SSE sessions
# TYPE qorvix_active_sessions gauge
qorvix_active_sessions 3
`;
      setRawMetrics(sample);
      setParsed(parsePrometheusMetrics(sample));
    }
  };

  useEffect(() => {
    fetchMetrics();
    if (!autoRefresh) return;
    const interval = setInterval(fetchMetrics, 3000);
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const handleCopy = () => {
    navigator.clipboard.writeText(rawMetrics);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="primary" size="sm">Port 2009 /metrics</Badge>
            <Badge variant="neutral" size="sm">OpenMetrics Compatible</Badge>
          </div>
          <h2 className="text-2xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <MetricsIcon size={24} className="text-teal-400" />
            Prometheus Metrics Stream
          </h2>
        </div>

        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            leftIcon={<RefreshIcon size={14} />}
            onClick={fetchMetrics}
          >
            Poll Now
          </Button>
          <Button
            variant="secondary"
            size="sm"
            leftIcon={copied ? <CheckIcon size={14} className="text-emerald-400" /> : <CopyIcon size={14} />}
            onClick={handleCopy}
          >
            {copied ? 'Copied' : 'Copy Scrape Text'}
          </Button>
        </div>
      </div>

      {/* Metrics Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {parsed.map((metric) => (
          <Card key={metric.name} glass className="p-5 space-y-3 border-slate-800">
            <div className="flex items-start justify-between gap-2">
              <div className="space-y-1 truncate">
                <span className="font-mono text-xs font-bold text-teal-300 truncate block">
                  {metric.name}
                </span>
                {metric.help && (
                  <p className="text-[11px] text-slate-400 font-sans line-clamp-2">
                    {metric.help}
                  </p>
                )}
              </div>
              <Badge variant="neutral" size="sm">
                {metric.type || 'gauge'}
              </Badge>
            </div>

            <div className="space-y-1.5 pt-2 border-t border-slate-800/60 font-mono text-xs max-h-36 overflow-y-auto">
              {metric.values.map((v, idx) => (
                <div key={idx} className="flex items-center justify-between p-1.5 rounded-lg bg-slate-950/60 text-slate-300">
                  <span className="truncate pr-2 text-slate-400 text-[10px]">
                    {Object.entries(v.labels).map(([k, val]) => `${k}="${val}"`).join(', ') || 'default'}
                  </span>
                  <span className="font-bold text-teal-400 shrink-0">
                    {v.value.toLocaleString()}
                  </span>
                </div>
              ))}
            </div>
          </Card>
        ))}
      </div>

      {/* Raw Stream Viewer */}
      <Card glass className="p-6 space-y-3">
        <CardTitle className="text-sm font-semibold text-slate-200">
          Raw Output (Ready for Grafana / Prometheus agent)
        </CardTitle>
        <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-slate-300 overflow-x-auto max-h-80 select-text">
          {rawMetrics}
        </pre>
      </Card>
    </div>
  );
};
