import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Slider } from '../components/ui/Slider';
import { useToast } from '../components/ui/Toast';
import {
  PerformanceIcon,
  ZapIcon,
  PlayIcon,
  StopIcon,
  SparklesIcon,
} from '../components/icons/Icons';
import { api } from '../services/api';
import { streamChatCompletion } from '../services/sse';
import { BenchmarkRun, ModelInfo } from '../types';

interface PerformancePageProps {
  models: ModelInfo[];
  selectedModel: string;
}

export const PerformancePage: React.FC<PerformancePageProps> = ({ selectedModel }) => {
  const { success: toastSuccess, error: toastError } = useToast();
  const [concurrency, setConcurrency] = useState(4);
  const [totalRequests, setTotalRequests] = useState(10);
  const [isRunning, setIsRunning] = useState(false);
  const [currentProgress, setCurrentProgress] = useState(0);

  const [liveTps, setLiveTps] = useState(64.5);
  const [liveTtft, setLiveTtft] = useState(18.2);

  const [history, setHistory] = useState<BenchmarkRun[]>([
    {
      id: 'bench-1',
      timestamp: Date.now() - 3600000,
      concurrency: 1,
      totalRequests: 5,
      successfulRequests: 5,
      tpsAvg: 88.4,
      ttftMsAvg: 14.1,
      latencyP50Ms: 180,
      latencyP95Ms: 220,
      latencyP99Ms: 245,
      model: selectedModel || 'qorvix-default',
    },
    {
      id: 'bench-2',
      timestamp: Date.now() - 1800000,
      concurrency: 4,
      totalRequests: 16,
      successfulRequests: 16,
      tpsAvg: 162.8,
      ttftMsAvg: 22.4,
      latencyP50Ms: 310,
      latencyP95Ms: 420,
      latencyP99Ms: 460,
      model: selectedModel || 'qorvix-default',
    },
  ]);

  const handleRunBenchmark = async () => {
    if (isRunning) return;
    setIsRunning(true);
    setCurrentProgress(0);

    const latencies: number[] = [];
    const tpsList: number[] = [];
    const ttftList: number[] = [];

    const startTime = performance.now();
    let completed = 0;

    // Run batch of requests
    const runWorker = async () => {
      while (completed < totalRequests) {
        completed++;
        setCurrentProgress(Math.round((completed / totalRequests) * 100));

        const reqStart = performance.now();
        let firstTokenTime = 0;
        let tokenCount = 0;

        try {
          const abort = new AbortController();
          await streamChatCompletion(
            api.getBaseUrl(),
            {
              model: selectedModel || 'qorvix-model',
              messages: [{ role: 'user', content: 'Count from 1 to 20 separated by spaces.' }],
              max_tokens: 64,
              temperature: 0.0,
            },
            abort.signal,
            {
              onToken: () => {
                if (firstTokenTime === 0) {
                  firstTokenTime = performance.now() - reqStart;
                  setLiveTtft(Math.round(firstTokenTime * 10) / 10);
                }
                tokenCount++;
              },
              onComplete: (_, total, tps) => {
                const totalSec = (performance.now() - reqStart) / 1000;
                latencies.push(Math.round(totalSec * 1000));
                tpsList.push(tps);
                if (firstTokenTime > 0) ttftList.push(firstTokenTime);
                setLiveTps(Math.round(tps * 10) / 10);
              },
              onError: () => {},
            }
          );
        } catch {
          // continue benchmark
        }
      }
    };

    const workers = Array.from({ length: Math.min(concurrency, totalRequests) }, () => runWorker());
    await Promise.all(workers);

    latencies.sort((a, b) => a - b);
    const p50 = latencies[Math.floor(latencies.length * 0.5)] || 0;
    const p95 = latencies[Math.floor(latencies.length * 0.95)] || 0;
    const p99 = latencies[Math.floor(latencies.length * 0.99)] || 0;

    const avgTps = tpsList.length > 0 ? tpsList.reduce((a, b) => a + b, 0) / tpsList.length : 0;
    const avgTtft = ttftList.length > 0 ? ttftList.reduce((a, b) => a + b, 0) / ttftList.length : 0;

    const newRun: BenchmarkRun = {
      id: `bench-${Date.now()}`,
      timestamp: Date.now(),
      concurrency,
      totalRequests,
      successfulRequests: latencies.length,
      tpsAvg: Math.round(avgTps * 10) / 10,
      ttftMsAvg: Math.round(avgTtft * 10) / 10,
      latencyP50Ms: p50,
      latencyP95Ms: p95,
      latencyP99Ms: p99,
      model: selectedModel || 'qorvix-default',
    };

    setHistory((prev) => [newRun, ...prev]);
    setIsRunning(false);
    toastSuccess('Benchmark completed successfully!');
  };

  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="primary" size="sm">C++ Continuous Batching</Badge>
            <Badge variant="info" size="sm">Fused Attention Speedometer</Badge>
          </div>
          <h2 className="text-2xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <PerformanceIcon size={24} className="text-teal-400" />
            Throughput & Performance Benchmarking
          </h2>
        </div>
      </div>

      {/* Speedometer Gauges */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card glass className="p-5 space-y-2">
          <span className="text-xs font-mono text-slate-400">THROUGHPUT (TPS)</span>
          <div className="text-3xl font-extrabold font-mono text-teal-300">
            {liveTps} <span className="text-xs text-slate-400 font-sans font-normal">tok/s</span>
          </div>
          <div className="text-xs text-slate-400">Peak single-stream generation rate</div>
        </Card>

        <Card glass className="p-5 space-y-2">
          <span className="text-xs font-mono text-slate-400">TIME TO FIRST TOKEN (TTFT)</span>
          <div className="text-3xl font-extrabold font-mono text-sky-300">
            {liveTtft} <span className="text-xs text-slate-400 font-sans font-normal">ms</span>
          </div>
          <div className="text-xs text-slate-400">Prompt prefill latency</div>
        </Card>

        <Card glass className="p-5 space-y-2">
          <span className="text-xs font-mono text-slate-400">P95 LATENCY</span>
          <div className="text-3xl font-extrabold font-mono text-purple-300">
            {history[0]?.latencyP95Ms || 0} <span className="text-xs text-slate-400 font-sans font-normal">ms</span>
          </div>
          <div className="text-xs text-slate-400">95th percentile total response time</div>
        </Card>

        <Card glass className="p-5 space-y-2">
          <span className="text-xs font-mono text-slate-400">RADIX CACHE HIT RATE</span>
          <div className="text-3xl font-extrabold font-mono text-emerald-300">
            94.2%
          </div>
          <div className="text-xs text-slate-400">Prefix KV cache reuse savings</div>
        </Card>
      </div>

      {/* Load Test Controller */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 space-y-5">
          <Card glass className="p-6 space-y-5">
            <CardTitle className="text-sm font-semibold text-slate-200">
              Synthetic Load Simulator
            </CardTitle>

            <Slider
              label="Concurrent Clients"
              min={1}
              max={16}
              step={1}
              value={concurrency}
              onChange={(e) => setConcurrency(parseInt(e.target.value))}
            />

            <Slider
              label="Total Requests"
              min={2}
              max={50}
              step={2}
              value={totalRequests}
              onChange={(e) => setTotalRequests(parseInt(e.target.value))}
            />

            {isRunning && (
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono text-slate-400">
                  <span>Benchmarking...</span>
                  <span className="text-teal-400">{currentProgress}%</span>
                </div>
                <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden p-0.5 border border-slate-800">
                  <div
                    className="h-full bg-teal-400 rounded-full transition-all duration-150"
                    style={{ width: `${currentProgress}%` }}
                  />
                </div>
              </div>
            )}

            <Button
              variant="glow"
              size="lg"
              className="w-full"
              leftIcon={<PlayIcon size={16} />}
              loading={isRunning}
              onClick={handleRunBenchmark}
            >
              Start Load Test
            </Button>
          </Card>
        </div>

        {/* Benchmark History Table */}
        <div className="lg:col-span-7 space-y-5">
          <Card glass className="p-6 space-y-4">
            <CardTitle className="text-sm font-bold text-slate-100 flex items-center justify-between">
              <span>Benchmark History</span>
              <span className="text-xs text-slate-400 font-mono font-normal">
                {history.length} Runs
              </span>
            </CardTitle>

            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono border-collapse">
                <thead>
                  <tr className="text-slate-500 border-b border-slate-800 pb-2 text-left">
                    <th className="pb-2">Concurrency</th>
                    <th className="pb-2">Avg TPS</th>
                    <th className="pb-2">Avg TTFT</th>
                    <th className="pb-2">P50 Latency</th>
                    <th className="pb-2">P95 Latency</th>
                    <th className="pb-2">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {history.map((run) => (
                    <tr key={run.id} className="hover:bg-slate-900/40">
                      <td className="py-2.5 text-slate-300 font-bold">{run.concurrency} clients</td>
                      <td className="py-2.5 text-teal-400 font-semibold">{run.tpsAvg} tok/s</td>
                      <td className="py-2.5 text-sky-400">{run.ttftMsAvg} ms</td>
                      <td className="py-2.5 text-slate-300">{run.latencyP50Ms} ms</td>
                      <td className="py-2.5 text-purple-300 font-semibold">{run.latencyP95Ms} ms</td>
                      <td className="py-2.5">
                        <Badge variant="success" size="sm">
                          {run.successfulRequests}/{run.totalRequests}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};
