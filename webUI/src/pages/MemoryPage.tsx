import React, { useState } from 'react';
import { Card, CardTitle } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { MemoryIcon, RefreshIcon, LayersIcon, ZapIcon } from '../components/icons/Icons';

export const MemoryPage: React.FC = () => {
  const [stats] = useState({
    vramTotalGb: 24.0,
    vramUsedGb: 6.8,
    ramTotalGb: 64.0,
    ramUsedGb: 14.2,
    spoolTotalGb: 500.0,
    spoolUsedGb: 28.5,
    kvPagesTotal: 32768,
    kvPagesUsed: 4096,
    fragmentationPct: 3.8,
  });

  const vramPct = Math.round((stats.vramUsedGb / stats.vramTotalGb) * 100);
  const ramPct = Math.round((stats.ramUsedGb / stats.ramTotalGb) * 100);
  const kvPct = Math.round((stats.kvPagesUsed / stats.kvPagesTotal) * 100);

  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="primary" size="sm">Zero-Copy Slab Allocator</Badge>
            <Badge variant="neutral" size="sm">3-Tier Storage Hierarchy</Badge>
          </div>
          <h2 className="text-2xl font-bold text-foreground tracking-tight flex items-center gap-2">
            <MemoryIcon size={24} className="text-amber-500" />
            Tiered Memory & VRAM Visualizer
          </h2>
        </div>
        <Button
          variant="outline"
          size="sm"
          leftIcon={<RefreshIcon size={14} />}
          onClick={() => {}}
        >
          Refresh Tiers
        </Button>
      </div>

      {/* Memory Tier Bars */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Tier 0: GPU VRAM */}
        <Card glass className="p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <span className="text-[11px] font-mono text-teal-600 dark:text-teal-400 uppercase font-bold tracking-wider">
                TIER 0 • LOW LATENCY
              </span>
              <h3 className="font-bold text-foreground text-base">GPU VRAM</h3>
            </div>
            <Badge variant="primary" size="sm">{vramPct}%</Badge>
          </div>

          <div className="space-y-2">
            <div className="h-3 w-full bg-secondary rounded-full overflow-hidden p-0.5 border border-border">
              <div
                className="h-full bg-teal-500 rounded-full transition-all duration-500"
                style={{ width: `${vramPct}%` }}
              />
            </div>
            <div className="flex justify-between text-xs font-mono text-muted-foreground">
              <span>{stats.vramUsedGb.toFixed(1)} GB Used</span>
              <span>{stats.vramTotalGb.toFixed(1)} GB Total</span>
            </div>
          </div>

          <p className="text-xs text-muted-foreground leading-relaxed pt-2 border-t border-border">
            Hosts active model tensor weights (Q4_K / Q8_0) and hot active KV cache blocks for zero-copy GEMV kernels.
          </p>
        </Card>

        {/* Tier 1: Host RAM */}
        <Card glass className="p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <span className="text-[11px] font-mono text-sky-600 dark:text-sky-400 uppercase font-bold tracking-wider">
                TIER 1 • SYSTEM RAM
              </span>
              <h3 className="font-bold text-foreground text-base">Host Memory</h3>
            </div>
            <Badge variant="info" size="sm">{ramPct}%</Badge>
          </div>

          <div className="space-y-2">
            <div className="h-3 w-full bg-secondary rounded-full overflow-hidden p-0.5 border border-border">
              <div
                className="h-full bg-sky-500 rounded-full transition-all duration-500"
                style={{ width: `${ramPct}%` }}
              />
            </div>
            <div className="flex justify-between text-xs font-mono text-muted-foreground">
              <span>{stats.ramUsedGb.toFixed(1)} GB Used</span>
              <span>{stats.ramTotalGb.toFixed(1)} GB Total</span>
            </div>
          </div>

          <p className="text-xs text-muted-foreground leading-relaxed pt-2 border-t border-border">
            Stages mmap'd GGUF files, pinned page-locked transfer buffers, Whisper audio spectra, and warm KV cache states.
          </p>
        </Card>

        {/* Tier 2: NVMe Spool */}
        <Card glass className="p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <span className="text-[11px] font-mono text-purple-600 dark:text-purple-400 uppercase font-bold tracking-wider">
                TIER 2 • PERSISTENT SPOOL
              </span>
              <h3 className="font-bold text-foreground text-base">NVMe Storage</h3>
            </div>
            <Badge variant="purple" size="sm">Active</Badge>
          </div>

          <div className="space-y-2">
            <div className="h-3 w-full bg-secondary rounded-full overflow-hidden p-0.5 border border-border">
              <div
                className="h-full bg-purple-500 rounded-full transition-all duration-500"
                style={{ width: `${(stats.spoolUsedGb / stats.spoolTotalGb) * 100}%` }}
              />
            </div>
            <div className="flex justify-between text-xs font-mono text-muted-foreground">
              <span>{stats.spoolUsedGb.toFixed(1)} GB Spooled</span>
              <span>{stats.spoolTotalGb.toFixed(1)} GB Total</span>
            </div>
          </div>

          <p className="text-xs text-muted-foreground leading-relaxed pt-2 border-t border-border">
            Cold tensor eviction spool and persistent prefix KV cache trees across long context multi-turn chat sessions.
          </p>
        </Card>
      </div>

      {/* KV Cache & Slab Diagnostics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card glass className="p-6 space-y-4">
          <CardTitle className="text-sm font-semibold text-foreground flex items-center gap-2">
            <LayersIcon size={18} className="text-teal-500" />
            Paged KV Cache Allocation
          </CardTitle>

          <div className="space-y-3">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-muted-foreground">Page Utilization</span>
              <span className="text-teal-600 dark:text-teal-400 font-bold">{stats.kvPagesUsed} / {stats.kvPagesTotal} pages ({kvPct}%)</span>
            </div>
            <div className="h-2.5 w-full bg-secondary rounded-full overflow-hidden p-0.5 border border-border">
              <div className="h-full bg-teal-500 rounded-full" style={{ width: `${kvPct}%` }} />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 pt-2 text-xs font-mono">
            <div className="p-3 rounded-xl bg-secondary border border-border">
              <span className="text-muted-foreground block text-[10px]">PAGE SIZE</span>
              <span className="text-foreground font-bold">16 Tokens / Page</span>
            </div>
            <div className="p-3 rounded-xl bg-secondary border border-border">
              <span className="text-muted-foreground block text-[10px]">RADIX TREE SHARING</span>
              <span className="text-emerald-600 dark:text-emerald-400 font-bold">Zero-Copy Active</span>
            </div>
          </div>
        </Card>

        <Card glass className="p-6 space-y-4">
          <CardTitle className="text-sm font-semibold text-foreground flex items-center gap-2">
            <ZapIcon size={18} className="text-amber-500" />
            Slab Allocator Health
          </CardTitle>

          <div className="space-y-3">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-muted-foreground">Fragmentation Index</span>
              <span className="text-emerald-600 dark:text-emerald-400 font-bold">{stats.fragmentationPct}% (Optimal)</span>
            </div>
            <div className="h-2.5 w-full bg-secondary rounded-full overflow-hidden p-0.5 border border-border">
              <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${stats.fragmentationPct * 5}%` }} />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 pt-2 text-xs font-mono">
            <div className="p-3 rounded-xl bg-secondary border border-border">
              <span className="text-muted-foreground block text-[10px]">SLAB CLASS BUCKETS</span>
              <span className="text-foreground font-bold">64B to 64MB Powers-of-2</span>
            </div>
            <div className="p-3 rounded-xl bg-secondary border border-border">
              <span className="text-muted-foreground block text-[10px]">ALLOCATION OVERHEAD</span>
              <span className="text-teal-600 dark:text-teal-400 font-bold">&lt; 1.2% Memory Loss</span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};
