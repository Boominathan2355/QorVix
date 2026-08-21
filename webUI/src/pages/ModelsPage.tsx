import React, { useState, useEffect } from 'react';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { ModelsIcon, RefreshIcon, CpuIcon, LayersIcon, HardDriveIcon } from '../components/icons/Icons';
import { api } from '../services/api';
import { ModelInfo } from '../types';

export const ModelsPage: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchModelsList = async () => {
    setIsLoading(true);
    const m = await api.getModels();
    setModels(m);
    setIsLoading(false);
  };

  useEffect(() => {
    fetchModelsList();
  }, []);

  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="primary" size="sm">GGUF v3 File Specification</Badge>
            <Badge variant="neutral" size="sm">Unified Model Registry</Badge>
          </div>
          <h2 className="text-2xl font-bold text-foreground tracking-tight flex items-center gap-2">
            <ModelsIcon size={24} className="text-teal-500" />
            Model Registry & Tensor Inspector
          </h2>
        </div>
        <Button
          variant="outline"
          size="sm"
          leftIcon={<RefreshIcon size={14} />}
          loading={isLoading}
          onClick={fetchModelsList}
        >
          Refresh Registry
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {models.map((model) => (
          <Card key={model.id} glass hover className="p-6 space-y-5">
            <div className="flex items-start justify-between">
              <div className="p-3 rounded-2xl bg-teal-500/10 border border-teal-500/20 text-teal-500 shadow-xs">
                <ModelsIcon size={24} />
              </div>
              <Badge variant={model.is_multimodal ? 'info' : 'primary'} size="sm">
                {model.is_multimodal ? 'Multimodal (Vision)' : 'Autoregressive Decoder'}
              </Badge>
            </div>

            <div className="space-y-1">
              <h3 className="font-bold text-base text-foreground truncate" title={model.id}>
                {model.id}
              </h3>
              <p className="text-xs text-muted-foreground font-mono">
                Owned by: {model.owned_by}
              </p>
            </div>

            <div className="p-4 rounded-xl bg-secondary/80 border border-border space-y-2.5 text-xs font-mono">
              <div className="flex items-center justify-between text-muted-foreground">
                <span className="flex items-center gap-1.5"><CpuIcon size={14} /> Backend</span>
                <span className="text-teal-600 dark:text-teal-400 font-bold">{model.backend || 'CUDA / Native'}</span>
              </div>
              <div className="flex items-center justify-between text-muted-foreground">
                <span className="flex items-center gap-1.5"><HardDriveIcon size={14} /> Quantization</span>
                <span className="text-foreground font-semibold">{model.quantization || 'Q4_K_M (4-bit block)'}</span>
              </div>
              <div className="flex items-center justify-between text-muted-foreground">
                <span className="flex items-center gap-1.5"><LayersIcon size={14} /> Context Window</span>
                <span className="text-foreground font-semibold">{model.context_length || '4096 / 8192'} tokens</span>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};
