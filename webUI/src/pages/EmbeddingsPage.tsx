import React, { useState } from 'react';
import { Card, CardTitle } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { useToast } from '../components/ui/Toast';
import {
  EmbeddingsIcon,
  SparklesIcon,
  LayersIcon,
} from '../components/icons/Icons';
import { api } from '../services/api';
import { EmbeddingResult } from '../types';

export const EmbeddingsPage: React.FC = () => {
  const { error: toastError, success: toastSuccess } = useToast();
  const [inputText, setInputText] = useState(
    "The quick brown fox jumps over the lazy dog.\nA speedy amber canine leaps above a resting hound.\nDeep learning inference on modern GPU architectures.\nQuantum computing algorithms for cryptography."
  );
  const [results, setResults] = useState<EmbeddingResult[]>([]);
  const [similarityMatrix, setSimilarityMatrix] = useState<number[][]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const calculateCosineSimilarity = (a: number[], b: number[]) => {
    let dot = 0, sumA = 0, sumB = 0;
    for (let i = 0; i < a.length; ++i) {
      dot += a[i] * b[i];
      sumA += a[i] * a[i];
      sumB += b[i] * b[i];
    }
    const denom = Math.sqrt(sumA) * Math.sqrt(sumB);
    return denom === 0 ? 0 : dot / denom;
  };

  const handleCompute = async () => {
    const lines = inputText
      .split('\n')
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    if (lines.length === 0 || isLoading) return;

    setIsLoading(true);
    try {
      const embeddings = await api.createEmbeddings(lines);
      setResults(embeddings);

      const matrix: number[][] = [];
      for (let i = 0; i < embeddings.length; ++i) {
        matrix[i] = [];
        for (let j = 0; j < embeddings.length; ++j) {
          matrix[i][j] = calculateCosineSimilarity(
            embeddings[i].vector,
            embeddings[j].vector
          );
        }
      }
      setSimilarityMatrix(matrix);
      toastSuccess(`Extracted ${embeddings.length} dense embeddings!`);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Embedding extraction failed', 'BERT Error');
    } finally {
      setIsLoading(false);
    }
  };

  const getHeatmapColor = (val: number) => {
    if (val >= 0.85) return 'bg-teal-500 text-slate-950 font-bold';
    if (val >= 0.65) return 'bg-teal-500/70 text-white dark:text-slate-950 font-semibold';
    if (val >= 0.45) return 'bg-teal-500/40 text-foreground';
    if (val >= 0.25) return 'bg-secondary text-foreground';
    return 'bg-muted text-muted-foreground';
  };

  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="success" size="sm">BERT Encoder Architecture</Badge>
            <Badge variant="primary" size="sm">L2 Normalized Vector Space</Badge>
          </div>
          <h2 className="text-2xl font-bold text-foreground tracking-tight flex items-center gap-2">
            <EmbeddingsIcon size={24} className="text-emerald-500" />
            Embeddings & Cosine Similarity Matrix
          </h2>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Sentences Input */}
        <div className="lg:col-span-5 space-y-5">
          <Card glass className="p-6 space-y-4">
            <CardTitle className="text-sm font-semibold text-foreground">
              Input Sentences (One per line)
            </CardTitle>

            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              rows={8}
              placeholder="Enter sentences to embed and compare..."
              className="w-full bg-background border border-border rounded-xl p-3 text-xs font-mono text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-teal-500/60 focus:ring-2 focus:ring-teal-500/20"
            />

            <Button
              variant="glow"
              size="lg"
              className="w-full"
              leftIcon={<SparklesIcon size={18} />}
              loading={isLoading}
              onClick={handleCompute}
            >
              Generate Embeddings & Matrix
            </Button>
          </Card>
        </div>

        {/* Right Column: Similarity Heatmap Matrix */}
        <div className="lg:col-span-7 space-y-5">
          <Card glass className="p-6 space-y-5 min-h-[460px]">
            <CardTitle className="text-sm font-bold text-foreground flex items-center gap-2">
              <LayersIcon size={18} className="text-teal-500" />
              Cosine Similarity Heatmap ({results.length} × {results.length})
            </CardTitle>

            {results.length > 0 ? (
              <div className="space-y-6">
                {/* Heatmap Grid */}
                <div className="overflow-x-auto border border-border rounded-xl bg-card p-4">
                  <table className="w-full border-collapse text-xs font-mono">
                    <thead>
                      <tr>
                        <th className="p-2 text-left text-muted-foreground">Sentence</th>
                        {results.map((_, idx) => (
                          <th key={idx} className="p-2 text-center text-teal-600 dark:text-teal-400 font-bold">
                            S{idx + 1}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {similarityMatrix.map((row, i) => (
                        <tr key={i} className="border-t border-border">
                          <td className="p-2 text-foreground max-w-[180px] truncate font-sans">
                            <span className="text-teal-600 dark:text-teal-400 font-mono font-bold mr-1.5">S{i + 1}:</span>
                            {results[i]?.text}
                          </td>
                          {row.map((val, j) => (
                            <td key={j} className="p-2 text-center">
                              <span
                                className={`inline-block px-2.5 py-1 rounded-lg text-xs transition-all ${getHeatmapColor(
                                  val
                                )}`}
                                title={`Similarity S${i + 1} - S${j + 1}: ${val.toFixed(4)}`}
                              >
                                {val.toFixed(2)}
                              </span>
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Vectors Inspector */}
                <div className="space-y-2">
                  <span className="text-xs font-mono font-bold text-muted-foreground uppercase tracking-wider">
                    Vector Dimensions & Norms
                  </span>
                  <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                    {results.map((r, idx) => (
                      <div
                        key={idx}
                        className="p-3 rounded-xl bg-secondary border border-border flex items-center justify-between text-xs font-mono"
                      >
                        <div className="flex items-center gap-2 truncate pr-3">
                          <Badge variant="primary" size="sm">S{idx + 1}</Badge>
                          <span className="truncate text-foreground font-sans">{r.text}</span>
                        </div>
                        <div className="flex items-center gap-3 shrink-0 text-muted-foreground">
                          <span>Dim: <b className="text-foreground">{r.dim}</b></span>
                          <span>Norm: <b className="text-teal-600 dark:text-teal-400">{r.norm.toFixed(4)}</b></span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-64 flex flex-col items-center justify-center text-center text-muted-foreground space-y-2">
                <EmbeddingsIcon size={40} className="text-muted-foreground/50" />
                <p className="text-xs font-mono">
                  Click 'Generate Embeddings' to compute representations
                </p>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
};
