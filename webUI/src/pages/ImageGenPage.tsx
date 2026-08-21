import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Slider } from '../components/ui/Slider';
import { Badge } from '../components/ui/Badge';
import { Modal } from '../components/ui/Modal';
import { useToast } from '../components/ui/Toast';
import {
  ImageIcon,
  SparklesIcon,
  DownloadIcon,
  ZapIcon,
  TrashIcon,
} from '../components/icons/Icons';
import { api } from '../services/api';
import { GeneratedImage } from '../types';

export const ImageGenPage: React.FC = () => {
  const { error: toastError, success: toastSuccess } = useToast();
  const [prompt, setPrompt] = useState('Cyberpunk neon city skyline at night with hyperdetailed reflections and volumetric fog, 8k resolution');
  const [negativePrompt, setNegativePrompt] = useState('blurry, low quality, distorted, artifacts, watermark');
  const [steps, setSteps] = useState(25);
  const [guidance, setGuidance] = useState(7.5);
  const [seed, setSeed] = useState<number>(-1);
  const [size, setSize] = useState<'512x512' | '768x512' | '512x768'>('512x512');
  const [isGenerating, setIsGenerating] = useState(false);

  const [gallery, setGallery] = useState<GeneratedImage[]>(() => {
    const saved = localStorage.getItem('qorvix_image_gallery');
    if (saved) {
      try { return JSON.parse(saved); } catch { /* ignore */ }
    }
    return [];
  });

  const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(null);

  const saveGallery = (newGallery: GeneratedImage[]) => {
    setGallery(newGallery);
    localStorage.setItem('qorvix_image_gallery', JSON.stringify(newGallery));
  };

  const handleGenerate = async () => {
    if (!prompt.trim() || isGenerating) return;

    setIsGenerating(true);
    const [w, h] = size.split('x').map(Number);
    const actualSeed = seed === -1 ? Math.floor(Math.random() * 2147483647) : seed;

    try {
      const img = await api.generateImage({
        prompt: prompt.trim(),
        negativePrompt: negativePrompt.trim(),
        steps,
        guidance,
        seed: actualSeed,
        width: w,
        height: h,
      });

      const updated = [img, ...gallery];
      saveGallery(updated);
      setSelectedImage(img);
      toastSuccess('Image synthesized successfully!');
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Generation failed', 'Stable Diffusion Error');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = (img: GeneratedImage) => {
    const a = document.createElement('a');
    a.href = img.url;
    a.download = `qorvix-${img.id}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updated = gallery.filter((g) => g.id !== id);
    saveGallery(updated);
    if (selectedImage?.id === id) setSelectedImage(null);
  };

  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="warning" size="sm">Stable Diffusion UNet + VAE</Badge>
            <Badge variant="primary" size="sm">Native C++ Euler Ancestral Sampler</Badge>
          </div>
          <h2 className="text-2xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <ImageIcon size={24} className="text-pink-400" />
            Stable Diffusion Studio
          </h2>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Prompt & Sampling Controls */}
        <div className="lg:col-span-5 space-y-5">
          <Card glass className="p-6 space-y-5">
            <CardTitle className="text-sm font-semibold text-slate-200">
              Prompt & Synthesis Settings
            </CardTitle>

            <div className="space-y-3">
              <div className="space-y-1.5">
                <label className="block text-xs font-medium text-slate-300">Prompt</label>
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="A cinematic futuristic cityscape..."
                  rows={3}
                  className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-teal-500/60 focus:ring-2 focus:ring-teal-500/20"
                />
              </div>

              <div className="space-y-1.5">
                <label className="block text-xs font-medium text-slate-400">Negative Prompt</label>
                <input
                  type="text"
                  value={negativePrompt}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  placeholder="blurry, distorted, low quality..."
                  className="w-full bg-slate-950/60 border border-slate-800 rounded-xl p-2.5 text-xs text-slate-300 placeholder:text-slate-600 focus:outline-none focus:border-teal-500/50"
                />
              </div>
            </div>

            <div className="space-y-4 pt-2 border-t border-slate-800/80">
              <Slider
                label="Diffusion Steps"
                min={5}
                max={50}
                step={1}
                value={steps}
                onChange={(e) => setSteps(parseInt(e.target.value))}
              />

              <Slider
                label="Guidance Scale (CFG)"
                min={1}
                max={20}
                step={0.5}
                value={guidance}
                valueDisplay={guidance.toFixed(1)}
                onChange={(e) => setGuidance(parseFloat(e.target.value))}
              />

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label className="block text-xs font-medium text-slate-300">Resolution</label>
                  <select
                    value={size}
                    onChange={(e) => setSize(e.target.value as typeof size)}
                    className="w-full bg-slate-950/70 border border-slate-800 rounded-xl p-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-teal-500/50"
                  >
                    <option value="512x512">512 × 512 (Square)</option>
                    <option value="768x512">768 × 512 (Landscape)</option>
                    <option value="512x768">512 × 768 (Portrait)</option>
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="block text-xs font-medium text-slate-300">Seed (-1 = Random)</label>
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value) || -1)}
                    className="w-full bg-slate-950/70 border border-slate-800 rounded-xl p-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-teal-500/50"
                  />
                </div>
              </div>
            </div>

            <Button
              variant="glow"
              size="lg"
              className="w-full"
              leftIcon={<SparklesIcon size={18} />}
              loading={isGenerating}
              disabled={!prompt.trim() || isGenerating}
              onClick={handleGenerate}
            >
              Synthesize Image
            </Button>
          </Card>
        </div>

        {/* Right Column: Generation Output & Gallery */}
        <div className="lg:col-span-7 space-y-5">
          <Card glass className="p-6 space-y-4 min-h-[480px]">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <CardTitle className="text-sm font-bold text-slate-100 flex items-center gap-2">
                <SparklesIcon size={16} className="text-pink-400" />
                Image Gallery ({gallery.length})
              </CardTitle>
            </div>

            {isGenerating ? (
              <div className="h-80 flex flex-col items-center justify-center text-center space-y-4">
                <span className="relative flex h-5 w-5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-pink-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-5 w-5 bg-pink-500" />
                </span>
                <div className="space-y-1">
                  <p className="text-sm font-bold text-slate-100">
                    Running UNet Denoising Steps ({steps} iterations)...
                  </p>
                  <p className="text-xs text-slate-500 font-mono">
                    VAE Latent Decoding onto host frame buffer
                  </p>
                </div>
              </div>
            ) : gallery.length === 0 ? (
              <div className="h-80 flex flex-col items-center justify-center text-center text-slate-500 space-y-2">
                <ImageIcon size={40} className="text-slate-700" />
                <p className="text-xs font-mono">No images synthesized yet. Enter a prompt to begin.</p>
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 max-h-[520px] overflow-y-auto pr-1">
                {gallery.map((img) => (
                  <div
                    key={img.id}
                    onClick={() => setSelectedImage(img)}
                    className="group relative aspect-square rounded-2xl overflow-hidden border border-slate-800 hover:border-teal-500/60 bg-slate-950 cursor-pointer shadow-lg transition-all"
                  >
                    <img
                      src={img.url}
                      alt={img.prompt}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-slate-950/90 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity p-3 flex flex-col justify-end">
                      <p className="text-[11px] text-slate-200 line-clamp-2 font-medium">
                        {img.prompt}
                      </p>
                      <div className="flex items-center justify-between mt-2 pt-1 border-t border-slate-700/60 text-[10px] text-slate-400 font-mono">
                        <span>{img.width}×{img.height}</span>
                        <button
                          onClick={(e) => handleDelete(img.id, e)}
                          className="hover:text-red-400 p-0.5"
                          title="Delete"
                        >
                          <TrashIcon size={12} />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Lightbox Modal */}
      {selectedImage && (
        <Modal
          isOpen={!!selectedImage}
          onClose={() => setSelectedImage(null)}
          title="Image Details"
          maxWidth="2xl"
        >
          <div className="space-y-4">
            <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 shadow-2xl flex items-center justify-center">
              <img
                src={selectedImage.url}
                alt={selectedImage.prompt}
                className="max-h-[60vh] w-auto object-contain"
              />
            </div>

            <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2 text-xs">
              <div className="space-y-1">
                <span className="text-slate-400 font-mono uppercase text-[10px]">Prompt</span>
                <p className="text-slate-200 font-medium">{selectedImage.prompt}</p>
              </div>
              {selectedImage.negativePrompt && (
                <div className="space-y-1">
                  <span className="text-slate-400 font-mono uppercase text-[10px]">Negative</span>
                  <p className="text-slate-400">{selectedImage.negativePrompt}</p>
                </div>
              )}
              <div className="flex flex-wrap gap-4 pt-2 border-t border-slate-800/80 font-mono text-slate-400 text-[11px]">
                <span>Steps: <b className="text-slate-200">{selectedImage.steps}</b></span>
                <span>CFG: <b className="text-slate-200">{selectedImage.guidance}</b></span>
                <span>Seed: <b className="text-slate-200">{selectedImage.seed}</b></span>
                <span>Size: <b className="text-slate-200">{selectedImage.width}×{selectedImage.height}</b></span>
              </div>
            </div>

            <div className="flex justify-end gap-3">
              <Button
                variant="primary"
                size="md"
                leftIcon={<DownloadIcon size={16} />}
                onClick={() => handleDownload(selectedImage)}
              >
                Download PNG
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};
