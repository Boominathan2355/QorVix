import React, { useState, useRef, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Tabs } from '../components/ui/Tabs';
import { useToast } from '../components/ui/Toast';
import {
  AudioIcon,
  UploadIcon,
  PlayIcon,
  StopIcon,
  CopyIcon,
  CheckIcon,
  SparklesIcon,
  ZapIcon,
} from '../components/icons/Icons';
import { api } from '../services/api';
import { AudioTranscriptionResult } from '../types';

export const AudioPage: React.FC = () => {
  const { error: toastError, success: toastSuccess } = useToast();
  const [task, setTask] = useState<'transcribe' | 'translate'>('transcribe');
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<AudioTranscriptionResult | null>(null);
  const [copied, setCopied] = useState(false);
  const [language, setLanguage] = useState('auto');

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

  // Clean up audio blob URL on unmount
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      if (audioContextRef.current) audioContextRef.current.close();
    };
  }, [audioUrl]);

  const startWaveformVisualizer = (stream: MediaStream) => {
    const audioCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
    audioContextRef.current = audioCtx;

    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    analyserRef.current = analyser;

    const source = audioCtx.createMediaStreamSource(stream);
    source.connect(analyser);

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animationFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);

      ctx.fillStyle = '#020617';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 2.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height * 0.9;

        const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
        gradient.addColorStop(0, '#0d9488');
        gradient.addColorStop(1, '#2dd4bf');

        ctx.fillStyle = gradient;
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

        x += barWidth + 1;
      }
    };

    draw();
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];

      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setAudioBlob(blob);
        if (audioUrl) URL.revokeObjectURL(audioUrl);
        setAudioUrl(URL.createObjectURL(blob));
        stream.getTracks().forEach((track) => track.stop());
        if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      };

      recorder.start();
      setIsRecording(true);
      startWaveformVisualizer(stream);
    } catch (err) {
      toastError('Could not access microphone. Check browser permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setAudioBlob(file);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(URL.createObjectURL(file));
    setResult(null);
  };

  const handleProcess = async () => {
    if (!audioBlob || isProcessing) return;

    setIsProcessing(true);
    setResult(null);

    try {
      let res: AudioTranscriptionResult;
      if (task === 'translate') {
        res = await api.translateAudio(audioBlob);
      } else {
        res = await api.transcribeAudio(audioBlob, language === 'auto' ? undefined : language);
      }
      setResult(res);
      toastSuccess('Speech processed successfully!');
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Transcription failed', 'Whisper Error');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleCopy = () => {
    if (!result?.text) return;
    navigator.clipboard.writeText(result.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="p-6 md:p-8 max-w-6xl mx-auto space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="purple" size="sm">Whisper Log-Mel Encoder-Decoder</Badge>
            <Badge variant="primary" size="sm">80 Mel Filterbanks • 16 kHz</Badge>
          </div>
          <h2 className="text-2xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <AudioIcon size={24} className="text-purple-400" />
            Whisper Speech & Audio Studio
          </h2>
        </div>

        <Tabs
          tabs={[
            { id: 'transcribe', label: 'Transcribe Speech' },
            { id: 'translate', label: 'Translate to English' },
          ]}
          activeTab={task}
          onChange={(t) => setTask(t as 'transcribe' | 'translate')}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Audio Capture / Upload */}
        <div className="lg:col-span-5 space-y-5">
          <Card glass className="p-6 space-y-5">
            <CardTitle className="text-sm font-semibold text-slate-200">
              Audio Source
            </CardTitle>

            {/* Live Waveform Canvas */}
            <div className="relative rounded-2xl overflow-hidden border border-slate-800 bg-slate-950 h-36 flex flex-col items-center justify-center shadow-inner">
              <canvas
                ref={canvasRef}
                width={380}
                height={144}
                className="w-full h-full object-cover"
              />
              {!isRecording && !audioUrl && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 text-xs font-mono space-y-1">
                  <AudioIcon size={24} className="text-slate-600" />
                  <span>Microphone idle</span>
                </div>
              )}
            </div>

            {/* Audio Controls */}
            <div className="flex items-center justify-center gap-3">
              {!isRecording ? (
                <Button
                  variant="primary"
                  size="md"
                  leftIcon={<PlayIcon size={16} />}
                  onClick={startRecording}
                  className="bg-red-500 hover:bg-red-400 text-slate-950 shadow-red-500/20"
                >
                  Record Microphone
                </Button>
              ) : (
                <Button
                  variant="danger"
                  size="md"
                  leftIcon={<StopIcon size={16} />}
                  onClick={stopRecording}
                  className="animate-pulse"
                >
                  Stop Recording
                </Button>
              )}

              <label className="inline-flex items-center justify-center font-medium transition-all duration-150 h-9.5 px-4 rounded-xl text-sm bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 cursor-pointer gap-2">
                <UploadIcon size={16} />
                <span>Upload File</span>
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>

            {/* Audio Playback Player */}
            {audioUrl && (
              <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-2">
                <div className="flex items-center justify-between text-xs font-mono text-slate-400">
                  <span>RECORDED AUDIO</span>
                  <span className="text-teal-400 font-semibold">Ready to process</span>
                </div>
                <audio src={audioUrl} controls className="w-full h-9 rounded-lg" />
              </div>
            )}

            {/* Language Selection */}
            {task === 'transcribe' && (
              <div className="space-y-1.5">
                <label className="block text-xs font-medium text-slate-300">Language</label>
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  className="w-full bg-slate-950/70 border border-slate-800 rounded-xl px-3.5 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-teal-500/50 cursor-pointer"
                >
                  <option value="auto">Auto Detect Language</option>
                  <option value="en">English (en)</option>
                  <option value="es">Spanish (es)</option>
                  <option value="fr">French (fr)</option>
                  <option value="de">German (de)</option>
                  <option value="zh">Chinese (zh)</option>
                  <option value="ja">Japanese (ja)</option>
                </select>
              </div>
            )}

            <Button
              variant="glow"
              size="lg"
              className="w-full"
              leftIcon={<SparklesIcon size={18} />}
              disabled={!audioBlob || isProcessing}
              loading={isProcessing}
              onClick={handleProcess}
            >
              {task === 'transcribe' ? 'Transcribe Audio' : 'Translate to English'}
            </Button>
          </Card>
        </div>

        {/* Right Column: Transcription Output */}
        <div className="lg:col-span-7 space-y-5">
          <Card glass className="p-6 space-y-4 min-h-[460px] flex flex-col">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <CardTitle className="text-sm font-bold text-slate-100 flex items-center gap-2">
                <SparklesIcon size={16} className="text-purple-400" />
                {task === 'transcribe' ? 'Transcription Text' : 'English Translation'}
              </CardTitle>
              {result && (
                <Button
                  variant="outline"
                  size="sm"
                  leftIcon={copied ? <CheckIcon size={14} className="text-emerald-400" /> : <CopyIcon size={14} />}
                  onClick={handleCopy}
                >
                  {copied ? 'Copied' : 'Copy Text'}
                </Button>
              )}
            </div>

            <div className="flex-1 flex flex-col justify-between">
              {result ? (
                <div className="space-y-4">
                  <div className="p-4 rounded-2xl bg-slate-950/70 border border-slate-800 text-slate-100 text-sm leading-relaxed font-sans select-text">
                    {result.text}
                  </div>

                  {/* Segment Breakdown with Timestamps */}
                  {result.segments && result.segments.length > 0 && (
                    <div className="space-y-2">
                      <div className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider">
                        Timestamped Segments
                      </div>
                      <div className="max-h-60 overflow-y-auto space-y-1.5 pr-1">
                        {result.segments.map((seg) => (
                          <div
                            key={seg.id}
                            className="p-2.5 rounded-xl bg-slate-900/60 border border-slate-800/60 flex items-start gap-3 text-xs"
                          >
                            <span className="font-mono text-teal-400 font-semibold bg-teal-950/50 px-2 py-0.5 rounded border border-teal-800/40 shrink-0">
                              {seg.start.toFixed(1)}s - {seg.end.toFixed(1)}s
                            </span>
                            <span className="text-slate-300 font-sans">{seg.text}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : isProcessing ? (
                <div className="h-full flex flex-col items-center justify-center text-center space-y-3 py-16">
                  <span className="relative flex h-4 w-4">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-4 w-4 bg-purple-500" />
                  </span>
                  <div className="space-y-1">
                    <p className="text-sm font-semibold text-slate-200">
                      Processing 80-channel Log-Mel spectrogram...
                    </p>
                    <p className="text-xs text-slate-500 font-mono">
                      Whisper cross-attention beam search running on C++ engine
                    </p>
                  </div>
                </div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-center space-y-2 py-16 text-slate-500">
                  <AudioIcon size={36} className="text-slate-700" />
                  <p className="text-xs font-mono">
                    Record microphone or upload an audio file to view transcription
                  </p>
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};
