import React, { useState } from 'react';
import { CopyIcon, CheckIcon } from '../icons/Icons';

export interface MarkdownViewProps {
  content: string;
  className?: string;
}

export const MarkdownView: React.FC<MarkdownViewProps> = ({ content, className = '' }) => {
  const renderFormattedText = (text: string) => {
    // Break into code blocks and normal paragraphs
    const parts = text.split(/(```[\s\S]*?```)/g);

    return parts.map((part, index) => {
      if (part.startsWith('```') && part.endsWith('```')) {
        const lines = part.slice(3, -3).trim().split('\n');
        const language = lines[0].trim();
        const code = (language && !part.startsWith('```\n') ? lines.slice(1) : lines).join('\n');

        return <CodeBlock key={index} code={code} language={language || 'text'} />;
      }

      return <RichTextChunk key={index} text={part} />;
    });
  };

  return (
    <div className={`prose prose-invert max-w-none text-slate-200 text-sm leading-relaxed space-y-3 ${className}`}>
      {renderFormattedText(content)}
    </div>
  );
};

const CodeBlock: React.FC<{ code: string; language: string }> = ({ code, language }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative my-3 rounded-xl border border-slate-800 bg-slate-950 overflow-hidden shadow-lg group">
      <div className="flex items-center justify-between px-4 py-2 bg-slate-900/80 border-b border-slate-800/80 text-xs text-slate-400 font-mono">
        <span className="uppercase tracking-wider font-semibold text-teal-400/90">{language}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-slate-800/60 hover:bg-slate-800 text-slate-300 hover:text-slate-100 transition-all text-xs"
        >
          {copied ? <CheckIcon size={13} className="text-emerald-400" /> : <CopyIcon size={13} />}
          <span>{copied ? 'Copied!' : 'Copy'}</span>
        </button>
      </div>
      <pre className="p-4 text-xs font-mono text-slate-200 overflow-x-auto selection:bg-teal-500/30">
        <code>{code}</code>
      </pre>
    </div>
  );
};

const RichTextChunk: React.FC<{ text: string }> = ({ text }) => {
  if (!text) return null;

  // Split lines
  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];

  for (let i = 0; i < lines.length; ++i) {
    const line = lines[i];

    if (line.startsWith('# ')) {
      elements.push(<h1 key={i} className="text-xl font-bold text-slate-100 mt-4 mb-2">{formatInline(line.slice(2))}</h1>);
    } else if (line.startsWith('## ')) {
      elements.push(<h2 key={i} className="text-lg font-bold text-slate-100 mt-3 mb-1.5">{formatInline(line.slice(3))}</h2>);
    } else if (line.startsWith('### ')) {
      elements.push(<h3 key={i} className="text-base font-semibold text-slate-100 mt-2 mb-1">{formatInline(line.slice(4))}</h3>);
    } else if (line.startsWith('- ') || line.startsWith('* ')) {
      elements.push(
        <li key={i} className="ml-4 list-disc text-slate-300 my-0.5">
          {formatInline(line.slice(2))}
        </li>
      );
    } else if (/^\d+\.\s/.test(line)) {
      const match = line.match(/^(\d+\.)\s(.*)/);
      elements.push(
        <li key={i} className="ml-4 list-decimal text-slate-300 my-0.5">
          {match ? formatInline(match[2]) : formatInline(line)}
        </li>
      );
    } else if (line.startsWith('> ')) {
      elements.push(
        <blockquote key={i} className="border-l-2 border-teal-500/60 pl-3.5 italic text-slate-400 my-2">
          {formatInline(line.slice(2))}
        </blockquote>
      );
    } else if (line.trim() === '') {
      elements.push(<div key={i} className="h-1.5" />);
    } else {
      elements.push(<p key={i} className="my-1">{formatInline(line)}</p>);
    }
  }

  return <>{elements}</>;
};

function formatInline(str: string): React.ReactNode {
  // Bold: **text** or __text__
  // Italic: *text* or _text_
  // Inline code: `text`
  const parts: React.ReactNode[] = [];
  let remaining = str;
  let keyIdx = 0;

  const regex = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let match;
  let lastIndex = 0;

  while ((match = regex.exec(remaining)) !== null) {
    if (match.index > lastIndex) {
      parts.push(remaining.substring(lastIndex, match.index));
    }
    const token = match[0];
    if (token.startsWith('`') && token.endsWith('`')) {
      parts.push(
        <code key={keyIdx++} className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700/60 text-teal-300 font-mono text-[13px]">
          {token.slice(1, -1)}
        </code>
      );
    } else if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(<strong key={keyIdx++} className="font-semibold text-slate-100">{token.slice(2, -2)}</strong>);
    } else if (token.startsWith('*') && token.endsWith('*')) {
      parts.push(<em key={keyIdx++} className="italic text-slate-200">{token.slice(1, -1)}</em>);
    }
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < remaining.length) {
    parts.push(remaining.substring(lastIndex));
  }

  return parts.length > 0 ? parts : str;
}
