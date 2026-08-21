import React from 'react';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'success' | 'warning' | 'danger' | 'info' | 'neutral' | 'primary' | 'purple';
  size?: 'sm' | 'md';
  pulse?: boolean;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  className = '',
  variant = 'neutral',
  size = 'md',
  pulse = false,
  ...props
}) => {
  const variants = {
    success: 'bg-emerald-950/60 text-emerald-300 border-emerald-800/50',
    warning: 'bg-amber-950/60 text-amber-300 border-amber-800/50',
    danger: 'bg-red-950/60 text-red-300 border-red-800/50',
    info: 'bg-sky-950/60 text-sky-300 border-sky-800/50',
    neutral: 'bg-slate-800/70 text-slate-300 border-slate-700/60',
    primary: 'bg-teal-950/60 text-teal-300 border-teal-800/50',
    purple: 'bg-purple-950/60 text-purple-300 border-purple-800/50',
  };

  const sizes = {
    sm: 'text-[11px] px-2 py-0.5 rounded-md gap-1 font-medium',
    md: 'text-xs px-2.5 py-1 rounded-lg gap-1.5 font-medium',
  };

  const dotColors = {
    success: 'bg-emerald-400',
    warning: 'bg-amber-400',
    danger: 'bg-red-400',
    info: 'bg-sky-400',
    neutral: 'bg-slate-400',
    primary: 'bg-teal-400',
    purple: 'bg-purple-400',
  };

  return (
    <span
      className={`inline-flex items-center border font-mono tracking-tight select-none ${variants[variant]} ${sizes[size]} ${className}`}
      {...props}
    >
      {pulse && (
        <span className="relative flex h-2 w-2">
          <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${dotColors[variant]}`} />
          <span className={`relative inline-flex rounded-full h-2 w-2 ${dotColors[variant]}`} />
        </span>
      )}
      {children}
    </span>
  );
};
