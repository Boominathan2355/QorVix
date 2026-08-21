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
    success: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/30',
    warning: 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30',
    danger: 'bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/30',
    info: 'bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/30',
    neutral: 'bg-secondary text-muted-foreground border-border',
    primary: 'bg-teal-500/10 text-teal-600 dark:text-teal-400 border-teal-500/30',
    purple: 'bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/30',
  };

  const sizes = {
    sm: 'text-[11px] px-2 py-0.5 rounded-md gap-1 font-medium',
    md: 'text-xs px-2.5 py-1 rounded-lg gap-1.5 font-medium',
  };

  const dotColors = {
    success: 'bg-emerald-500',
    warning: 'bg-amber-500',
    danger: 'bg-red-500',
    info: 'bg-sky-500',
    neutral: 'bg-slate-400',
    primary: 'bg-teal-500',
    purple: 'bg-purple-500',
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
