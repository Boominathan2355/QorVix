import React from 'react';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  glass?: boolean;
  hover?: boolean;
  glow?: boolean;
}

export const Card: React.FC<CardProps> = ({
  children,
  className = '',
  glass = true,
  hover = false,
  glow = false,
  ...props
}) => {
  return (
    <div
      className={`rounded-2xl border ${
        glass
          ? 'bg-slate-900/60 backdrop-blur-xl border-slate-800/80 shadow-xl shadow-black/20'
          : 'bg-slate-900 border-slate-800'
      } ${
        hover ? 'transition-all duration-200 hover:border-slate-700 hover:bg-slate-900/80 hover:shadow-2xl' : ''
      } ${glow ? 'border-teal-500/30 shadow-teal-500/10 shadow-lg' : ''} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

export const CardHeader: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  children,
  className = '',
  ...props
}) => {
  return (
    <div className={`p-5 pb-3 flex items-center justify-between border-b border-slate-800/40 ${className}`} {...props}>
      {children}
    </div>
  );
};

export const CardTitle: React.FC<React.HTMLAttributes<HTMLHeadingElement>> = ({
  children,
  className = '',
  ...props
}) => {
  return (
    <h3 className={`font-semibold text-slate-100 tracking-tight text-base flex items-center gap-2 ${className}`} {...props}>
      {children}
    </h3>
  );
};

export const CardDescription: React.FC<React.HTMLAttributes<HTMLParagraphElement>> = ({
  children,
  className = '',
  ...props
}) => {
  return (
    <p className={`text-xs text-slate-400 mt-0.5 ${className}`} {...props}>
      {children}
    </p>
  );
};

export const CardContent: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  children,
  className = '',
  ...props
}) => {
  return (
    <div className={`p-5 ${className}`} {...props}>
      {children}
    </div>
  );
};

export const CardFooter: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  children,
  className = '',
  ...props
}) => {
  return (
    <div className={`p-5 pt-3 border-t border-slate-800/40 flex items-center justify-between ${className}`} {...props}>
      {children}
    </div>
  );
};
