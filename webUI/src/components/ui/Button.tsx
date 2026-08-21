import React from 'react';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger' | 'glow';
  size?: 'sm' | 'md' | 'lg' | 'icon';
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      children,
      className = '',
      variant = 'primary',
      size = 'md',
      loading = false,
      disabled,
      leftIcon,
      rightIcon,
      ...props
    },
    ref
  ) => {
    const baseStyles =
      'inline-flex items-center justify-center font-medium transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500/50 disabled:opacity-50 disabled:pointer-events-none active:scale-[0.98] select-none';

    const variants = {
      primary:
        'bg-teal-500 hover:bg-teal-400 text-slate-950 font-semibold shadow-lg shadow-teal-500/20 hover:shadow-teal-500/35 border border-teal-400/30',
      secondary:
        'bg-slate-800/80 hover:bg-slate-700/80 text-slate-100 border border-slate-700/50 shadow-sm',
      outline:
        'border border-slate-700/80 hover:border-slate-500 bg-transparent hover:bg-slate-800/50 text-slate-200',
      ghost:
        'hover:bg-slate-800/60 text-slate-300 hover:text-slate-100',
      danger:
        'bg-red-500/20 hover:bg-red-500/30 text-red-300 border border-red-500/40 hover:border-red-500/60 shadow-sm',
      glow:
        'bg-gradient-to-r from-teal-500 to-emerald-500 hover:from-teal-400 hover:to-emerald-400 text-slate-950 font-semibold shadow-lg shadow-teal-500/30 border border-teal-300/40',
    };

    const sizes = {
      sm: 'text-xs h-8 px-3 rounded-lg gap-1.5',
      md: 'text-sm h-9.5 px-4 rounded-xl gap-2',
      lg: 'text-base h-11 px-5 rounded-xl gap-2.5',
      icon: 'h-9 w-9 rounded-xl p-0',
    };

    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
        {...props}
      >
        {loading ? (
          <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-current" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        ) : (
          leftIcon
        )}
        {children}
        {!loading && rightIcon}
      </button>
    );
  }
);
Button.displayName = 'Button';
