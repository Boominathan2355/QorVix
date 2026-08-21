import React from 'react';

export interface SliderProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  valueDisplay?: React.ReactNode;
  min?: number;
  max?: number;
  step?: number;
}

export const Slider: React.FC<SliderProps> = ({
  label,
  valueDisplay,
  min = 0,
  max = 100,
  step = 1,
  value,
  className = '',
  ...props
}) => {
  return (
    <div className="w-full space-y-1.5">
      <div className="flex justify-between items-center text-xs">
        {label && <span className="font-medium text-foreground">{label}</span>}
        <span className="font-mono text-teal-600 dark:text-teal-400 font-semibold bg-teal-500/10 px-2 py-0.5 rounded-md border border-teal-500/20">
          {valueDisplay !== undefined ? valueDisplay : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        className={`w-full cursor-pointer accent-teal-500 ${className}`}
        {...props}
      />
    </div>
  );
};
