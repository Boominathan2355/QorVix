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
        {label && <span className="font-medium text-slate-300">{label}</span>}
        <span className="font-mono text-teal-400 font-semibold bg-teal-950/50 px-2 py-0.5 rounded-md border border-teal-800/40">
          {valueDisplay !== undefined ? valueDisplay : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        className={`w-full cursor-pointer accent-teal-400 ${className}`}
        {...props}
      />
    </div>
  );
};
