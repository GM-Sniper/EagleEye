import React from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs) {
    return twMerge(clsx(inputs));
}

export const Card = ({ children, className, hover = false }) => {
    return (
        <div
            className={cn(
                "glass-card rounded-2xl p-6 text-white border border-white/5",
                hover && "hover:border-eagle-green/30 hover:bg-slate-800/80 hover:shadow-lg hover:shadow-eagle-green/10",
                className
            )}
        >
            {children}
        </div>
    );
};

export const CardHeader = ({ title, subtitle, action }) => (
    <div className="flex items-center justify-between mb-6">
        <div>
            <h3 className="text-lg font-semibold text-white">{title}</h3>
            {subtitle && <p className="text-sm text-gray-400 mt-1">{subtitle}</p>}
        </div>
        {action && <div>{action}</div>}
    </div>
);
