import React from 'react';
import { cn } from './Card';

export const Badge = ({ children, variant = 'default', className }) => {
    const variants = {
        default: "bg-gray-800 text-gray-300 border-gray-700",
        success: "bg-eagle-green/10 text-eagle-green border-eagle-green/20",
        warning: "bg-amber-500/10 text-amber-500 border-amber-500/20",
        error: "bg-red-500/10 text-red-500 border-red-500/20",
        info: "bg-blue-500/10 text-blue-500 border-blue-500/20",
    };

    return (
        <span
            className={cn(
                "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border",
                variants[variant],
                className
            )}
        >
            {children}
        </span>
    );
};
