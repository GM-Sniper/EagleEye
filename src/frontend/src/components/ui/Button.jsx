import React from 'react';
import { cn } from './Card'; // Reusing cn utility
import { Loader2 } from 'lucide-react';

export const Button = ({
    children,
    variant = 'primary',
    size = 'md',
    className,
    isLoading,
    disabled,
    ...props
}) => {
    const variants = {
        primary: "bg-eagle-green text-black font-semibold hover:bg-eagle-green-hover shadow-[0_0_15px_rgba(0,224,84,0.3)] hover:shadow-[0_0_25px_rgba(0,224,84,0.5)] border-transparent",
        secondary: "bg-transparent text-white border border-white/20 hover:bg-white/5 hover:border-white/40",
        ghost: "bg-transparent text-gray-400 hover:text-white hover:bg-white/5",
        danger: "bg-red-500/10 text-red-500 border border-red-500/20 hover:bg-red-500/20",
    };

    const sizes = {
        sm: "px-3 py-1.5 text-xs",
        md: "px-4 py-2 text-sm",
        lg: "px-6 py-3 text-base",
    };

    return (
        <button
            className={cn(
                "inline-flex items-center justify-center rounded-lg transition-all duration-200 active:scale-95 disabled:opacity-50 disabled:pointer-events-none",
                variants[variant],
                sizes[size],
                className
            )}
            disabled={isLoading || disabled}
            {...props}
        >
            {isLoading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            {children}
        </button>
    );
};
