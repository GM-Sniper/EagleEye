import React from 'react';
import { Card } from '../ui/Card';
import { ArrowUpRight, ArrowDownRight, TrendingUp } from 'lucide-react';
import { cn } from '../ui/Card';

export const StatCard = ({ title, value, subValue, trend, icon: Icon, color = "green" }) => {
    const isPositive = trend === 'up';

    const colorMap = {
        green: "text-eagle-green bg-eagle-green/10",
        neutral: "text-gray-400 bg-white/5",
        amber: "text-amber-500 bg-amber-500/10",
        purple: "text-purple-500 bg-purple-500/10",
    };

    return (
        <Card hover className="relative overflow-hidden group border-white/5">
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-sm font-medium text-gray-400">{title}</p>
                    <h3 className="text-2xl font-bold text-white mt-2 group-hover:scale-105 transition-transform origin-left">
                        {value}
                    </h3>
                </div>
                <div className={cn("p-2 rounded-lg transition-colors border border-white/5", colorMap[color] || colorMap.neutral)}>
                    <Icon className="w-5 h-5" />
                </div>
            </div>

            <div className="mt-4 flex items-center text-sm">
                {trend && (
                    <span className={cn(
                        "flex items-center font-medium",
                        isPositive ? "text-eagle-green" : "text-red-500"
                    )}>
                        {isPositive ? <ArrowUpRight className="w-4 h-4 mr-1" /> : <ArrowDownRight className="w-4 h-4 mr-1" />}
                        {subValue}
                    </span>
                )}
                {!trend && subValue && (
                    <span className="text-gray-500">{subValue}</span>
                )}
            </div>

            {/* Background Glow */}
            <div className={cn(
                "absolute -right-6 -bottom-6 w-24 h-24 rounded-full blur-2xl opacity-0 group-hover:opacity-20 transition-opacity duration-500",
                color === 'green' ? "bg-eagle-green" :
                    color === 'neutral' ? "bg-gray-500" :
                        color === 'amber' ? "bg-amber-500" : "bg-purple-500"
            )} />
        </Card>
    );
};
