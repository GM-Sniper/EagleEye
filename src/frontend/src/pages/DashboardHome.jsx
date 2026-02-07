import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardHeader } from '../components/ui/Card';
import { StatCard } from '../components/dashboard/StatCard';
import { ForecastChart } from '../components/dashboard/ForecastChart';
import { DashboardService } from '../services/api';
import { cn } from '../components/ui/Card';
import {
    DollarSign,
    ShoppingBag,
    Package,
    Activity,
    TrendingUp,
    AlertTriangle
} from 'lucide-react';

export const DashboardHome = () => {
    const navigate = useNavigate();
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const loadDashboard = async () => {
            try {
                const result = await DashboardService.getDashboard();
                setData(result);
            } catch (err) {
                console.error("Failed to load dashboard, using mock data", err);
                setError("Failed to connect to EagleEye Backend. Ensure API is running.");
            } finally {
                setLoading(false);
            }
        };

        loadDashboard();
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-[500px]">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-eagle-green"></div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-6 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-center">
                <AlertTriangle className="w-8 h-8 mx-auto mb-2" />
                <h3 className="font-semibold">Connection Error</h3>
                <p>{error}</p>
            </div>
        );
    }

    const { summary, forecast, weekly_pattern, hourly_pattern, top_items, abc_classification } = data;

    return (
        <div className="flex flex-col gap-6 w-full animate-in fade-in duration-700">
            {/* 1. KPI Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
                <StatCard
                    title="Total Revenue"
                    value={`${summary.total_revenue.toLocaleString()} DKK`}
                    subValue="+12.5% vs last week"
                    trend="up"
                    icon={DollarSign}
                    color="green"
                />
                <StatCard
                    title="14d Model Accuracy"
                    value={summary.model_accuracy}
                    subValue={`MAPE: ${summary.mape}`}
                    trend="up"
                    icon={Activity}
                    color="purple"
                />
                <StatCard
                    title="Active Items"
                    value={summary.unique_items}
                    subValue="System Analysis Ready"
                    trend="neutral"
                    icon={Package}
                    color="amber"
                />
                <StatCard
                    title="Total Orders"
                    value={summary.total_orders.toLocaleString()}
                    subValue="Across all regions"
                    trend="up"
                    icon={ShoppingBag}
                    color="neutral"
                />
            </div>

            {/* 2. Primary Analytics Tier */}
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                {/* Main Forecast - Taking 2/3 of wide screens */}
                <div className="xl:col-span-2 h-full">
                    <ForecastChart data={forecast} />
                </div>

                {/* Right Column: Mini Widgets stack */}
                <div className="flex flex-col gap-6 h-full">
                    {/* Top Performing Items */}
                    <Card className="flex-1 min-h-[300px]">
                        <CardHeader title="Top Items" subtitle="Highest revenue this week" />
                        <div className="mt-4 space-y-4">
                            {top_items.map((item, i) => (
                                <div key={i} className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 transition-colors group cursor-default">
                                    <div className="flex items-center gap-3">
                                        <div className="w-8 h-8 rounded-lg bg-eagle-green/20 flex items-center justify-center text-eagle-green font-bold text-xs ring-1 ring-eagle-green/30">
                                            {i + 1}
                                        </div>
                                        <div>
                                            <p className="text-sm font-semibold text-white group-hover:text-eagle-green transition-colors">{item.item_name}</p>
                                            <p className="text-xs text-gray-500">Tier {item.abc_class} Item</p>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-sm font-bold text-white">{item.revenue.toLocaleString()} DKK</p>
                                        <p className="text-[10px] text-gray-500 uppercase tracking-tighter">Revenue</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </Card>
                </div>
            </div>

            {/* 3. Secondary Analytics Tier */}
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6">
                {/* Weekly Traffic Patterns */}
                <Card className="xl:col-span-1">
                    <CardHeader title="Weekly Traffic" subtitle="Orders by day" />
                    <div className="mt-6 space-y-3">
                        {weekly_pattern.map((day) => (
                            <div key={day.day} className="space-y-1 group">
                                <div className="flex justify-between text-xs">
                                    <span className="text-gray-400 group-hover:text-white transition-colors">{day.day}</span>
                                    <span className="text-white font-medium">{day.orders}</span>
                                </div>
                                <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-eagle-green/50 rounded-full group-hover:bg-eagle-green transition-all duration-300"
                                        style={{ width: `${(day.orders / Math.max(...weekly_pattern.map(d => d.orders))) * 100}%` }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </Card>

                {/* Hourly Pattern Visualization */}
                <Card className="xl:col-span-2">
                    <CardHeader title="Hourly Activity" subtitle="Peak demand periods" />
                    <div className="mt-6 h-40 flex items-end gap-1">
                        {hourly_pattern.map((h, i) => (
                            <div
                                key={i}
                                className="flex-1 bg-eagle-green/20 rounded-t-sm hover:bg-eagle-green/60 transition-all cursor-help relative group"
                                style={{ height: `${(h.orders / Math.max(...hourly_pattern.map(p => p.orders))) * 100}%` }}
                            >
                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-zinc-900 border border-white/10 rounded text-[10px] opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-50">
                                    {h.hour}:00 - {h.orders} orders
                                </div>
                            </div>
                        ))}
                    </div>
                    <div className="mt-2 flex justify-between text-[10px] text-gray-500 uppercase tracking-widest px-1">
                        <span>12am</span>
                        <span>12pm</span>
                        <span>11pm</span>
                    </div>
                </Card>

                {/* ABC Distribution */}
                <Card className="xl:col-span-1 flex flex-col justify-between">
                    <div>
                        <CardHeader title="Inventory Tiers" subtitle="ABC Classification" />
                        <div className="mt-6 space-y-4">
                            {Object.entries(abc_classification).map(([tier, count]) => (
                                <div key={tier} className="flex items-center gap-4">
                                    <div className={cn(
                                        "w-10 h-10 rounded-xl flex items-center justify-center font-bold text-lg",
                                        tier === 'A' ? "bg-eagle-green/20 text-eagle-green ring-1 ring-eagle-green/50" :
                                            tier === 'B' ? "bg-amber-500/20 text-amber-500 ring-1 ring-amber-500/50" :
                                                "bg-zinc-800 text-gray-400 ring-1 ring-zinc-700"
                                    )}>
                                        {tier}
                                    </div>
                                    <div className="flex-1">
                                        <div className="flex justify-between items-baseline mb-1">
                                            <span className="text-xs font-medium text-gray-400">Class {tier}</span>
                                            <span className="text-sm font-bold text-white">{count} items</span>
                                        </div>
                                        <div className="h-1.5 w-full bg-white/5 rounded-full">
                                            <div
                                                className={cn(
                                                    "h-full rounded-full transition-all duration-500",
                                                    tier === 'A' ? "bg-eagle-green" : tier === 'B' ? "bg-amber-500" : "bg-zinc-500"
                                                )}
                                                style={{ width: `${(count / summary.unique_items) * 100}%` }}
                                            />
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                    <div className="mt-6 p-3 rounded-lg bg-white/5 border border-white/5 flex gap-3 text-xs text-gray-400">
                        <TrendingUp className="w-4 h-4 text-eagle-green shrink-0" />
                        <span>Class A items generate 80% of your total market revenue.</span>
                    </div>
                </Card>
            </div>

            {/* 4. Actionable Insights Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pb-6">
                <Card className="bg-gradient-to-br from-eagle-green/10 via-transparent to-transparent border-eagle-green/20 hover:border-eagle-green/40 transition-all group overflow-hidden relative">
                    <div className="p-2 relative z-10">
                        <div className="w-12 h-12 rounded-2xl bg-eagle-green/10 flex items-center justify-center mb-4 ring-1 ring-eagle-green/20 group-hover:scale-110 transition-transform">
                            <TrendingUp className="w-6 h-6 text-eagle-green" />
                        </div>
                        <h3 className="text-xl font-bold text-white mb-2">Smart Insights</h3>
                        <p className="text-gray-400 text-sm mb-6 max-w-md">
                            ML Models detected 5 high-potential items for margin optimization. Adjusting these could increase weekly profit by ~4.2%.
                        </p>
                        <button
                            onClick={() => navigate('/pricing')}
                            className="bg-eagle-green hover:bg-eagle-green-hover text-black px-6 py-2.5 rounded-xl font-bold text-sm transition-all hover:shadow-[0_0_20px_rgba(0,224,84,0.4)] active:scale-95"
                        >
                            Optimize Pricing
                        </button>
                    </div>
                    <div className="absolute -right-12 -bottom-12 w-64 h-64 bg-eagle-green/5 blur-[80px] rounded-full group-hover:bg-eagle-green/10 transition-all" />
                </Card>

                <Card className="bg-gradient-to-br from-amber-500/10 via-transparent to-transparent border-amber-500/20 hover:border-amber-500/40 transition-all group overflow-hidden relative">
                    <div className="p-2 relative z-10">
                        <div className="w-12 h-12 rounded-2xl bg-amber-500/10 flex items-center justify-center mb-4 ring-1 ring-amber-500/20 group-hover:scale-110 transition-transform">
                            <AlertTriangle className="w-6 h-6 text-amber-500" />
                        </div>
                        <h3 className="text-xl font-bold text-white mb-2">Restocking Priority</h3>
                        <p className="text-gray-400 text-sm mb-6 max-w-md">
                            3 critical ingredients are projected to fall below safety stock within the next 48 hours based on forecasted orders.
                        </p>
                        <button
                            onClick={() => navigate('/inventory')}
                            className="bg-amber-500 hover:bg-amber-600 text-black px-6 py-2.5 rounded-xl font-bold text-sm transition-all hover:shadow-[0_0_20px_rgba(245,158,11,0.4)] active:scale-95"
                        >
                            View Restock Plan
                        </button>
                    </div>
                    <div className="absolute -right-12 -bottom-12 w-64 h-64 bg-amber-500/5 blur-[80px] rounded-full group-hover:bg-amber-500/10 transition-all" />
                </Card>
            </div>
        </div>
    );
};
