import React, { useState } from "react";

// --- Types ---
interface StatItem {
    label: string;
    value: string;
    sub: string;
    trend: string;
    trendUp: boolean;
    alert?: boolean;
}

interface InventoryItem {
    id: number;
    name: string;
    stock: number;
    capacity: number;
    status: 'Critical' | 'Understocked' | 'Healthy' | 'Overstocked';
    mean: number;
    abc: string;
    color: 'emerald' | 'red' | 'blue' | 'orange';
}

export default function StockTerminal() {
    // State for data - currently empty as requested
    const [stats, setStats] = useState<StatItem[]>([]);
    const [inventory, setInventory] = useState<InventoryItem[]>([]);

    return (
        <div className="space-y-8 animate-in fade-in zoom-in duration-500">
            {/* Header / Title for this section */}
            <div>
                <h3 className="text-lg font-semibold text-white">Stock Operations</h3>
                <p className="text-sm text-slate-400">Manage real-time inventory levels and item intelligence.</p>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
                {stats.length > 0 ? (
                    stats.map((stat, i) => (
                        <div
                            key={i}
                            className={`group relative overflow-hidden rounded-2xl border p-5 transition-all duration-300 hover:shadow-lg hover:shadow-emerald-900/10 ${stat.alert
                                ? "bg-slate-900/40 border-red-500/20 hover:border-red-500/40"
                                : "bg-slate-900/40 border-slate-800 hover:border-emerald-500/30"
                                }`}
                        >
                            <div className="flex items-start justify-between">
                                <div>
                                    <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">{stat.label}</p>
                                    <h3 className="mt-2 text-3xl font-bold text-white tracking-tight">{stat.value}</h3>
                                    <p className="text-xs text-emerald-500/80 font-medium mt-1">{stat.sub}</p>
                                </div>
                                <div className={`flex items-center gap-1 text-xs font-semibold rounded-full px-2 py-1 ${stat.trendUp ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                                    {stat.trend}
                                </div>
                            </div>
                            {/* Decoration */}
                            <div className="absolute -right-6 -top-6 h-24 w-24 rounded-full bg-gradient-to-br from-white/5 to-white/0 blur-2xl transition-all group-hover:from-emerald-500/10" />
                        </div>
                    ))
                ) : (
                    <div className="col-span-full rounded-2xl border border-dashed border-slate-800 bg-slate-900/20 p-8 text-center text-slate-500">
                        <p>No statistics available. Connect data source.</p>
                    </div>
                )}
            </div>

            {/* Filters & Actions */}
            <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center bg-slate-900/50 p-1 rounded-xl border border-slate-800">
                    {['All Items', 'Critical', 'Understocked', 'Healthy', 'Overstocked'].map((filter, i) => (
                        <button key={filter} className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all ${i === 0 ? 'bg-slate-800 text-white shadow-sm' : 'text-slate-400 hover:text-white hover:bg-slate-800/50'}`}>
                            {filter}
                        </button>
                    ))}
                </div>

                <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-slate-500 uppercase tracking-widest mr-2">Found: {inventory.length} Units</span>
                    <button className="flex items-center justify-center h-8 w-8 rounded-lg border border-slate-800 text-slate-400 hover:text-white hover:bg-slate-800 transition-colors">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" /></svg>
                    </button>
                </div>
            </div>

            {/* Data Table */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/30 overflow-hidden backdrop-blur-sm">
                <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                        <thead>
                            <tr className="border-b border-slate-800 bg-slate-900/50 text-xs uppercase tracking-wider text-slate-500">
                                <th className="px-6 py-4 font-semibold">Item Intelligence</th>
                                <th className="px-6 py-4 font-semibold">Status</th>
                                <th className="px-6 py-4 font-semibold">Stock Level</th>
                                <th className="px-6 py-4 font-semibold">Daily Mean</th>
                                <th className="px-6 py-4 font-semibold">ABC Class</th>
                                <th className="px-6 py-4 font-semibold text-right">Action</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/50">
                            {inventory.length > 0 ? (
                                inventory.map((item, i) => (
                                    <tr key={i} className="group hover:bg-slate-800/30 transition-colors">
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-3">
                                                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-800 text-xs font-bold text-emerald-500 font-mono">
                                                    {item.id}
                                                </div>
                                                <div>
                                                    <p className="font-semibold text-slate-200">{item.name}</p>
                                                    <p className="text-xs text-slate-500">ID: 6535{item.id}</p>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className={`inline-flex items-center rounded-md px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${item.color === 'emerald' ? 'bg-emerald-400/10 text-emerald-400 ring-emerald-400/20' :
                                                item.color === 'red' ? 'bg-red-400/10 text-red-400 ring-red-400/20' :
                                                    item.color === 'blue' ? 'bg-blue-400/10 text-blue-400 ring-blue-400/20' :
                                                        'bg-orange-400/10 text-orange-400 ring-orange-400/20'
                                                }`}>
                                                {item.status.toUpperCase()}
                                            </span>
                                        </td>
                                        {/* ... (Rest of columns similar to original) ... */}
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td colSpan={6} className="px-6 py-12 text-center text-slate-500">
                                        <div className="flex flex-col items-center justify-center gap-2">
                                            <div className="h-10 w-10 rounded-full bg-slate-800/50 flex items-center justify-center text-slate-600">
                                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" /><polyline points="3.27 6.96 12 12.01 20.73 6.96" /><line x1="12" y1="22.08" x2="12" y2="12" /></svg>
                                            </div>
                                            <p>No inventory items found.</p>
                                        </div>
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
                {/* Pagination / Footer */}
                <div className="flex items-center justify-between border-t border-slate-800 bg-slate-900/50 px-6 py-3">
                    <p className="text-xs text-slate-500">Showing {inventory.length} of {inventory.length} items</p>
                    <div className="flex gap-2">
                        <button className="px-3 py-1 text-xs text-slate-400 hover:text-white disabled:opacity-50">Previous</button>
                        <button className="px-3 py-1 text-xs text-slate-400 hover:text-white">Next</button>
                    </div>
                </div>
            </div>
        </div>
    );
}
