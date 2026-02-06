import React from "react";

export default function CommandCenter() {
    return (
        <div className="space-y-6 animate-in fade-in zoom-in duration-500">
            {/* Welcome / Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h2 className="text-3xl font-black text-white tracking-tighter italic uppercase">System Overview</h2>
                    <p className="text-slate-400 text-sm">Real-time operational intelligence.</p>
                </div>
                <div className="flex items-center gap-3">
                    <span className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-xs font-medium text-emerald-400">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                        </span>
                        Operational
                    </span>
                    <button className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg border border-slate-700 transition-colors">
                        Generate Report
                    </button>
                </div>
            </div>

            {/* High-Level KPIs */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Network Health */}
                <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-6 backdrop-blur-sm">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-medium text-slate-400">Network Health</h3>
                        <svg className="text-emerald-500" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
                    </div>
                    <div className="text-3xl font-bold text-white">98.2%</div>
                    <div className="mt-2 h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-emerald-500 w-[98.2%]"></div>
                    </div>
                    <p className="mt-2 text-xs text-slate-500">All nodes active. Latency: 45ms</p>
                </div>

                {/* Global Inventory Value */}
                <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-6 backdrop-blur-sm">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-medium text-slate-400">Inventory Value</h3>
                        <svg className="text-blue-500" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="16" /><line x1="8" y1="12" x2="16" y2="12" /></svg>
                    </div>
                    <div className="text-3xl font-bold text-white">$1.2M</div>
                    <p className="mt-2 text-xs text-emerald-400 flex items-center gap-1">
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18" /><polyline points="17 6 23 6 23 12" /></svg>
                        +4.5% vs last week
                    </p>
                </div>

                {/* Pending Actions */}
                <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-6 backdrop-blur-sm">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-medium text-slate-400">Action Required</h3>
                        <svg className="text-orange-500" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
                    </div>
                    <div className="text-3xl font-bold text-white">12</div>
                    <p className="mt-2 text-xs text-orange-400">Critical alerts needing attention</p>
                </div>
            </div>

            {/* Predictive Intelligence */}
            <div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Demand Surge */}
                    <div className="relative overflow-hidden rounded-2xl border border-sky-500/30 bg-sky-900/10 p-6 backdrop-blur-sm group hover:border-sky-500/50 transition-colors">
                        <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                            <svg className="text-sky-400 w-24 h-24" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
                        </div>
                        <div className="relative z-10">
                            <div className="flex items-center gap-2 mb-2">
                                <span className="px-2 py-1 rounded-md bg-sky-500/20 text-sky-300 text-xs font-bold uppercase tracking-wider">Demand Surge</span>
                                <span className="text-xs text-sky-200/60">Confidence: 94%</span>
                            </div>
                            <h4 className="text-xl font-bold text-white mb-1">Product X-200</h4>
                            <p className="text-sky-200 text-sm mb-4">Demand expected to rise by <span className="font-bold text-white">40%</span> next week due to seasonal trends.</p>
                            <button className="text-xs font-medium text-sky-400 hover:text-sky-300 flex items-center gap-1 transition-colors">
                                View Forecast Details <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14" /><path d="m12 5 7 7-7 7" /></svg>
                            </button>
                        </div>
                    </div>

                    {/* Large Order Forecast */}
                    <div className="relative overflow-hidden rounded-2xl border border-blue-500/30 bg-blue-900/10 p-6 backdrop-blur-sm group hover:border-blue-500/50 transition-colors">
                        <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                            <svg className="text-blue-400 w-24 h-24" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" /><polyline points="3.27 6.96 12 12.01 20.73 6.96" /><line x1="12" y1="22.08" x2="12" y2="12" /></svg>
                        </div>
                        <div className="relative z-10">
                            <div className="flex items-center gap-2 mb-2">
                                <span className="px-2 py-1 rounded-md bg-blue-500/20 text-blue-300 text-xs font-bold uppercase tracking-wider">Incoming Bulk</span>
                                <span className="text-xs text-blue-200/60">Est: Friday</span>
                            </div>
                            <h4 className="text-xl font-bold text-white mb-1">Branch Y (Downtown)</h4>
                            <p className="text-blue-200 text-sm mb-4">Predicted large stock request of <span className="font-bold text-white">500 units</span> for Item Z.</p>
                            <button className="text-xs font-medium text-blue-400 hover:text-blue-300 flex items-center gap-1 transition-colors">
                                Prepare Allocation <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14" /><path d="m12 5 7 7-7 7" /></svg>
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Recent Activity / System Log */}
            <div className="grid grid-cols-1 gap-6">
                <div className="rounded-2xl border border-slate-800 bg-slate-900/30 flex flex-col backdrop-blur-sm">
                    <div className="border-b border-slate-800 px-6 py-4">
                        <h3 className="text-sm font-semibold text-white">Recent System Activity</h3>
                    </div>
                    <div className="p-4 space-y-4">
                        {[
                            { time: '10:42 AM', event: 'Stock update: Mellem box', user: 'Auto-Sync', type: 'info' },
                            { time: '09:15 AM', event: 'Critical Alert: Lille box stock low', user: 'System', type: 'alert' },
                            { time: '08:30 AM', event: 'User Login: Admin', user: 'Admin', type: 'info' },
                            { time: 'Yesterday', event: 'Database Backup Completed', user: 'System', type: 'success' },
                        ].map((log, i) => (
                            <div key={i} className="flex items-start gap-4 text-sm">
                                <span className="text-xs text-slate-500 w-16 pt-0.5">{log.time}</span>
                                <div className="flex-1">
                                    <p className={`font-medium ${log.type === 'alert' ? 'text-red-400' : log.type === 'success' ? 'text-emerald-400' : 'text-slate-300'}`}>
                                        {log.event}
                                    </p>
                                    <p className="text-xs text-slate-500">by {log.user}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                    <div className="mt-auto border-t border-slate-800 px-6 py-3">
                        <button className="text-xs text-emerald-400 hover:text-emerald-300 font-medium">View Full Log &rarr;</button>
                    </div>
                </div>
            </div>
        </div>
    );
}
