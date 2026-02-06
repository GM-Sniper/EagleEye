import React from "react";

export default function CommandCenter() {
    return (
        <div className="space-y-6 animate-in fade-in zoom-in duration-500">
            {/* Welcome / Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h2 className="text-2xl font-bold text-white tracking-tight">System Overview</h2>
                    <p className="text-slate-400 text-sm">Real-time operational intelligence and system health.</p>
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

            {/* Recent Activity / System Log */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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

                {/* Quick Actions Panel */}
                <div className="rounded-2xl border border-slate-800 bg-slate-900/30 flex flex-col backdrop-blur-sm">
                    <div className="border-b border-slate-800 px-6 py-4">
                        <h3 className="text-sm font-semibold text-white">Quick Actions</h3>
                    </div>
                    <div className="p-6 grid grid-cols-2 gap-4">
                        <button className="flex flex-col items-center justify-center gap-2 p-4 rounded-xl border border-slate-800 bg-slate-900/50 hover:bg-slate-800 transition-all group">
                            <div className="h-10 w-10 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-500 group-hover:bg-emerald-500 group-hover:text-white transition-colors">
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
                            </div>
                            <span className="text-sm font-medium text-slate-300">Add Item</span>
                        </button>
                        <button className="flex flex-col items-center justify-center gap-2 p-4 rounded-xl border border-slate-800 bg-slate-900/50 hover:bg-slate-800 transition-all group">
                            <div className="h-10 w-10 rounded-full bg-blue-500/10 flex items-center justify-center text-blue-500 group-hover:bg-blue-500 group-hover:text-white transition-colors">
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>
                            </div>
                            <span className="text-sm font-medium text-slate-300">Import CSV</span>
                        </button>
                        <button className="flex flex-col items-center justify-center gap-2 p-4 rounded-xl border border-slate-800 bg-slate-900/50 hover:bg-slate-800 transition-all group">
                            <div className="h-10 w-10 rounded-full bg-orange-500/10 flex items-center justify-center text-orange-500 group-hover:bg-orange-500 group-hover:text-white transition-colors">
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 22h14a2 2 0 0 0 2-2V7.5L14.5 2H6a2 2 0 0 0-2 2v4" /><polyline points="14 2 14 8 20 8" /><path d="M2 15h10" /><path d="M9 18l3-3-3-3" /></svg>
                            </div>
                            <span className="text-sm font-medium text-slate-300">Run Forecast</span>
                        </button>
                        <button className="flex flex-col items-center justify-center gap-2 p-4 rounded-xl border border-slate-800 bg-slate-900/50 hover:bg-slate-800 transition-all group">
                            <div className="h-10 w-10 rounded-full bg-slate-500/10 flex items-center justify-center text-slate-500 group-hover:bg-slate-500 group-hover:text-white transition-colors">
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33h.09a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" /></svg>
                            </div>
                            <span className="text-sm font-medium text-slate-300">Settings</span>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
