import React from "react";

// --- Mock Data ---
const FORECAST_DATA = [
    { day: 'Mon', actual: 40, predicted: 42 },
    { day: 'Tue', actual: 35, predicted: 38 },
    { day: 'Wed', actual: 50, predicted: 48 },
    { day: 'Thu', actual: 45, predicted: 55 },
    { day: 'Fri', actual: 60, predicted: 75 }, // Forecast starts diverging
    { day: 'Sat', actual: null, predicted: 85 },
    { day: 'Sun', actual: null, predicted: 90 },
];

const MOVERS = [
    { name: "Avocados (Hass)", change: "+200%", trend: "up", reason: "Viral Recipe Trend" },
    { name: "Sourdough Bread", change: "+45%", trend: "up", reason: "Weekend Spike" },
    { name: "Frozen Pizza", change: "-15%", trend: "down", reason: "Competitor Promo" },
    { name: "Almond Milk", change: "+12%", trend: "up", reason: "Steady Growth" },
];

export default function DemandProjection() {
    return (
        <div className="space-y-8 animate-in fade-in zoom-in duration-500">
            {/* Header */}
            <div>
                <h2 className="text-3xl font-black text-white tracking-tighter italic uppercase">Demand Intelligence</h2>
                <p className="text-sm text-slate-400">AI-driven forecasting and market velocity analysis.</p>
            </div>

            {/* AI Forecast Hero Section */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 backdrop-blur-sm relative overflow-hidden group">
                {/* Background Glow */}
                <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-blue-500/10 blur-3xl pointer-events-none" />

                <div className="flex items-center justify-between mb-6">
                    <div>
                        <h3 className="text-lg font-bold text-white flex items-center gap-2">
                            <div className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
                            7-Day Forecast
                        </h3>
                        <p className="text-xs text-slate-500">Actual vs. AI Predicted Velocity</p>
                    </div>
                    <div className="flex items-center gap-4 text-xs font-medium">
                        <div className="flex items-center gap-2">
                            <span className="w-3 h-1 bg-emerald-500 rounded-full"></span>
                            <span className="text-slate-300">Actual Sales</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="w-3 h-1 bg-blue-500 rounded-full border border-blue-500 border-dashed"></span>
                            <span className="text-blue-300">AI Prediction</span>
                        </div>
                    </div>
                </div>

                {/* Custom SVG Chart Area */}
                <div className="h-64 w-full relative">
                    {/* Y-Axis Grid Lines */}
                    <div className="absolute inset-0 flex flex-col justify-between text-xs text-slate-600">
                        {[100, 75, 50, 25, 0].map(val => (
                            <div key={val} className="flex items-center gap-2 w-full">
                                <span className="w-6 text-right opacity-50">{val}</span>
                                <div className="h-[1px] flex-1 bg-slate-800/50 border-t border-dashed border-slate-800"></div>
                            </div>
                        ))}
                    </div>

                    {/* Chart Bars/Lines Layer */}
                    <div className="absolute inset-0 pl-8 pt-2 pb-6 flex items-end justify-between px-4">
                        {FORECAST_DATA.map((data, i) => (
                            <div key={i} className="flex flex-col items-center gap-2 group/bar w-full">
                                <div className="relative w-full flex justify-center h-full items-end gap-1">
                                    {/* Actual Bar */}
                                    {data.actual !== null && (
                                        <div
                                            style={{ height: `${data.actual}%` }}
                                            className="w-3 bg-emerald-500 rounded-t-sm hover:opacity-80 transition-all"
                                        ></div>
                                    )}
                                    {/* Predicted Bar (Ghost) */}
                                    <div
                                        style={{ height: `${data.predicted}%` }}
                                        className={`w-3 rounded-t-sm transition-all relative ${data.actual === null ? 'bg-blue-500/80' : 'bg-transparent border-t-2 border-r-2 border-l-2 border-blue-500/30'}`}
                                    >
                                        {/* Tooltip on Hover */}
                                        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-slate-800 text-white text-[10px] rounded opacity-0 group-hover/bar:opacity-100 transition-opacity whitespace-nowrap z-10 pointer-events-none border border-slate-700">
                                            Scale: {data.predicted}
                                        </div>
                                    </div>
                                </div>
                                <span className="text-[10px] uppercase font-bold text-slate-500">{data.day}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Movers & Shakers Grid */}
            <div>
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                    <svg className="text-orange-500" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" /></svg>
                    Movers & Shakers
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {MOVERS.map((item, i) => (
                        <div key={i} className="group p-4 rounded-xl border border-slate-800 bg-slate-900/30 hover:bg-slate-800/50 transition-all">
                            <div className="flex items-start justify-between mb-2">
                                <div className={`p-2 rounded-lg ${item.trend === 'up' ? 'bg-emerald-500/10 text-emerald-500' : 'bg-red-500/10 text-red-500'}`}>
                                    {item.trend === 'up'
                                        ? <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18" /><polyline points="17 6 23 6 23 12" /></svg>
                                        : <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6" /><polyline points="17 18 23 18 23 12" /></svg>
                                    }
                                </div>
                                <span className={`text-lg font-black italic tracking-tighter ${item.trend === 'up' ? 'text-white' : 'text-slate-400'}`}>{item.change}</span>
                            </div>
                            <h4 className="font-bold text-slate-200">{item.name}</h4>
                            <p className="text-xs text-slate-500 mt-1 uppercase tracking-wide">{item.reason}</p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Prescriptive Action */}
            <div className="rounded-xl border border-amber-500/20 bg-amber-900/5 p-4 flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="h-10 w-10 rounded-full bg-amber-500/10 flex items-center justify-center text-amber-500">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
                    </div>
                    <div>
                        <h4 className="font-bold text-amber-200">Stockout Risk Detected</h4>
                        <p className="text-xs text-amber-200/60">High demand velocity on <b>Avocados</b> requires immediate restocking.</p>
                    </div>
                </div>
                <button className="px-4 py-2 bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 text-xs font-bold rounded-lg border border-amber-500/20 transition-colors uppercase tracking-wide">
                    Order Restock
                </button>
            </div>
        </div>
    );
}
