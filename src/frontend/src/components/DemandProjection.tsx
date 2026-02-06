import React, { useEffect, useState, useMemo } from "react";

// --- Mock Data ---
type Timeframe = 'Tomorrow' | 'Next 7 Days' | 'Next 14 Days' | 'Next 28 Days';

const FORECAST_DATA: Record<Timeframe, { label: string, actual: number | null, predicted: number }[]> = {
    'Tomorrow': [
        { label: '08:00', actual: 12, predicted: 15 },
        { label: '12:00', actual: 45, predicted: 48 },
        { label: '16:00', actual: 30, predicted: 35 },
        { label: '20:00', actual: null, predicted: 60 },
    ],
    'Next 7 Days': [
        { label: 'Mon', actual: 40, predicted: 42 },
        { label: 'Tue', actual: 35, predicted: 38 },
        { label: 'Wed', actual: 50, predicted: 48 },
        { label: 'Thu', actual: 45, predicted: 55 },
        { label: 'Fri', actual: 60, predicted: 75 },
        { label: 'Sat', actual: null, predicted: 85 },
        { label: 'Sun', actual: null, predicted: 90 },
    ],
    'Next 14 Days': Array.from({ length: 14 }, (_, i) => ({
        label: `Day ${i + 1}`,
        actual: i < 7 ? 40 + (i * 2) + Math.random() * 10 : null,
        predicted: 45 + (i * 3)
    })),
    'Next 28 Days': Array.from({ length: 14 }, (_, i) => ({
        label: `W${Math.floor(i / 7) + 1} D${(i % 7) + 1}`,
        actual: i < 10 ? 30 + (i * 1.5) : null,
        predicted: 35 + (i * 2)
    }))
};

const MOVERS = [
    { name: "Avocados (Hass)", change: "+200%", trend: "up", reason: "Viral Recipe Trend" },
    { name: "Sourdough Bread", change: "+45%", trend: "up", reason: "Weekend Spike" },
    { name: "Frozen Pizza", change: "-15%", trend: "down", reason: "Competitor Promo" },
    { name: "Almond Milk", change: "+12%", trend: "up", reason: "Steady Growth" },
];

const STORES = ["Global Hub #01", "Regional Center #04", "Retail Point #22"];
const ITEMS = ["All Categories", "Avocados (Hass)", "Almond Milk", "Organic Eggs"];

// --- Typewriter Component ---
const TypewriterText = ({ text, delay = 0 }: { text: string, delay?: number }) => {
    const [displayedText, setDisplayedText] = useState("");
    const [started, setStarted] = useState(false);

    useEffect(() => {
        const startTimeout = setTimeout(() => {
            setStarted(true);
        }, delay);
        return () => clearTimeout(startTimeout);
    }, [delay]);

    useEffect(() => {
        if (!started) return;
        let i = 0;
        const intervalId = setInterval(() => {
            setDisplayedText(text.slice(0, i + 1));
            i++;
            if (i >= text.length) clearInterval(intervalId);
        }, 30);
        return () => clearInterval(intervalId);
    }, [text, started]);

    return <span>{displayedText}</span>;
};

// --- Helper: Smooth Curve Path (Catmull-Rom Spline) ---
const getSmoothPath = (points: { x: number, y: number }[]) => {
    if (points.length < 2) return "";

    const tension = 0.3;
    let d = `M ${points[0].x},${points[0].y}`;

    for (let i = 0; i < points.length - 1; i++) {
        const p0 = points[i === 0 ? i : i - 1];
        const p1 = points[i];
        const p2 = points[i + 1];
        const p3 = points[i + 2] || p2;

        const cp1x = p1.x + (p2.x - p0.x) * tension;
        const cp1y = p1.y + (p2.y - p0.y) * tension;
        const cp2x = p2.x - (p3.x - p1.x) * tension;
        const cp2y = p2.y - (p3.y - p1.y) * tension;

        d += ` C ${cp1x},${cp1y} ${cp2x},${cp2y} ${p2.x},${p2.y}`;
    }
    return d;
};

export default function DemandProjection() {
    const [activeTimeframe, setActiveTimeframe] = useState<Timeframe>('Next 7 Days');
    const [selectedStore, setSelectedStore] = useState(STORES[0]);
    const [selectedItem, setSelectedItem] = useState(ITEMS[1]);
    const [animateChart, setAnimateChart] = useState(false);
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);
    const [storeSearch, setStoreSearch] = useState("");
    const [itemSearch, setItemSearch] = useState("");

    useEffect(() => {
        setAnimateChart(false);
        const timer = setTimeout(() => setAnimateChart(true), 100);
        return () => clearTimeout(timer);
    }, [activeTimeframe, selectedItem, selectedStore]);

    const data = FORECAST_DATA[activeTimeframe];

    const filteredStores = STORES.filter(store => store.toLowerCase().includes(storeSearch.toLowerCase()));
    const filteredItems = ITEMS.filter(item => item.toLowerCase().includes(itemSearch.toLowerCase()));

    // Calculate chart bounds
    const chartBounds = useMemo(() => {
        const allValues = data.flatMap(d => [d.actual, d.predicted].filter((v): v is number => v !== null));
        const min = Math.min(...allValues);
        const max = Math.max(...allValues);
        const padding = (max - min) * 0.2;
        return { min: min - padding, max: max + padding };
    }, [data]);

    const normalize = (val: number) => ((val - chartBounds.min) / (chartBounds.max - chartBounds.min)) * 100;

    const getPoints = (isActual: boolean): { x: number, y: number }[] => {
        return data.map((d, i) => {
            const val = isActual ? d.actual : d.predicted;
            if (val === null) return null;
            return {
                x: (i / (data.length - 1)) * 100,
                y: 100 - normalize(val)
            };
        }).filter((p): p is { x: number, y: number } => p !== null);
    };

    const actualPoints = getPoints(true);
    const predictedPoints = getPoints(false);

    // Y-axis labels
    const yLabels = useMemo(() => {
        const range = chartBounds.max - chartBounds.min;
        const step = range / 4;
        return Array.from({ length: 5 }, (_, i) => Math.round(chartBounds.max - step * i));
    }, [chartBounds]);

    return (
        <div className="space-y-8 animate-in fade-in zoom-in duration-500">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
                <div>
                    <h2 className="text-3xl font-black text-white tracking-tighter italic uppercase">Demand Intelligence</h2>
                    <p className="text-sm text-slate-400 font-medium">AI-driven forecasting and market velocity analysis.</p>
                </div>
                {/* Timeframe Filters */}
                <div className="bg-slate-900/80 p-1.5 rounded-xl border border-white/10 flex items-center backdrop-blur-md shadow-2xl">
                    {(['Tomorrow', 'Next 7 Days', 'Next 14 Days', 'Next 28 Days'] as Timeframe[]).map((tf) => (
                        <button
                            key={tf}
                            onClick={() => setActiveTimeframe(tf)}
                            className={`px-4 py-2 text-xs font-bold rounded-lg transition-all duration-300 ${activeTimeframe === tf ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/30 active:scale-95' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
                        >
                            {tf}
                        </button>
                    ))}
                </div>
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
                {/* Chart Section (col-span-2) */}
                <div className="lg:col-span-2 relative group h-[420px]">
                    {/* Background Layer */}
                    <div className="absolute inset-0 rounded-3xl border border-white/10 bg-slate-900/60 backdrop-blur-2xl overflow-hidden shadow-2xl">
                        <div className="absolute inset-0 opacity-10 pointer-events-none" style={{ backgroundImage: 'radial-gradient(circle at 2px 2px, #334155 1px, transparent 0)', backgroundSize: '24px 24px' }} />
                    </div>

                    {/* Content Layer */}
                    <div className="relative z-10 p-6 h-full flex flex-col justify-between">
                        <div className="flex items-center justify-between mb-6">
                            <div className="space-y-1">
                                <h3 className="text-xl font-bold text-white flex items-center gap-3">
                                    <span className="relative flex h-2.5 w-2.5">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-blue-500"></span>
                                    </span>
                                    {selectedItem} <span className="text-slate-500 font-light px-2 prose-sm">at</span> {selectedStore}
                                </h3>
                                <p className="text-xs text-slate-500 font-mono tracking-widest uppercase">{activeTimeframe} Forecast Velocity</p>
                            </div>
                            <div className="hidden sm:flex items-center gap-6">
                                <div className="flex items-center gap-2">
                                    <div className="w-8 h-0.5 bg-emerald-500 rounded-full"></div>
                                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Actual</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-8 h-0.5 border-t-2 border-dashed border-cyan-400"></div>
                                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Predicted</span>
                                </div>
                            </div>
                        </div>

                        {/* Chart Area */}
                        <div className="h-64 w-full relative">
                            {/* Y-Axis Labels */}
                            <div className="absolute left-0 top-0 bottom-8 w-10 flex flex-col justify-between text-[10px] font-mono text-slate-500 pointer-events-none">
                                {yLabels.map((val, i) => (
                                    <span key={i} className="text-right pr-2">{val}</span>
                                ))}
                            </div>

                            {/* Chart Visualization */}
                            <div className="absolute left-12 right-4 top-0 bottom-8">
                                {/* Grid Lines */}
                                <div className="absolute inset-0 flex flex-col justify-between pointer-events-none">
                                    {[0, 1, 2, 3, 4].map(i => (
                                        <div key={i} className="border-b border-slate-700/30" style={{ height: 0 }} />
                                    ))}
                                </div>

                                {/* SVG Chart */}
                                <svg
                                    className="absolute inset-0 w-full h-full overflow-visible"
                                    viewBox="0 0 100 100"
                                    preserveAspectRatio="none"
                                >
                                    <defs>
                                        <linearGradient id="actualFill" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="0%" stopColor="#10b981" stopOpacity="0.15" />
                                            <stop offset="100%" stopColor="#10b981" stopOpacity="0" />
                                        </linearGradient>
                                    </defs>

                                    {/* Actual Area Fill */}
                                    {actualPoints.length > 1 && (
                                        <path
                                            d={`${getSmoothPath(actualPoints)} L ${actualPoints[actualPoints.length - 1].x},100 L ${actualPoints[0].x},100 Z`}
                                            fill="url(#actualFill)"
                                            className="transition-opacity duration-700"
                                            style={{ opacity: animateChart ? 1 : 0 }}
                                        />
                                    )}

                                    {/* Predicted Line (Dashed, thin) */}
                                    {predictedPoints.length > 1 && (
                                        <path
                                            d={getSmoothPath(predictedPoints)}
                                            fill="none"
                                            stroke="#06b6d4"
                                            strokeWidth="1.5"
                                            strokeDasharray="4 3"
                                            strokeLinecap="round"
                                            className="transition-all duration-700"
                                            style={{
                                                strokeDashoffset: animateChart ? 0 : 500,
                                                opacity: 0.7
                                            }}
                                        />
                                    )}

                                    {/* Actual Line (Solid, thin) */}
                                    {actualPoints.length > 1 && (
                                        <path
                                            d={getSmoothPath(actualPoints)}
                                            fill="none"
                                            stroke="#10b981"
                                            strokeWidth="1.5"
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            className="transition-all duration-700"
                                            style={{
                                                strokeDasharray: 500,
                                                strokeDashoffset: animateChart ? 0 : 500
                                            }}
                                        />
                                    )}
                                </svg>

                                {/* Data Points Layer */}
                                <div className="absolute inset-0">
                                    {data.map((d, i) => {
                                        const xPos = (i / (data.length - 1)) * 100;
                                        const yActual = d.actual !== null ? 100 - normalize(d.actual) : null;
                                        const yPred = 100 - normalize(d.predicted);
                                        const isHovered = hoverIndex === i;

                                        return (
                                            <div
                                                key={i}
                                                className="absolute h-full transition-opacity duration-300"
                                                style={{ left: `${xPos}%`, opacity: animateChart ? 1 : 0 }}
                                            >
                                                {/* Hover Line */}
                                                {isHovered && (
                                                    <div className="absolute inset-y-0 w-px bg-slate-500/30 -ml-px" />
                                                )}

                                                {/* Predicted Point */}
                                                <div
                                                    className={`absolute w-2 h-2 -ml-1 rounded-full border border-cyan-400 bg-slate-900 transition-transform duration-200 ${isHovered ? 'scale-150' : 'scale-100'}`}
                                                    style={{ top: `${yPred}%`, marginTop: '-4px' }}
                                                />

                                                {/* Actual Point */}
                                                {yActual !== null && (
                                                    <div
                                                        className={`absolute w-2.5 h-2.5 -ml-1 rounded-full border-2 border-emerald-500 bg-slate-900 z-10 transition-transform duration-200 ${isHovered ? 'scale-150' : 'scale-100'}`}
                                                        style={{ top: `${yActual}%`, marginTop: '-5px' }}
                                                    />
                                                )}

                                                {/* Tooltip */}
                                                <div
                                                    className={`absolute bottom-full mb-3 left-1/2 -translate-x-1/2 z-50 transition-all duration-200 ${isHovered ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2 pointer-events-none'}`}
                                                >
                                                    <div className="bg-slate-800/95 border border-slate-600/50 rounded-lg px-3 py-2 shadow-xl whitespace-nowrap backdrop-blur-sm">
                                                        <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1.5 border-b border-slate-600/50 pb-1">{d.label}</div>
                                                        {d.actual !== null && (
                                                            <div className="flex items-center justify-between gap-4 text-xs">
                                                                <span className="text-emerald-400">Actual:</span>
                                                                <span className="font-mono font-bold text-white">{Math.round(d.actual)}</span>
                                                            </div>
                                                        )}
                                                        <div className="flex items-center justify-between gap-4 text-xs mt-0.5">
                                                            <span className="text-cyan-400">Pred:</span>
                                                            <span className="font-mono font-bold text-white">{Math.round(d.predicted)}</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>

                                {/* Hit Areas */}
                                <div className="absolute inset-0 flex cursor-crosshair">
                                    {data.map((_, i) => (
                                        <div
                                            key={i}
                                            className="flex-1 hover:bg-white/[0.02] transition-colors duration-150"
                                            onMouseEnter={() => setHoverIndex(i)}
                                            onMouseLeave={() => setHoverIndex(null)}
                                        />
                                    ))}
                                </div>
                            </div>

                            {/* X-Axis Labels */}
                            <div className="absolute left-12 right-4 bottom-0 h-6 flex justify-between text-[10px] font-bold uppercase tracking-widest text-slate-500">
                                {data.map((d, i) => (
                                    <span
                                        key={i}
                                        className={`transition-colors duration-200 ${hoverIndex === i ? 'text-white' : ''}`}
                                    >
                                        {d.label}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Sidebar Section (col-span-1) */}
                <div className="lg:col-span-1 space-y-6 self-start">
                    <div className="rounded-3xl border border-white/10 bg-slate-900/60 p-6 backdrop-blur-2xl shadow-xl transition-all duration-300 hover:border-white/20">
                        <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" /><circle cx="12" cy="10" r="3" /></svg>
                            Location Selection
                        </h4>

                        {/* Search Input */}
                        <div className="relative mb-3">
                            <input
                                type="text"
                                placeholder="Search locations..."
                                value={storeSearch}
                                onChange={(e) => setStoreSearch(e.target.value)}
                                className="w-full bg-slate-800/50 border border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-blue-500/50 transition-colors duration-200"
                            />
                            <svg className="absolute right-3 top-2.5 text-slate-500" xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>
                        </div>

                        <div className="space-y-2 max-h-48 overflow-y-auto pr-1 custom-scrollbar">
                            {filteredStores.map(store => (
                                <button
                                    key={store}
                                    onClick={() => setSelectedStore(store)}
                                    className={`w-full text-left px-4 py-3 rounded-xl text-xs font-bold transition-all duration-200 border ${selectedStore === store ? 'bg-blue-600 border-blue-400 text-white shadow-lg' : 'bg-white/5 border-transparent text-slate-400 hover:bg-white/[0.08]'}`}
                                >
                                    {store}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="rounded-3xl border border-white/10 bg-slate-900/60 p-6 backdrop-blur-2xl shadow-xl transition-all duration-300 hover:border-white/20">
                        <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="m21 21-4.3-4.3" /><circle cx="10" cy="10" r="7" /></svg>
                            SKU Navigator
                        </h4>

                        {/* Search Input */}
                        <div className="relative mb-3">
                            <input
                                type="text"
                                placeholder="Search products..."
                                value={itemSearch}
                                onChange={(e) => setItemSearch(e.target.value)}
                                className="w-full bg-slate-800/50 border border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-blue-500/50 transition-colors duration-200"
                            />
                            <svg className="absolute right-3 top-2.5 text-slate-500" xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>
                        </div>

                        <div className="space-y-1 max-h-48 overflow-y-auto pr-1 custom-scrollbar">
                            {filteredItems.map(item => (
                                <button
                                    key={item}
                                    onClick={() => setSelectedItem(item)}
                                    className={`w-full text-left px-4 py-2.5 rounded-lg text-[11px] font-bold transition-all duration-200 ${selectedItem === item ? 'text-blue-400 bg-blue-400/10 border-l-2 border-blue-400' : 'text-slate-500 hover:text-slate-300 hover:bg-white/5 border-l-2 border-transparent'}`}
                                >
                                    {item}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Movers & Shakers Section */}
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 delay-200">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                    <svg className="text-orange-500" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" /></svg>
                    Movers & Shakers
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {MOVERS.map((item, i) => (
                        <div key={i} className="group p-4 rounded-xl border border-slate-800 bg-slate-900/30 hover:bg-slate-800/50 transition-all duration-300 hover:border-slate-700">
                            <div className="flex items-start justify-between mb-2">
                                <div className={`p-2 rounded-lg transition-colors duration-200 ${item.trend === 'up' ? 'bg-emerald-500/10 text-emerald-500' : 'bg-red-500/10 text-red-500'}`}>
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

            {/* Prescriptive Action Section */}
            <div className="rounded-xl border border-amber-500/20 bg-amber-900/5 p-4 flex items-center justify-between animate-in fade-in slide-in-from-bottom-4 duration-500 delay-300">
                <div className="flex items-center gap-4">
                    <div className="h-10 w-10 rounded-full bg-amber-500/10 flex items-center justify-center text-amber-500">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
                    </div>
                    <div>
                        <h4 className="font-bold text-amber-200">
                            <TypewriterText text="Stockout Risk Detected" delay={1500} />
                        </h4>
                        <p className="text-xs text-amber-200/60 h-4">
                            <TypewriterText text="High demand velocity on Avocados requires immediate restocking." delay={2500} />
                        </p>
                    </div>
                </div>
                <button className="px-4 py-2 bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 text-xs font-bold rounded-lg border border-amber-500/20 transition-colors duration-200 uppercase">
                    Order Restock
                </button>
            </div>
        </div>
    );
}
