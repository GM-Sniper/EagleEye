import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import {
    TrendingUp,
    Package,
    AlertTriangle,
    Layers,
    Calendar,
    Clock,
    BarChart3,
    PieChart as PieChartIcon,
    ChevronRight,
    RefreshCw,
    Search,
    ArrowUpRight,
    ArrowDownRight,
    LayoutDashboard,
    Filter,
    CheckCircle2,
    XCircle,
    AlertCircle,
    HelpCircle,
    ArrowRight,
    X,
    ChevronDown,
    Activity,
    Box,
    Zap
} from 'lucide-react';
import {
    LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    BarChart, Bar, Cell, PieChart, Pie, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = 'http://localhost:8000';

// ============================================================================
// Components
// ============================================================================

const Badge = ({ status }) => {
    const styles = {
        HEALTHY: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20',
        UNDERSTOCKED: 'bg-amber-500/10 text-amber-500 border-amber-500/20',
        CRITICAL: 'bg-rose-500/10 text-rose-500 border-rose-500/20',
        OVERSTOCKED: 'bg-sky-500/10 text-sky-500 border-sky-500/20',
    };
    return (
        <span className={`px-2 py-1 rounded-full text-[10px] font-bold border ${styles[status] || 'bg-slate-500/10 text-slate-500'}`}>
            {status}
        </span>
    );
};

const OnboardingTour = ({ onComplete }) => {
    const [step, setStep] = useState(0);
    const steps = [
        {
            title: "Welcome to EagleEye AI",
            content: "Your mission-critical dashboard for inventory intelligence. Let's take a quick look at how to optimize your stock.",
            icon: TrendingUp,
        },
        {
            title: "Stock Health at a Glance",
            content: "Monitor Critical, Understocked, and Healthy items in real-time. Our AI predicts exactly when you'll run out.",
            icon: Activity,
        },
        {
            title: "Smart Forecasting",
            content: "The 7-day forecast uses advanced XGBoost models (8.34% MAPE accuracy) to predict future demand volume.",
            icon: BarChart3,
        },
        {
            title: "Actionable Insights",
            content: "Search through 26,000+ items and get specific reorder points and safety stock recommendations.",
            icon: Package,
        }
    ];

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm"
        >
            <motion.div
                initial={{ scale: 0.9, y: 20 }}
                animate={{ scale: 1, y: 0 }}
                className="glass-card max-w-md w-full p-8 rounded-3xl relative overflow-hidden"
            >
                <div className="absolute top-0 left-0 w-full h-1 bg-slate-800">
                    <motion.div
                        className="h-full bg-primary-500"
                        initial={{ width: "0%" }}
                        animate={{ width: `${((step + 1) / steps.length) * 100}%` }}
                    />
                </div>

                <div className="flex flex-col items-center text-center">
                    <div className="p-4 rounded-2xl bg-primary-500/10 text-primary-500 mb-6 animate-float">
                        {React.createElement(steps[step].icon, { size: 40 })}
                    </div>
                    <h3 className="text-2xl font-bold text-white mb-2">{steps[step].title}</h3>
                    <p className="text-slate-400 mb-8">{steps[step].content}</p>

                    <div className="flex items-center justify-between w-full">
                        <button
                            onClick={onComplete}
                            className="text-slate-500 text-sm hover:text-white transition-colors"
                        >
                            Skip tour
                        </button>
                        <button
                            onClick={() => step < steps.length - 1 ? setStep(s => s + 1) : onComplete()}
                            className="px-6 py-2 bg-primary-600 hover:bg-primary-500 text-white rounded-xl font-bold transition-all flex items-center space-x-2"
                        >
                            <span>{step === steps.length - 1 ? "Get Started" : "Next"}</span>
                            <ArrowRight size={16} />
                        </button>
                    </div>
                </div>
            </motion.div>
        </motion.div>
    );
};

// Item Detail Modal with Forecast
const ItemDetailModal = ({ item, loading, onClose }) => {
    if (!item && !loading) return null;

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-950/90 backdrop-blur-sm"
            onClick={onClose}
        >
            <motion.div
                initial={{ scale: 0.9, y: 20 }}
                animate={{ scale: 1, y: 0 }}
                exit={{ scale: 0.9, y: 20 }}
                className="glass-card max-w-4xl w-full max-h-[90vh] overflow-y-auto p-8 rounded-3xl relative"
                onClick={e => e.stopPropagation()}
            >
                {/* Close Button */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 p-2 rounded-xl hover:bg-slate-800 transition-colors text-slate-400 hover:text-white"
                >
                    <X size={24} />
                </button>

                {loading ? (
                    <div className="flex items-center justify-center h-64">
                        <Activity className="w-12 h-12 text-primary-500 animate-pulse" />
                    </div>
                ) : item && (
                    <div className="space-y-8">
                        {/* Header */}
                        <div className="flex items-start gap-6">
                            <div className="w-16 h-16 rounded-2xl bg-primary-500/10 flex items-center justify-center text-primary-500 font-black text-xl">
                                {item.item_id % 100}
                            </div>
                            <div className="flex-grow">
                                <h2 className="text-2xl font-black text-white italic tracking-tight">{item.item_name}</h2>
                                <p className="text-slate-500 text-sm font-bold mt-1">ID: {item.item_id}</p>
                                <div className="flex items-center gap-3 mt-3">
                                    <span className={`px-3 py-1 rounded-full text-xs font-black border ${item.status === 'HEALTHY' ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20' :
                                        item.status === 'CRITICAL' ? 'bg-rose-500/10 text-rose-500 border-rose-500/20' :
                                            item.status === 'UNDERSTOCKED' ? 'bg-amber-500/10 text-amber-500 border-amber-500/20' :
                                                'bg-sky-500/10 text-sky-500 border-sky-500/20'
                                        }`}>
                                        {item.status}
                                    </span>
                                    <span className={`px-2 py-1 rounded text-[10px] font-black ${item.abc_class === 'A' ? 'bg-primary-500 text-white' :
                                        item.abc_class === 'B' ? 'bg-violet-500 text-white' : 'bg-slate-700 text-slate-400'
                                        }`}>
                                        CLASS {item.abc_class}
                                    </span>
                                    <span className="px-3 py-1 rounded-full text-[10px] font-black bg-slate-800 text-slate-400 border border-slate-700">
                                        {item.model_used}
                                    </span>
                                </div>
                            </div>
                        </div>

                        {/* Stats Grid */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            {[
                                { label: 'Daily Mean', value: item.mean_daily?.toFixed(1), suffix: '/day' },
                                { label: 'Safety Stock', value: item.safety_stock?.toFixed(0), suffix: 'units' },
                                { label: 'Reorder Point', value: item.reorder_point?.toFixed(0), suffix: 'units' },
                                { label: 'Current Stock', value: item.current_stock?.toFixed(0), suffix: 'units' },
                            ].map(stat => (
                                <div key={stat.label} className="p-4 rounded-2xl bg-slate-900/50 border border-slate-800">
                                    <p className="text-[10px] font-black text-slate-500 tracking-widest uppercase">{stat.label}</p>
                                    <p className="text-xl font-black text-white mt-1 italic">
                                        {stat.value} <span className="text-xs text-slate-500 not-italic">{stat.suffix}</span>
                                    </p>
                                </div>
                            ))}
                        </div>

                        {/* Forecast Chart */}
                        {item.forecast && item.forecast.length > 0 && (
                            <div className="glass-card p-6 rounded-2xl border border-slate-800">
                                <div className="flex justify-between items-center mb-6">
                                    <div>
                                        <h3 className="text-lg font-black text-white tracking-widest italic">AI DEMAND FORECAST</h3>
                                        <p className="text-[10px] text-slate-500 font-bold tracking-wider mt-1">{item.forecast_days}-DAY PREDICTION HORIZON</p>
                                    </div>
                                    <div className="flex items-center gap-2 px-3 py-1.5 bg-primary-500/10 rounded-xl border border-primary-500/20">
                                        <Activity size={14} className="text-primary-500" />
                                        <span className="text-[10px] font-black text-primary-500 tracking-widest">LIVE</span>
                                    </div>
                                </div>
                                <div className="h-64">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={item.forecast}>
                                            <defs>
                                                <linearGradient id="itemForecast" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.4} />
                                                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                                                </linearGradient>
                                            </defs>
                                            <CartesianGrid strokeDasharray="5 5" vertical={false} stroke="#1e293b" />
                                            <XAxis dataKey="date" stroke="#475569" axisLine={false} tickLine={false} dy={10} fontSize={10} fontWeight="bold" />
                                            <YAxis stroke="#475569" axisLine={false} tickLine={false} dx={-10} fontSize={10} fontWeight="bold" />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
                                                itemStyle={{ fontWeight: 'bold' }}
                                            />
                                            <Area type="monotone" dataKey="upper_bound" stroke="transparent" fill="#0ea5e9" fillOpacity={0.1} />
                                            <Area type="monotone" dataKey="lower_bound" stroke="transparent" fill="#0ea5e9" fillOpacity={0.1} />
                                            <Area type="monotone" dataKey="predicted_demand" stroke="#0ea5e9" strokeWidth={3} fillOpacity={1} fill="url(#itemForecast)" />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}

                        {/* Historical Demand Chart */}
                        {item.history && item.history.length > 0 && (
                            <div className="glass-card p-6 rounded-2xl border border-slate-800">
                                <h3 className="text-lg font-black text-white tracking-widest italic mb-6">HISTORICAL DEMAND</h3>
                                <div className="h-48">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={item.history}>
                                            <CartesianGrid strokeDasharray="5 5" vertical={false} stroke="#1e293b" />
                                            <XAxis dataKey="date" stroke="#475569" axisLine={false} tickLine={false} dy={10} fontSize={9} fontWeight="bold" />
                                            <YAxis stroke="#475569" axisLine={false} tickLine={false} dx={-10} fontSize={10} fontWeight="bold" />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
                                            />
                                            <Bar dataKey="demand" fill="#8b5cf6" radius={[4, 4, 0, 0]} barSize={12} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}

                        {/* Recommendation */}
                        <div className="p-4 rounded-2xl bg-slate-900/50 border border-slate-800">
                            <p className="text-sm font-bold text-slate-300">{item.recommendation}</p>
                        </div>
                    </div>
                )}
            </motion.div>
        </motion.div>
    );
};

// ============================================================================
// Main Application
// ============================================================================

const Dashboard = () => {
    const [dashboardData, setDashboardData] = useState(null);
    const [inventoryPage, setInventoryPage] = useState({ items: [], summary: {} });
    const [loading, setLoading] = useState(true);
    const [invLoading, setInvLoading] = useState(false);
    const [activeTab, setActiveTab] = useState('overview');
    const [searchTerm, setSearchTerm] = useState('');
    const [statusFilter, setStatusFilter] = useState('');
    const [selectedItem, setSelectedItem] = useState(null);
    const [showOnboarding, setShowOnboarding] = useState(false);

    // Phase 7: Multi-Scenario & Multi-Window Backtesting
    const [backtestWindow, setBacktestWindow] = useState(7);
    const [backtestData, setBacktestData] = useState(null);
    const [backtestLoading, setBacktestLoading] = useState(false);
    const [forecastScenario, setForecastScenario] = useState('baseline');
    const [notification, setNotification] = useState(null);
    const [lastUpdated, setLastUpdated] = useState(new Date().toLocaleTimeString());

    // Item Detail Modal State
    const [selectedItemId, setSelectedItemId] = useState(null);
    const [itemDetail, setItemDetail] = useState(null);
    const [itemDetailLoading, setItemDetailLoading] = useState(false);

    useEffect(() => {
        const hasSeenTour = localStorage.getItem('eagleeye_tour_complete');
        if (!hasSeenTour) setShowOnboarding(true);
        fetchInitialData();
    }, []);

    const showMsg = (msg, type = 'success') => {
        setNotification({ msg, type });
        setTimeout(() => setNotification(null), 5000);
    };

    const fetchInitialData = async () => {
        setLoading(true);
        try {
            const resp = await axios.get(`${API_BASE}/dashboard`);
            setDashboardData(resp.data);
            setLastUpdated(new Date().toLocaleTimeString());

            // Also fetch initial inventory
            const invResp = await axios.get(`${API_BASE}/inventory/recommendations?top_n=20`);
            setInventoryPage(invResp.data);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const fetchInventory = async (search = '', status = '') => {
        setInvLoading(true);
        try {
            const url = `${API_BASE}/inventory/recommendations?top_n=200&search=${search}&status=${status}`;
            const resp = await axios.get(url);
            setInventoryPage(resp.data);
        } catch (err) {
            console.error(err);
        } finally {
            setInvLoading(false);
        }
    };

    const fetchBacktestData = async (window, scenario) => {
        setBacktestLoading(true);
        try {
            const response = await axios.get(`${API_BASE}/analytics/backtest?window_days=${window}&scenario=${scenario}`);
            setBacktestData(response.data);
        } catch (error) {
            console.error("Error fetching backtest data:", error);
            showMsg("BACKTEST ENGINE: UNAVAILABLE", "error");
        } finally {
            setBacktestLoading(false);
        }
    };

    const handleWindowChange = (window) => {
        setBacktestWindow(window);
        fetchBacktestData(window, forecastScenario);
    };

    useEffect(() => {
        if (activeTab === 'forecast') {
            fetchBacktestData(backtestWindow, forecastScenario);
        }
    }, [activeTab, backtestWindow, forecastScenario]);

    const handleGeneratePlan = async () => {
        try {
            const resp = await axios.post(`${API_BASE}/inventory/shipment-plan`);
            showMsg(resp.data.message);
        } catch (err) {
            showMsg("Failed to generate plan", "error");
        }
    };

    const handleRecalibrate = async () => {
        try {
            const resp = await axios.post(`${API_BASE}/analytics/recalibrate`);
            showMsg(resp.data.message);
        } catch (err) {
            showMsg("Calibration failed", "error");
        }
    };


    const handleSearch = (e) => {
        const val = e.target.value;
        setSearchTerm(val);
        const timer = setTimeout(() => fetchInventory(val, statusFilter), 500);
        return () => clearTimeout(timer);
    };

    const handleStatusFilter = (status) => {
        setStatusFilter(status);
        fetchInventory(searchTerm, status);
    };

    const fetchItemDetail = async (itemId) => {
        setItemDetailLoading(true);
        setSelectedItemId(itemId);
        try {
            const resp = await axios.get(`${API_BASE}/inventory/item/${itemId}`);
            setItemDetail(resp.data);
        } catch (err) {
            console.error(err);
            showMsg("Failed to load item details", "error");
            setSelectedItemId(null);
        } finally {
            setItemDetailLoading(false);
        }
    };

    const closeItemDetail = () => {
        setSelectedItemId(null);
        setItemDetail(null);
    };

    const completeTour = () => {
        setShowOnboarding(false);
        localStorage.setItem('eagleeye_tour_complete', 'true');
    };

    const Skeleton = ({ className }) => (
        <div className={`animate-pulse bg-slate-800/50 rounded-2xl ${className}`} />
    );

    const KPISkeleton = () => (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {[1, 2, 3, 4].map(i => (
                <div key={i} className="glass-card p-6 rounded-3xl h-40">
                    <Skeleton className="w-10 h-10 mb-4" />
                    <Skeleton className="w-1/2 h-4 mb-2" />
                    <Skeleton className="w-3/4 h-8" />
                </div>
            ))}
        </div>
    );

    const ChartSkeleton = () => (
        <div className="glass-card p-8 rounded-[2.5rem] h-[520px]">
            <div className="flex justify-between mb-10">
                <div>
                    <Skeleton className="w-48 h-6 mb-2" />
                    <Skeleton className="w-32 h-3" />
                </div>
                <Skeleton className="w-40 h-10" />
            </div>
            <Skeleton className="w-full h-80" />
        </div>
    );

    const TableSkeleton = () => (
        <div className="glass-card rounded-[2.5rem] overflow-hidden">
            <div className="p-8 border-b border-slate-800 flex justify-between">
                <Skeleton className="w-48 h-8" />
                <Skeleton className="w-64 h-8" />
            </div>
            <div className="p-8 space-y-4">
                {[1, 2, 3, 4, 5].map(i => (
                    <Skeleton key={i} className="w-full h-16" />
                ))}
            </div>
        </div>
    );

    const KPICard = ({ title, value, icon: Icon, trend, color }) => (
        <motion.div
            whileHover={{ y: -5 }}
            className="glass-card p-6 rounded-3xl"
        >
            <div className={`p-2 w-fit rounded-xl bg-${color}-500/10 text-${color}-500 mb-4`}>
                <Icon size={24} />
            </div>
            <p className="text-slate-400 text-xs font-bold uppercase tracking-wider">{title}</p>
            <h3 className="text-3xl font-black text-white mt-1 italic">{value}</h3>
            {trend && (
                <div className="mt-4 flex items-center space-x-1 text-xs text-emerald-400 font-bold">
                    <TrendingUp size={12} />
                    <span>{trend} increase</span>
                </div>
            )}
        </motion.div>
    );

    return (
        <div className="min-h-screen bg-[#060912] text-slate-300 font-sans selection:bg-primary-500/30">
            <AnimatePresence>
                {showOnboarding && <OnboardingTour onComplete={completeTour} />}
                {selectedItemId && (
                    <ItemDetailModal
                        item={itemDetail}
                        loading={itemDetailLoading}
                        onClose={closeItemDetail}
                    />
                )}
            </AnimatePresence>

            <div className="flex flex-col lg:flex-row min-h-screen">
                {/* Notifications */}
                <AnimatePresence>
                    {notification && (
                        <motion.div
                            initial={{ opacity: 0, y: 50, x: '-50%' }}
                            animate={{ opacity: 1, y: 0, x: '-50%' }}
                            exit={{ opacity: 0, y: 20, x: '-50%' }}
                            className={`fixed bottom-10 left-1/2 z-[200] px-8 py-4 rounded-2xl shadow-2xl backdrop-blur-md border ${notification.type === 'error' ? 'bg-rose-500/90 border-rose-400' : 'bg-emerald-500/90 border-emerald-400'
                                } text-white font-black text-sm tracking-widest italic flex items-center space-x-4`}
                        >
                            {notification.type === 'error' ? <XCircle size={20} /> : <CheckCircle2 size={20} />}
                            <span>{notification.msg}</span>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Sidebar */}
                <aside className="w-full lg:w-72 glass border-r border-slate-800 p-8 flex flex-col z-40">
                    <div className="flex items-center space-x-3 mb-12">
                        <div className="w-12 h-12 bg-gradient-to-br from-primary-400 to-primary-700 rounded-2xl flex items-center justify-center shadow-xl shadow-primary-500/20 rotate-3">
                            <TrendingUp className="text-white w-7 h-7" />
                        </div>
                        <div>
                            <h1 className="text-2xl font-black text-white tracking-tighter italic">EAGLEEYE</h1>
                            <p className="text-[10px] text-slate-500 font-bold tracking-[0.2em]">INTELLIGENCE v1.2</p>
                        </div>
                    </div>

                    <nav className="space-y-3 flex-grow">
                        {[
                            { id: 'overview', name: 'COMMAND CENTER', icon: LayoutDashboard },
                            { id: 'inventory', name: 'STOCK TERMINAL', icon: Layers },
                            { id: 'forecast', name: 'DEMAND PROJECTION', icon: BarChart3 },
                            { id: 'abc', name: 'REVENUE SEGMENTS', icon: PieChartIcon },
                        ].map(item => (
                            <button
                                key={item.id}
                                onClick={() => setActiveTab(item.id)}
                                className={`w-full group flex items-center space-x-4 px-6 py-4 rounded-2xl transition-all duration-300 ${activeTab === item.id
                                    ? 'bg-primary-600 text-white shadow-xl shadow-primary-900/40 translate-x-1'
                                    : 'text-slate-500 hover:text-white hover:bg-slate-800/40'
                                    }`}
                            >
                                <item.icon className={`w-5 h-5 transition-transform group-hover:scale-110 ${activeTab === item.id ? 'text-white' : 'text-slate-600'}`} />
                                <span className="font-black text-xs tracking-widest">{item.name}</span>
                                {activeTab === item.id && <motion.div layoutId="dot" className="w-1.5 h-1.5 rounded-full bg-white ml-auto" />}
                            </button>
                        ))}
                    </nav>

                    <div className="mt-8 p-6 rounded-3xl bg-slate-900/50 border border-slate-800">
                        <div className="flex items-center space-x-2 text-[10px] font-bold text-slate-500 mb-4 tracking-wider">
                            <Clock size={12} />
                            <span>LIVE FEED: {lastUpdated}</span>
                        </div>
                        <button
                            onClick={fetchInitialData}
                            className="w-full py-3 bg-slate-800 hover:bg-slate-700 rounded-xl text-xs font-black transition-all flex items-center justify-center space-x-2 border border-slate-700 hover:border-slate-500"
                        >
                            <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
                            <span>REFRESH CORE</span>
                        </button>
                    </div>
                </aside>

                {/* Content Area */}
                <main className="flex-grow p-6 lg:p-12 overflow-y-auto max-h-screen custom-scrollbar">

                    {/* Header */}
                    <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-12">
                        <div>
                            <h2 className="text-4xl font-black text-white tracking-tight italic">
                                {activeTab === 'overview' ? 'COMMAND CENTER' : activeTab.toUpperCase() + ' TERMINAL'}
                            </h2>
                            <p className="text-slate-500 font-medium mt-1">Deep Learning powered inventory optimization system</p>
                        </div>

                        <div className="flex items-center gap-4 w-full md:w-auto">
                            <div className="relative flex-grow md:flex-grow-0">
                                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                                <input
                                    type="text"
                                    placeholder="Query Item ID or Name..."
                                    onChange={handleSearch}
                                    className="bg-slate-900/80 border border-slate-800 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 rounded-2xl pl-12 pr-6 py-3.5 text-sm outline-none transition-all w-full md:w-80 font-medium"
                                />
                            </div>
                            <button
                                onClick={() => setShowOnboarding(true)}
                                className="w-12 h-12 flex items-center justify-center rounded-2xl glass border-slate-800 hover:border-slate-600 transition-all text-slate-400 hover:text-white"
                            >
                                <HelpCircle size={20} />
                            </button>
                        </div>
                    </header>

                    <AnimatePresence mode="wait">
                        {activeTab === 'overview' && (
                            <motion.div
                                key="overview"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: -20 }}
                                className="space-y-10"
                            >
                                {/* KPI Grid */}
                                {loading || !dashboardData ? (
                                    <KPISkeleton />
                                ) : (
                                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                                        <KPICard title="Projected Revenue" value={`${(dashboardData.summary.total_revenue / 1e6).toFixed(1)}M`} icon={TrendingUp} trend="12.5%" color="primary" />
                                        <KPICard title="Total Throughput" value={dashboardData.summary.total_orders.toLocaleString()} icon={Activity} trend="8.2%" color="emerald" />
                                        <KPICard title="Model Stability" value="91.7%" icon={Activity} color="violet" />
                                        <KPICard title="Inventory Breadth" value={dashboardData.summary.unique_items.toLocaleString()} icon={Box} color="orange" />
                                    </div>
                                )}

                                <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
                                    {/* Demand Projection */}
                                    {loading || !dashboardData ? (
                                        <div className="xl:col-span-2">
                                            <ChartSkeleton />
                                        </div>
                                    ) : (
                                        <div className="xl:col-span-2 glass-card p-8 rounded-[2.5rem] relative overflow-hidden">
                                            <div className="absolute top-0 right-0 p-8 opacity-5">
                                                <TrendingUp size={120} />
                                            </div>
                                            <div className="flex justify-between items-center mb-10">
                                                <div>
                                                    <h3 className="text-xl font-black text-white tracking-widest italic">DEMAND PROJECTION</h3>
                                                    <p className="text-slate-500 text-xs font-bold mt-1">7-DAY ANALYTICS HORIZON</p>
                                                </div>
                                                <div className="flex glass rounded-xl p-1.5 border-slate-800">
                                                    <button className="px-5 py-2 bg-primary-600 text-white rounded-lg text-[10px] font-black tracking-widest">DAILY</button>
                                                    <button className="px-5 py-2 text-slate-500 text-[10px] font-black tracking-widest hover:text-white transition-colors">WEEKLY</button>
                                                </div>
                                            </div>
                                            <div className="h-[380px] w-full">
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <AreaChart data={dashboardData.forecast}>
                                                        <defs>
                                                            <linearGradient id="colorDemand" x1="0" y1="0" x2="0" y2="1">
                                                                <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.4} />
                                                                <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                                                            </linearGradient>
                                                        </defs>
                                                        <CartesianGrid strokeDasharray="5 5" vertical={false} stroke="#1e293b" />
                                                        <XAxis dataKey="date" stroke="#475569" axisLine={false} tickLine={false} dy={15} fontSize={10} fontWeight="bold" />
                                                        <YAxis stroke="#475569" axisLine={false} tickLine={false} dx={-15} fontSize={10} fontWeight="bold" />
                                                        <Tooltip
                                                            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', boxShadow: '0 20px 25px -5px rgb(0 0 0 / 0.3)' }}
                                                            itemStyle={{ color: '#0ea5e9', fontWeight: 'bold' }}
                                                        />
                                                        <Area type="monotone" dataKey="predicted_demand" stroke="#0ea5e9" fillOpacity={1} fill="url(#colorDemand)" strokeWidth={4} />
                                                        <Area type="monotone" dataKey="upper_bound" stroke="transparent" fill="#0ea5e9" fillOpacity={0.05} />
                                                        <Area type="monotone" dataKey="lower_bound" stroke="transparent" fill="#0ea5e9" fillOpacity={0.05} />
                                                    </AreaChart>
                                                </ResponsiveContainer>
                                            </div>
                                        </div>
                                    )}

                                    {/* Stock Distribution */}
                                    {loading || !dashboardData ? (
                                        <ChartSkeleton />
                                    ) : (
                                        <div className="glass-card p-8 rounded-[2.5rem]">
                                            <h3 className="text-xl font-black text-white tracking-widest italic mb-2">REVENUE IMPACT</h3>
                                            <p className="text-slate-500 text-[10px] font-bold mb-10 tracking-widest">ABC CLASSIFICATION MATRIX</p>
                                            <div className="h-[280px] relative">
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <PieChart>
                                                        <Pie
                                                            data={[
                                                                { name: 'Class A', value: dashboardData.abc_classification.A, color: '#0ea5e9' },
                                                                { name: 'Class B', value: dashboardData.abc_classification.B, color: '#8b5cf6' },
                                                                { name: 'Class C', value: dashboardData.abc_classification.C, color: '#f59e0b' },
                                                            ]}
                                                            innerRadius={75}
                                                            outerRadius={105}
                                                            paddingAngle={10}
                                                            dataKey="value"
                                                            stroke="none"
                                                        >
                                                            <Cell fill="#0ea5e9" />
                                                            <Cell fill="#8b5cf6" />
                                                            <Cell fill="#f59e0b" />
                                                        </Pie>
                                                    </PieChart>
                                                </ResponsiveContainer>
                                                <div className="absolute inset-0 flex flex-col items-center justify-center">
                                                    <span className="text-4xl font-black text-white tracking-tighter italic">ABC</span>
                                                    <span className="text-[10px] text-slate-500 font-black tracking-[0.3em]">ANALYSIS</span>
                                                </div>
                                            </div>
                                            <div className="mt-8 space-y-4">
                                                {[
                                                    { label: 'TOP 80%', color: 'sky', cls: 'A' },
                                                    { label: 'MID 15%', color: 'violet', cls: 'B' },
                                                    { label: 'LOW 5%', color: 'orange', cls: 'C' }
                                                ].map(item => (
                                                    <div key={item.cls} className="flex items-center justify-between">
                                                        <div className="flex items-center space-x-3">
                                                            <div className={`w-3 h-3 rounded-full bg-${item.color}-500 shadow-[0_0_8px_rgba(var(--tw-shadow-color),0.4)] shadow-${item.color}-500/50`} />
                                                            <span className="text-[10px] font-black text-slate-400 tracking-widest">{item.label}</span>
                                                        </div>
                                                        <span className="text-xs font-black text-white italic">Class {item.cls}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Sub-Charts Section */}
                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                    {loading || !dashboardData ? (
                                        <ChartSkeleton />
                                    ) : (
                                        <div className="glass-card p-10 rounded-[2.5rem]">
                                            <div className="flex items-center space-x-4 mb-10">
                                                <div className="p-3 rounded-2xl bg-violet-500/10 text-violet-500 uppercase font-black text-xs tracking-widest italic">
                                                    Weekly
                                                </div>
                                                <h3 className="text-xl font-black text-white tracking-widest">TEMPORAL DEMAND CURVE</h3>
                                            </div>
                                            <div className="h-[300px]">
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <BarChart data={dashboardData.weekly_pattern}>
                                                        <XAxis dataKey="day" stroke="#475569" axisLine={false} tickLine={false} fontSize={10} fontWeight="black" dy={10} />
                                                        <Tooltip cursor={{ fill: '#1e293b' }} contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }} />
                                                        <Bar dataKey="orders" fill="#8b5cf6" radius={[8, 8, 0, 0]} barSize={24}>
                                                            {dashboardData.weekly_pattern.map((entry, index) => (
                                                                <Cell key={`cell-${index}`} fill={entry.day === 'Friday' ? '#0ea5e9' : '#1e293b'} />
                                                            ))}
                                                        </Bar>
                                                    </BarChart>
                                                </ResponsiveContainer>
                                            </div>
                                        </div>
                                    )}

                                    {loading || !dashboardData ? (
                                        <ChartSkeleton />
                                    ) : (
                                        <div className="glass-card p-10 rounded-[2.5rem]">
                                            <div className="flex items-center space-x-4 mb-10">
                                                <div className="p-3 rounded-2xl bg-orange-500/10 text-orange-500 uppercase font-black text-xs tracking-widest italic">
                                                    Hourly
                                                </div>
                                                <h3 className="text-xl font-black text-white tracking-widest">PEAK PERFORMANCE WINDOW</h3>
                                            </div>
                                            <div className="h-[300px]">
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <LineChart data={dashboardData.hourly_pattern}>
                                                        <XAxis dataKey="hour" stroke="#475569" axisLine={false} tickLine={false} fontSize={10} fontWeight="black" dy={10} />
                                                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }} />
                                                        <Line type="monotone" dataKey="orders" stroke="#f59e0b" strokeWidth={5} dot={false} animationDuration={2000} />
                                                    </LineChart>
                                                </ResponsiveContainer>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </motion.div>
                        )}

                        {activeTab === 'inventory' && (
                            <motion.div
                                key="inventory"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                className="space-y-8"
                            >
                                {/* Inventory Controls */}
                                <div className="flex flex-col md:flex-row justify-between gap-6 pb-2">
                                    <div className="flex bg-slate-900/50 p-1.5 rounded-2xl border border-slate-800 self-start">
                                        {['ALL', 'CRITICAL', 'UNDERSTOCKED', 'HEALTHY', 'OVERSTOCKED'].map(s => (
                                            <button
                                                key={s}
                                                onClick={() => handleStatusFilter(s === 'ALL' ? '' : s)}
                                                className={`px-5 py-2.5 rounded-xl text-[10px] font-black tracking-widest transition-all ${(statusFilter === s || (s === 'ALL' && !statusFilter))
                                                    ? 'bg-slate-700 text-white shadow-lg shadow-black/20'
                                                    : 'text-slate-500 hover:text-slate-300'
                                                    }`}
                                            >
                                                {s}
                                            </button>
                                        ))}
                                    </div>
                                    <div className="flex items-center space-x-4">
                                        <button
                                            onClick={handleGeneratePlan}
                                            className="px-6 py-2.5 bg-primary-600 hover:bg-primary-500 text-white rounded-xl text-[10px] font-black tracking-widest transition-all shadow-lg shadow-primary-900/20"
                                        >
                                            GENERATE SHIPMENT PLAN
                                        </button>
                                        <span className="text-xs font-bold text-slate-500 tracking-widest uppercase">Found: {inventoryPage.count} Units</span>
                                        <Filter className="w-5 h-5 text-slate-600" />
                                    </div>
                                </div>

                                {/* Inventory Table */}
                                {invLoading || inventoryPage.items.length === 0 ? (
                                    <TableSkeleton />
                                ) : (
                                    <div className="glass-card rounded-[2.5rem] overflow-hidden border border-slate-800/50">
                                        <div className="overflow-x-auto">
                                            <table className="w-full text-left border-collapse">
                                                <thead>
                                                    <tr className="border-b border-slate-800/50 bg-slate-900/20">
                                                        <th className="p-6 text-[10px] font-black tracking-[0.2em] text-slate-500 uppercase">Item Intelligence</th>
                                                        <th className="p-6 text-[10px] font-black tracking-[0.2em] text-slate-500 uppercase">Status</th>
                                                        <th className="p-6 text-[10px] font-black tracking-[0.2em] text-slate-500 uppercase">Stock Level</th>
                                                        <th className="p-6 text-[10px] font-black tracking-[0.2em] text-slate-500 uppercase">Daily Mean</th>
                                                        <th className="p-6 text-[10px] font-black tracking-[0.2em] text-slate-500 uppercase">ABC</th>
                                                        <th className="p-6"></th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {inventoryPage.items.length > 0 ? (
                                                        inventoryPage.items.map((item, idx) => (
                                                            <motion.tr
                                                                initial={{ opacity: 0, y: 10 }}
                                                                animate={{ opacity: 1, y: 0 }}
                                                                transition={{ delay: idx * 0.05 }}
                                                                key={item.item_id}
                                                                className="border-b border-slate-800/30 hover:bg-slate-800/20 transition-colors group"
                                                            >
                                                                <td className="p-6">
                                                                    <div className="flex items-center space-x-4">
                                                                        <div className="w-10 h-10 rounded-xl bg-slate-800/50 flex items-center justify-center text-primary-500 font-black text-xs">
                                                                            {item.item_id % 100}
                                                                        </div>
                                                                        <div>
                                                                            <p className="text-sm font-bold text-white group-hover:text-primary-400 transition-colors line-clamp-1 max-w-[200px]">{item.item_name}</p>
                                                                            <p className="text-[10px] text-slate-600 font-black uppercase mt-0.5 tracking-tighter">ID: {item.item_id}</p>
                                                                        </div>
                                                                    </div>
                                                                </td>
                                                                <td className="p-6">
                                                                    <Badge status={item.status} />
                                                                </td>
                                                                <td className="p-6 w-64">
                                                                    <div className="flex flex-col space-y-1.5">
                                                                        <div className="flex justify-between text-[10px] font-bold text-slate-500">
                                                                            <span>{item.current_stock.toFixed(0)} units</span>
                                                                            <span>{item.stock_percentage}%</span>
                                                                        </div>
                                                                        <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden border border-slate-700/50">
                                                                            <motion.div
                                                                                initial={{ width: 0 }}
                                                                                animate={{ width: `${item.stock_percentage}%` }}
                                                                                className={`h-full rounded-full ${item.status === 'CRITICAL' ? 'bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.4)]' :
                                                                                    item.status === 'UNDERSTOCKED' ? 'bg-amber-500' :
                                                                                        item.status === 'OVERSTOCKED' ? 'bg-sky-500' : 'bg-emerald-500'
                                                                                    }`}
                                                                            />
                                                                        </div>
                                                                    </div>
                                                                </td>
                                                                <td className="p-6">
                                                                    <div className="text-sm font-black text-white italic">{item.mean_daily.toFixed(1)} <span className="text-[10px] text-slate-500 not-italic">/day</span></div>
                                                                </td>
                                                                <td className="p-6">
                                                                    <div className={`w-6 h-6 rounded flex items-center justify-center text-[10px] font-black ${item.abc_class === 'A' ? 'bg-primary-500 text-white' :
                                                                        item.abc_class === 'B' ? 'bg-violet-500 text-white' : 'bg-slate-700 text-slate-400'
                                                                        }`}>
                                                                        {item.abc_class}
                                                                    </div>
                                                                </td>
                                                                <td className="p-6">
                                                                    <button
                                                                        onClick={() => fetchItemDetail(item.item_id)}
                                                                        className="p-2 rounded-lg hover:bg-slate-700 transition-colors text-slate-500 hover:text-white"
                                                                    >
                                                                        <ChevronRight size={18} />
                                                                    </button>
                                                                </td>
                                                            </motion.tr>
                                                        ))
                                                    ) : (
                                                        <tr>
                                                            <td colSpan="6" className="p-20 text-center">
                                                                <div className="flex flex-col items-center opacity-30">
                                                                    <Package size={40} className="mb-4" />
                                                                    <p className="font-black text-sm tracking-widest uppercase italic">Neural Network: No Matching Inventory Records Found</p>
                                                                </div>
                                                            </td>
                                                        </tr>
                                                    )}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                )}

                            </motion.div>
                        )}

                        {activeTab === 'forecast' && (
                            <motion.div
                                key="forecast"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                className="space-y-8"
                            >
                                <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
                                    {/* Accuracy Verification Chart */}
                                    <div className="xl:col-span-2 glass-card p-10 rounded-[2.5rem]">
                                        {backtestLoading || !dashboardData ? (
                                            <ChartSkeleton />
                                        ) : (
                                            <div className="flex flex-col h-full">
                                                <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-10 gap-6">
                                                    <div>
                                                        <h3 className="text-2xl font-black text-white tracking-widest italic">ACCURACY VERIFICATION</h3>
                                                        <p className="text-slate-500 text-xs font-bold mt-1 tracking-widest uppercase">
                                                            ACTUAL VS PREDICTED (LAST {backtestWindow} DAYS)
                                                        </p>
                                                    </div>

                                                    <div className="flex items-center p-1 bg-slate-900/60 rounded-xl border border-slate-800">
                                                        {[7, 14, 30].map(w => (
                                                            <button
                                                                key={w}
                                                                onClick={() => handleWindowChange(w)}
                                                                className={`px-4 py-2 rounded-lg text-[10px] font-black tracking-widest transition-all ${backtestWindow === w ? 'bg-primary-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'}`}
                                                            >
                                                                {w}D
                                                            </button>
                                                        ))}
                                                    </div>

                                                    <div className="flex items-center space-x-2 px-4 py-2 bg-emerald-500/10 rounded-xl border border-emerald-500/20">
                                                        <CheckCircle2 size={16} className="text-emerald-500" />
                                                        <span className="text-[10px] font-black text-emerald-500 tracking-widest uppercase">REAL-TIME VALIDATED</span>
                                                    </div>
                                                </div>
                                                <div className="h-[380px] w-full">
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <LineChart data={backtestData ? backtestData.comparison : dashboardData.accuracy_comparison}>
                                                            <CartesianGrid strokeDasharray="5 5" vertical={false} stroke="#1e293b" />
                                                            <XAxis dataKey="date" stroke="#475569" axisLine={false} tickLine={false} dy={15} fontSize={10} fontWeight="bold" />
                                                            <YAxis stroke="#475569" axisLine={false} tickLine={false} dx={-15} fontSize={10} fontWeight="bold" />
                                                            <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px' }} />
                                                            <Line type="monotone" dataKey="actual" stroke="#10b981" strokeWidth={4} dot={{ r: 4, fill: '#10b981' }} name="Actual Orders" />
                                                            <Line type="monotone" dataKey="predicted" stroke="#0ea5e9" strokeWidth={4} strokeDasharray="8 5" dot={{ r: 4, fill: '#0ea5e9' }} name="AI Prediction" />
                                                            <Legend wrapperStyle={{ paddingTop: '20px' }} />
                                                        </LineChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            </div>
                                        )}
                                    </div>

                                    {/* Model Stats */}
                                    <div className="glass-card p-10 rounded-[2.5rem]">
                                        {backtestLoading || !dashboardData ? (
                                            <KPISkeleton />
                                        ) : (
                                            <>
                                                <h3 className="text-2xl font-black text-white tracking-widest italic mb-2">STABILITY INDEX</h3>
                                                <p className="text-slate-500 font-bold text-[10px] mb-10 tracking-widest uppercase">Core performance metrics from last backtest window.</p>
                                                <div className="space-y-6">
                                                    {[
                                                        { label: `LIVE ACCURACY (${backtestWindow}D)`, value: backtestData ? backtestData.accuracy : dashboardData.summary.model_accuracy, color: 'emerald' },
                                                        { label: 'ERROR RATE (MAPE)', value: backtestData ? backtestData.mape : dashboardData.summary.mape, color: 'rose' },
                                                        { label: 'MODEL TYPE', value: 'XGBOOST v2', color: 'primary' },
                                                        { label: 'CONFIDENCE', value: '95% CI', color: 'violet' },
                                                    ].map(stat => (
                                                        <div key={stat.label} className="p-6 rounded-3xl bg-slate-900/40 border border-slate-800 hover:border-slate-700 transition-colors">
                                                            <div className="flex justify-between items-center">
                                                                <span className="text-[10px] font-black text-slate-500 tracking-widest uppercase italic">{stat.label}</span>
                                                                <span className={`text-xl font-black text-${stat.color}-400 italic`}>{stat.value}</span>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </>
                                        )}
                                    </div>
                                </div>

                                {/* Scenario Planning Card */}
                                <div className="glass-card p-10 rounded-[2.5rem] bg-gradient-to-br from-slate-900/30 to-primary-900/10 border-t border-primary-500/20 shadow-[0_-20px_40px_rgba(15,23,42,0.5)]">
                                    {loading || !dashboardData ? (
                                        <ChartSkeleton />
                                    ) : (
                                        <div className="flex flex-col gap-10">
                                            <div className="flex flex-col md:flex-row justify-between items-center gap-6">
                                                <div>
                                                    <h3 className="text-2xl font-black text-white tracking-widest italic mb-2">SCENARIO PLANNER</h3>
                                                    <p className="text-slate-500 font-bold text-[10px] tracking-widest uppercase">Select business outlook to adjust procurement recommendations.</p>
                                                </div>
                                                <div className="flex items-center p-1.5 bg-slate-950 rounded-2xl border border-slate-800 shadow-inner">
                                                    {[
                                                        { id: 'baseline', label: 'AI BASELINE' },
                                                        { id: 'optimistic', label: 'OPTIMISTIC' },
                                                        { id: 'conservative', label: 'CONSERVATIVE' }
                                                    ].map(s => (
                                                        <button
                                                            key={s.id}
                                                            onClick={() => setForecastScenario(s.id)}
                                                            className={`px-6 py-3 rounded-xl text-[10px] font-black tracking-widest transition-all ${forecastScenario === s.id ? 'bg-primary-600 text-white shadow-[0_0_15px_rgba(37,99,235,0.4)]' : 'text-slate-500 hover:text-slate-300'}`}
                                                        >
                                                            {s.label}
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>

                                            <div className="h-[300px] w-full bg-slate-950/40 p-6 rounded-3xl border border-slate-800/50">
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <LineChart data={backtestData ? backtestData.scenarios[forecastScenario] : []}>
                                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" />
                                                        <XAxis dataKey="date" stroke="#475569" fontSize={10} axisLine={false} tickLine={false} dy={10} />
                                                        <YAxis stroke="#475569" fontSize={10} axisLine={false} tickLine={false} dx={10} />
                                                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }} />
                                                        <Line type="monotone" dataKey="demand" stroke={forecastScenario === 'optimistic' ? '#10b981' : forecastScenario === 'conservative' ? '#f43f5e' : '#0ea5e9'} strokeWidth={4} dot={{ r: 4 }} animationDuration={1500} />
                                                    </LineChart>
                                                </ResponsiveContainer>
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Predictive Action Card */}
                                <div className="glass-card p-10 rounded-[2.5rem] bg-gradient-to-br from-slate-900/30 to-primary-900/10">
                                    {loading || !dashboardData ? (
                                        <KPISkeleton />
                                    ) : (
                                        <div className="flex flex-col md:flex-row items-center gap-10">
                                            <div className="p-8 rounded-full bg-primary-500/10 text-primary-500 border border-primary-500/20">
                                                <Activity size={48} className="animate-pulse" />
                                            </div>
                                            <div>
                                                <h3 className="text-2xl font-black text-white tracking-widest italic mb-2 uppercase">Neural Recommendation Engine</h3>
                                                <p className="text-slate-500 leading-relaxed text-sm max-w-2xl">
                                                    The model currently suggests the **{forecastScenario === 'baseline' ? 'Balanced' : forecastScenario === 'optimistic' ? 'Aggressive' : 'Protective'} Stocking Policy**.
                                                    Predicted demand volatility is **stable** over the next {backtestWindow} days.
                                                </p>
                                            </div>
                                            <button
                                                onClick={handleRecalibrate}
                                                disabled={loading}
                                                className="md:ml-auto px-10 py-5 bg-primary-600 hover:bg-primary-500 text-white rounded-2xl font-black text-xs tracking-[0.3em] transition-all shadow-xl shadow-primary-900/40 whitespace-nowrap disabled:opacity-50"
                                            >
                                                {loading ? 'PROCESSING...' : 'RE-CALIBRATE MODELS'}
                                            </button>
                                        </div>
                                    )}
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </main>

                {/* Item Details Modal */}
                <AnimatePresence>
                    {selectedItem && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-slate-950/80 backdrop-blur-sm"
                            onClick={() => setSelectedItem(null)}
                        >
                            <motion.div
                                initial={{ scale: 0.9, opacity: 0 }}
                                animate={{ scale: 1, opacity: 1 }}
                                exit={{ scale: 0.9, opacity: 0 }}
                                onClick={(e) => e.stopPropagation()}
                                className="w-full max-w-2xl bg-slate-900 border border-slate-700 rounded-[2rem] shadow-2xl overflow-hidden"
                            >
                                {/* Modal Header */}
                                <div className="p-8 border-b border-slate-800 flex justify-between items-start">
                                    <div>
                                        <div className="flex items-center space-x-3 mb-2">
                                            <Badge status={selectedItem.status} />
                                            <span className="text-xs font-black text-slate-500 tracking-widest uppercase">ID: {selectedItem.item_id}</span>
                                        </div>
                                        <h2 className="text-3xl font-black text-white italic tracking-tight">{selectedItem.item_name}</h2>
                                    </div>
                                    <button
                                        onClick={() => setSelectedItem(null)}
                                        className="p-2 rounded-full hover:bg-slate-800 text-slate-500 hover:text-white transition-colors"
                                    >
                                        <X size={24} />
                                    </button>
                                </div>

                                {/* Modal Content */}
                                <div className="p-8 grid grid-cols-1 md:grid-cols-2 gap-8">
                                    {/* Recommendation Section (Prominent) */}
                                    <div className="col-span-1 md:col-span-2 p-6 rounded-2xl bg-gradient-to-br from-slate-800/50 to-slate-900/50 border border-slate-700">
                                        <div className="flex items-start space-x-4">
                                            <div className="p-3 rounded-xl bg-primary-500/20 text-primary-400">
                                                <Zap size={24} />
                                            </div>
                                            <div>
                                                <h4 className="text-sm font-black text-slate-400 tracking-widest uppercase mb-1">AI RECOMMENDATION</h4>
                                                <p className="text-lg font-bold text-white text-shadow-sm">
                                                    {selectedItem.recommendation}
                                                </p>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Stats Grid */}
                                    <div className="space-y-6">
                                        <div>
                                            <p className="text-[10px] font-black text-slate-500 tracking-widest uppercase mb-2">CURRENT STOCK</p>
                                            <div className="flex items-baseline space-x-2">
                                                <span className="text-3xl font-black text-white">{selectedItem.current_stock.toFixed(0)}</span>
                                                <span className="text-xs font-bold text-slate-500">/ {selectedItem.capacity.toFixed(0)} Cap</span>
                                            </div>
                                            <div className="h-1.5 w-full bg-slate-800 rounded-full mt-3 overflow-hidden">
                                                <div
                                                    className={`h-full rounded-full ${selectedItem.status === 'CRITICAL' ? 'bg-rose-500' :
                                                        selectedItem.status === 'UNDERSTOCKED' ? 'bg-amber-500' :
                                                            selectedItem.status === 'OVERSTOCKED' ? 'bg-sky-500' : 'bg-emerald-500'
                                                        }`}
                                                    style={{ width: `${Math.min(selectedItem.stock_percentage, 100)}%` }}
                                                />
                                            </div>
                                        </div>

                                        <div className="flex justify-between items-center p-4 rounded-xl bg-slate-800/30 border border-slate-800">
                                            <span className="text-xs font-bold text-slate-400">Safety Stock</span>
                                            <span className="text-sm font-black text-white">{selectedItem.safety_stock.toFixed(1)}</span>
                                        </div>
                                    </div>

                                    <div className="space-y-6">
                                        <div>
                                            <p className="text-[10px] font-black text-slate-500 tracking-widest uppercase mb-2">DEMAND VELOCITY</p>
                                            <div className="flex items-baseline space-x-2">
                                                <span className="text-3xl font-black text-white">{selectedItem.mean_daily.toFixed(1)}</span>
                                                <span className="text-xs font-bold text-slate-500">units / day</span>
                                            </div>
                                        </div>

                                        <div className="flex justify-between items-center p-4 rounded-xl bg-slate-800/30 border border-slate-800">
                                            <span className="text-xs font-bold text-slate-400">Reorder Point</span>
                                            <span className="text-sm font-black text-white">{selectedItem.reorder_point.toFixed(1)}</span>
                                        </div>
                                    </div>
                                </div>

                                {/* Modal Footer */}
                                <div className="p-6 border-t border-slate-800 bg-slate-900/50 flex justify-end space-x-4">
                                    <button
                                        onClick={() => setSelectedItem(null)}
                                        className="px-6 py-3 rounded-xl text-xs font-black text-slate-400 hover:text-white transition-colors"
                                    >
                                        CLOSE
                                    </button>
                                    <button className="px-6 py-3 bg-primary-600 hover:bg-primary-500 text-white rounded-xl text-xs font-black tracking-widest shadow-lg shadow-primary-900/20 transition-all">
                                        EXECUTE ACTION
                                    </button>
                                </div>
                            </motion.div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            <style jsx global>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@100..900&display=swap');
        
        body {
          font-family: 'Outfit', sans-serif;
          overscroll-behavior: none;
        }

        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #060912;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #1e293b;
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #334155;
        }

        .glass {
          background: rgba(15, 23, 42, 0.8);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
        }

        .glass-card {
          background: rgba(15, 23, 42, 0.5);
          backdrop-filter: blur(16px);
          border: 1px solid rgba(255, 255, 255, 0.05);
          transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .glass-card:hover {
          background: rgba(15, 23, 42, 0.7);
          border: 1px solid rgba(255, 255, 255, 0.1);
          box-shadow: 0 40px 60px -15px rgba(0, 0, 0, 0.5);
        }

        @keyframes float {
          0% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-10px) rotate(2deg); }
          100% { transform: translateY(0px) rotate(0deg); }
        }

        .animate-float {
          animation: float 4s ease-in-out infinite;
        }
      `}</style>
        </div>
    );
};

export default Dashboard;
