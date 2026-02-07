import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { InventoryService } from '../services/api';
import {
    ChevronLeft,
    TrendingUp,
    Package,
    AlertTriangle,
    CheckCircle,
    ArrowUpRight,
    Target,
    Activity,
    ChefHat,
    ArrowDown
} from 'lucide-react';
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
    ReferenceLine
} from 'recharts';

export const ItemDetailsPage = () => {
    const { id } = useParams();
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchDetails = async () => {
            setLoading(true);
            try {
                const res = await InventoryService.getItemDetails(id);
                setData(res);
            } catch (err) {
                console.error("Failed to fetch item details", err);
                setError("Could not load item details. Please check the ID.");
            } finally {
                setLoading(false);
            }
        };
        fetchDetails();
    }, [id]);

    if (loading) return (
        <div className="flex items-center justify-center min-h-[400px]">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-eagle-green"></div>
        </div>
    );

    if (error || !data) return (
        <Card className="text-center py-12">
            <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-xl font-bold text-white mb-2">{error || "Item not found"}</h3>
            <Link to="/inventory">
                <Button variant="ghost" className="text-eagle-green">Return to Inventory</Button>
            </Link>
        </Card>
    );

    // Unified Chart Data: Merging History, Backtest Comparison, and Future Forecast
    const getChartData = () => {
        if (!data) return [];

        const backtestDates = new Set((data.backtest || []).map(b => b.date));

        // 1. Pure History (excluding points covered by backtest)
        const pureHistory = data.history
            .filter(h => !backtestDates.has(h.date))
            .map(h => ({
                date: h.date,
                actual: h.demand,
                predicted: null,
                type: 'history'
            }));

        // 2. Backtest Overlap (Actual vs Predicted)
        const backtest = (data.backtest || []).map(b => ({
            date: b.date,
            actual: b.actual,
            predicted: b.predicted,
            type: 'backtest'
        }));

        // 3. Future Forecast
        const future = data.forecast.map(f => ({
            date: f.date,
            actual: null,
            predicted: f.predicted_demand,
            upper: f.upper_bound,
            lower: f.lower_bound,
            type: 'future'
        }));

        return [...pureHistory, ...backtest, ...future];
    };

    const chartData = getChartData();

    const getAccuracyColor = (mape) => {
        if (mape < 10) return 'text-eagle-green';
        if (mape < 25) return 'text-yellow-400';
        return 'text-red-400';
    };

    return (
        <div className="space-y-6">
            {/* Header / Breadcrumbs */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                    <Link to="/inventory" className="p-2 hover:bg-white/5 rounded-full transition-colors">
                        <ChevronLeft className="w-6 h-6 text-gray-400" />
                    </Link>
                    <div>
                        <div className="flex items-center gap-2 mb-1">
                            <h2 className="text-3xl font-bold text-white">{data.item_name}</h2>
                            <Badge variant="neutral" className="bg-white/10 text-gray-300">Class {data.abc_class}</Badge>
                        </div>
                        <p className="text-gray-400 text-sm flex items-center gap-2">
                            <Package className="w-4 h-4" /> SKU: {id} • Category: Fresh Produce
                        </p>
                    </div>
                </div>
                <div className="flex gap-3">
                    <Button variant="ghost" className="border border-white/10 hover:bg-white/5">
                        Adjust Par Levels
                    </Button>
                    <Button className="bg-eagle-green text-black hover:bg-eagle-green/90">
                        Create Order
                    </Button>
                </div>
            </div>

            {/* KPI Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <Card className="p-4 bg-eagle-black-surface1 border-white/5 relative overflow-hidden group">
                    <div className="relative z-10">
                        <p className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-1">Current Stock</p>
                        <div className="flex items-end gap-2">
                            <h3 className="text-2xl font-bold text-white">{Math.round(data.current_stock)}</h3>
                            <span className="text-gray-500 mb-1 text-sm">/ {data.capacity} units</span>
                        </div>
                        <div className="mt-3 h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                            <div
                                className={`h-full rounded-full transition-all duration-1000 ${data.status === 'CRITICAL' ? 'bg-red-500' : 'bg-eagle-green'}`}
                                style={{ width: `${(data.current_stock / data.capacity) * 100}%` }}
                            />
                        </div>
                    </div>
                    <Package className="absolute -right-2 -bottom-2 w-16 h-16 text-white/5 group-hover:text-eagle-green/10 transition-colors" />
                </Card>

                <Card className="p-4 bg-eagle-black-surface1 border-white/5 relative overflow-hidden group">
                    <div className="relative z-10">
                        <p className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-1">Model Accuracy</p>
                        <div className="flex items-end gap-2">
                            <h3 className={`text-2xl font-bold ${data.accuracy !== 'N/A' ? getAccuracyColor(parseFloat(data.mape)) : 'text-gray-500'}`}>
                                {data.accuracy}
                            </h3>
                            <span className="text-gray-500 mb-1 text-sm">MAPE: {data.mape}</span>
                        </div>
                        <p className="mt-2 text-[10px] text-gray-500 flex items-center gap-1">
                            <Target className="w-3 h-3" /> Based on last 7 days of actuals
                        </p>
                    </div>
                    <Activity className="absolute -right-2 -bottom-2 w-16 h-16 text-white/5 group-hover:text-eagle-green/10 transition-colors" />
                </Card>

                <Card className="p-4 bg-eagle-black-surface1 border-white/5 relative overflow-hidden group">
                    <div className="relative z-10">
                        <p className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-1">Reorder Point</p>
                        <div className="flex items-end gap-2">
                            <h3 className="text-2xl font-bold text-white">{Math.round(data.reorder_point)}</h3>
                            <span className="text-gray-500 mb-1 text-sm">threshold</span>
                        </div>
                        <p className="mt-2 text-[10px] text-gray-500 flex items-center gap-1">
                            {data.current_stock <= data.reorder_point ?
                                <span className="text-red-400 flex items-center gap-1"><AlertTriangle className="w-3 h-3" /> Action required</span> :
                                <span className="text-eagle-green flex items-center gap-1"><CheckCircle className="w-3 h-3" /> Supply healthy</span>
                            }
                        </p>
                    </div>
                    <ArrowDown className="absolute -right-2 -bottom-2 w-16 h-16 text-white/5 group-hover:text-eagle-green/10 transition-colors" />
                </Card>

                <Card className="p-4 bg-eagle-black-surface1 border-white/5 relative overflow-hidden group">
                    <div className="relative z-10">
                        <p className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-1">Avg Daily Demand</p>
                        <div className="flex items-end gap-2">
                            <h3 className="text-2xl font-bold text-white">{data.mean_daily.toFixed(1)}</h3>
                            <span className="text-gray-500 mb-1 text-sm">units/day</span>
                        </div>
                        <p className="mt-2 text-[10px] text-gray-500 flex items-center gap-1">
                            <TrendingUp className="w-3 h-3" /> Stability: {(1 - data.std_daily / data.mean_daily).toFixed(2)}
                        </p>
                    </div>
                    <Activity className="absolute -right-2 -bottom-2 w-16 h-16 text-white/5 group-hover:text-eagle-green/10 transition-colors" />
                </Card>
            </div>

            {/* Main Content: Chart + Sidebar */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Demand Intelligence Chart */}
                <Card className="lg:col-span-2 p-6 h-[500px] flex flex-col">
                    <div className="flex items-center justify-between mb-6">
                        <div>
                            <h3 className="text-xl font-bold text-white flex items-center gap-2">
                                <Activity className="w-5 h-5 text-eagle-green" />
                                Demand Intelligence
                            </h3>
                            <p className="text-sm text-gray-400">30d History + 14d Hybrid AI Forecast</p>
                        </div>
                        <div className="flex gap-4">
                            <div className="flex items-center gap-2 text-xs">
                                <div className="w-3 h-3 bg-white/20 rounded-sm"></div>
                                <span className="text-gray-400">Actual</span>
                            </div>
                            <div className="flex items-center gap-2 text-xs">
                                <div className="w-3 h-3 bg-eagle-green rounded-sm"></div>
                                <span className="text-gray-400">Forecast</span>
                            </div>
                        </div>
                    </div>

                    <div className="flex-1 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                <defs>
                                    <linearGradient id="colorHistory" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#ffffff" stopOpacity={0.1} />
                                        <stop offset="95%" stopColor="#ffffff" stopOpacity={0} />
                                    </linearGradient>
                                    <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#00E054" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#00E054" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                                <XAxis
                                    dataKey="date"
                                    stroke="#52525b"
                                    fontSize={10}
                                    tickFormatter={(val) => val.split('-').slice(1).join('/')}
                                />
                                <YAxis stroke="#52525b" fontSize={10} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: 'rgb(24 24 27)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                                    itemStyle={{ fontSize: '12px' }}
                                />
                                <ReferenceLine x={data.history[data.history.length - 1]?.date} stroke="#00E054" strokeDasharray="3 3" label={{ position: 'top', value: 'Today', fill: '#00E054', fontSize: 10 }} />

                                <Area
                                    type="monotone"
                                    dataKey="actual"
                                    stroke="rgba(255,255,255,0.3)"
                                    fillOpacity={1}
                                    fill="url(#colorHistory)"
                                    strokeWidth={2}
                                    connectNulls
                                    activeDot={{ r: 4, fill: '#fff' }}
                                    name="Actual Demand"
                                />
                                <Area
                                    type="monotone"
                                    dataKey="predicted"
                                    stroke="#00E054"
                                    fillOpacity={1}
                                    fill="url(#colorForecast)"
                                    strokeWidth={3}
                                    name="AI Forecast"
                                />
                                {/* Confidence Band */}
                                <Area
                                    type="monotone"
                                    dataKey="upper"
                                    stroke="transparent"
                                    fill="#00E054"
                                    fillOpacity={0.05}
                                    name="Upper Bound"
                                />
                                <Area
                                    type="monotone"
                                    dataKey="lower"
                                    stroke="transparent"
                                    fill="#00E054"
                                    fillOpacity={0.05}
                                    name="Lower Bound"
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </Card>

                {/* Sidebar: Ingredients & Intelligence */}
                <div className="space-y-6">
                    {/* ML Insights */}
                    <Card className="p-5 border-eagle-green/20 bg-eagle-green/5">
                        <h4 className="text-white font-bold flex items-center gap-2 mb-3">
                            <Target className="w-4 h-4 text-eagle-green" />
                            AI Insight Layer
                        </h4>
                        <div className="space-y-3">
                            <p className="text-sm text-gray-300 leading-relaxed italic">
                                "{data.item_name} shows a strong correlation with weekend traffic spikes. Forecast suggests maintaining +15% safety stock buffer for the upcoming Friday."
                            </p>
                            <div className="pt-2 flex items-center justify-between border-t border-eagle-green/10">
                                <span className="text-[10px] text-gray-500 uppercase">Engine: {data.model_used}</span>
                                <span className="text-[10px] text-eagle-green font-bold">OPTIMIZED</span>
                            </div>
                        </div>
                    </Card>

                    {/* Composition / Ingredients */}
                    <Card className="p-5">
                        <h4 className="text-white font-bold flex items-center gap-2 mb-4">
                            <ChefHat className="w-4 h-4 text-eagle-green" />
                            Composition & Prep
                        </h4>
                        <div className="space-y-4">
                            {data.ingredients && data.ingredients.length > 0 ? (
                                data.ingredients.map((ing, i) => (
                                    <div key={i} className="flex items-center justify-between p-2 rounded bg-white/5 border border-white/5">
                                        <div className="flex flex-col">
                                            <span className="text-xs text-white">{ing.name}</span>
                                            <span className="text-[10px] text-gray-500">Need: {ing.quantity} {ing.unit} per unit</span>
                                        </div>
                                        <ArrowUpRight className="w-3 h-3 text-gray-500" />
                                    </div>
                                ))
                            ) : (
                                <p className="text-xs text-gray-500 text-center py-4">No bill of materials defined for this item.</p>
                            )}
                            <Link to="/inventory">
                                <Button size="sm" variant="ghost" className="w-full text-xs text-eagle-green hover:bg-eagle-green/5 mt-2">
                                    View Full Bill of Materials
                                </Button>
                            </Link>
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    );
};
