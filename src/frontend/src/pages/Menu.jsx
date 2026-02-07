import React, { useState, useEffect } from 'react';
import { Card, CardHeader } from '../components/ui/Card';
import { MenuService } from '../services/api';
import { TrendingUp, DollarSign, ShoppingCart, BarChart3 } from 'lucide-react';
import { Badge } from '../components/ui/Badge';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Cell
} from 'recharts';

export const MenuPage = () => {
    const [topItems, setTopItems] = useState([]);
    const [loading, setLoading] = useState(true);
    const [sortBy, setSortBy] = useState('revenue'); // 'revenue' or 'orders'

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            try {
                // Ensure MenuService is correctly handling the API response format
                const res = await MenuService.getTopItems(sortBy, 20);
                if (res && res.items) {
                    setTopItems(res.items);
                } else if (Array.isArray(res)) {
                    setTopItems(res);
                } else {
                    setTopItems([]);
                }
            } catch (err) {
                console.error("Failed to fetch demand data", err);
                setTopItems([]);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [sortBy]);

    // Prepare chart data (top 10 for visualization)
    const chartData = topItems.slice(0, 10).map(item => ({
        name: item.item_name?.length > 15 ? item.item_name.substring(0, 15) + '...' : item.item_name,
        value: sortBy === 'revenue' ? item.revenue : item.total_orders,
        fullName: item.item_name
    }));

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
                <div className="bg-slate-900/95 border border-white/10 p-3 rounded-lg shadow-xl backdrop-blur-md">
                    <p className="font-semibold text-white mb-1">{data.fullName}</p>
                    <p className="text-sm text-eagle-green">
                        {sortBy === 'revenue'
                            ? `${data.value.toLocaleString()} DKK`
                            : `${data.value.toLocaleString()} orders`}
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h2 className="text-2xl font-bold text-white">Demand Analysis</h2>
                    <p className="text-gray-400">Identify your highest-performing inventory items</p>
                </div>

                {/* Sort Toggle */}
                <div className="flex items-center gap-2 bg-white/5 p-1 rounded-xl border border-white/10">
                    <button
                        onClick={() => setSortBy('revenue')}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${sortBy === 'revenue'
                                ? 'bg-eagle-green text-black'
                                : 'text-gray-400 hover:text-white'
                            }`}
                    >
                        <DollarSign className="w-4 h-4 inline mr-1" />
                        By Revenue
                    </button>
                    <button
                        onClick={() => setSortBy('orders')}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${sortBy === 'orders'
                                ? 'bg-eagle-green text-black'
                                : 'text-gray-400 hover:text-white'
                            }`}
                    >
                        <ShoppingCart className="w-4 h-4 inline mr-1" />
                        By Orders
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Bar Chart - Top 10 */}
                <Card className="lg:col-span-2 min-h-[450px]">
                    <CardHeader
                        title="Top 10 Items"
                        subtitle={`Ranked by ${sortBy === 'revenue' ? 'total revenue' : 'order volume'}`}
                    />

                    {loading ? (
                        <div className="h-[350px] flex items-center justify-center">
                            <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-eagle-green"></div>
                        </div>
                    ) : (
                        <div className="h-[350px] mt-4">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#333" horizontal={true} vertical={false} />
                                    <XAxis type="number" stroke="#9ca3af" fontSize={10} tickFormatter={(val) => sortBy === 'revenue' ? `${(val / 1000).toFixed(0)}k` : val} />
                                    <YAxis type="category" dataKey="name" stroke="#9ca3af" fontSize={10} width={100} />
                                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
                                    <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
                                        {chartData.map((entry, index) => (
                                            <Cell
                                                key={`cell-${index}`}
                                                fill={index === 0 ? '#00E054' : index < 3 ? '#22c55e' : '#166534'}
                                            />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </Card>

                {/* Full List */}
                <Card className="max-h-[550px] overflow-y-auto">
                    <CardHeader title="All High-Demand Items" subtitle="Top 20 performers" />

                    <div className="mt-4 space-y-3">
                        {loading ? (
                            [1, 2, 3, 4, 5].map(i => (
                                <div key={i} className="h-16 bg-white/5 rounded-lg animate-pulse" />
                            ))
                        ) : topItems.length === 0 ? (
                            <div className="text-center py-8 text-gray-500">No data available</div>
                        ) : (
                            topItems.map((item, i) => (
                                <div
                                    key={item.item_id}
                                    className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 transition-colors group"
                                >
                                    <div className="flex items-center gap-3">
                                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold ${i < 3 ? 'bg-eagle-green/20 text-eagle-green ring-1 ring-eagle-green/30' : 'bg-white/10 text-gray-400'
                                            }`}>
                                            {i + 1}
                                        </div>
                                        <div>
                                            <p className="text-sm font-semibold text-white group-hover:text-eagle-green transition-colors truncate max-w-[150px]">
                                                {item.item_name}
                                            </p>
                                            <p className="text-xs text-gray-500">
                                                Class {item.abc_class} • {item.total_orders?.toLocaleString()} orders
                                            </p>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-sm font-bold text-white">
                                            {item.revenue?.toLocaleString()} DKK
                                        </p>
                                        <Badge variant={item.abc_class === 'A' ? 'success' : item.abc_class === 'B' ? 'warning' : 'neutral'} className="text-[10px]">
                                            {item.abc_class === 'A' ? 'High Value' : item.abc_class === 'B' ? 'Medium' : 'Standard'}
                                        </Badge>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </Card>
            </div>

            {/* Insight Banner */}
            <Card className="bg-gradient-to-r from-eagle-green/10 via-transparent to-transparent border-eagle-green/20">
                <div className="flex items-center gap-4">
                    <div className="p-3 rounded-xl bg-eagle-green/10 text-eagle-green">
                        <TrendingUp className="w-6 h-6" />
                    </div>
                    <div>
                        <h4 className="font-semibold text-white">Demand Insight</h4>
                        <p className="text-sm text-gray-400">
                            Your top 10 items account for approximately 80% of total revenue. Focus inventory efforts on maintaining optimal stock levels for these high-performers.
                        </p>
                    </div>
                </div>
            </Card>
        </div>
    );
};
