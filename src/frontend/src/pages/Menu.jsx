import React, { useState, useEffect } from 'react';
import { Card } from '../components/ui/Card';
import { MenuService } from '../services/api';
import { Star, AlertCircle, TrendingUp, TrendingDown, DollarSign, ChefHat } from 'lucide-react';
import {
    ScatterChart,
    Scatter,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
    Label,
    Cell
} from 'recharts';

export const MenuPage = () => {
    const [matrixData, setMatrixData] = useState([]);
    const [recommendations, setRecommendations] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const matrixRes = await MenuService.getMatrix();
                const recsRes = await MenuService.getRecommendations();

                // Backend returns: { matrix: { stars: [], puzzles: [], plowhorses: [], dogs: [] } }
                // Frontend needs: A flattened array with x, y coordinates
                const flattenedData = [];
                const categories = ['stars', 'puzzles', 'plowhorses', 'dogs'];

                categories.forEach(cat => {
                    const items = matrixRes.matrix[cat] || [];
                    items.forEach(item => {
                        flattenedData.push({
                            ...item,
                            name: item.item_name,
                            x: item.total_orders, // Use total_orders for X axis as per previous tooltip logic
                            y: item.revenue / item.total_orders, // Calculate margin proxy if needed, or use profitability_score
                            profit: item.revenue / item.total_orders,
                            orders: item.total_orders,
                            quadrant: cat.charAt(0).toUpperCase() + cat.slice(1).slice(0, -1) // e.g., 'stars' -> 'Star'
                        });
                    });
                });

                setMatrixData(flattenedData);
                setRecommendations(recsRes.recommendations);
            } catch (err) {
                console.error("Failed to fetch menu analytics", err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    const getQuadrantIcon = (quadrant) => {
        switch (quadrant) {
            case 'Star': return <Star className="w-5 h-5 text-yellow-400" fill="currentColor" />;
            case 'Plowhorse': return <DollarSign className="w-5 h-5 text-green-400" />;
            case 'Puzzle': return <TrendingDown className="w-5 h-5 text-orange-400" />; // High margin, low vol (potential)
            case 'Dog': return <AlertCircle className="w-5 h-5 text-red-400" />;
            default: return null;
        }
    };

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
                <div className="bg-slate-900/90 border border-white/10 p-3 rounded-lg shadow-xl backdrop-blur-md">
                    <p className="font-semibold text-white mb-1">{data.name}</p>
                    <div className="text-xs text-gray-300 space-y-1">
                        <p className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-eagle-green" />
                            Avg Revenue: {data.y.toFixed(2)} DKK
                        </p>
                        <p className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-blue-400" />
                            Volume: {data.x} orders
                        </p>
                        <p className="mt-1 font-medium text-white border-t border-white/10 pt-1">
                            {data.quadrant}
                        </p>
                    </div>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold text-white">Menu Engineering</h2>
                    <p className="text-gray-400">Optimize profitability using the BCG Matrix</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* BCG Matrix Chart */}
                <Card key={matrixData.length} className="lg:col-span-2 min-h-[500px] flex flex-col">
                    <div className="mb-4 flex items-center justify-between">
                        <h3 className="font-semibold text-lg">Profitability vs Popularity Matrix</h3>
                        <div className="flex gap-4 text-xs">
                            <span className="flex items-center text-yellow-400"><Star className="w-3 h-3 mr-1" fill="currentColor" /> Stars</span>
                            <span className="flex items-center text-green-400"><DollarSign className="w-3 h-3 mr-1" /> Plowhorses</span>
                            <span className="flex items-center text-orange-400"><TrendingUp className="w-3 h-3 mr-1" /> Puzzles</span>
                            <span className="flex items-center text-red-400"><AlertCircle className="w-3 h-3 mr-1" /> Dogs</span>
                        </div>
                    </div>

                    <div className="flex-1 w-full min-h-[400px]">
                        <ResponsiveContainer key={`chart-${matrixData.length}`} width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                <XAxis
                                    type="number"
                                    dataKey="x"
                                    name="Popularity"
                                    stroke="#9ca3af"
                                    label={{ value: 'Popularity (Orders)', position: 'bottom', fill: '#9ca3af', offset: 0 }}
                                />
                                <YAxis
                                    type="number"
                                    dataKey="y"
                                    name="Profitability"
                                    stroke="#9ca3af"
                                    label={{ value: 'Profitability (Margin $)', angle: -90, position: 'left', fill: '#9ca3af' }}
                                />
                                <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />

                                {/* Quadrant Lines (assuming centered around averages approx) */}
                                {/* These specific values would ideally come from the API averages */}
                                <ReferenceLine x={50} stroke="#4ade80" strokeDasharray="3 3" />
                                <ReferenceLine y={10} stroke="#4ade80" strokeDasharray="3 3" />

                                {/* Quadrant Labels */}
                                <ReferenceLine segment={[{ x: 80, y: 15 }, { x: 80, y: 15 }]} label={{ value: "STARS", fill: "#facc15", fontSize: 24, opacity: 0.2 }} />

                                <Scatter name="Menu Items" data={matrixData} fill="#8884d8">
                                    {matrixData.map((entry, index) => (
                                        <Cell
                                            key={`cell-${index}`}
                                            fill={
                                                entry.quadrant === 'Star' ? '#facc15' :
                                                    entry.quadrant === 'Plowhorse' ? '#4ade80' :
                                                        entry.quadrant === 'Puzzle' ? '#fb923c' : '#f87171'
                                            }
                                        />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </Card>

                {/* Recommendations Panel */}
                <div className="space-y-6">
                    <Card className="h-full overflow-y-auto max-h-[600px]">
                        <h3 className="font-semibold text-lg mb-4 flex items-center">
                            <ChefHat className="w-5 h-5 mr-2 text-eagle-green" />
                            Strategic Actions
                        </h3>
                        <div className="space-y-4">
                            {loading ? (
                                <div className="text-center py-8 text-gray-500">Loading insights...</div>
                            ) : recommendations.length === 0 ? (
                                <div className="text-center py-8 text-gray-500">No immediate actions required.</div>
                            ) : (
                                recommendations.map((rec, i) => (
                                    <div key={i} className="p-3 rounded-lg bg-white/5 border border-white/5 hover:bg-white/10 transition-colors">
                                        <div className="flex items-start justify-between mb-2">
                                            <h4 className="font-medium text-white text-sm">{rec.item_name}</h4>
                                            <span className={`text-[10px] px-2 py-0.5 rounded-full ${rec.action === 'PROMOTE' || rec.action === 'Promote' ? 'bg-yellow-500/20 text-yellow-400' :
                                                rec.action === 'RE-ENGINEER' || rec.action === 'Investigate' ? 'bg-orange-500/20 text-orange-400' :
                                                    'bg-red-500/20 text-red-400'
                                                }`}>
                                                {rec.action.toUpperCase()}
                                            </span>
                                        </div>
                                        <p className="text-xs text-gray-400">{rec.recommendation}</p>
                                        {rec.details && <p className="text-[10px] text-gray-500 mt-1 italic">{rec.details}</p>}
                                    </div>
                                ))
                            )}
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    );
};
