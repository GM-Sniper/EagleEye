import React, { useState, useEffect } from 'react';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { PricingService } from '../services/api';
import { Tag, TrendingUp, DollarSign, Percent } from 'lucide-react';

export const PricingPage = () => {
    const [discounts, setDiscounts] = useState([]);
    const [optimizations, setOptimizations] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const discountRes = await PricingService.getDiscounts();
                const optRes = await PricingService.getOptimization();
                setDiscounts(discountRes.recommendations);
                setOptimizations(optRes.optimizations);
            } catch (err) {
                console.error("Failed to fetch pricing data", err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-2xl font-bold text-white">Smart Pricing Engine</h2>
                <p className="text-gray-400">AI-driven discount and pricing recommendations</p>
            </div>

            {/* 1. Discount Recommendations for Overstocked Items */}
            <section>
                <div className="flex items-center gap-2 mb-4">
                    <Tag className="text-eagle-green w-5 h-5" />
                    <h3 className="text-xl font-semibold text-white">Clearance Recommendations</h3>
                    <Badge variant="info">{discounts.length} Items</Badge>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {loading ? (
                        [1, 2, 3].map(i => <Card key={i} className="h-48 animate-pulse bg-white/5" />)
                    ) : discounts.length === 0 ? (
                        <div className="col-span-full text-center py-8 text-gray-500">No clearance items found.</div>
                    ) : (
                        discounts.map((item) => (
                            <Card key={item.item_id} className="relative group border-eagle-green/20">
                                <div className="absolute top-4 right-4 animate-pulse">
                                    <Badge variant="success">Save {Math.round(item.suggested_discount_pct)}%</Badge>
                                </div>

                                <h4 className="font-semibold text-white text-lg pr-12 truncate">{item.item_name}</h4>
                                <div className="mt-2 space-y-2 text-sm text-gray-400">
                                    <div className="flex justify-between">
                                        <span>Stock Level:</span>
                                        <span className="text-white">{Math.round(item.current_stock)} / {Math.round(item.capacity)}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Excess:</span>
                                        <span className="text-red-400">+{Math.round(item.current_stock - item.capacity)} units</span>
                                    </div>
                                    <div className="pt-2 border-t border-white/10 flex justify-between items-center">
                                        <span className="text-eagle-green font-medium">Recover {Math.round(item.potential_revenue)} DKK</span>
                                        <span className="text-xs">Est. clear: {Math.round(item.days_to_clear_discounted)} days</span>
                                    </div>
                                </div>

                                <Button size="sm" className="w-full mt-4 bg-eagle-green/10 text-eagle-green hover:bg-eagle-green hover:text-black">
                                    Apply Discount
                                </Button>
                            </Card>
                        ))
                    )}
                </div>
            </section>

            {/* 2. Margin Optimization (Elasticity Based) */}
            <section>
                <div className="flex items-center gap-2 mb-4">
                    <TrendingUp className="text-eagle-green w-5 h-5" />
                    <h3 className="text-xl font-semibold text-white">Margin Optimization</h3>
                </div>

                <Card className="bg-gradient-to-r from-zinc-900/40 to-transparent border-white/5">
                    <div className="p-4 flex items-start gap-4">
                        <div className="p-3 rounded-full bg-eagle-green/10">
                            <DollarSign className="w-6 h-6 text-eagle-green" />
                        </div>
                        <div>
                            <h4 className="text-lg font-semibold text-white">Price Sensitivity Analysis</h4>
                            <p className="text-gray-400 max-w-2xl mt-1">
                                Our models are analyzing historical transaction data to calculate price elasticity for
                                top-selling items. Once complete, we will recommend price adjustments to maximize
                                total revenue without sacrificing volume.
                            </p>
                            <div className="mt-4 flex gap-2">
                                {optimizations.slice(0, 3).map((opt, i) => (
                                    <Badge key={i} variant="default" className="bg-white/5 text-gray-300 border-white/10">
                                        Analyze {opt.item_name || `Item #${opt.item_id}`}
                                    </Badge>
                                ))}
                            </div>
                        </div>
                    </div>
                </Card>
            </section>
        </div>
    );
};
