import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { Card, CardHeader } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { InventoryService } from '../services/api';
import { Search, Filter, AlertTriangle, CheckCircle, Package, Eye } from 'lucide-react';

export const InventoryPage = () => {
    const [searchParams, setSearchParams] = useSearchParams();
    const [items, setItems] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState(searchParams.get('search') || '');
    const [statusFilter, setStatusFilter] = useState('ALL');

    useEffect(() => {
        const querySearch = searchParams.get('search');
        if (querySearch && querySearch !== searchTerm) {
            setSearchTerm(querySearch);
        }
    }, [searchParams]);

    useEffect(() => {
        const fetchInventory = async () => {
            setLoading(true);
            try {
                // Use the filter endpoint which supports search and status
                const data = await InventoryService.getInventory({
                    status: statusFilter === 'ALL' ? null : statusFilter,
                    search_term: searchTerm || null,
                    limit: 100
                });
                setItems(data);
            } catch (err) {
                console.error("Failed to fetch inventory", err);
            } finally {
                setLoading(false);
            }
        };

        // Debounce search
        const timer = setTimeout(() => {
            fetchInventory();
        }, 300);

        return () => clearTimeout(timer);
    }, [searchTerm, statusFilter]);

    const getStatusBadge = (status) => {
        switch (status) {
            case 'CRITICAL': return <Badge variant="error" className="animate-pulse">CRITICAL</Badge>;
            case 'UNDERSTOCKED': return <Badge variant="warning">Low Stock</Badge>;
            case 'OVERSTOCKED': return <Badge variant="info">Overstocked</Badge>;
            default: return <Badge variant="success">Healthy</Badge>;
        }
    };

    const [isAddModalOpen, setIsAddModalOpen] = useState(false);

    return (
        <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h2 className="text-2xl font-bold text-white">Inventory Management</h2>
                    <p className="text-gray-400 text-sm">Real-time stock tracking and AI-powered optimization.</p>
                </div>
                <Button
                    onClick={() => setIsAddModalOpen(true)}
                    className="bg-eagle-green text-black hover:bg-eagle-green/90"
                >
                    <Package className="w-4 h-4 mr-2" />
                    Add Item
                </Button>
            </div>

            {/* Add Item Modal */}
            {isAddModalOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
                    <Card className="w-full max-w-md border-eagle-green/30 bg-eagle-black-surface1 p-6 animate-in zoom-in-95 duration-200">
                        <div className="flex justify-between items-center mb-6">
                            <h3 className="text-xl font-bold text-white">Add New Inventory Item</h3>
                            <button onClick={() => setIsAddModalOpen(false)} className="text-gray-400 hover:text-white">&times;</button>
                        </div>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-400 mb-1">Item Name</label>
                                <input type="text" className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-2 text-white focus:border-eagle-green/50 outline-none" placeholder="e.g. Fresh Tomatoes" />
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-400 mb-1">Initial Stock</label>
                                    <input type="number" className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-2 text-white outline-none" placeholder="100" />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-400 mb-1">Capacity</label>
                                    <input type="number" className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-2 text-white outline-none" placeholder="200" />
                                </div>
                            </div>
                            <div className="pt-4 flex gap-3">
                                <Button className="flex-1 bg-white/5 hover:bg-white/10" onClick={() => setIsAddModalOpen(false)}>Cancel</Button>
                                <Button className="flex-1 bg-eagle-green text-black" onClick={() => setIsAddModalOpen(false)}>Save Item</Button>
                            </div>
                        </div>
                    </Card>
                </div>
            )}

            {/* Filters */}
            <Card>
                <div className="flex flex-col md:flex-row gap-4 mb-6">
                    <div className="relative flex-1">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                        <input
                            type="text"
                            placeholder="Search items by name or ID..."
                            className="w-full bg-black/20 border border-white/10 rounded-lg pl-10 pr-4 py-2 text-sm text-white focus:outline-none focus:border-eagle-green/50 transition-colors"
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                        />
                    </div>
                    <div className="flex items-center gap-2">
                        <Filter className="w-4 h-4 text-gray-500" />
                        <select
                            className="bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-eagle-green/50"
                            value={statusFilter}
                            onChange={(e) => setStatusFilter(e.target.value)}
                        >
                            <option value="ALL">All Status</option>
                            <option value="CRITICAL">Critical</option>
                            <option value="UNDERSTOCKED">Understocked</option>
                            <option value="HEALTHY">Healthy</option>
                            <option value="OVERSTOCKED">Overstocked</option>
                        </select>
                    </div>
                </div>

                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="text-left border-b border-white/10">
                            <tr>
                                <th className="pb-4 px-4 font-semibold text-xs uppercase tracking-wider text-gray-500">Item</th>
                                <th className="pb-4 px-4 font-semibold text-xs uppercase tracking-wider text-gray-500">Status</th>
                                <th className="pb-4 px-4 font-semibold text-xs uppercase tracking-wider text-gray-500">Stock Level</th>
                                <th className="pb-4 px-4 font-semibold text-xs uppercase tracking-wider text-gray-500">Safety Stock</th>
                                <th className="pb-4 px-4 font-semibold text-xs uppercase tracking-wider text-gray-500">Recommendation</th>
                                <th className="pb-4 px-4 font-semibold text-xs uppercase tracking-wider text-gray-500 text-right">Action</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                            {loading ? (
                                [1, 2, 3, 4, 5].map(i => (
                                    <tr key={i} className="animate-pulse">
                                        <td colSpan="6" className="py-6 bg-white/5 rounded-lg m-2"></td>
                                    </tr>
                                ))
                            ) : items.length === 0 ? (
                                <tr>
                                    <td colSpan="6" className="py-12 text-center text-gray-500">
                                        No items found matching your filters.
                                    </td>
                                </tr>
                            ) : (
                                items.map((item) => (
                                    <tr key={item.item_id} className="group hover:bg-white/5 transition-colors border-l-2 border-transparent hover:border-eagle-green">
                                        <td className="py-4 px-4">
                                            <Link to={`/inventory/item/${item.item_id}`} className="flex flex-col group/link">
                                                <span className="font-medium text-white group-hover/link:text-eagle-green transition-colors">{item.item_name || 'N/A'}</span>
                                                <span className="text-xs text-gray-500">ID: {item.item_id || 'N/A'} • Class {item.abc_class || 'N/A'}</span>
                                            </Link>
                                        </td>
                                        <td className="py-4 px-4">
                                            <Badge variant={
                                                item.status === 'CRITICAL' ? 'danger' :
                                                    item.status === 'UNDERSTOCKED' ? 'warning' :
                                                        item.status === 'HEALTHY' ? 'success' : 'info'
                                            }>
                                                {item.status || 'UNKNOWN'}
                                            </Badge>
                                        </td>
                                        <td className="py-4 px-4">
                                            <div className="flex flex-col gap-1.5 w-32">
                                                <div className="flex justify-between text-[10px] text-gray-400">
                                                    <span>{Math.round(item.current_stock || 0)} units</span>
                                                    <span>{item.stock_percentage ? Math.min(item.stock_percentage, 100) : 0}%</span>
                                                </div>
                                                <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                                                    <div
                                                        className={`h-full rounded-full transition-all duration-500 ${item.status === 'CRITICAL' ? 'bg-red-500' :
                                                            item.status === 'UNDERSTOCKED' ? 'bg-amber-500' :
                                                                item.status === 'HEALTHY' ? 'bg-eagle-green' : 'bg-blue-500'
                                                            }`}
                                                        style={{ width: `${Math.min(item.stock_percentage || 0, 100)}%` }}
                                                    />
                                                </div>
                                            </div>
                                        </td>
                                        <td className="py-4 px-4 text-sm text-gray-400">
                                            {Math.round(item.reorder_point || 0)}
                                        </td>
                                        <td className="py-4 px-4 text-sm text-white italic">
                                            {item.recommendation || 'No recommendation'}
                                        </td>
                                        <td className="py-4 px-4 text-right flex items-center justify-end gap-2">
                                            <Link to={`/inventory/item/${item.item_id}`}>
                                                <Button variant="ghost" size="sm" className="text-gray-400 hover:text-white hover:bg-white/5">
                                                    <Eye className="w-4 h-4" />
                                                </Button>
                                            </Link>
                                            <Button variant="ghost" size="sm" className="text-eagle-green hover:bg-eagle-green/10">
                                                Restock
                                            </Button>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </Card>
        </div>
    );
};
