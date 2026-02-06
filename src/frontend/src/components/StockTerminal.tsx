import React, { useState, useMemo } from "react";

// --- Types ---
interface StatItem {
    label: string;
    value: string;
    sub: string;
    trend: string;
    trendUp: boolean;
    alert?: boolean;
}

interface InventoryItem {
    id: number;
    name: string;
    sku: string;
    stock: number;
    capacity: number;
    status: 'Critical' | 'Understocked' | 'Healthy' | 'Overstocked';
    store: string;
    demand: 'High' | 'Medium' | 'Low';
    mean: number;
    abc: string;
    color: 'emerald' | 'red' | 'blue' | 'orange';
}

// --- Mock Data ---
const MOCK_INVENTORY: InventoryItem[] = [
    { id: 101, name: "Fresh Milk 1L", sku: "DAI-001", stock: 42, capacity: 200, status: 'Critical', store: 'Downtown Branch', demand: 'High', mean: 150, abc: 'A', color: 'red' },
    { id: 102, name: "Sourdough Bread", sku: "BAK-042", stock: 85, capacity: 100, status: 'Healthy', store: 'Downtown Branch', demand: 'Medium', mean: 45, abc: 'B', color: 'emerald' },
    { id: 103, name: "Organic Bananas", sku: "PRO-112", stock: 200, capacity: 180, status: 'Overstocked', store: 'Uptown Hub', demand: 'Medium', mean: 80, abc: 'C', color: 'orange' },
    { id: 104, name: "Avocados (Hass)", sku: "PRO-205", stock: 15, capacity: 120, status: 'Critical', store: 'Uptown Hub', demand: 'High', mean: 60, abc: 'A', color: 'red' },
    { id: 105, name: "Greek Yogurt", sku: "DAI-105", stock: 45, capacity: 80, status: 'Understocked', store: 'Downtown Branch', demand: 'Medium', mean: 25, abc: 'B', color: 'blue' },
    { id: 106, name: "Cherry Tomatoes", sku: "PRO-331", stock: 120, capacity: 150, status: 'Healthy', store: 'Westside Market', demand: 'Low', mean: 15, abc: 'C', color: 'emerald' },
    { id: 107, name: "Ground Beef", sku: "MEA-002", stock: 0, capacity: 50, status: 'Critical', store: 'Westside Market', demand: 'High', mean: 30, abc: 'A', color: 'red' },
];

const MOCK_STATS: StatItem[] = [
    { label: "Total Stock Value", value: "$124,500", sub: "Across all locations", trend: "+12%", trendUp: true },
    { label: "Low Stock Alerts", value: "3 Items", sub: "Below safety stock", trend: "+1", trendUp: false, alert: true },
    { label: "Stock Accuracy", value: "99.8%", sub: "Cycle count verification", trend: "+0.2%", trendUp: true },
    { label: "Turnover Rate", value: "4.2x", sub: "Average monthly", trend: "-0.1%", trendUp: false },
];

export default function StockTerminal() {
    // State
    const [searchQuery, setSearchQuery] = useState("");
    const [statusFilter, setStatusFilter] = useState("All Items");
    const [storeFilter, setStoreFilter] = useState("All Stores");

    // Memoized Filters
    const filteredInventory = useMemo(() => {
        return MOCK_INVENTORY.filter(item => {
            const matchesSearch = item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                item.sku.toLowerCase().includes(searchQuery.toLowerCase()) ||
                item.id.toString().includes(searchQuery);
            const matchesStatus = statusFilter === "All Items" || item.status === statusFilter;
            const matchesStore = storeFilter === "All Stores" || item.store === storeFilter;

            return matchesSearch && matchesStatus && matchesStore;
        });
    }, [searchQuery, statusFilter, storeFilter]);

    // Unique Stores for Filter Dropdown
    const stores = ["All Stores", ...Array.from(new Set(MOCK_INVENTORY.map(item => item.store)))];

    return (
        <div className="space-y-8 animate-in fade-in zoom-in duration-500">
            {/* Header / Title for this section */}
            <div>
                <h2 className="text-3xl font-black text-white tracking-tighter italic uppercase">Stock Operations</h2>
                <p className="text-sm text-slate-400">Manage real-time inventory levels and item intelligence.</p>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
                {MOCK_STATS.map((stat, i) => (
                    <div
                        key={i}
                        className={`group relative overflow-hidden rounded-2xl border p-5 transition-all duration-300 hover:shadow-lg hover:shadow-emerald-900/10 ${stat.alert
                            ? "bg-slate-900/40 border-red-500/20 hover:border-red-500/40"
                            : "bg-slate-900/40 border-slate-800 hover:border-emerald-500/30"
                            }`}
                    >
                        <div className="flex items-start justify-between">
                            <div>
                                <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">{stat.label}</p>
                                <h3 className="mt-2 text-3xl font-bold text-white tracking-tight">{stat.value}</h3>
                                <p className="text-xs text-emerald-500/80 font-medium mt-1">{stat.sub}</p>
                            </div>
                            <div className={`flex items-center gap-1 text-xs font-semibold rounded-full px-2 py-1 ${stat.trendUp ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                                {stat.trend}
                            </div>
                        </div>
                        {/* Decoration */}
                        <div className="absolute -right-6 -top-6 h-24 w-24 rounded-full bg-gradient-to-br from-white/5 to-white/0 blur-2xl transition-all group-hover:from-emerald-500/10" />
                    </div>
                ))}
            </div>

            {/* Filters & Actions */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div className="flex flex-wrap items-center gap-4 w-full md:w-auto">
                    {/* Status Tabs */}
                    <div className="flex items-center bg-slate-900/50 p-1 rounded-xl border border-slate-800 overflow-x-auto max-w-full">
                        {['All Items', 'Critical', 'Understocked', 'Healthy', 'Overstocked'].map((filter) => (
                            <button
                                key={filter}
                                onClick={() => setStatusFilter(filter)}
                                className={`px-4 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${statusFilter === filter ? 'bg-emerald-600 text-white shadow-sm' : 'text-slate-400 hover:text-white hover:bg-slate-800/50'}`}
                            >
                                {filter}
                            </button>
                        ))}
                    </div>

                    {/* Store Filter Dropdown */}
                    <div className="relative">
                        <select
                            value={storeFilter}
                            onChange={(e) => setStoreFilter(e.target.value)}
                            className="appearance-none bg-slate-900/50 border border-slate-800 text-slate-300 text-xs font-medium rounded-xl pl-4 pr-10 py-2 hover:border-slate-700 focus:outline-none focus:border-emerald-500/50 transition-colors cursor-pointer"
                        >
                            {stores.map(store => (
                                <option key={store} value={store}>{store}</option>
                            ))}
                        </select>
                        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-slate-500">
                            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m6 9 6 6 6-6" /></svg>
                        </div>
                    </div>
                </div>

                {/* Search */}
                <div className="relative group w-full md:w-64">
                    <div className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-emerald-400 transition-colors">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>
                    </div>
                    <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search SKU, Item..."
                        className="w-full h-9 rounded-xl bg-slate-900/50 border border-slate-800 pl-9 pr-4 text-xs text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-all"
                    />
                </div>
            </div>

            {/* Data Table */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/30 overflow-hidden backdrop-blur-sm">
                <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                        <thead>
                            <tr className="border-b border-slate-800 bg-slate-900/50 text-xs uppercase tracking-wider text-slate-500">
                                <th className="px-6 py-4 font-semibold">Item Details</th>
                                <th className="px-6 py-4 font-semibold">Store</th>
                                <th className="px-6 py-4 font-semibold">Status</th>
                                <th className="px-6 py-4 font-semibold">Stock Level</th>
                                <th className="px-6 py-4 font-semibold">Demand Class</th>
                                <th className="px-6 py-4 font-semibold text-right">Action</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/50">
                            {filteredInventory.length > 0 ? (
                                filteredInventory.map((item) => (
                                    <tr key={item.id} className="group hover:bg-slate-800/30 transition-colors">
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-3">
                                                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-800 text-xs font-bold text-emerald-500 font-mono">
                                                    {item.abc}
                                                </div>
                                                <div>
                                                    <p className="font-semibold text-slate-200">{item.name}</p>
                                                    <p className="text-xs text-slate-500">{item.sku}</p>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-slate-400">
                                            {item.store}
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className={`inline-flex items-center rounded-md px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${item.color === 'emerald' ? 'bg-emerald-400/10 text-emerald-400 ring-emerald-400/20' :
                                                item.color === 'red' ? 'bg-red-400/10 text-red-400 ring-red-400/20' :
                                                    item.color === 'blue' ? 'bg-blue-400/10 text-blue-400 ring-blue-400/20' :
                                                        'bg-orange-400/10 text-orange-400 ring-orange-400/20'
                                                }`}>
                                                {item.status}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-2">
                                                <span className="text-slate-200 font-medium">{item.stock}</span>
                                                <span className="text-slate-500 text-xs">/ {item.capacity}</span>
                                            </div>
                                            <div className="w-24 h-1.5 bg-slate-800 rounded-full mt-1.5 overflow-hidden">
                                                <div
                                                    className={`h-full rounded-full ${item.color === 'red' ? 'bg-red-500' : item.color === 'blue' ? 'bg-blue-500' : 'bg-emerald-500'}`}
                                                    style={{ width: `${Math.min(100, (item.stock / item.capacity) * 100)}%` }}
                                                />
                                            </div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className={`text-xs font-medium px-2 py-1 rounded ${item.demand === 'High' ? 'bg-indigo-500/10 text-indigo-400' :
                                                item.demand === 'Medium' ? 'bg-slate-700/30 text-slate-400' :
                                                    'bg-slate-800/30 text-slate-500'
                                                }`}>
                                                {item.demand}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            <button className="text-slate-400 hover:text-emerald-400 transition-colors">
                                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="1" /><circle cx="19" cy="12" r="1" /><circle cx="5" cy="12" r="1" /></svg>
                                            </button>
                                        </td>
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td colSpan={6} className="px-6 py-12 text-center text-slate-500">
                                        <div className="flex flex-col items-center justify-center gap-2">
                                            <div className="h-10 w-10 rounded-full bg-slate-800/50 flex items-center justify-center text-slate-600">
                                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>
                                            </div>
                                            <p>No items match your search filters.</p>
                                        </div>
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
                {/* Pagination / Footer */}
                <div className="flex items-center justify-between border-t border-slate-800 bg-slate-900/50 px-6 py-3">
                    <p className="text-xs text-slate-500">Showing {filteredInventory.length} of {MOCK_INVENTORY.length} items</p>
                    <div className="flex gap-2">
                        <button className="px-3 py-1 text-xs text-slate-400 hover:text-white disabled:opacity-50" disabled>Previous</button>
                        <button className="px-3 py-1 text-xs text-slate-400 hover:text-white">Next</button>
                    </div>
                </div>
            </div>
        </div>
    );
}
