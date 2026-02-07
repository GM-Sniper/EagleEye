import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import {
    Search,
    Settings,
    LayoutDashboard,
    Package,
    ChefHat,
    Tag,
    TrendingUp
} from 'lucide-react';
import { Button } from '../ui/Button';
import { cn } from '../ui/Card';
import EagleLogo from '../../assets/eagleeye.png';

export const Header = () => {
    const navigate = useNavigate();
    const [searchTerm, setSearchTerm] = useState('');

    const links = [
        { name: 'Dashboard', to: '/', icon: LayoutDashboard },
        { name: 'Inventory', to: '/inventory', icon: Package },
        { name: 'Menu', to: '/menu', icon: ChefHat },
        { name: 'Pricing', to: '/pricing', icon: Tag },
        { name: 'Forecasting', to: '/forecasting', icon: TrendingUp },
    ];

    const handleSearch = (e) => {
        if (e.key === 'Enter' && searchTerm.trim()) {
            navigate(`/inventory?search=${encodeURIComponent(searchTerm)}`);
        }
    };

    return (
        <header className="h-20 sticky top-0 z-50 glass-panel border-b border-white/10 px-6 flex items-center justify-between backdrop-blur-md">
            {/* Logo Section */}
            <div className="flex items-center gap-3 min-w-fit">
                <img src={EagleLogo} alt="EagleEye" className="h-10 w-auto" />
                <span className="text-2xl font-bold tracking-tighter text-white">
                    Eagle<span className="text-eagle-green font-black">Eye</span>
                </span>
            </div>

            {/* Main Navigation */}
            <nav className="hidden lg:flex items-center bg-white/5 border border-white/10 rounded-full px-2 py-1.5 gap-1">
                {links.map((link) => (
                    <NavLink
                        key={link.to}
                        to={link.to}
                        className={({ isActive }) => cn(
                            "flex items-center gap-2 px-4 py-2 rounded-full transition-all duration-200 text-sm font-medium",
                            isActive
                                ? "bg-eagle-green text-black shadow-[0_0_15px_rgba(0,224,84,0.3)]"
                                : "text-gray-400 hover:text-white hover:bg-white/5"
                        )}
                    >
                        <link.icon className="w-4 h-4" />
                        {link.name}
                    </NavLink>
                ))}
            </nav>

            {/* Right Actions */}
            <div className="flex items-center gap-6">
                {/* Search */}
                <div className="relative hidden xl:block">
                    <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                    <input
                        type="text"
                        placeholder="Search Intelligence..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        onKeyDown={handleSearch}
                        className="bg-black/40 border border-white/10 rounded-full pl-11 pr-6 py-2.5 text-sm text-white focus:outline-none focus:border-eagle-green/50 w-72 placeholder:text-gray-600 transition-all focus:w-80"
                    />
                </div>

                <div className="flex items-center gap-2">
                    <Button variant="ghost" size="sm" className="rounded-full w-10 h-10 p-0 text-gray-400 hover:text-white" title="Settings">
                        <Settings className="w-5 h-5" />
                    </Button>

                    <div className="h-10 w-10 rounded-full bg-gradient-to-tr from-eagle-green/20 to-emerald-500/20 border border-eagle-green/30 flex items-center justify-center">
                        <span className="text-xs font-black text-eagle-green">AI</span>
                    </div>
                </div>
            </div>
        </header>
    );
};
