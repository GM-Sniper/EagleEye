import React from 'react';
import { Card } from '../components/ui/Card';

const PlaceholderPage = ({ title }) => (
    <div className="space-y-6">
        <Card className="min-h-[400px] flex items-center justify-center border-dashed border-2 border-white/10 bg-transparent">
            <div className="text-center space-y-2">
                <h2 className="text-2xl font-bold text-white">{title}</h2>
                <p className="text-gray-400">This module is under construction.</p>
            </div>
        </Card>
    </div>
);

// export const DashboardHome = () => <PlaceholderPage title="Dashboard Overview" />; 
// Commented out to use real component
// export const Inventory = () => <PlaceholderPage title="Inventory & Stock Management" />;
// Commented out to use real component
// export const MenuAnalytics = () => <PlaceholderPage title="Menu Analytics & Engineering" />;
// Commented out to use real component
// export const Pricing = () => <PlaceholderPage title="Smart Pricing & Discounts" />;
// Commented out to use real component
// export const Pricing = () => <PlaceholderPage title="Smart Pricing & Discounts" />;
// Commented out to use real component
// export const Forecasting = () => <PlaceholderPage title="Advanced Forecasting" />;
// Commented out to use real component
