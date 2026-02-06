"use client";

import React, { useState } from "react";
import CommandCenter from "../components/CommandCenter";
import StockTerminal from "../components/StockTerminal";
import DemandProjection from "../components/DemandProjection";

// --- Icons ---
function ActivityIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
    </svg>
  );
}

function BoxIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
      <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
      <line x1="12" y1="22.08" x2="12" y2="12" />
    </svg>
  );
}

function BarChartIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="12" y1="20" x2="12" y2="10" />
      <line x1="18" y1="20" x2="18" y2="4" />
      <line x1="6" y1="20" x2="6" y2="16" />
    </svg>
  );
}

function PieChartIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21.21 15.89A10 10 0 1 1 8 2.83" />
      <path d="M22 12A10 10 0 0 0 12 2v10z" />
    </svg>
  );
}

function SearchIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function BellIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
      <path d="M13.73 21a2 2 0 0 1-3.46 0" />
    </svg>
  );
}

function HelpCircleIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

// --- Components ---

function SidebarItem({
  icon: Icon,
  label,
  isActive,
  onClick,
}: {
  icon: React.ElementType;
  label: string;
  isActive?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`group flex w-full items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-all duration-200 ${isActive
        ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white shadow-lg shadow-emerald-900/20"
        : "text-slate-400 hover:bg-slate-800/50 hover:text-emerald-400"
        }`}
    >
      <Icon className={`h-5 w-5 ${isActive ? "text-white" : "text-slate-500 group-hover:text-emerald-400"}`} />
      <span>{label}</span>
      {isActive && <span className="ml-auto block h-1.5 w-1.5 rounded-full bg-white animate-pulse" />}
    </button>
  );
}

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState("Command Center");

  const renderContent = () => {
    switch (activeTab) {
      case "Command Center":
        return <CommandCenter />;
      case "Stock Terminal":
        return <StockTerminal />;
      case "Demand Projection":
        return <DemandProjection />;
      default:
        return <CommandCenter />;
    }
  };

  return (
    <div className="flex h-screen w-full bg-slate-950 text-foreground overflow-hidden flex-col">
      {/* Background */}
      <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 pointer-events-none -z-10" />

      {/* Top Navigation Bar */}
      <nav className="h-16 border-b border-white/5 bg-slate-900/80 backdrop-blur-xl flex items-center justify-between px-6 z-50">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-500">
            <img
              src="/Eagle_Eye.png"
              alt="Eagle Eye Logo"
              className="h-full w-full object-cover"
            />
          </div>
          <span className="text-lg font-bold text-white tracking-tight">Eagle<span className="text-emerald-500">Eye</span></span>
        </div>

        {/* Central Tabs */}
        <div className="hidden md:flex items-center gap-1 bg-slate-800/50 p-1 rounded-lg border border-white/5">
          <button
            onClick={() => setActiveTab("Command Center")}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${activeTab === "Command Center" ? "bg-emerald-500 text-white shadow-sm" : "text-slate-400 hover:text-white hover:bg-white/5"}`}
          >
            Overview
          </button>
          <button
            onClick={() => setActiveTab("Stock Terminal")}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${activeTab === "Stock Terminal" ? "bg-emerald-500 text-white shadow-sm" : "text-slate-400 hover:text-white hover:bg-white/5"}`}
          >
            Stock
          </button>
          <button
            onClick={() => setActiveTab("Demand Projection")}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${activeTab === "Demand Projection" ? "bg-emerald-500 text-white shadow-sm" : "text-slate-400 hover:text-white hover:bg-white/5"}`}
          >
            Demand
          </button>
        </div>

        {/* Right Actions */}
        <div className="flex items-center gap-4">
          <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/5 border border-emerald-500/20">
            <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
            <span className="text-xs font-medium text-emerald-400">System Online</span>
          </div>
          <button className="h-9 w-9 rounded-full bg-slate-800/50 border border-white/5 flex items-center justify-center text-slate-400 hover:text-white transition-colors">
            <BellIcon className="h-4 w-4" />
          </button>
          <div className="h-8 w-8 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-xs font-bold text-emerald-500">
            AD
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 overflow-auto p-6 lg:p-10 relative">
        <div className="max-w-7xl mx-auto space-y-6">


          <div key={activeTab} className="animate-slide-up-fade">
            {renderContent()}
          </div>
        </div>
      </main>
    </div>
  );
}