"use client";

import React, { useState } from "react";
import CommandCenter from "../components/CommandCenter";
import StockTerminal from "../components/StockTerminal";
import DemandProjection from "../components/DemandProjection";
import RevenueSegments from "../components/RevenueSegments";

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
      {isActive && <div className="ml-auto h-1.5 w-1.5 rounded-full bg-white animate-pulse" />}
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
      case "Revenue Segments":
        return <RevenueSegments />;
      default:
        return <CommandCenter />;
    }
  };

  return (
    <div className="flex h-screen w-full bg-background text-foreground overflow-hidden">
      {/* Sidebar */}
      <aside className="w-72 flex-shrink-0 border-r border-border/40 bg-card/50 backdrop-blur-xl p-6 flex flex-col gap-8">
        {/* Logo */}
        <div className="flex items-center gap-3 px-2">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-400 to-emerald-600 shadow-lg shadow-emerald-900/50">
            <svg
              className="text-white"
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white">
              EAGLE<span className="text-emerald-400">EYE</span>
            </h1>
            <p className="text-[10px] font-medium text-emerald-500/80 tracking-widest uppercase">
              Intelligence v1.2
            </p>
          </div>
        </div>

        {/* Nav Items */}
        <nav className="flex-1 space-y-2">
          <SidebarItem
            icon={ActivityIcon}
            label="Command Center"
            isActive={activeTab === "Command Center"}
            onClick={() => setActiveTab("Command Center")}
          />
          <SidebarItem
            icon={BoxIcon}
            label="Stock Terminal"
            isActive={activeTab === "Stock Terminal"}
            onClick={() => setActiveTab("Stock Terminal")}
          />
          <SidebarItem
            icon={BarChartIcon}
            label="Demand Projection"
            isActive={activeTab === "Demand Projection"}
            onClick={() => setActiveTab("Demand Projection")}
          />
          <SidebarItem
            icon={PieChartIcon}
            label="Revenue Segments"
            isActive={activeTab === "Revenue Segments"}
            onClick={() => setActiveTab("Revenue Segments")}
          />
        </nav>

        {/* Bottom Actions */}
        <div className="mt-auto">
          <div className="rounded-2xl bg-slate-900/50 border border-slate-800 p-4">
            <div className="flex items-center gap-3 mb-3">
              <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></div>
              <span className="text-xs font-medium text-emerald-400">System Online</span>
            </div>
            <button className="w-full rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-300 py-2.5 transition-colors border border-slate-700/50">
              Refresh Core
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content Wrapper */}
      <div className="flex-1 flex flex-col min-w-0 bg-background relative">
        {/* Background Gradients */}
        <div className="absolute top-0 left-0 w-full h-96 bg-gradient-to-b from-slate-900 to-transparent opacity-50 pointer-events-none" />
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />

        {/* Top Header */}
        <header className="sticky top-0 z-10 flex h-20 items-center justify-between px-8 py-4 backdrop-blur-md bg-background/0">
          <div>
            <h2 className="text-2xl font-bold text-white tracking-tight italic uppercase">
              {activeTab} <span className="text-slate-500 not-italic font-light">/</span> VIEW
            </h2>
            <p className="text-sm text-slate-400">
              Deep Learning powered inventory optimization system
            </p>
          </div>

          <div className="flex items-center gap-4">
            <div className="relative group">
              <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500 group-focus-within:text-emerald-400 transition-colors" />
              <input
                type="text"
                placeholder="Query Item ID or Name..."
                className="h-10 w-80 rounded-full bg-slate-900/50 border border-slate-800 pl-10 pr-4 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-all"
              />
            </div>

            <button className="h-10 w-10 rounded-full bg-slate-900/50 border border-slate-800 flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-800 transition-all">
              <BellIcon className="h-4 w-4" />
            </button>
            <button className="h-10 w-10 rounded-full bg-slate-900/50 border border-slate-800 flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-800 transition-all">
              <HelpCircleIcon className="h-4 w-4" />
            </button>
          </div>
        </header>

        {/* Dashboard Content Scroll Area */}
        <main className="flex-1 overflow-auto p-8 pt-4">
          {renderContent()}
        </main>
      </div>
    </div>
  );
}