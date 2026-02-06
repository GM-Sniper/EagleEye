import React from "react";

export default function DemandProjection() {
    return (
        <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center space-y-4 animate-in fade-in zoom-in duration-500">
            <div className="h-16 w-16 rounded-2xl bg-slate-800/50 flex items-center justify-center text-slate-600">
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="20" x2="12" y2="10" /><line x1="18" y1="20" x2="18" y2="4" /><line x1="6" y1="20" x2="6" y2="16" /></svg>
            </div>
            <div>
                <h3 className="text-xl font-bold text-white">Demand Projection</h3>
                <p className="text-slate-400 max-w-md mx-auto">AI-powered demand forecasting module is currently being configured. Check back later.</p>
            </div>
        </div>
    );
}
