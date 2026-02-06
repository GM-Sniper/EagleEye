import React from "react";

export default function RevenueSegments() {
    return (
        <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center space-y-4 animate-in fade-in zoom-in duration-500">
            <div className="h-16 w-16 rounded-2xl bg-slate-800/50 flex items-center justify-center text-slate-600">
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21.21 15.89A10 10 0 1 1 8 2.83" /><path d="M22 12A10 10 0 0 0 12 2v10z" /></svg>
            </div>
            <div>
                <h3 className="text-xl font-bold text-white">Revenue Segments</h3>
                <p className="text-slate-400 max-w-md mx-auto">Revenue analysis and segmentation tools are initializing.</p>
            </div>
        </div>
    );
}
