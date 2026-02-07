import React from 'react';
import { Outlet } from 'react-router-dom';
import { Header } from './Header';

export const Layout = () => {
    return (
        <div className="min-h-screen flex flex-col bg-eagle-black font-sans text-white">
            <Header />

            <main className="flex-1 w-full p-4 sm:p-6 lg:p-6 overflow-x-hidden">
                <div className="w-full space-y-8 animate-in fade-in slide-in-from-bottom-6 duration-700">
                    <Outlet />
                </div>
            </main>
        </div>
    );
};
