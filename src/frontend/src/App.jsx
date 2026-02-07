import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import { DashboardHome } from './pages/DashboardHome';
import { InventoryPage } from './pages/Inventory';
import { MenuPage } from './pages/Menu';
import { PricingPage } from './pages/Pricing';
import { ForecastingPage } from './pages/Forecasting';
import { ItemDetailsPage } from './pages/ItemDetails';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<DashboardHome />} />
          <Route path="inventory" element={<InventoryPage />} />
          <Route path="inventory/item/:id" element={<ItemDetailsPage />} />
          <Route path="menu" element={<MenuPage />} />
          <Route path="pricing" element={<PricingPage />} />
          <Route path="forecasting" element={<ForecastingPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
