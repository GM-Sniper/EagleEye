import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const DashboardService = {
    getDashboard: async () => {
        const response = await api.get('/dashboard');
        return response.data;
    },
};

export const InventoryService = {
    getInventory: async (params) => {
        // Map frontend search_term to backend search
        const apiParams = {
            ...params,
            search: params.search_term
        };
        delete apiParams.search_term;

        const response = await api.get('/inventory/recommendations', { params: apiParams });
        return response.data.items; // Return .items as backend wraps it in {items: [...]}
    },
    getIngredients: async () => {
        const response = await api.get('/ingredients/stock');
        return response.data;
    },
    getAlerts: async (days = 7) => {
        const response = await api.get('/ingredients/alerts', { params: { forecast_days: days } });
        return response.data;
    },
    getItemDetails: async (id) => {
        const response = await api.get(`/inventory/item/${id}`);
        return response.data;
    }
};

export const MenuService = {
    getMatrix: async () => {
        const response = await api.get('/analytics/menu-matrix');
        return response.data;
    },
    getTopItems: async (metric = 'revenue', limit = 10) => {
        const response = await api.get('/analytics/top-items', { params: { metric, limit } });
        return response.data;
    },
    getRecommendations: async () => {
        const response = await api.get('/analytics/menu-recommendations');
        return response.data;
    }
};

export const PricingService = {
    getDiscounts: async () => {
        const response = await api.get('/pricing/discounts');
        return response.data;
    },
    getOptimization: async () => {
        const response = await api.get('/pricing/optimization');
        return response.data;
    }
};

export const ForecastService = {
    getEvents: async (days = 14) => {
        const response = await api.get('/forecast/events', { params: { days_ahead: days } });
        return response.data;
    },
    getBacktest: async (days = 14) => {
        const response = await api.get('/analytics/backtest', { params: { window_days: days } });
        return response.data;
    }
};
