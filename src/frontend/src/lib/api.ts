const API_BASE = 'http://localhost:8002';

export interface ForecastItem {
    date: string;
    predicted: number;
    actual: number | null;
}

export async function getForecast(days: number = 7): Promise<{ forecast: ForecastItem[]; days: number }> {
    const res = await fetch(`${API_BASE}/api/forecast?days=${days}`);
    return res.json();
}

export async function getItems(limit: number = 50): Promise<{ items: { id: number; name: string }[]; total: number }> {
    const res = await fetch(`${API_BASE}/api/items?limit=${limit}`);
    return res.json();
}

export async function getPlaces(limit: number = 50): Promise<{ places: { id: number; name: string }[]; total: number }> {
    const res = await fetch(`${API_BASE}/api/places?limit=${limit}`);
    return res.json();
}

export async function getDashboard(): Promise<{ model_accuracy: number; wmape: number; total_predictions: number }> {
    const res = await fetch(`${API_BASE}/api/dashboard`);
    return res.json();
}

const api = { getForecast, getItems, getPlaces, getDashboard };
export default api;
