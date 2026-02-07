"""
Inventory Service for EagleEye
Calculates safety stock, reorder points, and provides optimization recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class InventoryService:
    """
    Service for inventory optimization and analysis.
    Provides safety stock calculations, reorder points, and ABC analysis.
    """
    
    def __init__(self, current_inventory: Optional[pd.DataFrame] = None, forecaster=None):
        """
        Initialize with optional current inventory and forecasting model.
        """
        self.current_inventory = current_inventory
        self.forecaster = forecaster

    def analyze_item(self, stats: pd.Series) -> Dict:
        """Calculate inventory recommendations for a single item."""
        # Mean and Std of daily demand
        mean = stats['mean_daily']
        std = stats['std_daily']
        item_id = stats['item_id']
        item_name = stats.get('item_name', f"Item {item_id}")
        
        # 0. Get Forecast (Hybrid Model)
        forecast_demand = None
        model_used = "Statistical (Mean)"
        
        if self.forecaster:
            try:
                # We need item history to predict. In a real app, we'd fetch it here.
                # For now, we assume the forecaster has access to global state or we pass it
                # But since HybridForecaster is stateful with item_stats, we can query it.
                # The HybridForecaster needs stats + history. 
                # Let's trust the mean calculation for ROP base, but add model metadata.
                # Ideally, ROP = (Forecast_Next_Lead_Time) + Safety_Stock
                
                # If we had the history DF here, we'd call: 
                # preds, model_used = self.forecaster.predict_item(item_id, history_df)
                # For this Hackathon MVP, we'll mark which model *would* be used based on the routing logic
                
                if hasattr(self.forecaster, 'item_stats'):
                     istats = self.forecaster.item_stats.get(item_id, {})
                     if istats.get('total_vol', 0) >= 150 and istats.get('history_days', 0) >= 90:
                         model_used = "Hybrid: Local (XGBoost)"
                     else:
                         model_used = "Hybrid: Global (Stacked)"
            except:
                pass

        # 1. Safety Stock (Z * std * sqrt(lead_time))
        # Lead time assumed to be 2 days for fresh flow
        safety_stock = 1.96 * std * np.sqrt(2)
        
        # 2. Reorder Point (Mean * lead_time + safety_stock)
        reorder_point = (mean * 2) + safety_stock
        
        # 3. Simulate current stock and capacity for demo purposes
        # Capacity is usually some multiple of max daily demand
        capacity = np.ceil(mean * 7) + 5
        
        # Synthetic current stock (between 0 and capacity)
        # We use a deterministic random seed based on item_id to keep it stable
        np.random.seed(int(item_id) % 1000)
        current_stock = np.random.uniform(0.1, 1.0) * capacity
        
        # Determine status
        if current_stock < (reorder_point * 0.5):
            status = "CRITICAL"
        elif current_stock < reorder_point:
            status = "UNDERSTOCKED"
        elif current_stock > (capacity * 0.9):
            status = "OVERSTOCKED"
        else:
            status = "HEALTHY"
            
        recommendation = self._get_recommendation(mean, std, current_stock, reorder_point)
        
        # Add ABC class if available in stats
        abc_class = stats.get('abc_class', 'N/A')
        
        return {
            'item_id': item_id,
            'item_name': item_name,
            'mean_daily': mean,
            'std_daily': std,
            'safety_stock': safety_stock,
            'reorder_point': reorder_point,
            'current_stock': current_stock,
            'capacity': capacity,
            'status': status,
            'abc_class': abc_class,
            'recommendation': recommendation,
            'model_type': model_used
        }

    def _get_recommendation(self, mean: float, std: float, current: float, rop: float) -> str:
        """Generate human-readable recommendation."""
        if current < (rop * 0.5):
            return "🔥 CRITICAL: Reorder immediately! Stock level is critically low."
        if current < rop:
            return f"⚠️ Order soon. Current stock below reorder point ({rop:.0f})."
        if current > (mean * 10):
            return "🧊 Overstocked. Pause reorders to reduce waste/holding costs."
        return "✅ Healthy stock level. No action needed."

    def analyze_all_items(self, item_stats: pd.DataFrame) -> pd.DataFrame:
        """Analyze all items in the provided stats dataframe."""
        # Ensure ABC classification is done if not present
        if 'abc_class' not in item_stats.columns:
            item_stats = self.classify_items_abc(item_stats)
            
        results = []
        for _, row in item_stats.iterrows():
            rec = self.analyze_item(row)
            results.append(rec)
        
        return pd.DataFrame(results)

    def classify_items_abc(self, item_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Classify items based on revenue using ABC analysis.
        A: Top 80% of revenue
        B: Next 15% of revenue
        C: Remaining 5% of revenue
        """
        df = item_stats.copy()
        
        # Ensure we have revenue
        if 'revenue' not in df.columns:
            df['revenue'] = df['total_qty'] * df.get('price', 10.0)
            
        # Sort by revenue descending
        df = df.sort_values(by='revenue', ascending=False)
        
        # Cumulative percentage
        df['cum_revenue'] = df['revenue'].cumsum()
        total_revenue = df['revenue'].sum()
        df['cum_perc'] = df['cum_revenue'] / total_revenue
        
        # Assign classes
        def assign_class(perc):
            if perc <= 0.80: return 'A'
            if perc <= 0.95: return 'B'
            return 'C'
            
        df['abc_class'] = df['cum_perc'].apply(assign_class)
        return df

    def get_summary_metrics(self, analysis_df: pd.DataFrame) -> Dict:
        """Generate high-level inventory metrics."""
        return {
            'total_items': len(analysis_df),
            'critical_items': len(analysis_df[analysis_df['status'] == 'CRITICAL']),
            'understocked_items': len(analysis_df[analysis_df['status'] == 'UNDERSTOCKED']),
            'overstocked_items': len(analysis_df[analysis_df['status'] == 'OVERSTOCKED']),
            'healthy_items': len(analysis_df[analysis_df['status'] == 'HEALTHY']),
            'abc_distribution': analysis_df['abc_class'].value_counts().to_dict()
        }

    def get_filtered_inventory(self, inventory_df: pd.DataFrame, status: str = None, search_term: str = None, limit: int = 1000) -> List[Dict]:
        """
        Efficiently filter inventory snapshot and generate recommendations.
        """
        # Work on a view/copy
        df = inventory_df

        # 1. Filter by Status
        if status and status.upper() != 'ALL':
            df = df[df['status'] == status.upper()]

        # 2. Filter by Search
        if search_term:
            term = search_term.lower()
            # Search in item_name or item_id
            # Ensure string types for safety
            mask = (
                df['item_name'].astype(str).str.lower().str.contains(term, na=False) | 
                df['item_id'].astype(str).str.contains(term)
            )
            df = df[mask]

        if df.empty:
            return []

        # 3. Generate Recommendations (Vectorized)
        # Copy to avoid SettingWithCopy warning on the slice
        df = df.copy()
        
        conditions = [
            df['status'] == 'CRITICAL',
            df['status'] == 'UNDERSTOCKED',
            df['status'] == 'OVERSTOCKED'
        ]
        choices = [
            "🔥 CRITICAL: Reorder immediately!",
            "⚠️ Order soon. Below reorder point.",
            "🧊 Overstocked. Pause reorders."
        ]
        df['recommendation'] = np.select(conditions, choices, default="✅ Healthy stock level.")

        # 4. Apply Discount Logic (User Request)
        # If Overstocked AND Discountable -> Recommend Discount
        # Ensure discountable column exists and handle NaNs
        if 'discountable' in df.columns:
            # check for 1 or True
            discount_mask = (df['status'] == 'OVERSTOCKED') & (df['discountable'].fillna(0).astype(int) == 1)
            df.loc[discount_mask, 'recommendation'] = "💸 Overstocked! Recommend: Promote/Discount"

        # 5. Formatting & Calculation
        # Add stock_percentage for frontend
        if 'capacity' in df.columns and 'current_stock' in df.columns:
             # Avoid division by zero
             df['stock_percentage'] = (df['current_stock'] / df['capacity'].replace(0, 1) * 100).round(1)
        
        # Limit results (optimization)
        if limit:
            df = df.head(limit)
            
        return df.to_dict('records')
