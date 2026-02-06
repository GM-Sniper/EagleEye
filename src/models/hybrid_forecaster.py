"""
EagleEye Hybrid Forecaster
Orchestrates the selection between Local (XGBoost) and Global (Stacked XGBoost) models
based on item volume and history depth to maximize accuracy across the portfolio.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .production_forecaster import ProductionForecaster
from .global_forecaster import GlobalForecaster

class HybridForecaster:
    """
    Intelligent router for demand forecasting.
    
    Strategy:
    1. HIGH VOLUME Items (> 100 orders, > 6 months history) -> Local Model
        Why: Needs specific tuning for sharp peaks/seasonality unique to this item.
    
    2. LOW VOLUME / CLUTTER Items -> Global Model
       Why: Not enough history for a local model to converge; benefits from transfer learning.
    """
    
    # Thresholds for "High Volume" classification
    # Derived from our analysis where "Lille Box" (High Vol) preferred Local
    # and "Ol Alm" (Low Vol) preferred Global.
    MIN_HISTORY_DAYS = 90
    MIN_TOTAL_volume = 150
    
    def __init__(self):
        self.local_model = ProductionForecaster()
        self.global_model = GlobalForecaster()
        self.item_stats = {} # Cache for routing decisions
        self.is_fitted = False
        
    def fit(self, orders: pd.DataFrame, order_items: pd.DataFrame):
        """
        Train both models.
        - Global Model trains on ALL data.
        - Local Model mainly fitted on aggregate for now, but we'll use instances per item if needed
          (Current ProductionForecaster implementation is designed for single-series, 
           so we might need to instantiate multiple for true local behavior, 
           or use the global model as the default and local for overrides).
        
        For this implementation:
        - Global Model is trained once on everything.
        - Local Models are trained ON-DEMAND or batched for top items.
        """
        print("🧠 HybridForecaster: Training Global Model...")
        # Prepare global stack
        daily_global = order_items.groupby(['date', 'item_id'])['quantity'].sum().reset_index()
        daily_global.columns = ['date', 'item_id', 'order_count']
        self.global_model.fit(daily_global)
        
        # Calculate item stats for routing
        stats = daily_global.groupby('item_id').agg({
            'order_count': ['sum', 'mean', 'std'],
            'date': lambda x: (x.max() - x.min()).days
        })
        # Flatten columns
        stats.columns = ['total_vol', 'mean_daily', 'std_daily', 'history_days']
        stats['cv'] = stats['std_daily'] / stats['mean_daily']
        stats = stats.fillna({'cv': 0})
        
        self.item_stats = stats.to_dict('index')
        
        self.is_fitted = True
        print(f"✅ HybridForecaster Ready. Routing {len(self.item_stats)} items.")
        return self
    
    def predict_item(self, item_id: int, item_history: pd.DataFrame, horizon_days: int = 14) -> Tuple[pd.DataFrame, str]:
        """
        Generate forecast for a specific item using the best model.
        Returns: (Forecast DataFrame, Model Name Used)
        """
        if not self.is_fitted:
            raise ValueError("HybridForecaster not fitted")
            
        stats = self.item_stats.get(item_id, {'total_vol': 0, 'history_days': 0, 'cv': 0})
        
        print(f"DEBUG: Item {item_id} | Vol: {stats['total_vol']} | Hist: {stats['history_days']}d | CV: {stats['cv']:.2f}")
        
        # --- ROUTING LOGIC (SMART) ---
        # 1. Must be "High Volume" to even consider Local
        is_high_volume = (
            stats['total_vol'] >= self.MIN_TOTAL_volume and 
            stats['history_days'] >= self.MIN_HISTORY_DAYS
        )
        
        # 2. Must be "Stable" (Low CV) to trust Local
        # If CV > 0.5, it's too erratic for a single series model; use Global to dampen noise.
        # Benchmark showed 0.62 was too high for Local (55% acc), but Global got 68%.
        is_stable = stats['cv'] < 0.5
        
        use_local = is_high_volume and is_stable
        
        print(f"DEBUG: Routing Decision -> HighVol={is_high_volume}, Stable={is_stable} -> Local={use_local}")
        # -----------------------------
        
        if use_local:
            # Train a dedicated local model just for this high-value item
            # (In production we'd cache these, but for now we fit on fly as it's fast for single series)
            local = ProductionForecaster()
            # Prepare data for local model (needs 'date', 'order_count')
            df = item_history.copy()
            if 'order_count' not in df.columns and 'quantity' in df.columns:
                df = df.rename(columns={'quantity': 'order_count'})
            
            try:
                print(f"DEBUG: Training Local Model on {len(df)} rows...")
                local.fit(df)
                preds = local.predict(horizon_days)
                print(f"DEBUG: Local Preds: {len(preds)} rows")
                return preds, "Local (XGBoost)"
            except Exception as e:
                print(f"⚠️ Local model failed for High-Vol item {item_id}, falling back to Global. Error: {e}")
                # Fallback to Global
                
        # Global Model Path (Refined item-level prediction)
        print(f"DEBUG: Predicting with Global Model for {item_id}...")
        preds = self.global_model.predict(item_id, horizon_days)
        if preds.empty:
            print(f"DEBUG: Global model returned empty. Using Naive.")
            # Cold start fallback (Naive)
            return self._naive_forecast(horizon_days), "Naive (Cold Start)"
            
        return preds, "Global (Stacked)"

    def _naive_forecast(self, days: int) -> pd.DataFrame:
        """Zero forecast for unknown items."""
        dates = [pd.Timestamp.now().date() + pd.Timedelta(days=i) for i in range(days)]
        return pd.DataFrame({
            'date': dates,
            'predicted_demand': [0]*days,
            'lower_bound': [0]*days,
            'upper_bound': [0]*days
        })
