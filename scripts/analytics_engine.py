"""
EagleEye Analytics Engine
Main engine that integrates all analytics components for inventory management.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.data_pipeline import DataPipeline
from models.forecaster import DemandForecaster, ItemDemandForecaster
from services.inventory_service import InventoryService


class AnalyticsEngine:
    """
    Main analytics engine for EagleEye inventory management.
    Combines demand forecasting with inventory optimization.
    """
    
    def __init__(self, data_dir: str = "Data"):
        self.data_dir = data_dir
        self.pipeline = DataPipeline(data_dir)
        self.store_forecaster = DemandForecaster()
        self.item_forecaster = ItemDemandForecaster()
        self.inventory_service = InventoryService()
        
        self._item_stats: Optional[pd.DataFrame] = None
        self._daily_demand: Optional[pd.DataFrame] = None
    
    def initialize(self):
        """Load data and prepare for analysis."""
        print("📥 Loading data...")
        self.pipeline.load_all()
        self.pipeline.load_core_tables()
        
        print("📊 Calculating item statistics...")
        self._item_stats = self.pipeline.get_item_stats()
        self._daily_demand = self.pipeline.get_daily_demand()
        
        print(f"✓ Loaded {len(self.pipeline.orders):,} orders")
        print(f"✓ Analyzed {len(self._item_stats):,} items")
        
        return self
    
    def train_store_forecaster(self) -> Dict[str, float]:
        """Train the store-level demand forecaster."""
        print("\n🔮 Training store-level forecaster...")
        
        if self._daily_demand is None:
            self._daily_demand = self.pipeline.get_daily_demand()
        
        self.store_forecaster.fit(self._daily_demand)
        
        # Evaluate
        metrics = self.store_forecaster.evaluate(self._daily_demand)
        print(f"   MAPE: {metrics['mape']:.1f}%")
        print(f"   RMSE: {metrics['rmse']:.1f}")
        
        return metrics
    
    def forecast_store_demand(self, horizon_days: int = 7) -> pd.DataFrame:
        """Generate store-level demand forecast."""
        if not self.store_forecaster.is_fitted:
            self.train_store_forecaster()
        
        forecast = self.store_forecaster.predict(horizon_days)
        return forecast
    
    def get_inventory_recommendations(
        self,
        top_n: int = 50,
        min_daily_demand: float = 1
    ) -> pd.DataFrame:
        """Get inventory recommendations for top items."""
        if self._item_stats is None:
            self._item_stats = self.pipeline.get_item_stats()
        
        # Filter to items with sufficient demand
        stats = self._item_stats[self._item_stats['mean_daily'] >= min_daily_demand].copy()
        stats = stats.nlargest(top_n, 'total_qty')
        
        # Generate recommendations
        recommendations = self.inventory_service.analyze_all_items(stats)
        
        return recommendations
    
    def get_critical_alerts(self) -> pd.DataFrame:
        """Get items that need immediate attention."""
        recommendations = self.get_inventory_recommendations(top_n=100)
        return self.inventory_service.get_alerts(recommendations)
    
    def get_abc_classification(self) -> pd.DataFrame:
        """Get ABC classification of all items."""
        if self._item_stats is None:
            self._item_stats = self.pipeline.get_item_stats()
        
        return self.inventory_service.classify_items_abc(self._item_stats)
    
    def get_demand_by_day_of_week(self) -> pd.DataFrame:
        """Get average demand by day of week."""
        orders = self.pipeline.orders
        
        dow_demand = orders.groupby('day_name').agg({
            'id': 'count',
            'total_amount': 'mean'
        }).reset_index()
        dow_demand.columns = ['day', 'avg_orders', 'avg_order_value']
        
        # Sort by day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_demand['day'] = pd.Categorical(dow_demand['day'], categories=day_order, ordered=True)
        dow_demand = dow_demand.sort_values('day')
        
        return dow_demand
    
    def get_peak_hours(self, top_n: int = 5) -> pd.DataFrame:
        """Get peak ordering hours."""
        orders = self.pipeline.orders
        
        hourly = orders.groupby('hour').agg({
            'id': 'count',
            'total_amount': 'sum'
        }).reset_index()
        hourly.columns = ['hour', 'order_count', 'revenue']
        
        return hourly.nlargest(top_n, 'order_count')
    
    def generate_dashboard_data(self) -> Dict:
        """Generate all data needed for the dashboard."""
        print("\n📊 Generating dashboard data...")
        
        # Basic stats
        orders = self.pipeline.orders
        
        data = {
            "summary": {
                "total_orders": int(len(orders)),
                "total_revenue": float(orders['total_amount'].sum()),
                "avg_order_value": float(orders['total_amount'].mean()),
                "unique_items": int(self._item_stats['item_id'].nunique()) if self._item_stats is not None else 0,
                "last_updated": datetime.now().isoformat()
            },
            "temporal": {
                "busiest_day": "Friday",
                "peak_hour": 16,
                "weekday_pct": float((~orders['is_weekend']).sum() / len(orders) * 100)
            },
            "demand_by_day": self.get_demand_by_day_of_week().to_dict(orient='records'),
            "peak_hours": self.get_peak_hours().to_dict(orient='records')
        }
        
        # Forecast
        try:
            forecast = self.forecast_store_demand(7)
            data["forecast"] = forecast.to_dict(orient='records')
        except Exception as e:
            data["forecast"] = []
            print(f"   ⚠️ Forecast error: {e}")
        
        # Top items
        if self._item_stats is not None:
            top_items = self._item_stats.nlargest(10, 'total_qty')[
                ['item_id', 'item_name', 'mean_daily', 'cv', 'total_qty']
            ]
            data["top_items"] = top_items.to_dict(orient='records')
        
        # ABC classification
        try:
            abc = self.get_abc_classification()
            abc_summary = abc.groupby('abc_class').agg({
                'item_id': 'count',
                'revenue': 'sum'
            }).reset_index()
            abc_summary.columns = ['class', 'item_count', 'revenue']
            data["abc_classification"] = abc_summary.to_dict(orient='records')
        except Exception as e:
            data["abc_classification"] = []
        
        return data
    
    def export_dashboard_data(self, output_path: str = "dashboard_data.json"):
        """Export dashboard data to JSON file."""
        data = self.generate_dashboard_data()
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\n💾 Dashboard data exported to {output_path}")
        return data


def main():
    """Run the analytics engine."""
    print("=" * 70)
    print("🦅 EAGLEEYE ANALYTICS ENGINE")
    print("=" * 70)
    
    engine = AnalyticsEngine(data_dir="../Data")
    engine.initialize()
    
    # Train forecaster
    metrics = engine.train_store_forecaster()
    
    # Generate 7-day forecast
    print("\n📈 7-Day Demand Forecast:")
    forecast = engine.forecast_store_demand(7)
    for _, row in forecast.iterrows():
        print(f"   {row['date'].strftime('%Y-%m-%d')}: {int(row['predicted_demand']):,} orders")
    
    # Get inventory recommendations
    print("\n📦 Top 10 Inventory Recommendations:")
    recommendations = engine.get_inventory_recommendations(top_n=10)
    for _, row in recommendations.iterrows():
        print(f"   {row['item_name'][:30]:30s} | ROP: {row['reorder_point']:>6.0f} | SS: {row['safety_stock']:>6.0f}")
    
    # ABC Analysis
    print("\n📊 ABC Classification Summary:")
    abc = engine.get_abc_classification()
    for cls in ['A', 'B', 'C']:
        class_items = abc[abc['abc_class'] == cls]
        print(f"   Class {cls}: {len(class_items):,} items ({len(class_items)/len(abc)*100:.1f}%)")
    
    # Export dashboard data
    engine.export_dashboard_data("dashboard_data.json")
    
    print("\n" + "=" * 70)
    print("✅ ANALYTICS ENGINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
