"""EagleEye Data Pipeline
Handles data loading, cleaning, and transformation for inventory analytics.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class DataPipeline:
    """Centralized data loading and transformation pipeline."""
    
    def __init__(self, data_dir: str = "Data"):
        self.data_dir = Path(data_dir)
        self.datasets: Dict[str, pd.DataFrame] = {}
        self._orders: Optional[pd.DataFrame] = None
        self._order_items: Optional[pd.DataFrame] = None
        self._items: Optional[pd.DataFrame] = None

        self._places: Optional[pd.DataFrame] = None
        self._inventory_snapshot: Optional[pd.DataFrame] = None
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from data directory."""
        for csv_file in self.data_dir.glob("*.csv"):
            name = csv_file.stem
            self.datasets[name] = pd.read_csv(csv_file, low_memory=False)
        return self.datasets
    
    def load_core_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and transform core tables: orders, order_items, items."""
        if not self.datasets:
            self.load_all()
        
        # Process orders
        self._orders = self._transform_orders(self.datasets['fct_orders'].copy())
        self._order_items = self.datasets['fct_order_items'].copy()
        self._items = self.datasets['dim_items'].copy()
        self._places = self.datasets['dim_places'].copy()
        
        return self._orders, self._order_items, self._items
    
    def _transform_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Transform orders with datetime features."""
        # Convert Unix timestamp to datetime
        orders['created_dt'] = pd.to_datetime(orders['created'], unit='s')
        orders['date'] = orders['created_dt'].dt.date
        orders['year'] = orders['created_dt'].dt.year
        orders['month'] = orders['created_dt'].dt.month
        orders['week'] = orders['created_dt'].dt.isocalendar().week
        orders['day_of_week'] = orders['created_dt'].dt.dayofweek
        orders['day_name'] = orders['created_dt'].dt.day_name()
        orders['hour'] = orders['created_dt'].dt.hour
        orders['is_weekend'] = orders['day_of_week'].isin([5, 6])
        
        # Clean numeric columns
        orders['total_amount'] = pd.to_numeric(orders['total_amount'], errors='coerce').fillna(0)
        
        return orders
    
    def get_daily_demand(self, place_id: Optional[int] = None) -> pd.DataFrame:
        """Get daily order counts, optionally filtered by place."""
        if self._orders is None:
            self.load_core_tables()
        
        orders = self._orders
        if place_id is not None:
            orders = orders[orders['place_id'] == place_id]
        
        daily = orders.groupby('date').agg({
            'id': 'count',
            'total_amount': 'sum'
        }).reset_index()
        daily.columns = ['date', 'order_count', 'revenue']
        daily['date'] = pd.to_datetime(daily['date'])
        
        return daily
    
    def get_item_daily_demand(self, item_id: Optional[int] = None) -> pd.DataFrame:
        """Get daily demand per item."""
        if self._order_items is None:
            self.load_core_tables()
        
        oi = self._order_items.copy()
        oi['date'] = pd.to_datetime(oi['created'], unit='s').dt.date
        
        if item_id is not None:
            oi = oi[oi['item_id'] == item_id]
        
        daily = oi.groupby(['date', 'item_id'])['quantity'].sum().reset_index()
        daily['date'] = pd.to_datetime(daily['date'])
        
        return daily
    
    def get_item_stats(self) -> pd.DataFrame:
        """Calculate demand statistics per item for inventory planning."""
        if self._order_items is None:
            self.load_core_tables()
        
        oi = self._order_items.copy()
        oi['date'] = pd.to_datetime(oi['created'], unit='s').dt.date
        
        # Daily demand per item
        daily_demand = oi.groupby(['item_id', 'date'])['quantity'].sum().reset_index()
        
        # Calculate stats
        stats = daily_demand.groupby('item_id')['quantity'].agg([
            ('mean_daily', 'mean'),
            ('std_daily', 'std'),
            ('max_daily', 'max'),
            ('min_daily', 'min'),
            ('total_qty', 'sum'),
            ('days_with_sales', 'count')
        ]).reset_index()
        
        # Coefficient of variation (demand variability)
        stats['cv'] = stats['std_daily'] / stats['mean_daily']
        stats['cv'] = stats['cv'].fillna(0)
        
        # Merge with item details (Left join to include items with zero sales)
        items = self._items[['id', 'title', 'price', 'discountable']].copy()
        items.columns = ['item_id', 'item_name', 'price', 'discountable']
        stats = items.merge(stats, on='item_id', how='left')
        
        # Fill zero sales items
        stats['mean_daily'] = stats['mean_daily'].fillna(0)
        stats['std_daily'] = stats['std_daily'].fillna(0)
        stats['total_qty'] = stats['total_qty'].fillna(0)
        stats['days_with_sales'] = stats['days_with_sales'].fillna(0)
        stats['cv'] = stats['cv'].fillna(0)
        
        return stats

    def get_inventory_snapshot(self) -> pd.DataFrame:
        """
        Generate or retrieve a cached snapshot of inventory with simulated stock levels.
        This ensures consistency across requests and enables fast filtering.
        """
        if self._inventory_snapshot is not None:
            return self._inventory_snapshot

        # 1. Get base stats
        stats = self.get_item_stats().copy()

        # 2. Vectorized simulation of stock levels
        # Lead time = 2 days, Safety Stock = 1.96 * std * sqrt(2)
        stats['safety_stock'] = 1.96 * stats['std_daily'] * np.sqrt(2)
        stats['reorder_point'] = (stats['mean_daily'] * 2) + stats['safety_stock']
        stats['capacity'] = np.ceil(stats['mean_daily'] * 7) + 5

        # Generate current stock (vectorized)
        # Use a stable random seed based on item_id to ensure reproducibility if re-run
        # But here we run it once and cache, so simple random is fine. 
        # However, to be safe against reload, we can seed.
        np.random.seed(42) 
        # We need a different random value for each item
        stats['current_stock'] = np.random.uniform(0.1, 1.0, size=len(stats)) * stats['capacity']
        
        # Determine status (vectorized)
        conditions = [
            stats['current_stock'] < (stats['reorder_point'] * 0.5),
            stats['current_stock'] < stats['reorder_point'],
            stats['current_stock'] > (stats['capacity'] * 0.9)
        ]
        choices = ['CRITICAL', 'UNDERSTOCKED', 'OVERSTOCKED']
        stats['status'] = np.select(conditions, choices, default='HEALTHY')



        # ABC Classification (Vectorized)
        # Revenue = Total Qty * Price
        # Handle missing prices/qty
        if 'total_qty' in stats.columns and 'price' in stats.columns:
             stats['revenue'] = stats['total_qty'].fillna(0) * stats['price'].fillna(0)
             # Sort for cumulative sum
             stats = stats.sort_values('revenue', ascending=False)
             stats['cum_revenue'] = stats['revenue'].cumsum()
             total_rev = stats['revenue'].sum()
             
             if total_rev > 0:
                 stats['cum_perc'] = stats['cum_revenue'] / total_rev
                 abc_conditions = [
                     stats['cum_perc'] <= 0.80,
                     stats['cum_perc'] <= 0.95
                 ]
                 stats['abc_class'] = np.select(abc_conditions, ['A', 'B'], default='C')
             else:
                 stats['abc_class'] = 'C'
        else:
            stats['abc_class'] = 'N/A'

        # Cache the result
        self._inventory_snapshot = stats
        return self._inventory_snapshot
    
    @property
    def orders(self) -> pd.DataFrame:
        if self._orders is None:
            self.load_core_tables()
        return self._orders
    
    @property
    def order_items(self) -> pd.DataFrame:
        if self._order_items is None:
            self.load_core_tables()
        return self._order_items
    
    @property
    def items(self) -> pd.DataFrame:
        if self._items is None:
            self.load_core_tables()
        return self._items
    
    @property
    def places(self) -> pd.DataFrame:
        if self._places is None:
            self.load_core_tables()
        return self._places


if __name__ == "__main__":
    # Test the pipeline
    pipeline = DataPipeline(data_dir="../Data")
    pipeline.load_all()
    
    print("Loaded datasets:", list(pipeline.datasets.keys()))
    
    orders, items, _ = pipeline.load_core_tables()
    print(f"\nOrders: {len(orders):,} rows")
    
    daily = pipeline.get_daily_demand()
    print(f"Daily demand: {len(daily):,} days")
    
    stats = pipeline.get_item_stats()
    print(f"Item stats: {len(stats):,} items")
    print("\nTop 5 items by volume:")
    print(stats.nlargest(5, 'total_qty')[['item_name', 'mean_daily', 'cv', 'total_qty']])
