"""
EagleEye Inventory Service
Calculates reorder points, safety stock, and inventory recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class StockStatus(Enum):
    """Inventory stock status levels."""
    CRITICAL = "critical"      # Below safety stock
    LOW = "low"               # Below reorder point
    NORMAL = "normal"         # Adequate stock
    OVERSTOCKED = "overstocked"  # Excess inventory


@dataclass
class InventoryRecommendation:
    """Inventory recommendation for an item."""
    item_id: int
    item_name: str
    current_stock: float
    safety_stock: float
    reorder_point: float
    reorder_quantity: float
    status: StockStatus
    days_of_stock: float
    recommendation: str


class InventoryService:
    """
    Inventory optimization service.
    Calculates safety stock, reorder points, and provides recommendations.
    """
    
    def __init__(
        self,
        service_level: float = 0.95,  # 95% service level
        lead_time_days: int = 2,       # Days to receive order
        review_period_days: int = 7    # Order review cycle
    ):
        self.service_level = service_level
        self.lead_time_days = lead_time_days
        self.review_period_days = review_period_days
        
        # Z-score for service level (95% = 1.645)
        self.z_score = self._get_z_score(service_level)
    
    def _get_z_score(self, service_level: float) -> float:
        """Get Z-score for service level."""
        z_scores = {
            0.90: 1.28,
            0.95: 1.645,
            0.99: 2.33
        }
        return z_scores.get(service_level, 1.645)
    
    def calculate_safety_stock(
        self,
        mean_demand: float,
        std_demand: float,
        lead_time_days: Optional[int] = None
    ) -> float:
        """
        Calculate safety stock using the formula:
        SS = Z × σ × √L
        where:
            Z = service level z-score
            σ = standard deviation of demand
            L = lead time in days
        """
        lead_time = lead_time_days or self.lead_time_days
        return self.z_score * std_demand * np.sqrt(lead_time)
    
    def calculate_reorder_point(
        self,
        mean_demand: float,
        safety_stock: float,
        lead_time_days: Optional[int] = None
    ) -> float:
        """
        Calculate reorder point using the formula:
        ROP = (D × L) + SS
        where:
            D = average daily demand
            L = lead time in days
            SS = safety stock
        """
        lead_time = lead_time_days or self.lead_time_days
        return (mean_demand * lead_time) + safety_stock
    
    def calculate_reorder_quantity(
        self,
        mean_demand: float,
        review_period_days: Optional[int] = None
    ) -> float:
        """
        Calculate economic order quantity (simplified).
        Uses review period demand + buffer.
        """
        review_period = review_period_days or self.review_period_days
        return mean_demand * (review_period + self.lead_time_days) * 1.1  # 10% buffer
    
    def analyze_item(
        self,
        item_id: int,
        item_name: str,
        mean_demand: float,
        std_demand: float,
        current_stock: float = 0
    ) -> InventoryRecommendation:
        """Generate inventory recommendation for a single item."""
        
        # Calculate metrics
        safety_stock = self.calculate_safety_stock(mean_demand, std_demand)
        reorder_point = self.calculate_reorder_point(mean_demand, safety_stock)
        reorder_qty = self.calculate_reorder_quantity(mean_demand)
        
        # Calculate days of stock remaining
        days_of_stock = current_stock / mean_demand if mean_demand > 0 else float('inf')
        
        # Determine status
        if current_stock <= safety_stock:
            status = StockStatus.CRITICAL
            recommendation = f"URGENT: Order {reorder_qty:.0f} units immediately"
        elif current_stock <= reorder_point:
            status = StockStatus.LOW
            recommendation = f"Order {reorder_qty:.0f} units soon"
        elif days_of_stock > 30:
            status = StockStatus.OVERSTOCKED
            recommendation = "Consider promotion to reduce excess stock"
        else:
            status = StockStatus.NORMAL
            recommendation = f"Next order in ~{days_of_stock - (self.lead_time_days):.0f} days"
        
        return InventoryRecommendation(
            item_id=item_id,
            item_name=item_name,
            current_stock=current_stock,
            safety_stock=round(safety_stock, 1),
            reorder_point=round(reorder_point, 1),
            reorder_quantity=round(reorder_qty, 1),
            status=status,
            days_of_stock=round(days_of_stock, 1),
            recommendation=recommendation
        )
    
    def analyze_all_items(
        self,
        item_stats: pd.DataFrame,
        current_inventory: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate inventory recommendations for all items.
        
        Args:
            item_stats: DataFrame with item_id, mean_daily, std_daily, item_name
            current_inventory: Optional DataFrame with item_id and current_stock
        """
        results = []
        
        for _, row in item_stats.iterrows():
            item_id = row['item_id']
            item_name = row.get('item_name', f'Item {item_id}')
            mean_demand = row.get('mean_daily', 0)
            std_demand = row.get('std_daily', 0)
            
            # Get current stock if available
            current_stock = 0
            if current_inventory is not None:
                stock_row = current_inventory[current_inventory['item_id'] == item_id]
                if len(stock_row) > 0:
                    current_stock = stock_row['current_stock'].values[0]
            
            if mean_demand > 0:
                rec = self.analyze_item(
                    item_id, item_name, mean_demand, std_demand, current_stock
                )
                results.append({
                    'item_id': rec.item_id,
                    'item_name': rec.item_name,
                    'mean_daily_demand': mean_demand,
                    'demand_variability': row.get('cv', 0),
                    'safety_stock': rec.safety_stock,
                    'reorder_point': rec.reorder_point,
                    'reorder_quantity': rec.reorder_quantity,
                    'current_stock': rec.current_stock,
                    'days_of_stock': rec.days_of_stock,
                    'status': rec.status.value,
                    'recommendation': rec.recommendation
                })
        
        return pd.DataFrame(results)
    
    def get_alerts(
        self,
        recommendations: pd.DataFrame,
        status_filter: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get items requiring attention."""
        if status_filter is None:
            status_filter = ['critical', 'low']
        
        alerts = recommendations[recommendations['status'].isin(status_filter)]
        return alerts.sort_values('days_of_stock')
    
    def classify_items_abc(self, item_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Classify items using ABC analysis.
        A: Top 20% of items (80% of value)
        B: Next 30% of items (15% of value)
        C: Bottom 50% of items (5% of value)
        """
        df = item_stats.copy()
        df['revenue'] = df['total_qty'] * df.get('price', 1)
        df = df.sort_values('revenue', ascending=False)
        
        df['cumulative_revenue'] = df['revenue'].cumsum()
        df['cumulative_pct'] = df['cumulative_revenue'] / df['revenue'].sum()
        
        def assign_class(pct):
            if pct <= 0.80:
                return 'A'
            elif pct <= 0.95:
                return 'B'
            else:
                return 'C'
        
        df['abc_class'] = df['cumulative_pct'].apply(assign_class)
        
        return df[['item_id', 'item_name', 'total_qty', 'revenue', 'abc_class']]


if __name__ == "__main__":
    # Test the inventory service
    print("Testing InventoryService...")
    
    service = InventoryService(
        service_level=0.95,
        lead_time_days=2,
        review_period_days=7
    )
    
    # Test single item
    rec = service.analyze_item(
        item_id=1,
        item_name="Chinabox Lille",
        mean_demand=72,  # From EDA: 72K/1000 days
        std_demand=25,
        current_stock=100
    )
    
    print(f"\nItem: {rec.item_name}")
    print(f"  Safety Stock: {rec.safety_stock}")
    print(f"  Reorder Point: {rec.reorder_point}")
    print(f"  Reorder Qty: {rec.reorder_quantity}")
    print(f"  Status: {rec.status.value}")
    print(f"  Days of Stock: {rec.days_of_stock}")
    print(f"  Recommendation: {rec.recommendation}")
    
    # Test ABC classification
    test_data = pd.DataFrame({
        'item_id': [1, 2, 3, 4, 5],
        'item_name': ['Item A', 'Item B', 'Item C', 'Item D', 'Item E'],
        'total_qty': [1000, 500, 200, 50, 10],
        'price': [100, 50, 30, 20, 10]
    })
    
    abc = service.classify_items_abc(test_data)
    print("\n\nABC Classification:")
    print(abc)
