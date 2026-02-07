"""
Pricing Service for EagleEye
Automated discount recommendations and pricing optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class PricingService:
    """
    Service for automated discount and pricing recommendations.
    Analyzes inventory levels and item profitability to suggest pricing actions.
    """
    
    def __init__(self, inventory_snapshot: pd.DataFrame):
        """
        Initialize with inventory snapshot.
        
        Args:
            inventory_snapshot: DataFrame from DataPipeline.get_inventory_snapshot()
        """
        self.inventory = inventory_snapshot.copy()
    
    def get_discount_recommendations(
        self, 
        max_discount_pct: int = 30,
        min_excess_ratio: float = 0.9
    ) -> List[Dict]:
        """
        Get automatic discount suggestions for overstocked items.
        
        Only recommends discounts for items where:
        - Status is 'OVERSTOCKED'
        - discountable flag is True
        
        Args:
            max_discount_pct: Maximum discount percentage to recommend
            min_excess_ratio: Minimum stock/capacity ratio to consider overstocked
            
        Returns:
            List of discount recommendations
        """
        # Filter for overstocked, discountable items
        df = self.inventory.copy()
        
        # Ensure discountable column exists and handle NaN/None
        if 'discountable' not in df.columns:
            df['discountable'] = True  # Default to discountable if not specified
        
        df['discountable'] = df['discountable'].fillna(True).astype(bool)
        
        overstocked = df[
            (df['status'] == 'OVERSTOCKED') & 
            (df['discountable'] == True)
        ]
        
        recommendations = []
        
        for _, row in overstocked.iterrows():
            # Calculate excess ratio (how much over capacity)
            capacity = row.get('capacity', row.get('mean_daily', 1) * 7)
            current = row.get('current_stock', 0)
            excess_ratio = current / max(capacity, 1)
            
            # Skip if not actually overstocked
            if excess_ratio < min_excess_ratio:
                continue
            
            # Calculate suggested discount (10-30% based on severity)
            # More excess = higher discount
            excess_severity = min((excess_ratio - min_excess_ratio) / 0.3, 1.0)  # 0-1 scale
            suggested_discount = int(10 + (excess_severity * (max_discount_pct - 10)))
            
            # Calculate potential revenue impact
            price = row.get('price', 0) or 0
            daily_demand = row.get('mean_daily', 0) or 0
            
            # Estimate demand increase from discount (simple elasticity model)
            demand_multiplier = 1 + (suggested_discount / 100) * 1.5  # 1.5x elasticity
            projected_daily_revenue_before = price * daily_demand
            projected_daily_revenue_after = (price * (1 - suggested_discount/100)) * (daily_demand * demand_multiplier)
            
            # Days to clear excess stock
            excess_units = max(0, current - (capacity * 0.7))
            days_to_clear = excess_units / max(daily_demand * demand_multiplier, 0.1)
            
            recommendations.append({
                'item_id': int(row.get('item_id', 0)),
                'item_name': str(row.get('item_name', 'Unknown'))[:50],
                'current_stock': round(float(current), 1),
                'capacity': round(float(capacity), 1),
                'excess_ratio': round(float(excess_ratio), 2),
                'current_price': round(float(price), 2),
                'suggested_discount_pct': suggested_discount,
                'discounted_price': round(float(price * (1 - suggested_discount/100)), 2),
                'estimated_days_to_clear': round(float(days_to_clear), 1),
                'abc_class': str(row.get('abc_class', 'N/A')),
                'reason': 'Overstock clearance - reduce waste and free up storage',
                'priority': 'HIGH' if excess_ratio > 1.1 else 'MEDIUM'
            })
        
        # Sort by excess ratio (most overstocked first)
        recommendations.sort(key=lambda x: x['excess_ratio'], reverse=True)
        
        return recommendations
    
    def get_pricing_optimization(self) -> List[Dict]:
        """
        Recommend pricing changes to maximize profitability.
        
        Analyzes:
        - High-demand + low-margin items → raise price
        - Low-demand + high-margin items → discount to boost volume
        - Items with high price elasticity → experiment with pricing
        
        Returns:
            List of pricing recommendations
        """
        df = self.inventory.copy()
        
        # Calculate relative metrics
        df['demand_percentile'] = df['mean_daily'].rank(pct=True)
        df['price_percentile'] = df['price'].rank(pct=True)
        
        recommendations = []
        
        for _, row in df.iterrows():
            demand_pct = row.get('demand_percentile', 0.5)
            price_pct = row.get('price_percentile', 0.5)
            abc_class = row.get('abc_class', 'C')
            
            action = None
            reason = None
            suggested_change = 0
            
            # High demand, low price → opportunity to raise price
            if demand_pct > 0.7 and price_pct < 0.3:
                action = 'RAISE_PRICE'
                reason = 'High demand item with below-average pricing'
                suggested_change = 10  # +10%
                priority = 'HIGH' if abc_class == 'A' else 'MEDIUM'
            
            # Low demand, high price → consider discounting
            elif demand_pct < 0.3 and price_pct > 0.7:
                action = 'CONSIDER_DISCOUNT'
                reason = 'Low demand may be due to high price point'
                suggested_change = -15  # -15%
                priority = 'MEDIUM'
            
            # Very low demand across the board → may need promotion or removal
            elif demand_pct < 0.1 and abc_class == 'C':
                action = 'REVIEW_ITEM'
                reason = 'Very low demand - consider promotion or menu removal'
                suggested_change = 0
                priority = 'LOW'
            
            if action:
                recommendations.append({
                    'item_id': int(row.get('item_id', 0)),
                    'item_name': str(row.get('item_name', 'Unknown'))[:50],
                    'current_price': round(float(row.get('price', 0)), 2),
                    'mean_daily_demand': round(float(row.get('mean_daily', 0)), 2),
                    'abc_class': abc_class,
                    'action': action,
                    'suggested_price_change_pct': suggested_change,
                    'new_price': round(float(row.get('price', 0)) * (1 + suggested_change/100), 2),
                    'reason': reason,
                    'priority': priority
                })
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 99))
        
        return recommendations
    
    def get_promotable_items(self, limit: int = 10) -> List[Dict]:
        """
        Get items that would benefit most from promotion.
        
        Identifies items with:
        - Good margins (high price relative to demand)
        - Room for demand growth
        - Discountable flag enabled
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of items suitable for promotion
        """
        df = self.inventory.copy()
        
        # Score items for promotion potential
        df['promotion_score'] = (
            df['price'].rank(pct=True) * 0.3 +  # Higher price = more room for discount
            (1 - df['mean_daily'].rank(pct=True)) * 0.4 +  # Lower demand = more growth potential
            (df['discountable'].fillna(True).astype(int)) * 0.3  # Must be discountable
        )
        
        # Get top promotion candidates
        top_items = df.nlargest(limit, 'promotion_score')
        
        return [
            {
                'item_id': int(row['item_id']),
                'item_name': str(row.get('item_name', 'Unknown'))[:50],
                'current_price': round(float(row.get('price', 0)), 2),
                'mean_daily_demand': round(float(row.get('mean_daily', 0)), 2),
                'promotion_score': round(float(row['promotion_score']), 3),
                'suggested_promotion': f"{int(15 + row['promotion_score'] * 10)}% off for 1 week",
                'expected_demand_increase': f"{int(30 + row['promotion_score'] * 40)}%"
            }
            for _, row in top_items.iterrows()
        ]
    
    def get_summary(self) -> Dict:
        """Get summary of pricing opportunities."""
        discounts = self.get_discount_recommendations()
        optimizations = self.get_pricing_optimization()
        
        return {
            'total_discount_opportunities': len(discounts),
            'high_priority_discounts': len([d for d in discounts if d['priority'] == 'HIGH']),
            'total_pricing_optimizations': len(optimizations),
            'raise_price_opportunities': len([o for o in optimizations if o['action'] == 'RAISE_PRICE']),
            'items_to_review': len([o for o in optimizations if o['action'] == 'REVIEW_ITEM']),
            'generated_at': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Test the service
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from services.data_pipeline import DataPipeline
    
    pipeline = DataPipeline(data_dir="../../Data")
    pipeline.load_all()
    pipeline.load_core_tables()
    
    snapshot = pipeline.get_inventory_snapshot()
    service = PricingService(snapshot)
    
    print("=== Pricing Service Test ===\n")
    
    print("--- Summary ---")
    print(service.get_summary())
    
    print("\n--- Discount Recommendations (Top 5) ---")
    for rec in service.get_discount_recommendations()[:5]:
        print(f"[{rec['priority']}] {rec['item_name']}: {rec['suggested_discount_pct']}% off")
    
    print("\n--- Pricing Optimizations (Top 5) ---")
    for opt in service.get_pricing_optimization()[:5]:
        print(f"[{opt['action']}] {opt['item_name']}: {opt['reason']}")
