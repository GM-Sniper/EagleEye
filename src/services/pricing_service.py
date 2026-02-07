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
    
    def __init__(self, inventory_snapshot: pd.DataFrame, forecaster=None):
        """
        Initialize with inventory snapshot.
        
        Args:
            inventory_snapshot: DataFrame from DataPipeline.get_inventory_snapshot()
            forecaster: Optional HybridForecaster instance for demand prediction
        """
        self.inventory = inventory_snapshot.copy()
        self.forecaster = forecaster
    
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
            min_excess_ratio: Minimum stock/capacity ratio
            
        Returns:
            List of discount recommendations
        """
        # Identify items that need clearing and suggest discounts.
        # Logic: 
        # 1. Base Discount = 1% for every 1% utilization > 80%.
        # 2. Final Discount = Base Discount / Predicted Daily Demand.
        
        # Filter for overstocked, discountable items
        df = self.inventory.copy()
        
        # Ensure discountable column exists and handle NaN/None
        if 'discountable' not in df.columns:
            df['discountable'] = True  # Default to discountable if not specified
        
        df['discountable'] = df['discountable'].fillna(True).astype(bool)
        
        # Filter for items with utilization > 80% (User definition of overstock for discount)
        # using capacity fallback if needed (though pipeline ensures capacity)
        # We calculate utilization first to filter
        
        # Ensure conversion to numeric
        df['current_stock'] = pd.to_numeric(df['current_stock'], errors='coerce').fillna(0)
        df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce').fillna(1) # avoid div0
        
        # Calculate utilization
        df['utilization'] = df['current_stock'] / df['capacity']
        
        # Filter: Utilization > 0.8 AND Discountable
        overstocked_mask = (df['utilization'] > 0.8) & (df['discountable'] == True)
        overstocked_df = df[overstocked_mask].copy()
        
        # Pre-sort by utilization to prioritize calculation for most critical items
        overstocked_df.sort_values('utilization', ascending=False, inplace=True)
        
        # Limit to top 400 candidates to keep performance high
        candidates = overstocked_df.head(400)
        
        recommendations = []
        
        # Optimization: Apply limit early if possible, but we need to calculate discounts to sort.
        # We process all overstocked (subset of total items), then sort and slice.
        
        for _, row in candidates.iterrows():
        
            # Calculate utilization ratio (Stock / Capacity)
            # User defined "stock divided by the demand" (interpreted as Capacity which is demand-based)
            # Threshold is 80%
            capacity = row.get('capacity', 0)
            if capacity <= 0:
                continue

            current = row.get('current_stock', 0)
            item_id = int(row.get('item_id', 0))
            
            # Utilization ratio (e.g. 0.9 = 90%)
            utilization = current / capacity
            excess_ratio = utilization # For compatibility with sorting logic
            
            # "above 80% of the stock"
            threshold = 0.80
            
            if utilization <= threshold:
                continue
                
            # "1% for every 1 percentage above 80%"
            # Excess percentage points
            excess_percentage_points = (utilization - threshold) * 100
            
            base_discount = int(round(excess_percentage_points))
            
            if base_discount <= 0:
                continue
            
            # --- INCORPORATE PREDICTED DEMAND ---
            predicted_daily_demand = 1.0 # Default fallback
            
            if self.forecaster:
                try:
                    # Use Global Model for speed (avoid training local model)
                    # Predict next 7 days
                    forecast = self.forecaster.global_model.predict(item_id, 7)
                    if not forecast.empty and 'predicted_demand' in forecast.columns:
                        predicted_daily_demand = forecast['predicted_demand'].mean()
                        # Ensure we don't divide by zero or negative
                        predicted_daily_demand = max(predicted_daily_demand, 0.1)
                except Exception as e:
                    print(f"Prediction failed for {item_id}: {e}")
                    # Fallback to mean_daily from snapshot if model fails
                    predicted_daily_demand = max(row.get('mean_daily', 1.0), 0.1)
            else:
                 predicted_daily_demand = max(row.get('mean_daily', 1.0), 0.1)
            
            # Final Formula: Discount / Predicted Demand
            # e.g. Base 20% / Demand 10 = 2%
            # e.g. Base 20% / Demand 0.5 = 40%
            
            adjusted_discount = base_discount / predicted_daily_demand
            
            suggested_discount = int(round(adjusted_discount))
            
            # Apply safety cap (default 30% from function arg)
            suggested_discount = min(suggested_discount, max_discount_pct)
            
            if suggested_discount <= 0:
                continue
            
            # Calculate potential revenue impact
            price = row.get('price', 0) or 0
            daily_demand = predicted_daily_demand # Use the predicted demand for consistency
            
            # Estimate demand increase from discount (simple elasticity model)
            demand_multiplier = 1 + (suggested_discount / 100) * 1.5
            projected_daily_revenue_before = price * daily_demand
            projected_daily_revenue_after = (price * (1 - suggested_discount/100)) * (daily_demand * demand_multiplier)
            
            # Days to clear excess stock (down to 80% level?)
            # Or just clear the excess? Assuming clear to threshold.
            target_stock = capacity * threshold
            excess_units = current - target_stock
            days_to_clear = excess_units / max(daily_demand * demand_multiplier, 0.1)
            
            recommendations.append({
                'item_id': item_id,
                'item_name': str(row.get('item_name', 'Unknown'))[:50],
                'current_stock': round(float(current), 1),
                'capacity': round(float(capacity), 1),
                'excess_ratio': round(float(excess_ratio), 2),
                'predicted_demand': round(float(predicted_daily_demand), 2),
                'current_price': round(float(price), 2),
                'suggested_discount_pct': suggested_discount,
                'discounted_price': round(float(price * (1 - suggested_discount/100)), 2),
                'estimated_days_to_clear': round(float(days_to_clear), 1),
                'abc_class': str(row.get('abc_class', 'N/A')),
                'reason': f'High utilization ({int(utilization*100)}%) vs Demand ({round(predicted_daily_demand, 1)})',
                'priority': 'HIGH' if suggested_discount > 15 else 'MEDIUM'
            })
        
        # Sort by discount magnitude
        recommendations.sort(key=lambda x: x['suggested_discount_pct'], reverse=True)
        
        # Limit to top 100 items to keep response light
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
