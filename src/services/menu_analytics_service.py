"""
Menu Analytics Service for EagleEye
Menu engineering matrix and performance analytics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class MenuAnalyticsService:
    """
    Service for menu engineering and performance analytics.
    Implements BCG-style matrix analysis for menu optimization.
    """
    
    def __init__(self, item_stats: pd.DataFrame, orders: Optional[pd.DataFrame] = None):
        """
        Initialize with item statistics.
        
        Args:
            item_stats: DataFrame from DataPipeline.get_item_stats()
            orders: Optional orders DataFrame for additional analysis
        """
        self.stats = item_stats.copy()
        self.orders = orders
        self._prepare_metrics()
    
    def _prepare_metrics(self):
        """Pre-calculate metrics for analysis."""
        df = self.stats
        
        # Calculate revenue if not present
        if 'revenue' not in df.columns:
            df['revenue'] = df['total_qty'].fillna(0) * df['price'].fillna(0)
        
        # Normalize metrics to 0-1 scale
        df['popularity'] = df['total_qty'] / df['total_qty'].max() if df['total_qty'].max() > 0 else 0
        df['profitability'] = df['revenue'] / df['revenue'].max() if df['revenue'].max() > 0 else 0
        
        # Calculate contribution margins (using price as proxy when cost unavailable)
        # Higher price items assumed to have higher margins
        df['margin_proxy'] = df['price'].rank(pct=True)
        
        self.stats = df
    
    def get_top_items(
        self, 
        metric: str = 'revenue', 
        limit: int = 20
    ) -> List[Dict]:
        """
        Get top performing items by specified metric.
        
        Args:
            metric: 'revenue', 'orders', or 'avg_daily'
            limit: Maximum number of items to return
            
        Returns:
            List of top items with performance data
        """
        df = self.stats.copy()
        
        # Map metric to column
        metric_map = {
            'revenue': 'revenue',
            'orders': 'total_qty',
            'avg_daily': 'mean_daily'
        }
        
        sort_col = metric_map.get(metric, 'revenue')
        
        if sort_col not in df.columns:
            sort_col = 'total_qty'
        
        top = df.nlargest(limit, sort_col)
        
        return [
            {
                'rank': i + 1,
                'item_id': int(row.get('item_id', 0)),
                'item_name': str(row.get('item_name', 'Unknown'))[:50],
                'total_orders': int(row.get('total_qty', 0)),
                'mean_daily': round(float(row.get('mean_daily', 0)), 2),
                'price': round(float(row.get('price', 0)), 2),
                'revenue': round(float(row.get('revenue', 0)), 2),
                'abc_class': str(row.get('abc_class', 'N/A'))
            }
            for i, (_, row) in enumerate(top.iterrows())
        ]
    
    def get_menu_engineering_matrix(self) -> Dict[str, List[Dict]]:
        """
        Classify items into BCG-style menu engineering matrix.
        
        Categories:
        - STARS: High popularity + High profitability → Keep & promote
        - PUZZLES: Low popularity + High profitability → Market better
        - PLOWHORSES: High popularity + Low profitability → Re-engineer
        - DOGS: Low popularity + Low profitability → Consider removing
        
        Returns:
            Dict with category names as keys and item lists as values
        """
        df = self.stats.copy()
        
        # Calculate medians for thresholds
        pop_median = df['popularity'].median()
        prof_median = df['profitability'].median()
        
        def classify(row):
            high_pop = row['popularity'] >= pop_median
            high_prof = row['profitability'] >= prof_median
            
            if high_pop and high_prof:
                return 'stars'
            elif not high_pop and high_prof:
                return 'puzzles'
            elif high_pop and not high_prof:
                return 'plowhorses'
            else:
                return 'dogs'
        
        df['category'] = df.apply(classify, axis=1)
        
        result = {}
        
        for category in ['stars', 'puzzles', 'plowhorses', 'dogs']:
            cat_items = df[df['category'] == category].nlargest(20, 'revenue')
            
            result[category] = [
                {
                    'item_id': int(row.get('item_id', 0)),
                    'item_name': str(row.get('item_name', 'Unknown'))[:50],
                    'popularity_score': round(float(row['popularity']), 3),
                    'profitability_score': round(float(row['profitability']), 3),
                    'total_orders': int(row.get('total_qty', 0)),
                    'revenue': round(float(row.get('revenue', 0)), 2),
                    'price': round(float(row.get('price', 0)), 2)
                }
                for _, row in cat_items.iterrows()
            ]
        
        return result
    
    def get_recommendations(self) -> List[Dict]:
        """
        Generate actionable menu recommendations.
        
        Returns:
            List of recommendations with actions and reasoning
        """
        matrix = self.get_menu_engineering_matrix()
        recommendations = []
        
        # STARS: Keep doing what's working
        for item in matrix['stars'][:5]:
            recommendations.append({
                'item_id': item['item_id'],
                'item_name': item['item_name'],
                'category': 'STAR',
                'action': 'MAINTAIN',
                'priority': 'HIGH',
                'recommendation': 'High performer - maintain quality and visibility',
                'details': f"Generating ${item['revenue']:,.0f} with {item['total_orders']} orders"
            })
        
        # PUZZLES: Need better marketing
        for item in matrix['puzzles'][:5]:
            recommendations.append({
                'item_id': item['item_id'],
                'item_name': item['item_name'],
                'category': 'PUZZLE',
                'action': 'PROMOTE',
                'priority': 'MEDIUM',
                'recommendation': 'High profit but low sales - increase visibility',
                'details': f"Only {item['total_orders']} orders but ${item['price']:.2f} price point"
            })
        
        # PLOWHORSES: Need re-engineering
        for item in matrix['plowhorses'][:5]:
            recommendations.append({
                'item_id': item['item_id'],
                'item_name': item['item_name'],
                'category': 'PLOWHORSE',
                'action': 'RE-ENGINEER',
                'priority': 'MEDIUM',
                'recommendation': 'Popular but low margin - consider price increase or cost reduction',
                'details': f"Popular ({item['total_orders']} orders) but only ${item['price']:.2f}"
            })
        
        # DOGS: Consider removal
        for item in matrix['dogs'][:5]:
            recommendations.append({
                'item_id': item['item_id'],
                'item_name': item['item_name'],
                'category': 'DOG',
                'action': 'ELIMINATE',
                'priority': 'LOW',
                'recommendation': 'Low volume and low profit - consider removing from menu',
                'details': f"Only {item['total_orders']} orders, ${item['revenue']:,.0f} revenue"
            })
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: (priority_order.get(x['priority'], 99), -x['item_id']))
        
        return recommendations
    
    def get_summary(self) -> Dict:
        """Get summary of menu performance."""
        matrix = self.get_menu_engineering_matrix()
        
        total_revenue = self.stats['revenue'].sum()
        
        return {
            'total_items': len(self.stats),
            'total_revenue': round(float(total_revenue), 2),
            'stars_count': len(matrix['stars']),
            'stars_revenue_pct': round(
                sum(i['revenue'] for i in matrix['stars']) / max(total_revenue, 1) * 100, 1
            ),
            'puzzles_count': len(matrix['puzzles']),
            'plowhorses_count': len(matrix['plowhorses']),
            'dogs_count': len(matrix['dogs']),
            'recommendations_count': len(self.get_recommendations()),
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
    
    stats = pipeline.get_item_stats()
    service = MenuAnalyticsService(stats)
    
    print("=== Menu Analytics Service Test ===\n")
    
    print("--- Summary ---")
    print(service.get_summary())
    
    print("\n--- Top 5 Items by Revenue ---")
    for item in service.get_top_items(metric='revenue', limit=5):
        print(f"#{item['rank']} {item['item_name']}: ${item['revenue']:,.0f}")
    
    print("\n--- Menu Engineering Matrix ---")
    matrix = service.get_menu_engineering_matrix()
    for category, items in matrix.items():
        print(f"\n{category.upper()}: {len(items)} items")
        for item in items[:3]:
            print(f"  - {item['item_name']}")
    
    print("\n--- Top Recommendations ---")
    for rec in service.get_recommendations()[:5]:
        print(f"[{rec['action']}] {rec['item_name']}: {rec['recommendation']}")
