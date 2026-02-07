"""
Ingredient Service for EagleEye
Handles bill-of-materials tracking and ingredient consumption projections.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


class IngredientService:
    """
    Service for ingredient and bill-of-materials management.
    Projects ingredient consumption based on demand forecasts.
    """
    
    # Default ingredient mappings for common menu item categories
    # In production, this would come from dim_bill_of_materials.csv
    DEFAULT_INGREDIENT_TEMPLATES = {
        'beverage': [
            {'ingredient': 'Base Syrup', 'unit': 'ml', 'qty_per_item': 30},
            {'ingredient': 'Water/Milk', 'unit': 'ml', 'qty_per_item': 200},
            {'ingredient': 'Ice', 'unit': 'g', 'qty_per_item': 50},
        ],
        'food': [
            {'ingredient': 'Main Protein', 'unit': 'g', 'qty_per_item': 150},
            {'ingredient': 'Side Carb', 'unit': 'g', 'qty_per_item': 100},
            {'ingredient': 'Vegetables', 'unit': 'g', 'qty_per_item': 50},
            {'ingredient': 'Sauce', 'unit': 'ml', 'qty_per_item': 20},
        ],
        'dessert': [
            {'ingredient': 'Sugar', 'unit': 'g', 'qty_per_item': 25},
            {'ingredient': 'Flour/Base', 'unit': 'g', 'qty_per_item': 50},
            {'ingredient': 'Dairy', 'unit': 'ml', 'qty_per_item': 30},
        ],
        'default': [
            {'ingredient': 'Primary Ingredient', 'unit': 'units', 'qty_per_item': 1},
        ]
    }
    
    def __init__(self, items_df: pd.DataFrame, bom_df: Optional[pd.DataFrame] = None):
        """
        Initialize with items data and optional bill-of-materials.
        
        Args:
            items_df: DataFrame with item details (id, title, category, etc.)
            bom_df: Optional DataFrame with bill-of-materials mapping
        """
        self.items = items_df.copy()
        self.bom = bom_df if bom_df is not None else self._generate_mock_bom()
        self._ingredient_stock = self._initialize_ingredient_stock()
    
    def _generate_mock_bom(self) -> pd.DataFrame:
        """Generate mock bill-of-materials when actual data is unavailable."""
        records = []
        
        for _, item in self.items.iterrows():
            item_id = item.get('id') or item.get('item_id')
            item_name = str(item.get('title') or item.get('item_name', '')).lower()
            
            # Determine category based on item name keywords
            if any(kw in item_name for kw in ['coffee', 'tea', 'juice', 'soda', 'water', 'drink', 'øl', 'beer']):
                category = 'beverage'
            elif any(kw in item_name for kw in ['cake', 'cookie', 'ice cream', 'dessert', 'sweet']):
                category = 'dessert'
            elif any(kw in item_name for kw in ['burger', 'pizza', 'sandwich', 'salad', 'wrap', 'meal']):
                category = 'food'
            else:
                category = 'default'
            
            template = self.DEFAULT_INGREDIENT_TEMPLATES.get(category, self.DEFAULT_INGREDIENT_TEMPLATES['default'])
            
            for ing in template:
                records.append({
                    'item_id': item_id,
                    'ingredient_name': ing['ingredient'],
                    'unit': ing['unit'],
                    'quantity_per_item': ing['qty_per_item']
                })
        
        return pd.DataFrame(records)
    
    def _initialize_ingredient_stock(self) -> Dict[str, Dict]:
        """Initialize simulated ingredient stock levels."""
        np.random.seed(123)  # Reproducible random values
        
        unique_ingredients = self.bom['ingredient_name'].unique()
        stock = {}
        
        for ing in unique_ingredients:
            # Calculate average daily consumption
            avg_daily = self.bom[self.bom['ingredient_name'] == ing]['quantity_per_item'].mean() * 100
            capacity = avg_daily * 14  # 2 weeks capacity
            
            stock[ing] = {
                'current_stock': round(np.random.uniform(0.3, 1.0) * capacity, 2),
                'capacity': round(capacity, 2),
                'unit': self.bom[self.bom['ingredient_name'] == ing]['unit'].iloc[0],
                'avg_daily_consumption': round(avg_daily, 2),
                'reorder_point': round(avg_daily * 3, 2)  # 3 days buffer
            }
        
        return stock
    
    def get_composition(self, item_id: int) -> List[Dict]:
        """
        Get ingredient composition for a specific menu item.
        
        Args:
            item_id: The ID of the menu item
            
        Returns:
            List of ingredients with quantities required per item
        """
        item_bom = self.bom[self.bom['item_id'] == item_id]
        
        return [
            {
                'ingredient': row['ingredient_name'],
                'quantity': row['quantity_per_item'],
                'unit': row['unit']
            }
            for _, row in item_bom.iterrows()
        ]
    
    def calculate_ingredient_consumption(
        self, 
        item_id: int, 
        quantity: int
    ) -> Dict[str, float]:
        """
        Calculate total ingredient consumption for producing N units of an item.
        
        Args:
            item_id: The menu item ID
            quantity: Number of units to produce
            
        Returns:
            Dict mapping ingredient names to total quantities needed
        """
        composition = self.get_composition(item_id)
        
        return {
            ing['ingredient']: ing['quantity'] * quantity
            for ing in composition
        }
    
    def project_ingredient_demand(
        self, 
        forecast_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Project ingredient needs based on item-level demand forecast.
        
        Args:
            forecast_df: DataFrame with columns ['item_id', 'date', 'predicted_demand']
            
        Returns:
            DataFrame with projected ingredient consumption per day
        """
        if 'item_id' not in forecast_df.columns:
            raise ValueError("forecast_df must contain 'item_id' column")
        
        records = []
        
        for _, row in forecast_df.iterrows():
            item_id = row['item_id']
            demand = row.get('predicted_demand', row.get('demand', 0))
            date = row.get('date')
            
            consumption = self.calculate_ingredient_consumption(item_id, demand)
            
            for ing_name, qty in consumption.items():
                records.append({
                    'date': date,
                    'ingredient': ing_name,
                    'projected_consumption': qty,
                    'item_id': item_id
                })
        
        result = pd.DataFrame(records)
        
        if not result.empty:
            # Aggregate by date and ingredient
            result = result.groupby(['date', 'ingredient']).agg({
                'projected_consumption': 'sum'
            }).reset_index()
        
        return result
    
    def get_ingredient_stock(self) -> List[Dict]:
        """Get current ingredient stock levels."""
        return [
            {
                'ingredient': name,
                **data,
                'status': self._get_stock_status(data)
            }
            for name, data in self._ingredient_stock.items()
        ]
    
    def _get_stock_status(self, data: Dict) -> str:
        """Determine stock status based on current level."""
        ratio = data['current_stock'] / data['capacity']
        
        if data['current_stock'] < data['reorder_point'] * 0.5:
            return 'CRITICAL'
        elif data['current_stock'] < data['reorder_point']:
            return 'LOW'
        elif ratio > 0.9:
            return 'OVERSTOCKED'
        return 'HEALTHY'
    
    def get_restocking_alerts(
        self, 
        forecast_days: int = 7
    ) -> List[Dict]:
        """
        Get alerts for ingredients that will run out based on projected consumption.
        
        Args:
            forecast_days: Number of days to project ahead
            
        Returns:
            List of alert dictionaries for ingredients needing restocking
        """
        alerts = []
        
        for name, data in self._ingredient_stock.items():
            projected_consumption = data['avg_daily_consumption'] * forecast_days
            remaining_after = data['current_stock'] - projected_consumption
            days_until_empty = data['current_stock'] / max(data['avg_daily_consumption'], 0.1)
            
            if remaining_after < 0:
                alerts.append({
                    'ingredient': name,
                    'current_stock': data['current_stock'],
                    'unit': data['unit'],
                    'projected_consumption': round(projected_consumption, 2),
                    'days_until_empty': round(days_until_empty, 1),
                    'suggested_order_qty': round(data['capacity'] - data['current_stock'], 2),
                    'severity': 'CRITICAL' if days_until_empty < 3 else 'WARNING',
                    'message': f"Will run out in ~{days_until_empty:.1f} days. Order {data['capacity'] - data['current_stock']:.0f} {data['unit']}."
                })
            elif data['current_stock'] < data['reorder_point']:
                alerts.append({
                    'ingredient': name,
                    'current_stock': data['current_stock'],
                    'unit': data['unit'],
                    'projected_consumption': round(projected_consumption, 2),
                    'days_until_empty': round(days_until_empty, 1),
                    'suggested_order_qty': round(data['capacity'] * 0.5, 2),
                    'severity': 'INFO',
                    'message': f"Below reorder point. Consider ordering {data['capacity'] * 0.5:.0f} {data['unit']}."
                })
        
        # Sort by severity
        severity_order = {'CRITICAL': 0, 'WARNING': 1, 'INFO': 2}
        alerts.sort(key=lambda x: severity_order.get(x['severity'], 99))
        
        return alerts


if __name__ == "__main__":
    # Test the service
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from services.data_pipeline import DataPipeline
    
    pipeline = DataPipeline(data_dir="../../Data")
    pipeline.load_all()
    pipeline.load_core_tables()
    
    service = IngredientService(pipeline.items)
    
    print("=== Ingredient Service Test ===\n")
    
    # Get composition of first item
    first_item_id = pipeline.items['id'].iloc[0]
    print(f"Composition of item {first_item_id}:")
    print(service.get_composition(first_item_id))
    
    print("\n--- Restocking Alerts ---")
    alerts = service.get_restocking_alerts(forecast_days=7)
    for alert in alerts[:5]:
        print(f"[{alert['severity']}] {alert['ingredient']}: {alert['message']}")
