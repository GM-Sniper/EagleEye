
import sys
from pathlib import Path
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from services.data_pipeline import DataPipeline
from models.production_forecaster import ProductionForecaster
from datetime import timedelta
import os
import seaborn as sns

def analyze_top_items():
    print("🚀 Starting Top Item Analysis...")
    
    # 1. Load Data
    pipeline = DataPipeline(data_dir='../Data')
    pipeline.load_core_tables()
    
    # Use Order Items for item-level analysis
    print("📊 Aggregating Item-Level Demand...")
    order_items = pipeline.order_items.copy()
    
    # Ensure date column exists
    if 'date' not in order_items.columns:
        order_items['created_dt'] = pd.to_datetime(order_items['created'], unit='s')
        order_items['date'] = order_items['created_dt'].dt.date
    
    order_items['date'] = pd.to_datetime(order_items['date'])

    # Find Top 5 Items by Total Volume
    item_stats = order_items.groupby('item_id')['quantity'].sum().sort_values(ascending=False)
    top_5_items = item_stats.head(5).index.tolist()
    
    print(f"🏆 Top 5 Items (IDs): {top_5_items}")
    
    # Get item names for better plotting
    items_df = pipeline.items
    item_map = items_df.set_index('id')['title'].to_dict()
    
    results = []
    
    # Set style
    plt.style.use('dark_background')
    
    for item_id in top_5_items:
        item_name = item_map.get(item_id, f"Item {item_id}")
        print(f"\n🔍 Analyzing: {item_name} (ID: {item_id})")
        
        # Filter for this item
        item_data = order_items[order_items['item_id'] == item_id]
        
        # Daily aggregation
        daily = item_data.groupby('date')['quantity'].sum().reset_index()
        daily.columns = ['date', 'order_count'] # Rename for model compatibility (model expects 'order_count')
        
        # Fill missing dates
        all_dates = pd.date_range(start=daily['date'].min(), end=daily['date'].max())
        daily = daily.set_index('date').reindex(all_dates, fill_value=0).reset_index()
        daily.columns = ['date', 'order_count']
        
        # Backtest Split (Last 14 Days)
        max_date = daily['date'].max()
        cutoff_date = max_date - timedelta(days=14)
        
        train = daily[daily['date'] < cutoff_date]
        test = daily[daily['date'] >= cutoff_date]
        
        if len(train) < 28:
            print(f"⚠️ Not enough history for {item_name}, skipping...")
            continue
            
        print(f"   Train size: {len(train)} days | Test size: {len(test)} days")
            
        # Fit Model
        try:
            model = ProductionForecaster()
            model.fit(train)
            
            # Predict
            preds = model.predict(horizon_days=14)
            
            # Compare
            comparison = pd.merge(test, preds, on='date', how='inner')
            comparison['abs_error'] = abs(comparison['order_count'] - comparison['predicted_demand'])
            
            # Handle zero actuals for MAPE (use SMAPE-like logic or just cap denominator)
            comparison['ape'] = comparison['abs_error'] / np.maximum(comparison['order_count'], 1)
            
            mape = comparison['ape'].mean() * 100
            accuracy = max(0, 100 - mape)
            
            print(f"   ✅ Accuracy: {accuracy:.2f}% | MAPE: {mape:.2f}%")
            
            results.append({
                'item_name': item_name,
                'accuracy': accuracy,
                'mape': mape
            })
            
            # Plot
            plt.figure(figsize=(12, 6))
            
            # Plot history context (last 30 days of training)
            recent_train = train.tail(30)
            plt.plot(recent_train['date'], recent_train['order_count'], label='History', color='gray', alpha=0.5)
            
            plt.plot(comparison['date'], comparison['order_count'], label='Actual', marker='o', color='#10b981', linewidth=2)
            plt.plot(comparison['date'], comparison['predicted_demand'], label='AI Prediction', marker='x', linestyle='--', color='#22d3ee', linewidth=2)
            plt.fill_between(comparison['date'], comparison['lower_bound'], comparison['upper_bound'], color='#22d3ee', alpha=0.15, label='95% CI')
            
            plt.title(f"{item_name}\nAccuracy: {accuracy:.1f}% | MAPE: {mape:.1f}%", fontsize=14, pad=20)
            plt.suptitle(f"14-Day Backtest ({cutoff_date.date()} - {max_date.date()})", fontsize=10, color='gray', y=0.92)
            plt.legend()
            plt.grid(True, alpha=0.1)
            
            # Save
            filename = f"top_item_{len(results)}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            print(f"   📷 Saved chart to {filename}")
            plt.close()
            
        except Exception as e:
            print(f"   ❌ Validation failed: {str(e)}")

    print("\n📝 Final Summary:")
    print("-" * 50)
    for res in results:
        print(f"{res['item_name']:<30} | Acc: {res['accuracy']:.1f}% | MAPE: {res['mape']:.1f}%")

if __name__ == "__main__":
    analyze_top_items()