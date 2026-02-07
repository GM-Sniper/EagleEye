
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from services.data_pipeline import DataPipeline
from models.hybrid_forecaster import HybridForecaster
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

def benchmark_hybrid_model():
    print("🚀 Starting Hybrid Model Benchmark...")
    
    # 1. Load Data
    pipeline = DataPipeline(data_dir='../Data')
    pipeline.load_core_tables()
    
    # Use Order Items
    print("📊 Loading Item History...")
    order_items = pipeline.order_items.copy()
    items_df = pipeline.items
    item_map = items_df.set_index('id')['title'].to_dict()
    orders = pipeline.orders.copy()
    
    if 'date' not in order_items.columns:
        order_items['created_dt'] = pd.to_datetime(order_items['created'], unit='s')
        order_items['date'] = order_items['created_dt'].dt.date
    order_items['date'] = pd.to_datetime(order_items['date'])
    
    # Identify Test Items (Validated IDs from Debug)
    test_items = {
        615418: "Øl Alm. (Problem Item - Hybrid Test)", 
        653549: "Lille Box (Flagship - Local Strength)",
        480007: "Alm Øl (Top Vol - Local Strength)",
        653567: "Mellem Box (Med Vol)"
    }
    
    # Split Train/Test (Global Split)
    max_date = order_items['date'].max()
    cutoff_date = max_date - timedelta(days=14)
    
    train_orders = orders[pd.to_datetime(orders['date']) < cutoff_date]
    train_items = order_items[order_items['date'] < cutoff_date]
    
    test_items_df = order_items[order_items['date'] >= cutoff_date]

    print("DEBUG: Checking Training Data Volume...")
    vol_stats = train_items.groupby('item_id')['quantity'].sum().sort_values(ascending=False)
    
    print("\nTop 10 Items by Volume:")
    for i, (tid, vol) in enumerate(vol_stats.head(10).items()):
        name = item_map.get(tid, "Unknown")
        print(f"{i+1}. ID {tid}: {vol} units - {name}")

    print("\nDEBUG: Specific IDs in Training Data:")
    for tid in test_items.keys():
        print(f"ID {tid}: {vol_stats.get(tid, 0)} units")

    print("🧠 Training Hybrid Model (Global + Local Routing)...")
    hybrid = HybridForecaster()
    hybrid.fit(train_orders, train_items)
    
    results = []
    
    print("\n⚖️  Benchmarking Hybrid Routing...")
    print(f"{'Item Name':<30} | {'Model Used':<20} | {'Acc':<8}")
    print("-" * 70)
    
    plt.style.use('dark_background')
    
    for item_id, item_name in test_items.items():
        # Get Ground Truth
        truth = test_items_df[test_items_df['item_id'] == item_id].groupby('date')['quantity'].sum().reset_index()
        truth.columns = ['date', 'order_count']
        
        # Get History for this item
        item_history = train_items[train_items['item_id'] == item_id].groupby('date')['quantity'].sum().reset_index()
        item_history.columns = ['date', 'order_count']
        
        # Predict using Hybrid Model
        try:
            preds, model_name = hybrid.predict_item(item_id, item_history, horizon_days=14)
            
            # Merge
            comparison = pd.merge(truth, preds, on='date', how='inner')
            comparison['abs_error'] = abs(comparison['order_count'] - comparison['predicted_demand'])
            comparison['ape'] = comparison['abs_error'] / np.maximum(comparison['order_count'], 1)
            
            mape = comparison['ape'].mean() * 100
            accuracy = max(0, 100 - mape)
            
            print(f"{item_name:<30} | {model_name:<20} | {accuracy:.1f}%")
            
            results.append({
                'item_name': item_name,
                'model': model_name,
                'accuracy': accuracy,
                'mape': mape
            })
            
            # Plot
            plt.figure(figsize=(12, 6))
            train_subset = item_history.tail(60)
            
            plt.plot(train_subset['date'], train_subset['order_count'], label='History', color='gray', alpha=0.5)
            plt.plot(comparison['date'], comparison['order_count'], label='Actual', marker='o', color='#10b981', linewidth=2)
            plt.plot(comparison['date'], comparison['predicted_demand'], label=f'Hybrid ({model_name})', marker='x', linestyle='--', color='#f472b6', linewidth=2)
            
            plt.title(f"Hybrid Forecast: {item_name}\nModel: {model_name} | Accuracy: {accuracy:.1f}%", fontsize=14, pad=20)
            plt.legend()
            plt.grid(True, alpha=0.1)
            plt.savefig(f"hybrid_bench_{item_id}.png", bbox_inches='tight', dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"{item_name:<30} | ❌ Failed: {str(e)}")

    print("\n📝 Hybrid Strategy Summary:")
    print("-" * 50)
    for res in results:
        status = "✅ Golden Rule (>70%)" if res['accuracy'] >= 70 else "⚠️ Needs work"
        print(f"{res['item_name']:<30} ({res['model']}) : {res['accuracy']:.1f}%  {status}")

if __name__ == "__main__":
    benchmark_hybrid_model()
