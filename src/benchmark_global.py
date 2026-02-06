
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from services.data_pipeline import DataPipeline
from models.global_forecaster import GlobalForecaster
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

def benchmark_global_model():
    print("🚀 Starting Global Model Benchmark...")
    
    # 1. Load Data
    pipeline = DataPipeline(data_dir='../Data')
    pipeline.load_core_tables()
    
    # Use Order Items
    print("📊 Loading Item History...")
    order_items = pipeline.order_items.copy()
    items_df = pipeline.items
    item_map = items_df.set_index('id')['title'].to_dict()
    
    if 'date' not in order_items.columns:
        order_items['created_dt'] = pd.to_datetime(order_items['created'], unit='s')
        order_items['date'] = order_items['created_dt'].dt.date
    order_items['date'] = pd.to_datetime(order_items['date'])
    
    # Identify Top 50 Items for Global Training (to keep benchmark fast)
    item_stats = order_items.groupby('item_id')['quantity'].sum().sort_values(ascending=False)
    top_50_items = item_stats.head(50).index.tolist()
    top_5_items = top_50_items[:5]
    
    print(f"📉 Filtering to Top 50 Items for Benchmark (from {len(item_stats)} total items)...")
    order_items = order_items[order_items['item_id'].isin(top_50_items)]
    
    # Prepare Global Dataset (Daily aggregation per item)
    print("📚 Creating Global Stacked Dataset...")
    daily_global = order_items.groupby(['date', 'item_id'])['quantity'].sum().reset_index()
    daily_global.columns = ['date', 'item_id', 'order_count']
    
    # Fill missing dates for every item (Critical for lags)
    # We do a Cartesian product of all dates x all items to ensure continuous time series
    all_dates = pd.date_range(start=daily_global['date'].min(), end=daily_global['date'].max())
    all_items = daily_global['item_id'].unique()
    
    # Create full grid
    mux = pd.MultiIndex.from_product([all_dates, all_items], names=['date', 'item_id'])
    daily_global = daily_global.set_index(['date', 'item_id']).reindex(mux, fill_value=0).reset_index()
    
    # Split Train/Test (Global Split)
    max_date = daily_global['date'].max()
    cutoff_date = max_date - timedelta(days=14)
    
    train_global = daily_global[daily_global['date'] < cutoff_date]
    test_global = daily_global[daily_global['date'] >= cutoff_date]
    
    print(f"   Global Train: {len(train_global)} rows | Max Date: {train_global['date'].max()}")
    print("🎯 Training Global Model (XGBoost on all items)...")
    
    global_model = GlobalForecaster()
    global_model.fit(train_global)
    
    results = []
    
    # Evaluate on Top 5
    print("\n⚖️  Benchmarking Top 5 Items...")
    plt.style.use('dark_background')
    
    for item_id in top_5_items:
        item_name = item_map.get(item_id, f"Item {item_id}")
        print(f"\n🔍 Evaluating: {item_name}")
        
        # Get Ground Truth
        truth = test_global[test_global['item_id'] == item_id].copy()
        
        # Predict using Global Model
        try:
            preds = global_model.predict(item_id=item_id, horizon_days=14)
            
            # Merge
            comparison = pd.merge(truth, preds, on='date', how='inner')
            comparison['abs_error'] = abs(comparison['order_count'] - comparison['predicted_demand'])
            comparison['ape'] = comparison['abs_error'] / np.maximum(comparison['order_count'], 1)
            
            mape = comparison['ape'].mean() * 100
            accuracy = max(0, 100 - mape)
            
            print(f"   ✅ Global Model Accuracy: {accuracy:.2f}% | MAPE: {mape:.2f}%")
            
            results.append({
                'item_name': item_name,
                'accuracy': accuracy,
                'mape': mape
            })
            
            # Plot
            plt.figure(figsize=(12, 6))
            train_subset = train_global[train_global['item_id'] == item_id].tail(30)
            
            plt.plot(train_subset['date'], train_subset['order_count'], label='History', color='gray', alpha=0.5)
            plt.plot(comparison['date'], comparison['order_count'], label='Actual', marker='o', color='#10b981', linewidth=2)
            plt.plot(comparison['date'], comparison['predicted_demand'], label='Global Model Prediction', marker='x', linestyle='--', color='#f472b6', linewidth=2)
            
            plt.title(f"Global Model: {item_name}\nAccuracy: {accuracy:.1f}%", fontsize=14, pad=20)
            plt.legend()
            plt.grid(True, alpha=0.1)
            plt.savefig(f"global_bench_{item_id}.png", bbox_inches='tight', dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")

    print("\n📝 Final Global Model Results:")
    print("-" * 50)
    for res in results:
        print(f"{res['item_name']:<30} | Acc: {res['accuracy']:.1f}%")

if __name__ == "__main__":
    benchmark_global_model()
