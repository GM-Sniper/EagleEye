import pandas as pd
import numpy as np
from datetime import timedelta
from services.data_pipeline import DataPipeline
from models.production_forecaster import ProductionForecaster
import matplotlib.pyplot as plt

def run_verification():
    print("=" * 70)
    print("🔭 EAGLEEYE PREDICTION VERIFICATION (ACTUAL VS PREDICTED)")
    print("=" * 70)

    # 1. Load Data
    pipeline = DataPipeline(data_dir="../Data")
    pipeline.load_all()
    pipeline.load_core_tables()
    daily_demand = pipeline.get_daily_demand()
    daily_demand['date'] = pd.to_datetime(daily_demand['date'])

    # 2. Split Data (Hide the last 7 days from the model)
    max_date = daily_demand['date'].max()
    test_start_date = max_date - timedelta(days=6)
    
    train_data = daily_demand[daily_demand['date'] < test_start_date]
    actual_data = daily_demand[daily_demand['date'] >= test_start_date]

    print(f"Training on data up to: {train_data['date'].max().strftime('%Y-%m-%d')}")
    print(f"Verifying on dates:    {test_start_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

    # 3. Train Model
    forecaster = ProductionForecaster()
    forecaster.fit(train_data)

    # 4. Predict
    predictions = forecaster.predict(horizon_days=7)
    
    # 5. Compare
    comparison = pd.merge(
        predictions, 
        actual_data[['date', 'order_count']], 
        on='date'
    )
    comparison.columns = ['Date', 'Day', 'Predicted', 'Lower', 'Upper', 'Actual']
    comparison['Error %'] = (abs(comparison['Actual'] - comparison['Predicted']) / comparison['Actual'] * 100).round(2)

    print("\n📊 DAILY COMPARISON:")
    print("-" * 80)
    print(comparison[['Date', 'Day', 'Actual', 'Predicted', 'Error %']].to_string(index=False))
    
    avg_mape = comparison['Error %'].mean()
    print("-" * 80)
    print(f"Average Error (MAPE) for this week: {avg_mape:.2f}%")
    print("=" * 70)

    if avg_mape < 15:
        print("✅ VERIFIED: Model is performing within acceptable high-accuracy bounds.")
    else:
        print("⚠️ NOTICE: High variance this week, check for unusual events in original data.")

if __name__ == "__main__":
    run_verification()
