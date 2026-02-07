"""
EagleEye Model Evaluation Framework
Comprehensive backtesting and accuracy testing for demand forecasting models.
"""

import sys
from pathlib import Path
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from services.data_pipeline import DataPipeline
from models.forecaster import DemandForecaster

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Model evaluation framework for demand forecasting.
    Implements backtesting, cross-validation, and comprehensive metrics.
    """
    
    def __init__(self):
        self.results: Dict[str, Dict] = {}
        self.backtest_results: List[Dict] = []
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive forecasting metrics."""
        
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE (handle zeros)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        
        # Weighted MAPE
        wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
        
        # R-squared
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
        
        # Bias (positive = over-forecasting)
        bias = np.mean(y_pred - y_true)
        bias_pct = (bias / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else 0
        
        # Tracking signal (cumulative error / MAD)
        cumulative_error = np.sum(y_pred - y_true)
        mad = np.mean(np.abs(y_true - y_pred))
        tracking_signal = cumulative_error / mad if mad > 0 else 0
        
        return {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'mape': round(mape, 2),
            'wmape': round(wmape, 2),
            'r2': round(r2, 4),
            'bias': round(bias, 2),
            'bias_pct': round(bias_pct, 2),
            'tracking_signal': round(tracking_signal, 2),
            'n_samples': len(y_true)
        }
    
    def backtest(
        self,
        data: pd.DataFrame,
        train_size: int = 60,
        test_size: int = 7,
        n_backtests: int = 10,
        target_col: str = 'order_count'
    ) -> pd.DataFrame:
        """
        Perform rolling window backtesting.
        
        Args:
            data: Time series data with 'date' and target column
            train_size: Number of days for training
            test_size: Number of days to forecast (test period)
            n_backtests: Number of backtest iterations
            target_col: Column to forecast
        """
        data = data.sort_values('date').reset_index(drop=True)
        total_size = train_size + test_size
        
        if len(data) < total_size:
            raise ValueError(f"Insufficient data: need {total_size}, have {len(data)}")
        
        # Calculate step size for rolling window
        step = max(1, (len(data) - total_size) // (n_backtests - 1)) if n_backtests > 1 else 1
        
        results = []
        
        for i in range(n_backtests):
            start_idx = i * step
            end_idx = start_idx + total_size
            
            if end_idx > len(data):
                break
            
            subset = data.iloc[start_idx:end_idx]
            train = subset.iloc[:train_size]
            test = subset.iloc[train_size:]
            
            # Train model
            forecaster = DemandForecaster()
            forecaster.fit(train, target_col)
            
            # Predict
            forecast = forecaster.predict(horizon_days=test_size)
            
            # Calculate metrics
            y_true = test[target_col].values
            y_pred = forecast['predicted_demand'].values[:len(y_true)]
            
            metrics = self.calculate_metrics(y_true, y_pred)
            metrics['backtest_id'] = i + 1
            metrics['train_start'] = train['date'].min()
            metrics['train_end'] = train['date'].max()
            metrics['test_start'] = test['date'].min()
            metrics['test_end'] = test['date'].max()
            
            results.append(metrics)
            
            print(f"   Backtest {i+1}/{n_backtests}: MAPE={metrics['mape']:.1f}%, RMSE={metrics['rmse']:.0f}")
        
        self.backtest_results = results
        return pd.DataFrame(results)
    
    def evaluate_by_horizon(
        self,
        data: pd.DataFrame,
        horizons: List[int] = [1, 3, 7, 14],
        target_col: str = 'order_count'
    ) -> pd.DataFrame:
        """Evaluate forecast accuracy at different horizons."""
        
        # Use last portion for testing
        train_size = int(len(data) * 0.8)
        train = data.iloc[:train_size]
        test = data.iloc[train_size:]
        
        forecaster = DemandForecaster()
        forecaster.fit(train, target_col)
        
        results = []
        
        for horizon in horizons:
            if horizon > len(test):
                continue
            
            forecast = forecaster.predict(horizon_days=horizon)
            y_true = test[target_col].values[:horizon]
            y_pred = forecast['predicted_demand'].values[:len(y_true)]
            
            metrics = self.calculate_metrics(y_true, y_pred)
            metrics['horizon_days'] = horizon
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def evaluate_by_day_of_week(
        self,
        data: pd.DataFrame,
        target_col: str = 'order_count'
    ) -> pd.DataFrame:
        """Evaluate forecast accuracy by day of week."""
        
        # Train on 80% of data
        train_size = int(len(data) * 0.8)
        train = data.iloc[:train_size].copy()
        test = data.iloc[train_size:].copy()
        
        # Ensure date is datetime
        train['date'] = pd.to_datetime(train['date'])
        test['date'] = pd.to_datetime(test['date'])
        test['day_of_week'] = test['date'].dt.day_name()
        
        # Train model
        forecaster = DemandForecaster()
        forecaster.fit(train, target_col)
        
        # Generate predictions for test period
        forecast = forecaster.predict(horizon_days=len(test))
        test['predicted'] = forecast['predicted_demand'].values[:len(test)]
        
        # Calculate metrics by day of week
        results = []
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            day_data = test[test['day_of_week'] == day]
            if len(day_data) > 0:
                y_true = day_data[target_col].values
                y_pred = day_data['predicted'].values
                metrics = self.calculate_metrics(y_true, y_pred)
                metrics['day_of_week'] = day
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    def generate_evaluation_report(self, backtest_df: pd.DataFrame) -> str:
        """Generate a text report of model evaluation."""
        
        report = []
        report.append("=" * 70)
        report.append("📊 MODEL EVALUATION REPORT")
        report.append("=" * 70)
        
        # Overall metrics
        report.append("\n📈 OVERALL PERFORMANCE (Averaged across backtests)")
        report.append("-" * 50)
        
        avg_metrics = {
            'MAPE (%)': backtest_df['mape'].mean(),
            'WMAPE (%)': backtest_df['wmape'].mean(),
            'RMSE': backtest_df['rmse'].mean(),
            'MAE': backtest_df['mae'].mean(),
            'R²': backtest_df['r2'].mean(),
            'Bias (%)': backtest_df['bias_pct'].mean()
        }
        
        for metric, value in avg_metrics.items():
            report.append(f"   {metric:15s}: {value:>10.2f}")
        
        # Performance rating
        mape = avg_metrics['MAPE (%)']
        if mape < 20:
            rating = "🌟 EXCELLENT"
            recommendation = "Model is production-ready"
        elif mape < 35:
            rating = "✅ GOOD"
            recommendation = "Model can be deployed with monitoring"
        elif mape < 50:
            rating = "⚠️ MODERATE"
            recommendation = "Consider additional features or ensemble methods"
        else:
            rating = "❌ POOR"
            recommendation = "Significant model improvements needed"
        
        report.append(f"\n   Rating: {rating}")
        report.append(f"   Recommendation: {recommendation}")
        
        # Stability analysis
        report.append("\n📉 STABILITY ANALYSIS")
        report.append("-" * 50)
        
        mape_std = backtest_df['mape'].std()
        report.append(f"   MAPE Std Dev: {mape_std:.2f}%")
        report.append(f"   MAPE Range: {backtest_df['mape'].min():.1f}% - {backtest_df['mape'].max():.1f}%")
        
        if mape_std < 10:
            report.append("   Stability: STABLE ✓")
        elif mape_std < 20:
            report.append("   Stability: MODERATE")
        else:
            report.append("   Stability: UNSTABLE ⚠️")
        
        # Bias analysis
        report.append("\n📊 BIAS ANALYSIS")
        report.append("-" * 50)
        
        avg_bias = avg_metrics['Bias (%)']
        if abs(avg_bias) < 5:
            report.append(f"   Model is well-calibrated (bias: {avg_bias:.1f}%)")
        elif avg_bias > 0:
            report.append(f"   Model tends to OVER-FORECAST by {avg_bias:.1f}%")
        else:
            report.append(f"   Model tends to UNDER-FORECAST by {abs(avg_bias):.1f}%")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


def run_model_evaluation():
    """Run comprehensive model evaluation."""
    
    print("=" * 70)
    print("🔬 EAGLEEYE MODEL ACCURACY TESTING")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading data...")
    pipeline = DataPipeline(data_dir="../Data")
    pipeline.load_all()
    pipeline.load_core_tables()
    
    daily_demand = pipeline.get_daily_demand()
    daily_demand = daily_demand.sort_values('date')
    
    print(f"   Total days of data: {len(daily_demand)}")
    print(f"   Date range: {daily_demand['date'].min()} to {daily_demand['date'].max()}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Run backtesting
    print("\n🔄 Running Rolling Window Backtests...")
    backtest_results = evaluator.backtest(
        daily_demand,
        train_size=60,
        test_size=7,
        n_backtests=10
    )
    
    # Generate report
    report = evaluator.generate_evaluation_report(backtest_results)
    print(report)
    
    # Evaluate by horizon
    print("\n📅 ACCURACY BY FORECAST HORIZON")
    print("-" * 50)
    horizon_results = evaluator.evaluate_by_horizon(daily_demand)
    for _, row in horizon_results.iterrows():
        print(f"   {int(row['horizon_days'])}-day forecast: MAPE={row['mape']:.1f}%, RMSE={row['rmse']:.0f}")
    
    # Evaluate by day of week
    print("\n📆 ACCURACY BY DAY OF WEEK")
    print("-" * 50)
    dow_results = evaluator.evaluate_by_day_of_week(daily_demand)
    for _, row in dow_results.iterrows():
        print(f"   {row['day_of_week']:10s}: MAPE={row['mape']:.1f}%")
    
    # Save results
    backtest_results.to_csv('backtest_results.csv', index=False)
    horizon_results.to_csv('horizon_results.csv', index=False)
    dow_results.to_csv('dow_results.csv', index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    
    avg_mape = backtest_results['mape'].mean()
    print(f"\n   Average MAPE: {avg_mape:.1f}%")
    print(f"   Best Backtest: {backtest_results['mape'].min():.1f}%")
    print(f"   Worst Backtest: {backtest_results['mape'].max():.1f}%")
    
    # Recommendations
    print("\n🎯 RECOMMENDATIONS")
    print("-" * 50)
    if avg_mape > 50:
        print("   1. Consider adding external features (weather, holidays)")
        print("   2. Try different model architectures (Prophet, LSTM)")
        print("   3. Filter outliers in historical data")
    elif avg_mape > 35:
        print("   1. Add more lag features or rolling statistics")
        print("   2. Consider ensemble methods")
        print("   3. Segment forecast by location")
    else:
        print("   1. Model performance is acceptable")
        print("   2. Consider fine-tuning hyperparameters")
        print("   3. Monitor performance in production")
    
    print("\n💾 Results saved to:")
    print("   - backtest_results.csv")
    print("   - horizon_results.csv")
    print("   - dow_results.csv")
    
    print("\n" + "=" * 70)
    print("✅ MODEL EVALUATION COMPLETE")
    print("=" * 70)
    
    return backtest_results, horizon_results, dow_results


if __name__ == "__main__":
    run_model_evaluation()