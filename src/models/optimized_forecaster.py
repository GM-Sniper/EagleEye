"""
EagleEye Optimized Forecaster V2
Focus on proven best performers: XGBoost and GBR.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import warnings
import logging

warnings.filterwarnings('ignore')


class OptimizedForecaster:
    """
    Optimized forecaster using XGBoost + GBR stacking.
    Based on iteration 1 findings: XGBoost and GBR perform best.
    """
    
    def __init__(self, recent_days: int = 90):
        self.recent_days = recent_days
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.is_fitted = False
        self._feature_cols = []
        self._last_data = None
        self.validation_mape = None
    
    def _filter_recent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to only recent days of data."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        cutoff = df['date'].max() - timedelta(days=self.recent_days)
        return df[df['date'] >= cutoff].reset_index(drop=True)
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create optimized feature set."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Time features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['month'] = df['date'].dt.month
        
        # Binary (proven important)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        
        # Cyclical (important for periodicity)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, target_col: str = 'order_count') -> pd.DataFrame:
        """Add lag features - focus on most predictive lags."""
        df = df.copy()
        
        # Key lags
        for lag in [1, 2, 7, 14]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Same day last week ratio
        df['lag_7_ratio'] = df[target_col].shift(1) / df[target_col].shift(8).replace(0, 1)
        
        # Rolling statistics (most important)
        for window in [7, 14]:
            df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).std()
        
        # EWM (exponential weighted) - captures recent trends
        df['ewm_7'] = df[target_col].shift(1).ewm(span=7, min_periods=1).mean()
        
        # Trend
        df['trend'] = df['rolling_mean_7'] - df['rolling_mean_14']
        
        return df
    
    def _get_feature_cols(self) -> List[str]:
        """Get optimized feature columns."""
        return [
            'day_of_week', 'day_of_month', 'week_of_year', 'month',
            'is_weekend', 'is_friday', 'is_monday',
            'day_sin', 'day_cos',
            'lag_1', 'lag_2', 'lag_7', 'lag_14',
            'lag_7_ratio',
            'rolling_mean_7', 'rolling_mean_14',
            'rolling_std_7', 'rolling_std_14',
            'ewm_7', 'trend'
        ]
    
    def fit(self, daily_data: pd.DataFrame, target_col: str = 'order_count'):
        """Train stacked XGBoost + GBR model."""
        
        # Filter recent data
        df = self._filter_recent_data(daily_data)
        print(f"   Training on {len(df)} days")
        
        df = self._create_features(df)
        df = self._add_lag_features(df, target_col)
        df = df.dropna()
        
        self._feature_cols = [c for c in self._get_feature_cols() if c in df.columns]
        X = df[self._feature_cols].fillna(0)
        y = df[target_col]
        
        # Replace infinities
        X = X.replace([np.inf, -np.inf], 0)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Stacking ensemble
        estimators = [
            ('xgb', xgb.XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )),
            ('gbr', GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                random_state=42
            ))
        ]
        
        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=RidgeCV(),
            cv=5,
            passthrough=True,
            n_jobs=-1
        )
        
        print("   Training stacked XGB+GBR model...")
        self.model.fit(X_scaled, y)
        
        self._last_data = df
        self.is_fitted = True
        
        # Validate on last 20%
        self._validate(df, target_col)
        
        return self
    
    def _validate(self, df: pd.DataFrame, target_col: str):
        """Validate on held-out portion."""
        val_size = int(len(df) * 0.2)
        train = df.iloc[:-val_size]
        val = df.iloc[-val_size:]
        
        X_val = val[self._feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_val_scaled = self.scaler.transform(X_val)
        y_val = val[target_col].values
        
        y_pred = self.model.predict(X_val_scaled)
        y_pred = np.maximum(y_pred, 0)
        
        mask = y_val != 0
        self.validation_mape = np.mean(np.abs((y_val[mask] - y_pred[mask]) / y_val[mask])) * 100
        print(f"   Validation MAPE: {self.validation_mape:.1f}%")
    
    def predict(self, horizon_days: int = 7) -> pd.DataFrame:
        """Generate forecast."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        last_date = self._last_data['date'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(horizon_days)]
        
        predictions = []
        current_data = self._last_data.copy()
        
        for future_date in future_dates:
            new_row = pd.DataFrame({'date': [future_date], 'order_count': [np.nan]})
            temp_df = pd.concat([current_data, new_row], ignore_index=True)
            temp_df = self._create_features(temp_df)
            temp_df = self._add_lag_features(temp_df, 'order_count')
            
            X_pred = temp_df[self._feature_cols].iloc[-1:].fillna(0).replace([np.inf, -np.inf], 0)
            X_pred_scaled = self.scaler.transform(X_pred)
            
            pred = max(0, round(self.model.predict(X_pred_scaled)[0]))
            predictions.append({'date': future_date, 'predicted_demand': pred})
            
            current_data = pd.concat([
                current_data,
                pd.DataFrame({'date': [future_date], 'order_count': [pred]})
            ], ignore_index=True)
        
        return pd.DataFrame(predictions)
    
    def cross_validate(self, daily_data: pd.DataFrame, target_col: str = 'order_count', n_splits: int = 5) -> Dict:
        """Detailed cross-validation."""
        df = self._filter_recent_data(daily_data)
        df = self._create_features(df)
        df = self._add_lag_features(df, target_col)
        df = df.dropna()
        
        feature_cols = [c for c in self._get_feature_cols() if c in df.columns]
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = df[target_col]
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mapes = []
        rmses = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train XGBoost (faster for CV)
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred = np.maximum(y_pred, 0)
            
            mask = y_test != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
                mapes.append(mape)
            
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            rmses.append(rmse)
            
            print(f"   Fold {fold+1}: MAPE={mape:.1f}%, RMSE={rmse:.0f}")
        
        return {
            'mape_mean': np.mean(mapes),
            'mape_std': np.std(mapes),
            'rmse_mean': np.mean(rmses),
            'mapes': mapes
        }


def run_v2_optimization():
    """Run V2 optimization."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from services.data_pipeline import DataPipeline
    
    print("=" * 70)
    print("🚀 EAGLEEYE MODEL OPTIMIZATION V2")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading data...")
    pipeline = DataPipeline(data_dir="../Data")
    pipeline.load_all()
    pipeline.load_core_tables()
    daily_demand = pipeline.get_daily_demand()
    
    best_mape = float('inf')
    best_config = None
    results_log = []
    
    # Test different window sizes
    for recent_days in [60, 90, 120, 150, 180]:
        print(f"\n{'='*50}")
        print(f"Testing: {recent_days} days")
        print('='*50)
        
        forecaster = OptimizedForecaster(recent_days=recent_days)
        
        print("\n📊 Cross-validation:")
        cv_results = forecaster.cross_validate(daily_demand)
        
        print(f"\n   Mean MAPE: {cv_results['mape_mean']:.1f}% ± {cv_results['mape_std']:.1f}%")
        
        results_log.append({
            'recent_days': recent_days,
            'mape_mean': cv_results['mape_mean'],
            'mape_std': cv_results['mape_std']
        })
        
        if cv_results['mape_mean'] < best_mape:
            best_mape = cv_results['mape_mean']
            best_config = {'recent_days': recent_days}
    
    print("\n" + "=" * 70)
    print("📊 ALL RESULTS")
    print("=" * 70)
    for r in results_log:
        marker = "🏆" if r['recent_days'] == best_config['recent_days'] else "  "
        print(f"{marker} {r['recent_days']} days: MAPE = {r['mape_mean']:.1f}% ± {r['mape_std']:.1f}%")
    
    print("\n" + "=" * 70)
    print(f"🏆 BEST: {best_config['recent_days']} days with MAPE = {best_mape:.1f}%")
    print("=" * 70)
    
    # Final training
    print("\n🎯 Final model training...")
    final_forecaster = OptimizedForecaster(**best_config)
    final_forecaster.fit(daily_demand)
    
    print("\n📈 7-Day Forecast:")
    forecast = final_forecaster.predict(7)
    for _, row in forecast.iterrows():
        print(f"   {row['date'].strftime('%Y-%m-%d %A'):25s}: {int(row['predicted_demand']):>6,} orders")
    
    return final_forecaster, best_config, best_mape


if __name__ == "__main__":
    run_v2_optimization()
