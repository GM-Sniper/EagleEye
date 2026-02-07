"""
EagleEye Advanced Forecaster
Multi-model ensemble for maximum forecast accuracy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from prophet import Prophet
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').disabled = True


class AdvancedForecaster:
    """
    Advanced ensemble forecaster combining multiple models for best accuracy.
    """
    
    def __init__(self, recent_months: int = 6):
        """
        Args:
            recent_months: Only use last N months of data for training
        """
        self.recent_months = recent_months
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._feature_cols = []
        self._last_data = None
        self.best_model = None
        self.metrics_history = []
    
    def _filter_recent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to only recent months of data."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        cutoff = df['date'].max() - pd.DateOffset(months=self.recent_months)
        return df[df['date'] >= cutoff].reset_index(drop=True)
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive time-series features."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Basic time features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        
        # Cyclical encoding
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, target_col: str = 'order_count') -> pd.DataFrame:
        """Add comprehensive lag and rolling features."""
        df = df.copy()
        
        # Lag features (previous days)
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Same day last week, 2 weeks ago, 3 weeks ago
        df['lag_7_diff'] = df[target_col].shift(1) - df[target_col].shift(8)
        
        # Rolling means
        for window in [3, 7, 14, 21, 28]:
            df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).std()
            df[f'rolling_min_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).min()
            df[f'rolling_max_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).max()
        
        # Expanding mean (all history)
        df['expanding_mean'] = df[target_col].shift(1).expanding(min_periods=1).mean()
        
        # Trend features
        df['trend_7'] = df['rolling_mean_7'] - df['rolling_mean_14']
        df['trend_14'] = df['rolling_mean_14'] - df['rolling_mean_28']
        
        # Volatility
        df['volatility_7'] = df['rolling_std_7'] / df['rolling_mean_7'].replace(0, 1)
        
        return df
    
    def _get_feature_cols(self) -> List[str]:
        """Get list of feature columns."""
        return [
            # Time features
            'day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter',
            'is_weekend', 'is_friday', 'is_month_start', 'is_month_end',
            'day_sin', 'day_cos', 'month_sin', 'month_cos', 'week_sin', 'week_cos',
            # Lag features
            'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_21', 'lag_28',
            'lag_7_diff',
            # Rolling features
            'rolling_mean_3', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_21', 'rolling_mean_28',
            'rolling_std_3', 'rolling_std_7', 'rolling_std_14', 'rolling_std_21', 'rolling_std_28',
            'rolling_min_7', 'rolling_max_7',
            'expanding_mean',
            'trend_7', 'trend_14', 'volatility_7'
        ]
    
    def fit(self, daily_data: pd.DataFrame, target_col: str = 'order_count'):
        """Train all models in the ensemble."""
        
        # Filter to recent data
        df = self._filter_recent_data(daily_data)
        print(f"   Training on {len(df)} days of recent data")
        
        df = self._create_features(df)
        df = self._add_lag_features(df, target_col)
        df = df.dropna()
        
        self._feature_cols = [c for c in self._get_feature_cols() if c in df.columns]
        X = df[self._feature_cols].fillna(0)
        y = df[target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train multiple models
        print("   Training Gradient Boosting...")
        self.models['gbr'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.models['gbr'].fit(X_scaled, y)
        
        print("   Training XGBoost...")
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        self.models['xgb'].fit(X_scaled, y)
        
        print("   Training Random Forest...")
        self.models['rf'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.models['rf'].fit(X_scaled, y)
        
        print("   Training Ridge Regression...")
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['ridge'].fit(X_scaled, y)
        
        # Train Prophet
        print("   Training Prophet...")
        prophet_df = df[['date', target_col]].rename(columns={'date': 'ds', target_col: 'y'})
        self.models['prophet'] = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        self.models['prophet'].fit(prophet_df)
        
        self._last_data = df
        self.is_fitted = True
        
        # Calculate weights based on validation performance
        self._calculate_weights(df, target_col)
        
        return self
    
    def _calculate_weights(self, df: pd.DataFrame, target_col: str):
        """Calculate model weights based on validation performance."""
        # Use last 20% for validation
        val_size = int(len(df) * 0.2)
        train = df.iloc[:-val_size]
        val = df.iloc[-val_size:]
        
        X_val = val[self._feature_cols].fillna(0)
        X_val_scaled = self.scaler.transform(X_val)
        y_val = val[target_col].values
        
        errors = {}
        
        for name, model in self.models.items():
            if name == 'prophet':
                future = pd.DataFrame({'ds': val['date']})
                pred = model.predict(future)['yhat'].values
            else:
                pred = model.predict(X_val_scaled)
            
            pred = np.maximum(pred, 0)
            mape = np.mean(np.abs((y_val - pred) / np.maximum(y_val, 1))) * 100
            errors[name] = mape
            print(f"      {name}: MAPE = {mape:.1f}%")
        
        # Convert errors to weights (lower error = higher weight)
        total_inv_error = sum(1/e for e in errors.values())
        self.weights = {name: (1/e) / total_inv_error for name, e in errors.items()}
        
        # Find best model
        self.best_model = min(errors, key=errors.get)
        print(f"\n   Best model: {self.best_model} (MAPE: {errors[self.best_model]:.1f}%)")
        print(f"   Ensemble weights: {', '.join(f'{k}:{v:.2f}' for k,v in self.weights.items())}")
    
    def predict(self, horizon_days: int = 7, method: str = 'ensemble') -> pd.DataFrame:
        """
        Generate forecast.
        
        Args:
            horizon_days: Number of days to forecast
            method: 'ensemble', 'best', or specific model name
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        last_date = self._last_data['date'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(horizon_days)]
        
        predictions = []
        current_data = self._last_data.copy()
        
        for future_date in future_dates:
            # Create features for future date
            new_row = pd.DataFrame({'date': [future_date], 'order_count': [np.nan]})
            temp_df = pd.concat([current_data, new_row], ignore_index=True)
            temp_df = self._create_features(temp_df)
            temp_df = self._add_lag_features(temp_df, 'order_count')
            
            X_pred = temp_df[self._feature_cols].iloc[-1:].fillna(0)
            X_pred_scaled = self.scaler.transform(X_pred)
            
            if method == 'ensemble':
                # Weighted ensemble prediction
                pred = 0
                for name, model in self.models.items():
                    if name == 'prophet':
                        future_df = pd.DataFrame({'ds': [future_date]})
                        model_pred = model.predict(future_df)['yhat'].values[0]
                    else:
                        model_pred = model.predict(X_pred_scaled)[0]
                    pred += self.weights[name] * max(0, model_pred)
            elif method == 'best':
                model = self.models[self.best_model]
                if self.best_model == 'prophet':
                    future_df = pd.DataFrame({'ds': [future_date]})
                    pred = model.predict(future_df)['yhat'].values[0]
                else:
                    pred = model.predict(X_pred_scaled)[0]
            else:
                model = self.models[method]
                if method == 'prophet':
                    future_df = pd.DataFrame({'ds': [future_date]})
                    pred = model.predict(future_df)['yhat'].values[0]
                else:
                    pred = model.predict(X_pred_scaled)[0]
            
            pred = max(0, round(pred))
            predictions.append({'date': future_date, 'predicted_demand': pred})
            
            # Add prediction to data for next iteration
            current_data = pd.concat([
                current_data,
                pd.DataFrame({'date': [future_date], 'order_count': [pred]})
            ], ignore_index=True)
        
        return pd.DataFrame(predictions)
    
    def evaluate(self, daily_data: pd.DataFrame, target_col: str = 'order_count') -> Dict[str, float]:
        """Comprehensive model evaluation with cross-validation."""
        df = self._filter_recent_data(daily_data)
        df = self._create_features(df)
        df = self._add_lag_features(df, target_col)
        df = df.dropna()
        
        feature_cols = [c for c in self._get_feature_cols() if c in df.columns]
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        tscv = TimeSeriesSplit(n_splits=5)
        results = {name: [] for name in ['gbr', 'xgb', 'rf', 'ridge', 'prophet', 'ensemble']}
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            gbr = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            gbr.fit(X_train_scaled, y_train)
            
            xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
            xgb_model.fit(X_train_scaled, y_train)
            
            rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_scaled, y_train)
            
            # Prophet
            prophet_train = df.iloc[train_idx][['date', target_col]].rename(columns={'date': 'ds', target_col: 'y'})
            prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            prophet.fit(prophet_train)
            
            # Predictions
            preds = {
                'gbr': gbr.predict(X_test_scaled),
                'xgb': xgb_model.predict(X_test_scaled),
                'rf': rf.predict(X_test_scaled),
                'ridge': ridge.predict(X_test_scaled),
                'prophet': prophet.predict(pd.DataFrame({'ds': df.iloc[test_idx]['date']}))['yhat'].values
            }
            
            # Ensemble (equal weights for CV)
            preds['ensemble'] = np.mean([preds['gbr'], preds['xgb'], preds['rf'], preds['prophet']], axis=0)
            
            # Calculate MAPE for each
            for name, pred in preds.items():
                pred = np.maximum(pred, 0)
                mask = y_test != 0
                if mask.sum() > 0:
                    mape = np.mean(np.abs((y_test[mask] - pred[mask]) / y_test[mask])) * 100
                    results[name].append(mape)
        
        # Average across folds
        final_results = {name: np.mean(mapes) for name, mapes in results.items()}
        return final_results


def run_optimization():
    """Run iterative optimization to find best model configuration."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from services.data_pipeline import DataPipeline
    
    print("=" * 70)
    print("🚀 EAGLEEYE MODEL OPTIMIZATION")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading data...")
    pipeline = DataPipeline(data_dir="../Data")
    pipeline.load_all()
    pipeline.load_core_tables()
    daily_demand = pipeline.get_daily_demand()
    
    best_mape = float('inf')
    best_config = None
    
    # Try different configurations
    configs = [
        {'recent_months': 3},
        {'recent_months': 6},
        {'recent_months': 9},
        {'recent_months': 12},
    ]
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Testing config: {config}")
        print('='*50)
        
        forecaster = AdvancedForecaster(**config)
        forecaster.fit(daily_demand)
        
        # Evaluate
        print("\n📊 Cross-validation results:")
        cv_results = forecaster.evaluate(daily_demand)
        for model, mape in sorted(cv_results.items(), key=lambda x: x[1]):
            print(f"   {model:10s}: MAPE = {mape:.1f}%")
        
        ensemble_mape = cv_results['ensemble']
        if ensemble_mape < best_mape:
            best_mape = ensemble_mape
            best_config = config
    
    print("\n" + "=" * 70)
    print(f"🏆 BEST CONFIGURATION: {best_config}")
    print(f"   Ensemble MAPE: {best_mape:.1f}%")
    print("=" * 70)
    
    # Final training with best config
    print("\n🎯 Final training with best configuration...")
    final_forecaster = AdvancedForecaster(**best_config)
    final_forecaster.fit(daily_demand)
    
    # Generate forecast
    print("\n📈 7-Day Forecast:")
    forecast = final_forecaster.predict(7, method='ensemble')
    for _, row in forecast.iterrows():
        print(f"   {row['date'].strftime('%Y-%m-%d')}: {int(row['predicted_demand']):,} orders")
    
    return final_forecaster, best_config, best_mape


if __name__ == "__main__":
    run_optimization()
