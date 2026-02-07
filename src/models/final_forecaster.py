"""
EagleEye Final Optimized Forecaster V3
Maximum accuracy through hyperparameter tuning and grid search.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


class FinalForecaster:
    """
    Final production forecaster with maximum accuracy.
    """
    
    def __init__(self, recent_days: int = 60, xgb_params: Dict = None):
        self.recent_days = recent_days
        self.xgb_params = xgb_params or {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbosity': 0
        }
        self.model = None
        self.scaler = RobustScaler()
        self.is_fitted = False
        self._feature_cols = []
        self._last_data = None
        self.validation_mape = None
    
    def _filter_recent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        cutoff = df['date'].max() - timedelta(days=self.recent_days)
        return df[df['date'] >= cutoff].reset_index(drop=True)
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Time
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        
        # Binary
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_sunday'] = (df['day_of_week'] == 6).astype(int)
        
        # Cyclical
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, target_col: str = 'order_count') -> pd.DataFrame:
        df = df.copy()
        
        # Lags
        for lag in [1, 2, 7, 14]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Same day patterns
        df['lag_7_ratio'] = df[target_col].shift(1) / df[target_col].shift(8).replace(0, 1)
        df['lag_14_ratio'] = df[target_col].shift(1) / df[target_col].shift(15).replace(0, 1)
        
        # Rolling
        for window in [7, 14]:
            df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).std()
        
        # EWM
        df['ewm_7'] = df[target_col].shift(1).ewm(span=7, min_periods=1).mean()
        df['ewm_3'] = df[target_col].shift(1).ewm(span=3, min_periods=1).mean()
        
        # Trend
        df['trend'] = df['rolling_mean_7'] - df['rolling_mean_14']
        
        return df
    
    def _get_feature_cols(self) -> List[str]:
        return [
            'day_of_week', 'day_of_month', 'week_of_year',
            'is_weekend', 'is_friday', 'is_monday', 'is_sunday',
            'day_sin', 'day_cos',
            'lag_1', 'lag_2', 'lag_7', 'lag_14',
            'lag_7_ratio', 'lag_14_ratio',
            'rolling_mean_7', 'rolling_mean_14',
            'rolling_std_7', 'rolling_std_14',
            'ewm_7', 'ewm_3', 'trend'
        ]
    
    def fit(self, daily_data: pd.DataFrame, target_col: str = 'order_count'):
        df = self._filter_recent_data(daily_data)
        df = self._create_features(df)
        df = self._add_lag_features(df, target_col)
        df = df.dropna()
        
        self._feature_cols = [c for c in self._get_feature_cols() if c in df.columns]
        X = df[self._feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = df[target_col]
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(X_scaled, y)
        
        self._last_data = df
        self.is_fitted = True
        
        return self
    
    def predict(self, horizon_days: int = 7) -> pd.DataFrame:
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
    
    def cross_validate(self, daily_data: pd.DataFrame, target_col: str = 'order_count') -> Dict:
        df = self._filter_recent_data(daily_data)
        df = self._create_features(df)
        df = self._add_lag_features(df, target_col)
        df = df.dropna()
        
        feature_cols = [c for c in self._get_feature_cols() if c in df.columns]
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = df[target_col]
        
        tscv = TimeSeriesSplit(n_splits=5)
        mapes = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = xgb.XGBRegressor(**self.xgb_params)
            model.fit(X_train_scaled, y_train)
            y_pred = np.maximum(model.predict(X_test_scaled), 0)
            
            mask = y_test != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
                mapes.append(mape)
        
        return {'mape_mean': np.mean(mapes), 'mape_std': np.std(mapes), 'mapes': mapes}


def run_hyperparameter_search():
    """Grid search for best hyperparameters."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from services.data_pipeline import DataPipeline
    
    print("=" * 70)
    print("🔬 EAGLEEYE HYPERPARAMETER OPTIMIZATION V3")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading data...")
    pipeline = DataPipeline(data_dir="../Data")
    pipeline.load_all()
    pipeline.load_core_tables()
    daily_demand = pipeline.get_daily_demand()
    
    best_mape = float('inf')
    best_config = None
    results = []
    
    # Grid search
    param_grid = [
        # Window size variations
        {'recent_days': 45, 'xgb_params': {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05}},
        {'recent_days': 60, 'xgb_params': {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05}},
        {'recent_days': 60, 'xgb_params': {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.03}},
        {'recent_days': 60, 'xgb_params': {'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.08}},
        {'recent_days': 75, 'xgb_params': {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05}},
        # Higher regularization
        {'recent_days': 60, 'xgb_params': {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05, 'reg_alpha': 0.5, 'reg_lambda': 2.0}},
    ]
    
    for i, config in enumerate(param_grid):
        print(f"\n{'='*50}")
        print(f"Config {i+1}/{len(param_grid)}: {config['recent_days']} days, depth={config['xgb_params'].get('max_depth', 4)}, lr={config['xgb_params'].get('learning_rate', 0.05)}")
        print('='*50)
        
        xgb_params = {
            'n_estimators': config['xgb_params'].get('n_estimators', 200),
            'max_depth': config['xgb_params'].get('max_depth', 4),
            'learning_rate': config['xgb_params'].get('learning_rate', 0.05),
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 3,
            'reg_alpha': config['xgb_params'].get('reg_alpha', 0.1),
            'reg_lambda': config['xgb_params'].get('reg_lambda', 1.0),
            'random_state': 42,
            'verbosity': 0
        }
        
        forecaster = FinalForecaster(recent_days=config['recent_days'], xgb_params=xgb_params)
        cv = forecaster.cross_validate(daily_demand)
        
        print(f"   MAPE: {cv['mape_mean']:.2f}% ± {cv['mape_std']:.2f}%")
        print(f"   Folds: {[f'{m:.1f}%' for m in cv['mapes']]}")
        
        results.append({
            'config': config,
            'mape_mean': cv['mape_mean'],
            'mape_std': cv['mape_std']
        })
        
        if cv['mape_mean'] < best_mape:
            best_mape = cv['mape_mean']
            best_config = config
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    
    for r in sorted(results, key=lambda x: x['mape_mean']):
        marker = "🏆" if r['config'] == best_config else "  "
        print(f"{marker} MAPE={r['mape_mean']:.2f}% ± {r['mape_std']:.1f}% | days={r['config']['recent_days']}, depth={r['config']['xgb_params'].get('max_depth', 4)}")
    
    print("\n" + "=" * 70)
    print(f"🏆 BEST CONFIGURATION")
    print(f"   Days: {best_config['recent_days']}")
    print(f"   Params: {best_config['xgb_params']}")
    print(f"   MAPE: {best_mape:.2f}%")
    print("=" * 70)
    
    # Final model
    print("\n🎯 Training final model...")
    final_xgb_params = {
        'n_estimators': best_config['xgb_params'].get('n_estimators', 200),
        'max_depth': best_config['xgb_params'].get('max_depth', 4),
        'learning_rate': best_config['xgb_params'].get('learning_rate', 0.05),
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_weight': 3,
        'reg_alpha': best_config['xgb_params'].get('reg_alpha', 0.1),
        'reg_lambda': best_config['xgb_params'].get('reg_lambda', 1.0),
        'random_state': 42,
        'verbosity': 0
    }
    
    final = FinalForecaster(recent_days=best_config['recent_days'], xgb_params=final_xgb_params)
    final.fit(daily_demand)
    
    print("\n📈 7-Day Forecast:")
    forecast = final.predict(7)
    for _, row in forecast.iterrows():
        print(f"   {row['date'].strftime('%Y-%m-%d %A'):25s}: {int(row['predicted_demand']):>6,} orders")
    
    # Get feature importance
    print("\n📊 Top Features:")
    importance = pd.DataFrame({
        'feature': final._feature_cols,
        'importance': final.model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.head(10).iterrows():
        bar = "█" * int(row['importance'] * 40)
        print(f"   {row['feature']:20s}: {bar}")
    
    return final, best_config, best_mape


if __name__ == "__main__":
    run_hyperparameter_search()
