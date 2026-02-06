"""
EagleEye Demand Forecaster
Time-series forecasting for inventory demand prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')


class DemandForecaster:
    """
    Demand forecasting engine using gradient boosting with time features.
    Optimized for restaurant/retail inventory forecasting.
    """
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importance: Dict[str, float] = {}
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-series features for forecasting."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Time features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        
        # Cyclical encoding for periodic features
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, target_col: str = 'order_count') -> pd.DataFrame:
        """Add lagged features for time series."""
        df = df.copy()
        
        # Lag features
        for lag in [1, 7, 14, 28]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling means
        for window in [7, 14, 28]:
            df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
        
        return df
    
    def fit(self, daily_data: pd.DataFrame, target_col: str = 'order_count'):
        """
        Train the forecasting model.
        
        Args:
            daily_data: DataFrame with 'date' and target column
            target_col: Name of the column to predict
        """
        df = self._create_features(daily_data)
        df = self._add_lag_features(df, target_col)
        df = df.dropna()
        
        feature_cols = [
            'day_of_week', 'day_of_month', 'week_of_year', 'month',
            'is_weekend', 'is_month_start', 'is_month_end',
            'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'lag_1', 'lag_7', 'lag_14', 'lag_28',
            'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
            'rolling_std_7', 'rolling_std_14', 'rolling_std_28'
        ]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Store feature importance
        self.feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
        self._feature_cols = feature_cols
        self._last_data = df
        
        return self
    
    def predict(self, horizon_days: int = 7) -> pd.DataFrame:
        """
        Generate demand forecast for the next N days.
        
        Args:
            horizon_days: Number of days to forecast
            
        Returns:
            DataFrame with date and predicted demand
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Start from last date in training data
        last_date = self._last_data['date'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(horizon_days)]
        
        predictions = []
        current_data = self._last_data.copy()
        
        for future_date in future_dates:
            # Create features for the future date
            new_row = pd.DataFrame({'date': [future_date], 'order_count': [np.nan]})
            temp_df = pd.concat([current_data, new_row], ignore_index=True)
            temp_df = self._create_features(temp_df)
            temp_df = self._add_lag_features(temp_df, 'order_count')
            
            # Get the last row (our prediction date)
            X_pred = temp_df[self._feature_cols].iloc[-1:].fillna(0)
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Predict
            pred = max(0, self.model.predict(X_pred_scaled)[0])
            predictions.append({'date': future_date, 'predicted_demand': round(pred)})
            
            # Add prediction to data for next iteration
            current_data = pd.concat([
                current_data, 
                pd.DataFrame({'date': [future_date], 'order_count': [pred]})
            ], ignore_index=True)
        
        return pd.DataFrame(predictions)
    
    def evaluate(self, daily_data: pd.DataFrame, target_col: str = 'order_count') -> Dict[str, float]:
        """Evaluate model using time series cross-validation."""
        df = self._create_features(daily_data)
        df = self._add_lag_features(df, target_col)
        df = df.dropna()
        
        feature_cols = self._feature_cols if hasattr(self, '_feature_cols') else [
            'day_of_week', 'day_of_month', 'week_of_year', 'month',
            'is_weekend', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_14'
        ]
        
        X = df[[c for c in feature_cols if c in df.columns]]
        y = df[target_col]
        
        tscv = TimeSeriesSplit(n_splits=5)
        mapes = []
        rmses = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # MAPE
            mask = y_test != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
                mapes.append(mape)
            
            # RMSE
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            rmses.append(rmse)
        
        return {
            'mape': np.mean(mapes) if mapes else 0,
            'rmse': np.mean(rmses),
            'cv_folds': len(rmses)
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as a sorted DataFrame."""
        if not self.feature_importance:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in self.feature_importance.items()
        ])
        return df.sort_values('importance', ascending=False)


class ItemDemandForecaster:
    """Forecaster for individual item demand."""
    
    def __init__(self):
        self.models: Dict[int, DemandForecaster] = {}
    
    def fit_item(self, item_id: int, daily_demand: pd.DataFrame):
        """Fit a forecaster for a specific item."""
        item_data = daily_demand[daily_demand['item_id'] == item_id].copy()
        if len(item_data) < 30:
            return None
        
        # Aggregate by date
        item_daily = item_data.groupby('date')['quantity'].sum().reset_index()
        item_daily.columns = ['date', 'order_count']
        
        forecaster = DemandForecaster()
        forecaster.fit(item_daily, 'order_count')
        self.models[item_id] = forecaster
        
        return forecaster
    
    def predict_item(self, item_id: int, horizon_days: int = 7) -> Optional[pd.DataFrame]:
        """Generate forecast for a specific item."""
        if item_id not in self.models:
            return None
        return self.models[item_id].predict(horizon_days)


if __name__ == "__main__":
    # Test the forecaster
    print("Testing DemandForecaster...")
    
    # Generate synthetic data
    dates = pd.date_range(start='2023-01-01', end='2024-01-31', freq='D')
    np.random.seed(42)
    
    # Simulate demand with weekly seasonality and trend
    base = 100
    trend = np.linspace(0, 50, len(dates))
    weekly = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    noise = np.random.normal(0, 15, len(dates))
    demand = base + trend + weekly + noise
    demand = np.maximum(demand, 0)
    
    df = pd.DataFrame({'date': dates, 'order_count': demand})
    
    # Fit and predict
    forecaster = DemandForecaster()
    forecaster.fit(df)
    
    forecast = forecaster.predict(horizon_days=7)
    print("\n7-Day Forecast:")
    print(forecast)
    
    # Evaluate
    metrics = forecaster.evaluate(df)
    print(f"\nModel Performance: MAPE={metrics['mape']:.1f}%, RMSE={metrics['rmse']:.1f}")
    
    # Feature importance
    print("\nTop 5 Features:")
    print(forecaster.get_feature_importance().head())
