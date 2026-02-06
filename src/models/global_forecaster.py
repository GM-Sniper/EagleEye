"""
EagleEye Global Forecaster
Trains a single XGBoost model across ALL items to learn shared temporal patterns.
Solves the "Cold Start" / "Low Volume" problem by transfer learning from high-volume items.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sklearn.preprocessing import RobustScaler, LabelEncoder
import xgboost as xgb
import joblib
import warnings

warnings.filterwarnings('ignore')

class GlobalForecaster:
    """
    Global Time-Series Model (Many-to-One).
    Trains on stacked item history to learn generalizable demand patterns.
    """
    
    OPTIMAL_DAYS = 90  # Needs more history than local models to learn cross-patterns
    
    XGB_PARAMS = {
        'n_estimators': 500,    # More trees for more complex data
        'max_depth': 6,         # Deeper trees to capture item-specific nuances
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror', # Robust for general regression
        'eval_metric': 'mae',
        'n_jobs': -1,
        'random_state': 42
    }
    
    def __init__(self, recent_days: int = 90):
        self.recent_days = recent_days
        self.model = xgb.XGBRegressor(**self.XGB_PARAMS)
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self._feature_cols = []
        self._last_data = None
        self.bias_factors = {} # Per-item bias correction
        
    def _create_features(self, df: pd.DataFrame, target: str = 'order_count') -> pd.DataFrame:
        """Create features with strict GroupBy handling to prevent leakage."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. Global Time Features (Shared)
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # 2. Item-Specific Features (Grouped)
        # We must group by item_id for any lag/rolling operation
        grouped = df.groupby('item_id')[target]
        
        # Lags (Log-space) - shift per group
        for lag in [1, 2, 7, 14, 21, 28]:
            df[f'lag_{lag}'] = df.groupby('item_id')[target].shift(lag)
            df[f'log_lag_{lag}'] = np.log1p(df[f'lag_{lag}'])
        
        # Rolling Stats
        for window in [7, 14, 28]:
            # Use transform to ensure calculations stay within item groups and alignment is preserved
            df[f'rolling_mean_{window}'] = df.groupby('item_id')[target].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby('item_id')[target].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
            )
            
        # 3. Item Metadata Features (The "Global" magic)
        # Relative price or volume tier can be added here if available
        # For now, we rely on the item_id embedding via label encoding
            
        return df

    def prepare_data(self, df: pd.DataFrame, target: str = 'order_count') -> pd.DataFrame:
        """Filter and prepare stacked dataframe."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter deep history
        min_date = df['date'].max() - timedelta(days=self.recent_days + 30) # +30 for lags
        df = df[df['date'] >= min_date].copy()
        
        # Encode Item IDs
        if 'item_id_encoded' not in df.columns:
            # Fit or transform depending on training state
            if not self.is_fitted:
                 df['item_id_encoded'] = self.label_encoder.fit_transform(df['item_id'])
            else:
                 # Handle new items safely
                 known_items = set(self.label_encoder.classes_)
                 df = df[df['item_id'].isin(known_items)].copy()
                 df['item_id_encoded'] = self.label_encoder.transform(df['item_id'])
                 
        return df

    def fit(self, stack_data: pd.DataFrame, target: str = 'order_count'):
        """Train global model on stacked data."""
        print(f"DEBUG: fit() called with {len(stack_data)} rows. Columns: {stack_data.columns}")
        df = self.prepare_data(stack_data, target)
        print(f"DEBUG: After prepare_data: {len(df)} rows. Date Range: {df['date'].min()} to {df['date'].max()}")
        
        # Feature Engineering
        df = self._create_features(df, target)
        print(f"DEBUG: After create_features: {len(df)} rows")
        
        # Log-Transform Target
        df['target_log'] = np.log1p(df[target])
        
        # Define Features (moved up)
        self._feature_cols = [
            'item_id_encoded',
            'day_sin', 'day_cos', 'is_weekend',
            'log_lag_1', 'log_lag_2', 'log_lag_7', 'log_lag_14',
            'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7'
        ]
        
        print("DEBUG: NaN counts per feature:")
        print(df[self._feature_cols].isna().sum())
        
        # Drop NaNs created by lags
        df = df.dropna(subset=self._feature_cols + ['target_log'])
        print(f"DEBUG: After dropna: {len(df)} rows")
        
        if len(df) == 0:
             print("DEBUG: Head of dataframe before drop:")
             print(df.head())
             raise ValueError("Training data is empty after creating features and dropping NaNs. Check date range vs lags.")

        X = df[self._feature_cols]
        y = df['target_log']
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self._last_data = stack_data # Save for recursive prediction
        
        # Calculate Per-Item Bias Factors (Calibration)
        # Predict on training set
        preds_log = self.model.predict(X_scaled)
        df['pred'] = np.expm1(preds_log)
        
        # Median bias per item
        df['bias'] = df[target] / np.maximum(df['pred'], 1)
        bias_factors = df.groupby('item_id')['bias'].median()
        self.bias_factors = bias_factors.to_dict()
        
        print(f"✅ Global Model Fitted on {len(df)} rows across {df['item_id'].nunique()} items")
        return self

    def predict(self, item_id: int, horizon_days: int = 14) -> pd.DataFrame:
        """Recursive forecast for a SPECIFIC item using the global model."""
        if not self.is_fitted:
             raise ValueError("Model not fitted")
             
        # Get history for this item
        item_history = self._last_data[self._last_data['item_id'] == item_id].copy()
        if item_history.empty:
            return pd.DataFrame()
            
        last_date = pd.to_datetime(item_history['date'].max())
        future_dates = [last_date + timedelta(days=i+1) for i in range(horizon_days)]
        
        encoded_id = self.label_encoder.transform([item_id])[0]
        bias = self.bias_factors.get(item_id, 1.0)
        bias = np.clip(bias, 0.8, 1.2) # Safety clip
        
        predictions = []
        current_data = item_history.copy()
        
        for future_date in future_dates:
            # Append placeholder row
            new_row = pd.DataFrame({
                'date': [future_date],
                'item_id': [item_id],
                'order_count': [np.nan]
            })
            temp_df = pd.concat([current_data, new_row], ignore_index=True)
            
            # Re-generate features (this is expensive but necessary for recursion)
            # In production, we would optimize this to only update the tail
            temp_encoded = self.prepare_data(temp_df)
            temp_feat = self._create_features(temp_encoded)
            
            # Predict next step
            row = temp_feat.iloc[-1:]
            X_pred = row[self._feature_cols]
            X_scaled = self.scaler.transform(X_pred)
            
            y_log = self.model.predict(X_scaled)[0]
            pred = max(0, round(np.expm1(y_log) * bias))
            
            predictions.append({
                'date': future_date,
                'predicted_demand': pred
            })
            
            # Fill actual with prediction for next loop
            current_data = pd.concat([
                current_data,
                pd.DataFrame({'date': [future_date], 'item_id': [item_id], 'order_count': [pred]})
            ], ignore_index=True)
            
        return pd.DataFrame(predictions)
