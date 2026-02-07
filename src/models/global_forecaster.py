"""EagleEye Global Forecaster.

Trains a single XGBoost model across ALL items to learn shared temporal patterns.
Designed to work well for low-volume items by transferring signal from the full portfolio.

Notes on accuracy
- For time-series features, calendar-day lags (t-7 days) are only correct when the
    underlying series is dense daily. This implementation can optionally resample
    each item to a complete daily index and fill missing days with 0 to avoid
    "row-lag" leakage/shift errors.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Optional
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
        # Slightly over-provision estimators and rely on early stopping.
        'n_estimators': 1500,
        'max_depth': 6,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        # XGBoost 3.x sklearn API: early stopping is configured on the estimator
        # (and triggered by providing eval_set in fit()).
        'early_stopping_rounds': 50,
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0,
    }
    
    def __init__(
        self,
        recent_days: int = 90,
        *,
        fill_missing_days: bool = True,
        max_resampled_rows: int = 2_000_000,
        verbose: bool = False,
    ):
        self.recent_days = recent_days
        self.fill_missing_days = fill_missing_days
        self.max_resampled_rows = int(max_resampled_rows)
        self.verbose = verbose
        self.model = xgb.XGBRegressor(**self.XGB_PARAMS)
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self._feature_cols = []
        self._last_data = None
        self.bias_factors = {} # Per-item bias correction
        self.item_static_features: Optional[pd.DataFrame] = None
        self.global_dow_mean: Dict[int, float] = {}

    def _log(self, message: str):
        if self.verbose:
            print(message)

    @staticmethod
    def _normalize_and_aggregate(
        df: pd.DataFrame,
        *,
        target: str,
    ) -> pd.DataFrame:
        """Ensure one row per (item_id, date) with normalized date."""
        out = df.copy()
        out['date'] = pd.to_datetime(out['date']).dt.normalize()
        out[target] = pd.to_numeric(out[target], errors='coerce').fillna(0.0)
        out['item_id'] = pd.to_numeric(out['item_id'], errors='coerce')
        out = out.dropna(subset=['item_id']).copy()
        out['item_id'] = out['item_id'].astype(int)
        out = (
            out.groupby(['item_id', 'date'], as_index=False)[target]
            .sum()
            .sort_values(['item_id', 'date'])
            .reset_index(drop=True)
        )
        return out

    @staticmethod
    def _estimate_resampled_rows(df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        ranges = df.groupby('item_id')['date'].agg(['min', 'max'])
        days = (ranges['max'] - ranges['min']).dt.days + 1
        days = days.clip(lower=0).fillna(0)
        return int(days.sum())

    def _resample_to_daily(self, df: pd.DataFrame, *, target: str) -> pd.DataFrame:
        """Resample each item to a dense daily series; missing days -> 0."""
        if df.empty:
            return df

        estimate = self._estimate_resampled_rows(df)
        if estimate > self.max_resampled_rows:
            self._log(
                f"⚠️ Skipping daily resample: would create ~{estimate:,} rows "
                f"(limit {self.max_resampled_rows:,})."
            )
            return df

        frames: list[pd.DataFrame] = []
        for item_id, g in df.groupby('item_id', sort=False):
            start = g['date'].min()
            end = g['date'].max()
            idx = pd.date_range(start=start, end=end, freq='D')
            g2 = g.set_index('date').reindex(idx)
            g2.index.name = 'date'
            g2 = g2.reset_index()
            g2['item_id'] = int(item_id)
            g2[target] = g2[target].fillna(0.0)
            frames.append(g2[['item_id', 'date', target]])

        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(['item_id', 'date']).reset_index(drop=True)
        return out

    def _build_item_static_features(self, df: pd.DataFrame, *, target: str) -> pd.DataFrame:
        """Per-item static signals that help the global model generalize."""
        feats = (
            df.groupby('item_id')[target]
            .agg(total_vol='sum', mean_daily='mean', std_daily='std')
            .reset_index()
        )
        feats['std_daily'] = feats['std_daily'].fillna(0.0)
        feats['cv'] = (feats['std_daily'] / feats['mean_daily'].replace(0, np.nan)).fillna(0.0)
        feats['log_total_vol'] = np.log1p(feats['total_vol'].fillna(0.0))
        feats['log_mean_daily'] = np.log1p(feats['mean_daily'].fillna(0.0))
        return feats
        
    def _create_features(self, df: pd.DataFrame, target: str = 'order_count') -> pd.DataFrame:
        """Create features with strict GroupBy handling to prevent leakage."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        
        # 1. Global Time Features (Shared)
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # 2. Item-Specific Features (Grouped)
        # We must group by item_id for any lag/rolling operation
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

        # If static per-item features are available, merge them in (no leakage: they are
        # computed from the training window only in fit() and reused in predict()).
        if self.item_static_features is not None:
            df = df.merge(self.item_static_features, on='item_id', how='left')
            for col in ['total_vol', 'mean_daily', 'std_daily', 'cv', 'log_total_vol', 'log_mean_daily']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
        return df

    def prepare_data(self, df: pd.DataFrame, target: str = 'order_count') -> pd.DataFrame:
        """Filter and prepare stacked dataframe."""
        df = df.copy()
        df = self._normalize_and_aggregate(df, target=target)
        
        # Filter deep history
        min_date = df['date'].max() - timedelta(days=self.recent_days + 30)  # +30 for lags
        df = df[df['date'] >= min_date].copy()

        # Optional: make each item series dense daily so lag_7 = t-7 days (not t-7 rows).
        if self.fill_missing_days:
            df = self._resample_to_daily(df, target=target)
        
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
        self._log(f"fit(): {len(stack_data):,} input rows")
        df = self.prepare_data(stack_data, target)
        self._log(f"After prepare_data: {len(df):,} rows; date {df['date'].min()} → {df['date'].max()}")

        # Static item features (computed once on training window; reused in predict)
        self.item_static_features = self._build_item_static_features(df, target=target)

        # Store a simple seasonal fallback for cold-start/unseen items
        tmp = df.copy()
        tmp['day_of_week'] = tmp['date'].dt.dayofweek
        self.global_dow_mean = (
            tmp.groupby('day_of_week')[target].mean().fillna(0.0).to_dict()
        )
        
        # Feature Engineering
        df = self._create_features(df, target)
        self._log(f"After create_features: {len(df):,} rows")
        
        # Log-Transform Target
        df['target_log'] = np.log1p(df[target])
        
        # Define Features
        self._feature_cols = [
            'item_id_encoded',
            'day_sin', 'day_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos', 'is_weekend',
            'log_lag_1', 'log_lag_2', 'log_lag_7', 'log_lag_14',
            'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7',
            # Static per-item signals
            'log_total_vol', 'log_mean_daily', 'cv'
        ]

        for c in self._feature_cols:
            if c not in df.columns:
                df[c] = 0.0
        
        if self.verbose:
            self._log("NaN counts per feature:")
            self._log(str(df[self._feature_cols].isna().sum()))
        
        # Drop NaNs created by lags
        df = df.dropna(subset=self._feature_cols + ['target_log'])
        self._log(f"After dropna: {len(df):,} rows")
        
        if len(df) == 0:
            raise ValueError(
                "Training data is empty after feature engineering. "
                "Check date coverage vs lag windows / resampling settings."
            )

        X = df[self._feature_cols]
        y = df['target_log']
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Train with a time-based holdout for early stopping.
        # This is a pragmatic approach for stacked series.
        df_sorted = df.sort_values('date')
        cutoff = df_sorted['date'].quantile(0.9)
        train_mask = df_sorted['date'] <= cutoff

        X_train = df_sorted.loc[train_mask, self._feature_cols]
        y_train = df_sorted.loc[train_mask, 'target_log']
        X_val = df_sorted.loc[~train_mask, self._feature_cols]
        y_val = df_sorted.loc[~train_mask, 'target_log']

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if len(X_val) else None

        if X_val_scaled is not None and len(X_val_scaled) > 0:
            self.model.fit(
                X_train_scaled,
                y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train_scaled, y_train)

        self.is_fitted = True
        self._last_data = df[['item_id', 'date', target]].copy()  # Save prepared history
        
        # Calculate Per-Item Bias Factors (Calibration)
        # Predict on training set
        # Calibrate per-item bias on the full prepared set (more stable with more rows)
        X_all_scaled = self.scaler.transform(df_sorted[self._feature_cols])
        preds_log = self.model.predict(X_all_scaled)
        df['pred'] = np.expm1(preds_log)
        
        # Median bias per item
        df['bias'] = df[target] / np.maximum(df['pred'], 1)
        bias_factors = df.groupby('item_id')['bias'].median()
        self.bias_factors = bias_factors.to_dict()
        
        self._log(f"✅ Global Model fitted on {len(df):,} rows across {df['item_id'].nunique():,} items")
        return self

    def predict(self, item_id: int, horizon_days: int = 14) -> pd.DataFrame:
        """Recursive forecast for a SPECIFIC item using the global model."""
        if not self.is_fitted:
             raise ValueError("Model not fitted")

        # Cold-start / unseen item_id fallback
        if item_id not in set(self.label_encoder.classes_):
            return self._seasonal_naive_forecast(horizon_days)
             
        # Get history for this item
        item_history = self._last_data[self._last_data['item_id'] == item_id].copy()
        if item_history.empty:
            return self._seasonal_naive_forecast(horizon_days)
            
        last_date = pd.to_datetime(item_history['date'].max())
        future_dates = [last_date + timedelta(days=i+1) for i in range(horizon_days)]
        
        encoded_id = int(self.label_encoder.transform([item_id])[0])
        bias = self.bias_factors.get(item_id, 1.0)
        bias = np.clip(bias, 0.8, 1.2) # Safety clip
        
        predictions = []
        current_data = item_history.copy()
        # Ensure dense daily history for correct calendar lags during recursion
        if self.fill_missing_days:
            current_data = self._normalize_and_aggregate(current_data, target='order_count')
            current_data = self._resample_to_daily(current_data, target='order_count')
        
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
            temp_encoded = self.prepare_data(temp_df, target='order_count')
            temp_feat = self._create_features(temp_encoded, target='order_count')
            
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

    def _seasonal_naive_forecast(self, horizon_days: int) -> pd.DataFrame:
        """Fallback forecast based on global mean by day-of-week."""
        start = pd.Timestamp.now().normalize()
        dates = [start + pd.Timedelta(days=i + 1) for i in range(int(horizon_days))]
        preds = []
        for d in dates:
            dow = int(d.dayofweek)
            base = float(self.global_dow_mean.get(dow, 0.0))
            preds.append({'date': d, 'predicted_demand': int(round(max(0.0, base)))})
        return pd.DataFrame(preds)
