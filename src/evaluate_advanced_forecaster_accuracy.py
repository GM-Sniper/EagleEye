"""Evaluate AdvancedForecaster accuracy on overall daily demand.

This avoids loading all CSVs via DataPipeline.load_all() (which can be slow) and
instead reads only Data/fct_orders.csv to build a dense daily order_count series.

Run from repo root:
  .venv/Scripts/python.exe src/evaluate_advanced_forecaster_accuracy.py

Outputs:
- BEST_MODEL, ensemble weights
- VAL20_*: last-20% (val tail) MAPE per model + ensemble (same slice used for weighting)
- HOLDOUT_*: strict time holdout evaluation (fit on train-only, forecast next N days)
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from models.advanced_forecaster import AdvancedForecaster
except Exception:
    from src.models.advanced_forecaster import AdvancedForecaster  # type: ignore


def _dense_daily_orders_from_fct_orders(path: Path) -> pd.DataFrame:
    orders = pd.read_csv(path, low_memory=False)
    created = pd.to_datetime(orders.get("created"), unit="s", errors="coerce")
    dates = created.dt.normalize()

    daily = (
        pd.DataFrame({"date": dates})
        .dropna()
        .groupby("date")
        .size()
        .rename("order_count")
        .reset_index()
        .sort_values("date")
    )

    if daily.empty:
        return daily

    idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(idx).fillna(0).reset_index()
    daily.columns = ["date", "order_count"]
    daily["order_count"] = daily["order_count"].astype(float)
    return daily


def _mape_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(y_true, 1.0)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def _wmape_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def _smape_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-days", type=int, default=28, help="Strict holdout size in days.")
    parser.add_argument("--recent-months", type=int, default=6, help="Training window inside AdvancedForecaster.")
    args = parser.parse_args()

    daily = _dense_daily_orders_from_fct_orders(Path("Data") / "fct_orders.csv")
    if daily.empty:
        raise RuntimeError("No rows in daily demand series.")

    forecaster = AdvancedForecaster(recent_months=int(args.recent_months))
    forecaster.fit(daily, target_col="order_count")

    # Validation slice used for weight calculation (tail 20%).
    df = forecaster._last_data
    val_size = int(len(df) * 0.2)
    val = df.iloc[-val_size:].copy()

    X_val = val[forecaster._feature_cols].fillna(0)
    X_val_scaled = forecaster.scaler.transform(X_val)
    y_val = val["order_count"].to_numpy(dtype=float)

    preds: dict[str, np.ndarray] = {}
    for name, model in forecaster.models.items():
        if name == "prophet":
            future = pd.DataFrame({"ds": val["date"]})
            p = model.predict(future)["yhat"].to_numpy(dtype=float)
        else:
            p = model.predict(X_val_scaled).astype(float)
        preds[name] = np.maximum(p, 0.0)

    ens = np.zeros_like(y_val, dtype=float)
    for name, p in preds.items():
        ens += float(forecaster.weights.get(name, 0.0)) * p

    print("BEST_MODEL", forecaster.best_model)
    print("WEIGHTS", {k: round(float(v), 3) for k, v in forecaster.weights.items()})
    print("VAL20_MAPE_ENSEMBLE", round(_mape_pct(y_val, ens), 3))
    print("VAL20_MAPE_BY_MODEL", {k: round(_mape_pct(y_val, p), 3) for k, p in preds.items()})

    # Strict holdout: train on data before cutoff, then forecast next val_days.
    val_days = int(args.val_days)
    max_date = pd.to_datetime(daily["date"]).max()
    cutoff = max_date - timedelta(days=val_days)
    train = daily[daily["date"] <= cutoff].copy()
    test = daily[daily["date"] > cutoff].copy()

    strict = AdvancedForecaster(recent_months=int(args.recent_months))
    strict.fit(train, target_col="order_count")
    fc = strict.predict(horizon_days=len(test), method="ensemble")

    out = test[["date", "order_count"]].rename(columns={"order_count": "y_true"}).merge(
        fc.rename(columns={"predicted_demand": "y_pred"}),
        on="date",
        how="left",
    )
    out["y_pred"] = out["y_pred"].fillna(0.0).astype(float)
    y_true2 = out["y_true"].to_numpy(dtype=float)
    y_pred2 = out["y_pred"].to_numpy(dtype=float)

    print("HOLDOUT_VAL_DAYS", val_days)
    print("HOLDOUT_CUTOFF_DATE", str(pd.to_datetime(cutoff).date()))
    print("HOLDOUT_MAPE_PCT", round(_mape_pct(y_true2, y_pred2), 3))
    print("HOLDOUT_WMAPE_PCT", round(_wmape_pct(y_true2, y_pred2), 3))
    print("HOLDOUT_SMAPE_PCT", round(_smape_pct(y_true2, y_pred2), 3))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
