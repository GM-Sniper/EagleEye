"""Train and evaluate FinalForecaster with plots + metrics.

This script mirrors the artifact outputs from src/train_xgb_item_demand_ensemble.py,
but for the univariate daily forecaster (FinalForecaster) and optionally for
per-place and per-item series.

Run from repo root:
  python src/train_final_forecaster_reports.py --val-days 28

Optional:
  python src/train_final_forecaster_reports.py --val-days 28 --top-places 6 --top-items 12
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def _wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    denom = np.maximum(np.abs(y_true), 1.0)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dense_daily(df: pd.DataFrame, *, date_col: str, value_col: str) -> pd.DataFrame:
    """Ensure a dense daily index with missing days filled as 0."""
    out = df[[date_col, value_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out = out.groupby(date_col, as_index=False)[value_col].sum().sort_values(date_col)

    if out.empty:
        return out

    idx = pd.date_range(out[date_col].min(), out[date_col].max(), freq="D")
    out = out.set_index(date_col).reindex(idx)
    out.index.name = date_col
    out[value_col] = out[value_col].fillna(0.0)
    out = out.reset_index()
    return out


def _plot_timeseries(df: pd.DataFrame, out_path: Path, *, title: str, y_col_true: str, y_col_pred: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df[y_col_true], label="Actual", linewidth=2)
    plt.plot(df["date"], df[y_col_pred], label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_scatter(df: pd.DataFrame, out_path: Path, *, title: str, y_col_true: str, y_col_pred: str) -> None:
    y_true = df[y_col_true].to_numpy(dtype=float)
    y_pred = df[y_col_pred].to_numpy(dtype=float)

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(y_true, y_pred, s=12, alpha=0.25)
    max_v = float(np.max([np.max(y_true) if len(y_true) else 1.0, np.max(y_pred) if len(y_pred) else 1.0, 1.0]))
    plt.plot([0, max_v], [0, max_v], linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _aggregate_for_granularity(pred_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = pred_df[["date", "y_true", "y_pred"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    out = (
        df.groupby(pd.Grouper(key="date", freq=freq))[["y_true", "y_pred"]]
        .sum()
        .reset_index()
        .sort_values("date")
    )
    return out


def _metrics_from_pred_df(pred_df: pd.DataFrame) -> dict[str, float]:
    y_true = pred_df["y_true"].to_numpy(dtype=float)
    y_pred = pred_df["y_pred"].to_numpy(dtype=float)
    return {
        "rows": int(len(pred_df)),
        "mae": _mae(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
        "mape": _mape(y_true, y_pred),
        "wmape": _wmape(y_true, y_pred),
        "smape": _smape(y_true, y_pred),
        "mape_pct": _mape(y_true, y_pred) * 100.0,
        "wmape_pct": _wmape(y_true, y_pred) * 100.0 if np.isfinite(_wmape(y_true, y_pred)) else float("nan"),
        "smape_pct": _smape(y_true, y_pred) * 100.0 if np.isfinite(_smape(y_true, y_pred)) else float("nan"),
        "zero_true_pct": float(np.mean(y_true == 0.0)) if len(y_true) else float("nan"),
    }


@dataclass(frozen=True)
class SeriesEvalResult:
    key_name: str
    key_value: int
    rows: int
    sum_true: float
    sum_pred: float
    wmape_pct: float
    smape_pct: float
    mae: float
    rmse: float


def _evaluate_one_series(
    series_df: pd.DataFrame,
    *,
    val_days: int,
    recent_days: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit FinalForecaster on train part, recursively predict val horizon."""
    # Local imports to avoid path issues when running from repo root.
    try:
        from models.final_forecaster import FinalForecaster
    except Exception:
        from src.models.final_forecaster import FinalForecaster  # type: ignore

    df = series_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date")

    max_date = df["date"].max()
    if pd.isna(max_date):
        raise RuntimeError("Empty series")

    cutoff = max_date - timedelta(days=int(val_days) - 1)
    train_df = df[df["date"] < cutoff].copy()
    val_df = df[df["date"] >= cutoff].copy()

    if train_df.empty or val_df.empty:
        raise RuntimeError(f"Train/val split empty: train={len(train_df)}, val={len(val_df)}")

    # Forecaster expects column named order_count.
    train_fit = train_df[["date", "order_count"]].copy()

    forecaster = FinalForecaster(recent_days=int(recent_days))
    # Ensure the model uses this seed if supported.
    if isinstance(getattr(forecaster, "xgb_params", None), dict):
        forecaster.xgb_params = dict(forecaster.xgb_params)
        forecaster.xgb_params["random_state"] = int(seed)

    forecaster.fit(train_fit, target_col="order_count")

    horizon = int(len(val_df))
    pred = forecaster.predict(horizon_days=horizon)

    pred_out = val_df[["date", "order_count"]].copy().rename(columns={"order_count": "y_true"})
    pred_out = pred_out.merge(pred.rename(columns={"predicted_demand": "y_pred"}), on="date", how="left")
    pred_out["y_pred"] = pred_out["y_pred"].fillna(0.0).astype(float)
    pred_out["y_true"] = pred_out["y_true"].astype(float)

    metrics = _metrics_from_pred_df(pred_out)
    metrics.update(
        {
            "val_days": int(val_days),
            "cutoff_date": str(cutoff.date()),
            "max_date": str(max_date.date()),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
        }
    )

    return pred_out, metrics


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--val-days", type=int, default=28, help="Holdout size in days (by date).")
    parser.add_argument("--recent-days", type=int, default=60, help="Training window inside FinalForecaster.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--top-places", type=int, default=6, help="Evaluate top-N places by order volume.")
    parser.add_argument("--top-items", type=int, default=12, help="Evaluate top-N items by quantity volume.")

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("artifacts") / "final_forecaster"),
        help="Directory to write metrics/predictions/plots.",
    )

    args = parser.parse_args()

    out_dir = _ensure_dir(Path(args.output_dir))
    plots_dir = _ensure_dir(out_dir / "plots")

    # Local imports to avoid path issues when running from repo root.
    try:
        from services.data_pipeline import DataPipeline
    except Exception:
        from src.services.data_pipeline import DataPipeline  # type: ignore

    pipeline = DataPipeline(data_dir="Data")
    pipeline.load_all()
    pipeline.load_core_tables()

    # === Overall daily demand ===
    daily = pipeline.get_daily_demand()
    daily = daily.rename(columns={"order_count": "order_count"})
    daily = _dense_daily(daily, date_col="date", value_col="order_count")

    pred_out, metrics = _evaluate_one_series(
        daily,
        val_days=int(args.val_days),
        recent_days=int(args.recent_days),
        seed=int(args.seed),
    )

    pred_out.to_csv(out_dir / "val_predictions.csv", index=False)

    # Plots (overall)
    _plot_timeseries(
        pred_out,
        plots_dir / "agg_actual_vs_pred.png",
        title=f"FinalForecaster: Actual vs predicted (val last {int(args.val_days)} days)",
        y_col_true="y_true",
        y_col_pred="y_pred",
    )
    _plot_scatter(
        pred_out,
        plots_dir / "scatter_actual_vs_pred.png",
        title="FinalForecaster validation scatter: actual vs predicted",
        y_col_true="y_true",
        y_col_pred="y_pred",
    )

    # Weekly + monthly rollups
    weekly = _aggregate_for_granularity(pred_out, freq="W")
    monthly = _aggregate_for_granularity(pred_out, freq="MS")

    if not weekly.empty:
        _plot_timeseries(
            weekly,
            plots_dir / "agg_weekly_actual_vs_pred.png",
            title="FinalForecaster: weekly totals (val window)",
            y_col_true="y_true",
            y_col_pred="y_pred",
        )
        _plot_scatter(
            weekly,
            plots_dir / "scatter_weekly_actual_vs_pred.png",
            title="FinalForecaster weekly scatter (val window)",
            y_col_true="y_true",
            y_col_pred="y_pred",
        )

    if not monthly.empty:
        _plot_timeseries(
            monthly,
            plots_dir / "agg_monthly_actual_vs_pred.png",
            title="FinalForecaster: monthly totals (val window)",
            y_col_true="y_true",
            y_col_pred="y_pred",
        )
        _plot_scatter(
            monthly,
            plots_dir / "scatter_monthly_actual_vs_pred.png",
            title="FinalForecaster monthly scatter (val window)",
            y_col_true="y_true",
            y_col_pred="y_pred",
        )

    metrics["weekly"] = _metrics_from_pred_df(weekly) if not weekly.empty else {"rows": 0}
    metrics["monthly"] = _metrics_from_pred_df(monthly) if not monthly.empty else {"rows": 0}

    # === Per-place evaluation (top-N by total orders) ===
    per_place_metrics: list[SeriesEvalResult] = []
    per_place_pred_rows: list[pd.DataFrame] = []

    top_places_n = max(0, int(args.top_places))
    if top_places_n > 0 and "place_id" in pipeline.orders.columns:
        place_counts = (
            pipeline.orders.groupby("place_id", as_index=False)["id"]
            .count()
            .rename(columns={"id": "n_orders"})
            .sort_values("n_orders", ascending=False)
        )
        top_place_ids = place_counts.head(top_places_n)["place_id"].astype(int).tolist()

        for i, place_id in enumerate(top_place_ids):
            place_daily = pipeline.get_daily_demand(place_id=int(place_id))
            place_daily = _dense_daily(place_daily, date_col="date", value_col="order_count")
            if len(place_daily) < (int(args.val_days) + 30):
                continue

            pred_place, m_place = _evaluate_one_series(
                place_daily,
                val_days=int(args.val_days),
                recent_days=int(args.recent_days),
                seed=int(args.seed) + i + 1,
            )

            pred_place = pred_place.copy()
            pred_place["place_id"] = int(place_id)
            per_place_pred_rows.append(pred_place)

            per_place_metrics.append(
                SeriesEvalResult(
                    key_name="place_id",
                    key_value=int(place_id),
                    rows=int(m_place.get("rows", len(pred_place))),
                    sum_true=float(pred_place["y_true"].sum()),
                    sum_pred=float(pred_place["y_pred"].sum()),
                    wmape_pct=float(m_place.get("wmape_pct", float("nan"))),
                    smape_pct=float(m_place.get("smape_pct", float("nan"))),
                    mae=float(m_place.get("mae", float("nan"))),
                    rmse=float(m_place.get("rmse", float("nan"))),
                )
            )

            _plot_timeseries(
                pred_place,
                plots_dir / f"place_{int(place_id)}_actual_vs_pred.png",
                title=f"Place {int(place_id)}: actual vs predicted (val window)",
                y_col_true="y_true",
                y_col_pred="y_pred",
            )

    if per_place_pred_rows:
        pd.concat(per_place_pred_rows, ignore_index=True).to_csv(out_dir / "place_val_predictions.csv", index=False)

    if per_place_metrics:
        pd.DataFrame([m.__dict__ for m in per_place_metrics]).to_csv(out_dir / "place_level_metrics.csv", index=False)

    # === Per-item evaluation (top-N by quantity) ===
    per_item_metrics: list[SeriesEvalResult] = []
    per_item_pred_rows: list[pd.DataFrame] = []

    top_items_n = max(0, int(args.top_items))
    if top_items_n > 0 and "item_id" in pipeline.order_items.columns:
        oi = pipeline.order_items.copy()
        oi["quantity"] = pd.to_numeric(oi.get("quantity", 0), errors="coerce").fillna(0.0)
        item_volume = (
            oi.groupby("item_id", as_index=False)["quantity"]
            .sum()
            .rename(columns={"quantity": "qty_sum"})
            .sort_values("qty_sum", ascending=False)
        )
        top_item_ids = item_volume.head(top_items_n)["item_id"].astype(int).tolist()

        for i, item_id in enumerate(top_item_ids):
            item_daily = pipeline.get_item_daily_demand(item_id=int(item_id))
            if item_daily.empty:
                continue

            # Standardize to expected schema.
            item_daily = item_daily.rename(columns={"quantity": "order_count"})
            item_daily = _dense_daily(item_daily, date_col="date", value_col="order_count")

            if len(item_daily) < (int(args.val_days) + 30):
                continue

            pred_item, m_item = _evaluate_one_series(
                item_daily,
                val_days=int(args.val_days),
                recent_days=int(args.recent_days),
                seed=int(args.seed) + 100 + i,
            )

            pred_item = pred_item.copy()
            pred_item["item_id"] = int(item_id)
            per_item_pred_rows.append(pred_item)

            per_item_metrics.append(
                SeriesEvalResult(
                    key_name="item_id",
                    key_value=int(item_id),
                    rows=int(m_item.get("rows", len(pred_item))),
                    sum_true=float(pred_item["y_true"].sum()),
                    sum_pred=float(pred_item["y_pred"].sum()),
                    wmape_pct=float(m_item.get("wmape_pct", float("nan"))),
                    smape_pct=float(m_item.get("smape_pct", float("nan"))),
                    mae=float(m_item.get("mae", float("nan"))),
                    rmse=float(m_item.get("rmse", float("nan"))),
                )
            )

            _plot_timeseries(
                pred_item,
                plots_dir / f"item_{int(item_id)}_actual_vs_pred.png",
                title=f"Item {int(item_id)}: actual vs predicted (val window)",
                y_col_true="y_true",
                y_col_pred="y_pred",
            )

    if per_item_pred_rows:
        pd.concat(per_item_pred_rows, ignore_index=True).to_csv(out_dir / "item_val_predictions.csv", index=False)

    if per_item_metrics:
        pd.DataFrame([m.__dict__ for m in per_item_metrics]).to_csv(out_dir / "item_level_metrics.csv", index=False)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved outputs to:", str(out_dir))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
