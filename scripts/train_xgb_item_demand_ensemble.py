"""Train and evaluate an XGBoost ensemble for item demand.

This script uses the engineered feature table produced by
DataPipeline.build_item_place_item_demand_features(), performs a time-based
holdout split, trains an ensemble of XGBoost models, and saves:
  - metrics JSON
  - predictions CSV (validation only)
  - prediction plots (PNG)

Run from repo root:
  python src/train_xgb_item_demand_ensemble.py --val-days 28 --n-models 5

Optional for quicker smoke tests:
  python src/train_xgb_item_demand_ensemble.py --val-days 28 --n-models 3 --sample-frac 0.2
"""

import sys
from pathlib import Path
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from pandas.api.types import is_bool_dtype, is_numeric_dtype


def _wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
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


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    drop_cols = {
        "demand_qty",
        "date",
        # ABC features are computed using the full history in the feature builder.
        # To avoid lookahead leakage, drop them unless explicitly recomputed train-only.
        "abc_class",
        "abc_A",
        "abc_B",
        "abc_C",
        # Explicitly drop leaky/target-derived columns if present.
        "share_of_basket_qty",
        "share_of_basket_revenue",
        "place_total_item_qty",
        "place_total_item_revenue",
        "place_unique_items_sold",
    }

    # Heuristic guardrail: drop same-day transactional aggregates unless lagged/rolling.
    # (These commonly allow inferring demand_qty for the same date.)
    leaky_substrings = (
        "revenue", "cost", "margin", "discount", "commission",
        "unit_price", "unit_cost",
        "n_orders", "n_order_lines", "order_lines", "order_count",
        "basket", "total_amount",
    )

    def _is_probably_leaky(col: str) -> bool:
        c = col.lower()
        if ("lag_" in c) or ("roll_" in c):
            return False
        return any(s in c for s in leaky_substrings)

    feature_cols: list[str] = []
    for c in df.columns:
        if c in drop_cols:
            continue
        if c.startswith("Unnamed:"):
            continue
        if _is_probably_leaky(c):
            continue
        if not (is_numeric_dtype(df[c]) or is_bool_dtype(df[c])):
            continue
        feature_cols.append(c)

    seen: set[str] = set()
    out: list[str] = []
    for c in feature_cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _downsample(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    if frac >= 1.0:
        return df
    frac = max(0.0, min(1.0, float(frac)))
    if frac <= 0.0:
        return df.iloc[:0].copy()
    return df.sample(frac=frac, replace=False, random_state=seed).reset_index(drop=True)


def _plot_aggregate_timeseries(
    val_df: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
) -> None:
    plot_df = (
        val_df.groupby("date", as_index=False)[["y_true", "y_pred"]]
        .sum()
        .sort_values("date")
    )

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["date"], plot_df["y_true"], label="Actual", linewidth=2)
    plt.plot(plot_df["date"], plot_df["y_pred"], label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Total demand_qty")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_scatter(
    val_df: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
) -> None:
    y_true = val_df["y_true"].to_numpy(dtype=float)
    y_pred = val_df["y_pred"].to_numpy(dtype=float)

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.25)
    max_v = float(np.max([np.max(y_true), np.max(y_pred), 1.0]))
    plt.plot([0, max_v], [0, max_v], linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Actual demand_qty")
    plt.ylabel("Predicted demand_qty")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _aggregate_for_granularity(pred_out: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate y_true/y_pred over time.

    Expects columns: date, y_true, y_pred.
    """
    df = pred_out[["date", "y_true", "y_pred"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    out = (
        df.groupby(pd.Grouper(key="date", freq=freq), as_index=False)[["y_true", "y_pred"]]
        .sum()
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
        "wmape": _wmape(y_true, y_pred),
        "smape": _smape(y_true, y_pred),
        "wmape_pct": _wmape(y_true, y_pred) * 100.0 if np.isfinite(_wmape(y_true, y_pred)) else float("nan"),
        "smape_pct": _smape(y_true, y_pred) * 100.0 if np.isfinite(_smape(y_true, y_pred)) else float("nan"),
        "zero_true_pct": float(np.mean(y_true == 0.0)) if len(y_true) else float("nan"),
    }


def _plot_top_series(
    val_df: pd.DataFrame,
    out_path: Path,
    *,
    top_n: int,
) -> None:
    # Pick top (place_id, item_id) pairs by actual demand in validation.
    grp = (
        val_df.groupby(["place_id", "item_id"], as_index=False)["y_true"]
        .sum()
        .sort_values("y_true", ascending=False)
        .head(int(top_n))
    )
    if grp.empty:
        return

    pairs = list(zip(grp["place_id"].tolist(), grp["item_id"].tolist()))

    n = len(pairs)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.4 * nrows), sharex=False)
    axes = np.array(axes).reshape(-1)

    for i, (place_id, item_id) in enumerate(pairs):
        ax = axes[i]
        s = val_df[(val_df["place_id"] == place_id) & (val_df["item_id"] == item_id)].copy()
        s = s.sort_values("date")
        ax.plot(s["date"], s["y_true"], label="Actual", linewidth=2)
        ax.plot(s["date"], s["y_pred"], label="Predicted", linewidth=2)
        ax.set_title(f"place_id={place_id}, item_id={item_id}")
        ax.set_ylabel("demand_qty")
        ax.tick_params(axis="x", rotation=25)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_top_items_daily_series(
    pred_out: pd.DataFrame,
    out_path: Path,
    *,
    top_n: int,
) -> None:
    """Plot daily actual vs predicted aggregated per item across places."""
    item_day = (
        pred_out.groupby(["date", "item_id"], as_index=False)[["y_true", "y_pred"]]
        .sum()
        .sort_values(["item_id", "date"])
    )
    top_items = (
        item_day.groupby("item_id", as_index=False)["y_true"]
        .sum()
        .sort_values("y_true", ascending=False)
        .head(int(top_n))["item_id"]
        .tolist()
    )
    if not top_items:
        return

    n = len(top_items)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.4 * nrows), sharex=False)
    axes = np.array(axes).reshape(-1)

    for i, item_id in enumerate(top_items):
        ax = axes[i]
        s = item_day[item_day["item_id"] == item_id].sort_values("date")
        ax.plot(s["date"], s["y_true"], label="Actual", linewidth=2)
        ax.plot(s["date"], s["y_pred"], label="Predicted", linewidth=2)
        ax.set_title(f"item_id={item_id}")
        ax.set_ylabel("demand_qty")
        ax.tick_params(axis="x", rotation=25)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_top_places_daily_series(
    pred_out: pd.DataFrame,
    out_path: Path,
    *,
    top_n: int,
) -> None:
    """Plot daily actual vs predicted aggregated per place across items."""
    place_day = (
        pred_out.groupby(["date", "place_id"], as_index=False)[["y_true", "y_pred"]]
        .sum()
        .sort_values(["place_id", "date"])
    )
    top_places = (
        place_day.groupby("place_id", as_index=False)["y_true"]
        .sum()
        .sort_values("y_true", ascending=False)
        .head(int(top_n))["place_id"]
        .tolist()
    )
    if not top_places:
        return

    n = len(top_places)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.4 * nrows), sharex=False)
    axes = np.array(axes).reshape(-1)

    for i, place_id in enumerate(top_places):
        ax = axes[i]
        s = place_day[place_day["place_id"] == place_id].sort_values("date")
        ax.plot(s["date"], s["y_true"], label="Actual", linewidth=2)
        ax.plot(s["date"], s["y_pred"], label="Predicted", linewidth=2)
        ax.set_title(f"place_id={place_id}")
        ax.set_ylabel("demand_qty")
        ax.tick_params(axis="x", rotation=25)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _per_item_metrics(pred_out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item_id, g in pred_out.groupby("item_id", sort=False):
        y_true = g["y_true"].to_numpy(dtype=float)
        y_pred = g["y_pred"].to_numpy(dtype=float)
        rows.append(
            {
                "item_id": int(item_id),
                "rows": int(len(g)),
                "nonzero_true_rows": int(np.sum(y_true > 0)),
                "sum_true": float(np.sum(y_true)),
                "sum_pred": float(np.sum(y_pred)),
                "mae": _mae(y_true, y_pred),
                "rmse": _rmse(y_true, y_pred),
                "wmape": _wmape(y_true, y_pred),
                "smape": _smape(y_true, y_pred),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["wmape_pct"] = out["wmape"] * 100.0
    out["smape_pct"] = out["smape"] * 100.0
    out = out.sort_values(["sum_true", "wmape"], ascending=[False, True]).reset_index(drop=True)
    return out


def _per_place_metrics(pred_out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for place_id, g in pred_out.groupby("place_id", sort=False):
        y_true = g["y_true"].to_numpy(dtype=float)
        y_pred = g["y_pred"].to_numpy(dtype=float)
        rows.append(
            {
                "place_id": float(place_id),
                "rows": int(len(g)),
                "nonzero_true_rows": int(np.sum(y_true > 0)),
                "sum_true": float(np.sum(y_true)),
                "sum_pred": float(np.sum(y_pred)),
                "mae": _mae(y_true, y_pred),
                "rmse": _rmse(y_true, y_pred),
                "wmape": _wmape(y_true, y_pred),
                "smape": _smape(y_true, y_pred),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["wmape_pct"] = out["wmape"] * 100.0
    out["smape_pct"] = out["smape"] * 100.0
    out = out.sort_values(["sum_true", "wmape"], ascending=[False, True]).reset_index(drop=True)
    return out


def _plot_item_wmape_bars(item_metrics: pd.DataFrame, out_path: Path, *, top_n: int) -> None:
    if item_metrics.empty:
        return
    top = item_metrics.sort_values("sum_true", ascending=False).head(int(top_n)).copy()
    top = top.sort_values("wmape_pct", ascending=False)
    plt.figure(figsize=(12, 5))
    plt.bar(top["item_id"].astype(str), top["wmape_pct"].astype(float))
    plt.title(f"Per-item WMAPE% (top {len(top)} items by volume)")
    plt.xlabel("item_id")
    plt.ylabel("WMAPE%")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_place_wmape_bars(place_metrics: pd.DataFrame, out_path: Path, *, top_n: int) -> None:
    if place_metrics.empty:
        return
    top = place_metrics.sort_values("sum_true", ascending=False).head(int(top_n)).copy()
    top = top.sort_values("wmape_pct", ascending=False)
    plt.figure(figsize=(12, 5))
    plt.bar(top["place_id"].astype(str), top["wmape_pct"].astype(float))
    plt.title(f"Per-place WMAPE% (top {len(top)} places by volume)")
    plt.xlabel("place_id")
    plt.ylabel("WMAPE%")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--val-days", type=int, default=28, help="Holdout size in days (by date).")
    parser.add_argument("--n-models", type=int, default=5, help="Number of ensemble members.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")

    parser.add_argument("--min-date", type=str, default=None, help="Optional min date (YYYY-MM-DD).")
    parser.add_argument("--max-date", type=str, default=None, help="Optional max date (YYYY-MM-DD).")

    parser.add_argument(
        "--place-ids",
        type=str,
        default=None,
        help="Comma-separated list of place_ids to filter (optional).",
    )
    parser.add_argument(
        "--item-ids",
        type=str,
        default=None,
        help="Comma-separated list of item_ids to filter (optional).",
    )

    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional downsampling fraction applied to TRAIN rows only.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("artifacts") / "xgb_item_demand"),
        help="Directory to write metrics/predictions/plots.",
    )

    args = parser.parse_args()

    out_dir = _ensure_dir(Path(args.output_dir))
    plots_dir = _ensure_dir(out_dir / "plots")

    # Local imports to avoid path issues when running from repo root.
    try:
        from services.data_pipeline import DataPipeline
        from services.demand_feature_builder import DemandFeatureConfig
    except Exception:
        from src.services.data_pipeline import DataPipeline  # type: ignore
        from src.services.demand_feature_builder import DemandFeatureConfig  # type: ignore

    place_ids = None
    if args.place_ids:
        place_ids = [int(x.strip()) for x in args.place_ids.split(",") if x.strip()]

    item_ids = None
    if args.item_ids:
        item_ids = [int(x.strip()) for x in args.item_ids.split(",") if x.strip()]

    pipeline = DataPipeline(data_dir="Data")
    pipeline.load_all()
    pipeline.load_core_tables()

    config = DemandFeatureConfig(fill_missing_days=True)
    df = pipeline.build_item_place_item_demand_features(
        min_date=args.min_date,
        max_date=args.max_date,
        place_ids=place_ids,
        item_ids=item_ids,
        config=config,
    )

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    max_date = df["date"].max()
    if pd.isna(max_date):
        raise RuntimeError("Feature table is empty; check filters or input data.")

    val_days = int(args.val_days)
    cutoff = max_date - timedelta(days=val_days - 1)

    train_df = df[df["date"] < cutoff].copy()
    val_df = df[df["date"] >= cutoff].copy()

    if train_df.empty or val_df.empty:
        raise RuntimeError(
            f"Train/val split produced empty set: train={len(train_df)}, val={len(val_df)}. "
            "Try lowering --val-days or relaxing date filters."
        )

    train_df = _downsample(train_df, frac=float(args.sample_frac), seed=int(args.seed))

    print("\n=== Feature table columns (df.columns) ===")
    print(f"n_cols={df.shape[1]}")
    print(df.columns.tolist())
    print("=== End columns ===\n")

    feature_cols = _select_feature_columns(df)

    print("=== Selected feature columns (feature_cols) ===")
    print(f"n_features={len(feature_cols)}")
    print(feature_cols)
    print("=== End selected features ===\n")

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df["demand_qty"].astype(float).to_numpy()

    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_val = val_df["demand_qty"].astype(float).to_numpy()

    # Target transform for stability.
    y_train_t = np.log1p(y_train)

    models: list[XGBRegressor] = []
    val_preds_t = []

    base_params: dict[str, Any] = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "random_state": int(args.seed),
        "n_jobs": -1,
    }

    for i in range(int(args.n_models)):
        params = dict(base_params)
        params["random_state"] = int(args.seed) + i
        # small jitter for diversity
        params["subsample"] = float(np.clip(0.75 + 0.05 * (i % 3), 0.6, 0.95))
        params["colsample_bytree"] = float(np.clip(0.75 + 0.05 * ((i + 1) % 3), 0.6, 0.95))

        model = XGBRegressor(**params)
        model.fit(X_train, y_train_t)
        models.append(model)

        pred_t = model.predict(X_val)
        val_preds_t.append(pred_t)

    val_pred_t_mean = np.mean(np.vstack(val_preds_t), axis=0)
    y_pred = np.expm1(val_pred_t_mean)
    y_pred = np.maximum(y_pred, 0.0)

    metrics = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "n_features": int(len(feature_cols)),
        "val_days": int(val_days),
        "cutoff_date": str(cutoff.date()),
        "max_date": str(max_date.date()),
        "mae": _mae(y_val, y_pred),
        "rmse": _rmse(y_val, y_pred),
        "wmape": _wmape(y_val, y_pred),
        "smape": _smape(y_val, y_pred),
        "wmape_pct": _wmape(y_val, y_pred) * 100.0 if np.isfinite(_wmape(y_val, y_pred)) else float("nan"),
        "smape_pct": _smape(y_val, y_pred) * 100.0 if np.isfinite(_smape(y_val, y_pred)) else float("nan"),
        "zero_true_pct": float(np.mean(y_val == 0.0)),
    }

    # Save validation predictions for analysis.
    pred_out = val_df[["date", "place_id", "item_id", "demand_qty"]].copy()
    pred_out = pred_out.rename(columns={"demand_qty": "y_true"})
    pred_out["y_pred"] = y_pred
    pred_out.to_csv(out_dir / "val_predictions.csv", index=False)

    # Plots
    _plot_aggregate_timeseries(
        pred_out,
        plots_dir / "agg_actual_vs_pred.png",
        title=f"Aggregate actual vs predicted (val: last {val_days} days)",
    )
    _plot_scatter(
        pred_out,
        plots_dir / "scatter_actual_vs_pred.png",
        title="Validation scatter: actual vs predicted",
    )
    _plot_top_series(
        pred_out,
        plots_dir / "top_series_actual_vs_pred.png",
        top_n=6,
    )

    # Per-item inspection outputs (aggregated across places)
    _plot_top_items_daily_series(
        pred_out,
        plots_dir / "top_items_daily_actual_vs_pred.png",
        top_n=12,
    )

    item_metrics = _per_item_metrics(pred_out)
    item_metrics.to_csv(out_dir / "item_level_metrics.csv", index=False)
    _plot_item_wmape_bars(
        item_metrics,
        plots_dir / "item_wmape_top_items.png",
        top_n=20,
    )

    # Per-place inspection outputs (aggregated across items)
    place_metrics = _per_place_metrics(pred_out)
    place_metrics.to_csv(out_dir / "place_level_metrics.csv", index=False)
    _plot_top_places_daily_series(
        pred_out,
        plots_dir / "top_places_daily_actual_vs_pred.png",
        top_n=12,
    )
    _plot_place_wmape_bars(
        place_metrics,
        plots_dir / "place_wmape_top_places.png",
        top_n=20,
    )

    # Weekly + monthly rollups (company totals) for additional reporting.
    weekly = _aggregate_for_granularity(pred_out, freq="W")
    monthly = _aggregate_for_granularity(pred_out, freq="MS")

    if not weekly.empty:
        _plot_aggregate_timeseries(
            weekly,
            plots_dir / "agg_weekly_actual_vs_pred.png",
            title=f"Aggregate actual vs predicted (weekly totals; val window)",
        )
        _plot_scatter(
            weekly,
            plots_dir / "scatter_weekly_actual_vs_pred.png",
            title="Validation scatter (weekly totals): actual vs predicted",
        )

    if not monthly.empty:
        _plot_aggregate_timeseries(
            monthly,
            plots_dir / "agg_monthly_actual_vs_pred.png",
            title=f"Aggregate actual vs predicted (monthly totals; val window)",
        )
        _plot_scatter(
            monthly,
            plots_dir / "scatter_monthly_actual_vs_pred.png",
            title="Validation scatter (monthly totals): actual vs predicted",
        )

    metrics["weekly"] = _metrics_from_pred_df(weekly) if not weekly.empty else {"rows": 0}
    metrics["monthly"] = _metrics_from_pred_df(monthly) if not monthly.empty else {"rows": 0}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved outputs to:", str(out_dir))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())