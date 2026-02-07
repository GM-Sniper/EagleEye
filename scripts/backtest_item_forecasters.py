"""Backtest item-level forecasters and export plots similar to artifacts.

This script builds a dense daily item-demand table, performs a strict time
holdout, evaluates multiple forecasters, and exports:
- metrics.json
- item_level_metrics.csv
- val_predictions.csv
- plots/*.png

Run from repo root:
  .venv/Scripts/python.exe scripts/backtest_item_forecasters.py --val-days 28 --max-items 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from services.data_pipeline import DataPipeline
from models.global_forecaster import GlobalForecaster
from models.hybrid_forecaster import HybridForecaster
from models.production_forecaster import ProductionForecaster


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _plot_aggregate_timeseries(val_df: pd.DataFrame, out_path: Path, *, title: str) -> None:
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
    plt.ylabel("Total demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_scatter(val_df: pd.DataFrame, out_path: Path, *, title: str) -> None:
    y_true = val_df["y_true"].to_numpy(dtype=float)
    y_pred = val_df["y_pred"].to_numpy(dtype=float)

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.25)
    max_v = float(np.max([np.max(y_true), np.max(y_pred), 1.0]))
    plt.plot([0, max_v], [0, max_v], linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Actual demand")
    plt.ylabel("Predicted demand")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _aggregate_for_granularity(pred_out: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = pred_out[["date", "y_true", "y_pred"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    out = (
        df.groupby(pd.Grouper(key="date", freq=freq), as_index=False)[["y_true", "y_pred"]]
        .sum()
        .sort_values("date")
    )
    return out


def _plot_top_items_series(val_df: pd.DataFrame, out_path: Path, *, top_n: int) -> None:
    item_day = (
        val_df.groupby(["date", "item_id"], as_index=False)[["y_true", "y_pred"]]
        .sum()
        .sort_values(["item_id", "date"])
    )
    top_items = (
        item_day.groupby("item_id", as_index=False)["y_true"]
        .sum()
        .sort_values("y_true", ascending=False)
        .head(int(top_n))
    )
    if top_items.empty:
        return

    items = top_items["item_id"].tolist()
    n = len(items)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.4 * nrows), sharex=False)
    axes = np.array(axes).reshape(-1)

    for i, item_id in enumerate(items):
        ax = axes[i]
        s = item_day[item_day["item_id"] == item_id].copy().sort_values("date")
        ax.plot(s["date"], s["y_true"], label="Actual", linewidth=2)
        ax.plot(s["date"], s["y_pred"], label="Predicted", linewidth=2)
        ax.set_title(f"item_id={item_id}")
        ax.set_ylabel("demand")
        ax.tick_params(axis="x", rotation=25)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _dense_item_series(df: pd.DataFrame, item_id: int) -> pd.DataFrame:
    s = df[df["item_id"] == item_id].copy()
    s["date"] = pd.to_datetime(s["date"]).dt.normalize()
    s = s.sort_values("date")
    if s.empty:
        return s

    idx = pd.date_range(s["date"].min(), s["date"].max(), freq="D")
    s = s.set_index("date").reindex(idx)
    s.index.name = "date"
    s = s.reset_index()
    s["item_id"] = item_id
    s["order_count"] = s["order_count"].fillna(0.0)
    return s[["date", "item_id", "order_count"]]


def _metrics_from_pred_df(pred_df: pd.DataFrame) -> Dict[str, float]:
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


def _predict_global(
    model: GlobalForecaster,
    item_ids: Iterable[int],
    horizon_days: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for item_id in item_ids:
        preds = model.predict(int(item_id), horizon_days=int(horizon_days))
        if preds.empty:
            continue
        preds = preds.copy()
        preds["item_id"] = int(item_id)
        frames.append(preds)
    if not frames:
        return pd.DataFrame(columns=["date", "predicted_demand", "item_id"])
    return pd.concat(frames, ignore_index=True)


def _predict_local(
    train_df: pd.DataFrame,
    item_ids: Iterable[int],
    horizon_days: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for item_id in item_ids:
        series = train_df[train_df["item_id"] == item_id].copy()
        if len(series) < 30:
            continue
        local = ProductionForecaster()
        local.fit(series[["date", "order_count"]], target="order_count")
        preds = local.predict(horizon_days=int(horizon_days))
        preds = preds.rename(columns={"predicted_demand": "predicted_demand"})
        preds["item_id"] = int(item_id)
        frames.append(preds)
    if not frames:
        return pd.DataFrame(columns=["date", "predicted_demand", "item_id"])
    return pd.concat(frames, ignore_index=True)


def _predict_hybrid(
    model: HybridForecaster,
    train_df: pd.DataFrame,
    item_ids: Iterable[int],
    horizon_days: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for item_id in item_ids:
        series = train_df[train_df["item_id"] == item_id].copy()
        if series.empty:
            continue
        preds, _ = model.predict_item(int(item_id), series[["date", "order_count"]], horizon_days=int(horizon_days))
        preds = preds.copy()
        preds["item_id"] = int(item_id)
        frames.append(preds)
    if not frames:
        return pd.DataFrame(columns=["date", "predicted_demand", "item_id"])
    return pd.concat(frames, ignore_index=True)


def _export_model_outputs(
    model_name: str,
    out_dir: Path,
    val_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    top_n_items: int,
) -> Dict[str, float]:
    out_dir = _ensure_dir(out_dir)
    plots_dir = _ensure_dir(out_dir / "plots")

    pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.normalize()
    val_df["date"] = pd.to_datetime(val_df["date"]).dt.normalize()

    merged = val_df.merge(pred_df, on=["date", "item_id"], how="left")
    merged["y_pred"] = merged["predicted_demand"].fillna(0.0).astype(float)
    merged = merged.rename(columns={"order_count": "y_true"})

    merged.to_csv(out_dir / "val_predictions.csv", index=False)

    # Item-level metrics
    item_metrics = []
    for item_id, g in merged.groupby("item_id"):
        metrics = _metrics_from_pred_df(g)
        metrics["item_id"] = int(item_id)
        item_metrics.append(metrics)
    metrics_df = pd.DataFrame(item_metrics).sort_values("wmape")
    metrics_df.to_csv(out_dir / "item_level_metrics.csv", index=False)

    # Overall metrics
    overall = _metrics_from_pred_df(merged)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    # Plots
    _plot_aggregate_timeseries(merged, plots_dir / "agg_actual_vs_pred.png", title=f"{model_name}: Daily Total")
    _plot_scatter(merged, plots_dir / "scatter_actual_vs_pred.png", title=f"{model_name}: Daily Scatter")

    weekly = _aggregate_for_granularity(merged, "W")
    _plot_aggregate_timeseries(weekly, plots_dir / "agg_weekly_actual_vs_pred.png", title=f"{model_name}: Weekly Total")
    _plot_scatter(weekly, plots_dir / "scatter_weekly_actual_vs_pred.png", title=f"{model_name}: Weekly Scatter")

    monthly = _aggregate_for_granularity(merged, "ME")
    _plot_aggregate_timeseries(monthly, plots_dir / "agg_monthly_actual_vs_pred.png", title=f"{model_name}: Monthly Total")
    _plot_scatter(monthly, plots_dir / "scatter_monthly_actual_vs_pred.png", title=f"{model_name}: Monthly Scatter")

    _plot_top_items_series(merged, plots_dir / "top_items_actual_vs_pred.png", top_n=top_n_items)

    return overall


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-days", type=int, default=28)
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--top-items-plot", type=int, default=6)
    parser.add_argument("--out-dir", type=str, default=str(Path("artifacts") / "forecaster_backtest"))
    parser.add_argument("--models", type=str, default="global,hybrid,local")
    parser.add_argument("--max-resampled-rows", type=int, default=2_000_000)
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pipeline = DataPipeline(data_dir=str(Path("Data")))
    pipeline.load_all()
    pipeline.load_core_tables()

    oi = pipeline.order_items.copy()
    oi["date"] = pd.to_datetime(oi["created"], unit="s").dt.normalize()
    daily = oi.groupby(["date", "item_id"])["quantity"].sum().reset_index()
    daily.columns = ["date", "item_id", "order_count"]

    item_counts = daily.groupby("item_id")["order_count"].sum().sort_values(ascending=False)
    top_items = item_counts.head(int(args.max_items)).index.tolist()

    dense_frames = []
    for item_id in top_items:
        dense_frames.append(_dense_item_series(daily, int(item_id)))
    dense = pd.concat(dense_frames, ignore_index=True)

    max_date = pd.to_datetime(dense["date"]).max()
    cutoff = max_date - pd.Timedelta(days=int(args.val_days))

    train_df = dense[dense["date"] <= cutoff].copy()
    val_df = dense[dense["date"] > cutoff].copy()

    model_names = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    summaries: Dict[str, Dict[str, float]] = {}

    if "global" in model_names:
        global_model = GlobalForecaster(
            recent_days=90,
            fill_missing_days=True,
            max_resampled_rows=int(args.max_resampled_rows),
            verbose=False,
        )
        global_model.fit(train_df)
        preds = _predict_global(global_model, top_items, int(args.val_days))
        summaries["global"] = _export_model_outputs(
            "GlobalForecaster",
            out_root / "global",
            val_df,
            preds,
            top_n_items=int(args.top_items_plot),
        )

    if "local" in model_names:
        preds = _predict_local(train_df, top_items, int(args.val_days))
        summaries["local"] = _export_model_outputs(
            "ProductionForecaster (Local)",
            out_root / "local",
            val_df,
            preds,
            top_n_items=int(args.top_items_plot),
        )

    if "hybrid" in model_names:
        orders = pipeline.orders.copy()
        orders = orders[pd.to_datetime(orders["created"], unit="s") <= cutoff]
        oi_train = oi[oi["date"] <= cutoff].copy()
        hybrid = HybridForecaster()
        hybrid.fit(orders, oi_train)
        preds = _predict_hybrid(hybrid, train_df, top_items, int(args.val_days))
        summaries["hybrid"] = _export_model_outputs(
            "HybridForecaster",
            out_root / "hybrid",
            val_df,
            preds,
            top_n_items=int(args.top_items_plot),
        )

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print("Wrote results to:", str(out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
