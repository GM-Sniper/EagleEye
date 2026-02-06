"""Leakage sanity checks for the demand feature table.

This is a pragmatic audit to catch common leakage patterns:
- rolling windows including same-day target
- same-day place traffic features (place_orders/place_revenue)

Run from repo root, example:
  python src/leakage_audit.py --val-days 28 --min-date 2023-08-20 --max-date 2024-02-16 --place-ids 94025
"""

from __future__ import annotations

import argparse
from datetime import timedelta

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-days", type=int, default=28)
    ap.add_argument("--min-date", type=str, default=None)
    ap.add_argument("--max-date", type=str, default=None)
    ap.add_argument("--place-ids", type=str, default=None)
    ap.add_argument("--item-ids", type=str, default=None)
    args = ap.parse_args()

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

    pipe = DataPipeline(data_dir="Data")
    pipe.load_all()
    orders, order_items, items = pipe.load_core_tables()

    df = pipe.build_item_place_item_demand_features(
        min_date=args.min_date,
        max_date=args.max_date,
        place_ids=place_ids,
        item_ids=item_ids,
        config=DemandFeatureConfig(fill_missing_days=True),
    )

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    max_date = df["date"].max()
    cutoff = max_date - timedelta(days=int(args.val_days) - 1)
    val = df[df["date"] >= cutoff].copy()

    print(f"Rows total={len(df):,} val={len(val):,} cutoff={cutoff.date()}")

    # 1) Check that rolling windows don't trivially contain same-day target.
    suspicious = []
    for c in df.columns:
        if c.startswith("demand_qty_roll_"):
            s = val[["demand_qty", c]].dropna()
            if len(s) == 0:
                continue
            eq_rate = float(np.mean(np.isclose(s["demand_qty"].to_numpy(), s[c].to_numpy())))
            corr = float(np.corrcoef(s["demand_qty"].to_numpy(), s[c].to_numpy())[0, 1]) if len(s) > 2 else float("nan")
            suspicious.append((c, eq_rate, corr))

    suspicious = sorted(suspicious, key=lambda x: (-x[1], -abs(x[2]) if np.isfinite(x[2]) else 0.0))
    print("\nTop roll features by equality-to-target rate (val):")
    for c, eq_rate, corr in suspicious[:10]:
        print(f"  {c}: eq_rate={eq_rate:.4f} corr={corr:.4f}")

    # 2) Check that place_orders aligns with previous day's true orders count.
    # Recompute true daily orders per place from raw orders.
    ord_df = orders.copy()
    ord_df["date"] = pd.to_datetime(ord_df["date"]).dt.normalize()
    if place_ids is not None and "place_id" in ord_df.columns:
        ord_df = ord_df[ord_df["place_id"].isin(place_ids)]
    true_place_day = (
        ord_df.groupby(["date", "place_id"], as_index=False)["id"].count().rename(columns={"id": "true_place_orders"})
    )
    true_place_day["date"] = pd.to_datetime(true_place_day["date"]).dt.normalize()

    check = val[["date", "place_id"]].drop_duplicates().merge(
        val[["date", "place_id", "place_orders"]].drop_duplicates(),
        on=["date", "place_id"],
        how="left",
    )

    # Join true orders for t-1
    true_prev = true_place_day.copy()
    true_prev["date"] = true_prev["date"] + pd.Timedelta(days=1)
    check = check.merge(true_prev, on=["date", "place_id"], how="left")

    if "place_orders" in check.columns:
        a = pd.to_numeric(check["place_orders"], errors="coerce").fillna(0).to_numpy(dtype=float)
        b = pd.to_numeric(check["true_place_orders"], errors="coerce").fillna(0).to_numpy(dtype=float)
        mae = float(np.mean(np.abs(a - b))) if len(a) else float("nan")
        exact = float(np.mean(a == b)) if len(a) else float("nan")
        print("\nplace_orders lag check (val place-days):")
        print(f"  MAE(place_orders vs true_orders[t-1]) = {mae:.4f}")
        print(f"  exact_match_rate = {exact:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
