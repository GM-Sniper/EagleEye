"""Inventory planning from item-demand predictions.

This module bridges demand forecasting outputs (per item_id) to inventory needs.

Inputs:
- Demand predictions at grain (date, place_id, item_id), typically from
  src/train_xgb_item_demand_ensemble.py (validation predictions) or a future
  inference job.
- dim_skus.csv for current on-hand quantities and low stock thresholds.
- dim_bill_of_materials.csv to convert a sold item SKU into ingredient SKUs.

Notes
- This repo's Data/fct_inventory_reports.csv currently contains only a header
  row in the provided workspace snapshot, so we use dim_skus.quantity as the
  available on-hand signal.
- If a predicted item_id cannot be mapped to a parent SKU (dim_skus.item_id is
  null for many ingredient SKUs), it will be skipped in the BOM expansion.
"""

import sys
from pathlib import Path
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InventoryPlanConfig:
    prediction_qty_col: str = "y_pred"
    min_qty: float = 0.0


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def build_ingredient_requirements(
    predictions: pd.DataFrame,
    skus: pd.DataFrame,
    bom: pd.DataFrame,
    *,
    config: InventoryPlanConfig = InventoryPlanConfig(),
) -> pd.DataFrame:
    """Return ingredient-level requirements and shortages.

    Parameters
    - predictions: Must have columns item_id and config.prediction_qty_col.
    - skus: dim_skus with id, item_id, title, quantity, low_stock_threshold, unit.
    - bom: dim_bill_of_materials with parent_sku_id, sku_id, quantity.

    Output columns
    - ingredient_sku_id, ingredient_title, unit
    - required_qty, on_hand_qty, shortage_qty
    - low_stock_threshold
    """

    if "item_id" not in predictions.columns:
        raise ValueError("predictions must have column 'item_id'")
    if config.prediction_qty_col not in predictions.columns:
        raise ValueError(f"predictions must have column '{config.prediction_qty_col}'")

    for col in ["id", "title", "quantity", "low_stock_threshold", "unit"]:
        if col not in skus.columns:
            raise ValueError(f"dim_skus missing required column '{col}'")

    for col in ["parent_sku_id", "sku_id", "quantity"]:
        if col not in bom.columns:
            raise ValueError(f"dim_bill_of_materials missing required column '{col}'")

    pred = predictions[["item_id", config.prediction_qty_col]].copy()
    pred[config.prediction_qty_col] = pd.to_numeric(pred[config.prediction_qty_col], errors="coerce").fillna(0.0)
    pred = pred[pred[config.prediction_qty_col] > float(config.min_qty)]

    skus_map = skus.copy()
    if "item_id" in skus_map.columns:
        skus_map["item_id"] = pd.to_numeric(skus_map["item_id"], errors="coerce")

    # Map item_id -> parent_sku_id using dim_skus.item_id (keep first SKU if multiple).
    parent = (
        skus_map.dropna(subset=["item_id"])
        .sort_values(["item_id", "id"])
        .drop_duplicates(subset=["item_id"], keep="first")
        .rename(columns={"id": "parent_sku_id"})[["item_id", "parent_sku_id"]]
    )

    pred = pred.merge(parent, on="item_id", how="left")
    pred = pred.dropna(subset=["parent_sku_id"]).copy()
    pred["parent_sku_id"] = pred["parent_sku_id"].astype(int)

    # Expand to ingredients via BOM.
    bom2 = bom[["parent_sku_id", "sku_id", "quantity"]].copy()
    bom2["parent_sku_id"] = pd.to_numeric(bom2["parent_sku_id"], errors="coerce")
    bom2["sku_id"] = pd.to_numeric(bom2["sku_id"], errors="coerce")
    bom2["quantity"] = pd.to_numeric(bom2["quantity"], errors="coerce").fillna(0.0)
    bom2 = bom2.dropna(subset=["parent_sku_id", "sku_id"]).copy()
    bom2["parent_sku_id"] = bom2["parent_sku_id"].astype(int)
    bom2["sku_id"] = bom2["sku_id"].astype(int)

    expanded = pred.merge(bom2, on="parent_sku_id", how="inner")
    expanded["required_qty"] = expanded[config.prediction_qty_col].astype(float) * expanded["quantity"].astype(float)

    req = (
        expanded.groupby("sku_id", as_index=False)["required_qty"]
        .sum()
        .rename(columns={"sku_id": "ingredient_sku_id"})
    )

    inv = skus_map.rename(columns={"id": "ingredient_sku_id", "quantity": "on_hand_qty"}).copy()
    inv["ingredient_sku_id"] = pd.to_numeric(inv["ingredient_sku_id"], errors="coerce")
    inv = inv.dropna(subset=["ingredient_sku_id"]).copy()
    inv["ingredient_sku_id"] = inv["ingredient_sku_id"].astype(int)

    out = req.merge(
        inv[["ingredient_sku_id", "title", "unit", "on_hand_qty", "low_stock_threshold"]],
        on="ingredient_sku_id",
        how="left",
    )

    out["on_hand_qty"] = pd.to_numeric(out["on_hand_qty"], errors="coerce").fillna(0.0)
    out["low_stock_threshold"] = pd.to_numeric(out["low_stock_threshold"], errors="coerce").fillna(0.0)

    out["shortage_qty"] = np.maximum(0.0, out["required_qty"].astype(float) - out["on_hand_qty"].astype(float))
    out = out.rename(columns={"title": "ingredient_title"})

    # Order: most critical shortages first.
    out = out.sort_values(["shortage_qty", "required_qty"], ascending=[False, False]).reset_index(drop=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Create an ingredient-level inventory plan from demand predictions.")
    ap.add_argument("--predictions", type=str, required=True, help="CSV with item_id and y_pred (or chosen qty col).")
    ap.add_argument("--skus", type=str, default="Data/dim_skus.csv", help="Path to dim_skus.csv")
    ap.add_argument("--bom", type=str, default="Data/dim_bill_of_materials.csv", help="Path to dim_bill_of_materials.csv")
    ap.add_argument("--qty-col", type=str, default="y_pred", help="Column in predictions to use as quantity")
    ap.add_argument("--min-qty", type=float, default=0.0, help="Minimum predicted qty to include")
    ap.add_argument("--out", type=str, default=str(Path("artifacts") / "inventory_plan.csv"), help="Output CSV path")

    args = ap.parse_args()

    preds = _read_csv(args.predictions)
    skus = _read_csv(args.skus)
    bom = _read_csv(args.bom)

    plan = build_ingredient_requirements(
        preds,
        skus,
        bom,
        config=InventoryPlanConfig(prediction_qty_col=args.qty_col, min_qty=float(args.min_qty)),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out_path, index=False)
    print("Wrote:", str(out_path))
    print("Rows:", len(plan))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())