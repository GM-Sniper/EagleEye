"""
EagleEye Demand Feature Builder

Creates a supervised learning table for item demand forecasting at the
(date, place_id, item_id) grain.

This module intentionally focuses on feature engineering. Model selection
and training are handled separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import warnings


@dataclass(frozen=True)
class DemandFeatureConfig:
    # Time-series features
    lags: Tuple[int, ...] = (1, 7, 14, 28)
    rolling_windows: Tuple[int, ...] = (7, 14, 28)

    # Categorical mix features from orders (shares for top-K categories)
    top_k_mix_categories: int = 5
    mix_columns: Tuple[str, ...] = ("type", "channel", "source", "payment_method")

    # Calendar
    add_holidays: bool = True
    holiday_country: str = "DK"

    # Correct rolling features require missing days = 0.
    # This can be memory-heavy; a safety threshold is enforced.
    fill_missing_days: bool = True
    max_resampled_rows: int = 5_000_000


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace(0, np.nan)
    out = numer / denom
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def _sanitize_category(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "__nan__"
    text = str(value).strip().lower()
    if not text:
        return "__empty__"
    cleaned = []
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")
    out = "".join(cleaned)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "__empty__"


def _entropy_from_counts(counts: pd.Series) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts.astype(float) / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _compute_mix_features(
    orders: pd.DataFrame,
    group_cols: Sequence[str],
    col: str,
    top_categories: Sequence[object],
) -> pd.DataFrame:
    if col not in orders.columns:
        return pd.DataFrame(columns=[*group_cols])

    df = orders[[*group_cols, col]].copy()
    df[col] = df[col].astype("object")

    # Shares for top categories
    ct = (
        df.groupby([*group_cols, col])
        .size()
        .rename("count")
        .reset_index()
    )
    total = ct.groupby(list(group_cols))["count"].sum().rename("total")
    ct = ct.merge(total.reset_index(), on=list(group_cols), how="left")
    ct["share"] = _safe_div(ct["count"], ct["total"]).astype(float)

    keep = ct[col].isin(list(top_categories))
    ct_keep = ct.loc[keep, [*group_cols, col, "share"]]
    ct_keep[col] = ct_keep[col].map(_sanitize_category)
    wide = ct_keep.pivot_table(
        index=list(group_cols),
        columns=col,
        values="share",
        fill_value=0.0,
        aggfunc="sum",
    )
    wide.columns = [f"{col}_share_{c}" for c in wide.columns]
    wide = wide.reset_index()

    # Entropy (diversity) feature
    ent = (
        df.groupby(list(group_cols))[col]
        .value_counts(dropna=False)
        .groupby(level=list(range(len(group_cols))))
        .apply(_entropy_from_counts)
        .rename(f"{col}_entropy")
        .reset_index()
    )

    out = wide.merge(ent, on=list(group_cols), how="outer")
    return out


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_month"] = ((df["day_of_month"] - 1) // 7 + 1).astype(int)

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = (df["day_of_month"] <= 5).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= 25).astype(int)
    df["is_payday_like"] = df["day_of_month"].isin([1, 15, 25]).astype(int)
    df["is_payday_like"] = ((df["is_payday_like"] == 1) | (df["is_month_end"] == 1)).astype(int)

    # Cyclical encodings
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["woy_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 53)
    df["woy_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 53)

    return df


def _add_holiday_features(df: pd.DataFrame, country: str) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    years = sorted(df["date"].dt.year.unique().tolist())
    if not years:
        df["is_holiday"] = 0
        return df

    try:
        from importlib import import_module

        make_holidays_df = import_module("prophet.make_holidays").make_holidays_df
    except Exception:
        warnings.warn("prophet is not available; skipping holiday features")
        df["is_holiday"] = 0
        return df

    try:
        holidays = make_holidays_df(year_list=years, country=country)
        holidays = holidays.rename(columns={"ds": "date"})
        holidays["date"] = pd.to_datetime(holidays["date"]).dt.normalize()
        holidays = holidays[["date"]].drop_duplicates()
        df = df.merge(holidays.assign(is_holiday=1), on="date", how="left")
        df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
        return df
    except Exception:
        warnings.warn("Failed to compute holiday features; skipping")
        df["is_holiday"] = 0
        return df


def _estimate_resampled_rows(df: pd.DataFrame, group_cols: Sequence[str], date_col: str) -> int:
    ranges = df.groupby(list(group_cols))[date_col].agg(["min", "max"])
    days = (ranges["max"] - ranges["min"]).dt.days + 1
    days = days.clip(lower=0).fillna(0)
    return int(days.sum())


def _resample_to_daily(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    date_col: str,
    numeric_cols: Sequence[str],
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df = df.sort_values([*group_cols, date_col])

    frames: list[pd.DataFrame] = []
    for keys, g in df.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        start = g[date_col].min()
        end = g[date_col].max()
        idx = pd.date_range(start=start, end=end, freq="D")
        g2 = g.set_index(date_col).reindex(idx)
        g2.index.name = date_col
        g2 = g2.reset_index()

        for c, v in zip(group_cols, keys):
            g2[c] = v

        for c in numeric_cols:
            if c in g2.columns:
                g2[c] = g2[c].fillna(0)

        frames.append(g2)

    if not frames:
        return df.iloc[:0].copy()
    return pd.concat(frames, ignore_index=True)


def _add_lag_and_rolling(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    target_col: str,
    lags: Iterable[int],
    windows: Iterable[int],
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([*group_cols, "date"])

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby(list(group_cols))[target_col].shift(lag).fillna(0)

    for window in windows:
        # IMPORTANT: rolling features must not include the current day's target.
        # Use a 1-step shifted series so that features at date t only depend on <= t-1.
        prev = df.groupby(list(group_cols))[target_col].shift(1).fillna(0)
        df["__prev_target__"] = prev
        roll = df.groupby(list(group_cols))["__prev_target__"].rolling(window=window, min_periods=1)
        df[f"{target_col}_roll_mean_{window}"] = roll.mean().reset_index(level=list(range(len(group_cols))), drop=True)
        df[f"{target_col}_roll_std_{window}"] = roll.std().reset_index(level=list(range(len(group_cols))), drop=True).fillna(0)
        df[f"{target_col}_roll_sum_{window}"] = roll.sum().reset_index(level=list(range(len(group_cols))), drop=True)
        df[f"{target_col}_roll_nonzero_{window}"] = (
            df.groupby(list(group_cols))["__prev_target__"]
            .rolling(window=window, min_periods=1)
            .apply(lambda x: float(np.count_nonzero(np.asarray(x) > 0)), raw=True)
            .reset_index(level=list(range(len(group_cols))), drop=True)
        )
        df[f"{target_col}_roll_sales_freq_{window}"] = _safe_div(
            df[f"{target_col}_roll_nonzero_{window}"], pd.Series(window, index=df.index)
        )

    df = df.drop(columns=["__prev_target__"], errors="ignore")

    return df


def _add_exact_daily_lags_by_merge(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    date_col: str,
    target_col: str,
    lags: Iterable[int],
) -> pd.DataFrame:
    """Adds exact calendar-day lags without requiring a dense daily index.

    For each lag L, the feature at date D equals the target value at date D-L.
    Missing days are treated as 0.
    """
    df = df.copy()
    base = df[[date_col, *group_cols, target_col]].copy()
    base[date_col] = pd.to_datetime(base[date_col]).dt.normalize()

    for lag in lags:
        shifted = base.copy()
        shifted[date_col] = shifted[date_col] + pd.Timedelta(days=lag)
        feat = f"{target_col}_lag_{lag}"
        shifted = shifted.rename(columns={target_col: feat})
        df = df.merge(shifted[[date_col, *group_cols, feat]], on=[date_col, *group_cols], how="left")
        df[feat] = df[feat].fillna(0)
    return df


def _add_sparse_time_window_rollups(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    date_col: str,
    target_col: str,
    windows: Iterable[int],
) -> pd.DataFrame:
    """Adds rolling features using time windows on a sparse daily table.

    Assumes one row per (date, group). Missing dates are absent (not zero rows).
    Rolling sums over a time window are still correct (missing days contribute 0).
    Rolling means/std are computed assuming missing days are zeros by dividing by
    the window length in days.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df = df.sort_values([*group_cols, date_col])

    frames: list[pd.DataFrame] = []
    for keys, g in df.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        g = g.sort_values(date_col).copy()
        g = g.set_index(date_col)
        # IMPORTANT: exclude the current day's target from rolling features.
        s = g[target_col].astype(float).shift(1).fillna(0)
        s2 = (s ** 2).astype(float)

        for w in windows:
            win = f"{int(w)}D"
            roll_sum = s.rolling(win, min_periods=1).sum()
            roll_sumsq = s2.rolling(win, min_periods=1).sum()
            mean = roll_sum / float(w)
            mean_sq = roll_sumsq / float(w)
            var = (mean_sq - (mean ** 2)).clip(lower=0)
            std = np.sqrt(var)

            g[f"{target_col}_roll_sum_{w}"] = roll_sum.values
            g[f"{target_col}_roll_mean_{w}"] = mean.values
            g[f"{target_col}_roll_std_{w}"] = std.values
            nonzero = s.rolling(win, min_periods=1).apply(lambda x: float(np.count_nonzero(np.asarray(x) > 0)), raw=True)
            g[f"{target_col}_roll_nonzero_{w}"] = nonzero.values
            g[f"{target_col}_roll_sales_freq_{w}"] = (nonzero / float(w)).values

        g2 = g.reset_index()
        for c, v in zip(group_cols, keys):
            g2[c] = v
        frames.append(g2)

    if not frames:
        return df.iloc[:0].copy()
    return pd.concat(frames, ignore_index=True)


def _add_lifecycle_features(df: pd.DataFrame, group_cols: Sequence[str], target_col: str) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([*group_cols, "date"])

    frames: list[pd.DataFrame] = []
    for keys, g in df.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        g = g.sort_values("date").copy()
        sold = g[target_col] > 0
        if sold.any():
            first_sale = g.loc[sold, "date"].iloc[0]
        else:
            first_sale = g["date"].iloc[0]
        g["days_since_first_sale"] = (g["date"] - first_sale).dt.days.astype(int)

        last_sale_date = pd.NaT
        days_since_last: list[int] = []
        zero_streak: list[int] = []
        streak = 0
        for d, is_sold in zip(g["date"], sold):
            if is_sold:
                last_sale_date = d
                streak = 0
            else:
                streak += 1
            if pd.isna(last_sale_date):
                days_since_last.append(9999)
            else:
                days_since_last.append(int((d - last_sale_date).days))
            zero_streak.append(streak)

        g["days_since_last_sale"] = days_since_last
        g["zero_sales_streak"] = zero_streak

        for c, v in zip(group_cols, keys):
            g[c] = v

        frames.append(g)

    if not frames:
        return df.iloc[:0].copy()
    return pd.concat(frames, ignore_index=True)


def _select_and_prepare_item_features(items: pd.DataFrame) -> pd.DataFrame:
    items = items.copy()
    if "id" not in items.columns:
        raise ValueError("dim_items must contain 'id'")

    out = pd.DataFrame({"item_id": items["id"]})
    for col in [
        "section_id",
        "type",
        "vat",
        "price",
        "discountable",
        "manage_inventory",
        "variable_price",
        "delivery",
        "eat_in",
        "takeaway",
        "voucher",
        "status",
        "deleted",
    ]:
        if col in items.columns:
            out[f"item_{col}"] = items[col]

    # Numeric coercions
    for col in ["item_vat", "item_price"]:
        if col in out.columns:
            out[col] = _to_numeric(out[col]).fillna(0)

    # Boolean-ish to int
    for col in [
        "item_discountable",
        "item_manage_inventory",
        "item_variable_price",
        "item_delivery",
        "item_eat_in",
        "item_takeaway",
        "item_voucher",
        "item_deleted",
    ]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.lower().isin(["1", "true", "t", "yes"]).astype(int)

    # Frequency encodings for key categoricals (keep raw IDs too)
    for col in ["item_section_id", "item_type", "item_status"]:
        if col in out.columns:
            counts = out[col].value_counts(dropna=False)
            out[f"{col}_freq"] = out[col].map(counts).fillna(0).astype(float)

    return out


def _select_and_prepare_place_features(places: pd.DataFrame) -> pd.DataFrame:
    places = places.copy()
    if "id" not in places.columns:
        raise ValueError("dim_places must contain 'id'")

    out = pd.DataFrame({"place_id": places["id"]})
    for col in [
        "active",
        "activated",
        "bankrupt",
        "binding_period",
        "chain_id",
        "country",
        "vat",
        "vip_threshold",
        "visit_duration",
        "votes",
        "waiting_time",
    ]:
        if col in places.columns:
            out[f"place_{col}"] = places[col]

    for col in [
        "place_binding_period",
        "place_vat",
        "place_vip_threshold",
        "place_visit_duration",
        "place_votes",
        "place_waiting_time",
    ]:
        if col in out.columns:
            out[col] = _to_numeric(out[col]).fillna(0)

    for col in ["place_active", "place_activated", "place_bankrupt"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.lower().isin(["1", "true", "t", "yes"]).astype(int)

    for col in ["place_chain_id", "place_country"]:
        if col in out.columns:
            counts = out[col].value_counts(dropna=False)
            out[f"{col}_freq"] = out[col].map(counts).fillna(0).astype(float)

    return out


class DemandFeatureBuilder:
    """Builds a demand forecasting dataset at (date, place_id, item_id)."""

    def __init__(self, config: DemandFeatureConfig = DemandFeatureConfig()):
        self.config = config

    def build(
        self,
        orders: pd.DataFrame,
        order_items: pd.DataFrame,
        items: pd.DataFrame,
        places: pd.DataFrame,
        *,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        place_ids: Optional[Sequence[int]] = None,
        item_ids: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame at grain (date, place_id, item_id) with:
          - target: demand_qty
          - price/margin/discount/commission aggregates
          - place traffic features + channel mix shares
          - calendar + holiday features
          - lag/rolling features
          - lifecycle / intermittency proxy features
          - static item/place metadata + frequency encodings
          - rolling popularity rank within place
          - ABC class (global, from historical revenue)
        """

        if "id" not in orders.columns:
            raise ValueError("orders must contain 'id'")
        if "order_id" not in order_items.columns:
            raise ValueError("order_items must contain 'order_id'")
        if "item_id" not in order_items.columns:
            raise ValueError("order_items must contain 'item_id'")
        if "quantity" not in order_items.columns:
            raise ValueError("order_items must contain 'quantity'")

        ord_df = orders.copy()
        ord_df["date"] = pd.to_datetime(ord_df["date"]).dt.normalize()
        ord_df["total_amount"] = _to_numeric(ord_df.get("total_amount", 0)).fillna(0)

        if place_ids is not None and "place_id" in ord_df.columns:
            ord_df = ord_df[ord_df["place_id"].isin(place_ids)]

        if min_date is not None:
            ord_df = ord_df[ord_df["date"] >= pd.to_datetime(min_date)]
        if max_date is not None:
            ord_df = ord_df[ord_df["date"] <= pd.to_datetime(max_date)]

        oi = order_items.copy()
        oi["created_dt"] = pd.to_datetime(oi["created"], unit="s", errors="coerce")
        oi["date"] = oi["created_dt"].dt.normalize()

        if item_ids is not None:
            oi = oi[oi["item_id"].isin(item_ids)]

        if min_date is not None:
            oi = oi[oi["date"] >= pd.to_datetime(min_date)]
        if max_date is not None:
            oi = oi[oi["date"] <= pd.to_datetime(max_date)]

        join_cols = ["id", "place_id", "date", "total_amount"]
        for c in ["type", "channel", "source", "payment_method", "customer_mobile_phone"]:
            if c in ord_df.columns:
                join_cols.append(c)
        join_cols = list(dict.fromkeys(join_cols))

        oi = oi.merge(
            ord_df[join_cols],
            left_on="order_id",
            right_on="id",
            how="inner",
            suffixes=("", "_order"),
        )

        # Monetary line features
        oi["quantity"] = _to_numeric(oi["quantity"]).fillna(0)
        oi["unit_price"] = _to_numeric(oi.get("price", 0)).fillna(0)
        oi["unit_cost"] = _to_numeric(oi.get("cost", 0)).fillna(0)
        oi["discount_amount"] = _to_numeric(oi.get("discount_amount", 0)).fillna(0)
        oi["commission_amount"] = _to_numeric(oi.get("commission_amount", 0)).fillna(0)

        oi["line_revenue"] = oi["unit_price"] * oi["quantity"]
        oi["line_cost"] = oi["unit_cost"] * oi["quantity"]
        oi["line_margin"] = oi["line_revenue"] - oi["line_cost"]

        # Per (date, place, item)
        key = ["date", "place_id", "item_id"]
        grp = oi.groupby(key, sort=False)
        weighted_price_sum = (oi["unit_price"] * oi["quantity"]).groupby([oi["date"], oi["place_id"], oi["item_id"]]).sum()
        weighted_cost_sum = (oi["unit_cost"] * oi["quantity"]).groupby([oi["date"], oi["place_id"], oi["item_id"]]).sum()

        agg = grp.agg(
            demand_qty=("quantity", "sum"),
            n_order_lines=("id", "count"),
            n_orders=("order_id", "nunique"),
            revenue_sum=("line_revenue", "sum"),
            cost_sum=("line_cost", "sum"),
            margin_sum=("line_margin", "sum"),
            discount_sum=("discount_amount", "sum"),
            commission_sum=("commission_amount", "sum"),
            unit_price_std=("unit_price", "std"),
        ).reset_index()

        agg["unit_price_std"] = agg["unit_price_std"].fillna(0)
        w = pd.DataFrame(weighted_price_sum.rename("_wprice")).reset_index()
        w = w.merge(pd.DataFrame(weighted_cost_sum.rename("_wcost")).reset_index(), on=key, how="left")
        agg = agg.merge(w, on=key, how="left")
        agg["unit_price_wavg"] = _safe_div(agg["_wprice"].fillna(0), agg["demand_qty"])
        agg["unit_cost_wavg"] = _safe_div(agg["_wcost"].fillna(0), agg["demand_qty"])
        agg = agg.drop(columns=["_wprice", "_wcost"], errors="ignore")
        agg["unit_margin"] = agg["unit_price_wavg"] - agg["unit_cost_wavg"]
        agg["unit_margin_pct"] = _safe_div(agg["unit_margin"], agg["unit_price_wavg"])
        agg["discount_per_unit"] = _safe_div(agg["discount_sum"], agg["demand_qty"])
        agg["commission_per_unit"] = _safe_div(agg["commission_sum"], agg["demand_qty"])
        agg["discount_rate"] = _safe_div(agg["discount_sum"], (agg["revenue_sum"] + agg["discount_sum"]))
        agg["commission_rate"] = _safe_div(agg["commission_sum"], agg["revenue_sum"])
        agg["avg_line_qty"] = _safe_div(agg["demand_qty"], agg["n_order_lines"])

        # Decide whether to build a dense daily index (includes zero-demand days).
        group_cols = ["place_id", "item_id"]
        fill_missing = bool(self.config.fill_missing_days)
        if fill_missing:
            approx_rows = _estimate_resampled_rows(agg[["date", *group_cols]].drop_duplicates(), group_cols, "date")
            if approx_rows > self.config.max_resampled_rows:
                warnings.warn(
                    f"Resampling would create ~{approx_rows:,} rows; disabling fill_missing_days to avoid memory issues. "
                    "Filter place_ids/item_ids or raise max_resampled_rows to force a dense daily index (includes zeros)."
                )
                fill_missing = False

        if fill_missing:
            # Resample numeric columns; missing days become 0 for demand + monetary values.
            numeric_cols = [c for c in agg.columns if c not in ["date", *group_cols]]
            df = agg.drop_duplicates(subset=["date", *group_cols], keep="last")
            df = _resample_to_daily(df, group_cols, "date", numeric_cols)
        else:
            df = agg

        # Place-day traffic features
        place_day = ord_df.groupby(["date", "place_id"]).agg(
            place_orders=("id", "count"),
            place_revenue=("total_amount", "sum"),
            place_aov=("total_amount", "mean"),
        ).reset_index()
        place_day["place_aov"] = place_day["place_aov"].fillna(0)

        if "customer_mobile_phone" in ord_df.columns:
            uniq = (
                ord_df.groupby(["date", "place_id"])["customer_mobile_phone"]
                .nunique(dropna=True)
                .rename("place_unique_customers")
                .reset_index()
            )
            place_day = place_day.merge(uniq, on=["date", "place_id"], how="left")
            place_day["place_unique_customers"] = place_day["place_unique_customers"].fillna(0)

        # Mix features (shares + entropy)
        for mix_col in self.config.mix_columns:
            if mix_col in ord_df.columns:
                top = ord_df[mix_col].value_counts(dropna=False).head(self.config.top_k_mix_categories).index.tolist()
                mix = _compute_mix_features(ord_df, ["date", "place_id"], mix_col, top)
                place_day = place_day.merge(mix, on=["date", "place_id"], how="left")

        # Fill any mix NAs with 0
        mix_feature_cols = [c for c in place_day.columns if c not in ["date", "place_id", "place_orders", "place_revenue", "place_aov", "place_unique_customers"]]
        for c in mix_feature_cols:
            place_day[c] = place_day[c].fillna(0)

        # Shift place-day features by 1 day per place to avoid same-day leakage.
        # At date t, these represent information available as-of t-1.
        place_day = place_day.sort_values(["place_id", "date"]).copy()
        pd_feature_cols = [c for c in place_day.columns if c not in ["date", "place_id"]]
        place_day[pd_feature_cols] = (
            place_day.groupby("place_id", sort=False)[pd_feature_cols]
            .shift(1)
            .fillna(0)
        )

        # Merge lagged place-day to item-demand
        df = df.merge(place_day, on=["date", "place_id"], how="left")
        for c in ["place_orders", "place_revenue", "place_aov"]:
            if c in df.columns:
                df[c] = df[c].fillna(0)

        # NOTE: Basket share features based on same-day demand_qty are leaky and are intentionally omitted.

        # Calendar + holidays
        df = _add_calendar_features(df)
        if self.config.add_holidays:
            df = _add_holiday_features(df, self.config.holiday_country)

        # Static item/place features
        item_feat = _select_and_prepare_item_features(items)
        place_feat = _select_and_prepare_place_features(places)
        df = df.merge(item_feat, on="item_id", how="left")
        df = df.merge(place_feat, on="place_id", how="left")

        # Place size proxies without future leakage:
        # use expanding means of *lagged* daily traffic features per place.
        place_sizes = place_day.sort_values(["place_id", "date"]).copy()
        place_sizes["place_avg_daily_orders"] = (
            place_sizes.groupby("place_id", sort=False)["place_orders"]
            .expanding(min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        place_sizes["place_avg_daily_revenue"] = (
            place_sizes.groupby("place_id", sort=False)["place_revenue"]
            .expanding(min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df = df.merge(
            place_sizes[["date", "place_id", "place_avg_daily_orders", "place_avg_daily_revenue"]],
            on=["date", "place_id"],
            how="left",
        )
        df[["place_avg_daily_orders", "place_avg_daily_revenue"]] = df[["place_avg_daily_orders", "place_avg_daily_revenue"]].fillna(0)

        # ABC class (global) from historical item revenue
        try:
            from services.inventory_service import InventoryService

            # Build item_stats-like frame
            item_stats = (
                df.groupby("item_id").agg(
                    total_qty=("demand_qty", "sum"),
                    mean_daily=("demand_qty", "mean"),
                    std_daily=("demand_qty", "std"),
                    price=("unit_price_wavg", "mean"),
                ).reset_index()
            )
            item_stats["std_daily"] = item_stats["std_daily"].fillna(0)
            item_stats["cv"] = _safe_div(item_stats["std_daily"], item_stats["mean_daily"])
            # Minimal item_name for compatibility
            item_stats["item_name"] = ""

            abc = InventoryService().classify_items_abc(
                item_stats.rename(columns={"total_qty": "total_qty", "price": "price", "item_id": "item_id", "item_name": "item_name"})
            )
            abc_map = abc.set_index("item_id")["abc_class"].to_dict()
            df["abc_class"] = df["item_id"].map(abc_map).fillna("C")
            df["abc_A"] = (df["abc_class"] == "A").astype(int)
            df["abc_B"] = (df["abc_class"] == "B").astype(int)
            df["abc_C"] = (df["abc_class"] == "C").astype(int)
        except Exception:
            df["abc_class"] = "C"
            df["abc_A"] = 0
            df["abc_B"] = 0
            df["abc_C"] = 1

        # Lag/rolling + lifecycle features
        if fill_missing:
            df = _add_lag_and_rolling(df, group_cols, "demand_qty", self.config.lags, self.config.rolling_windows)
            df = _add_lifecycle_features(df, group_cols, "demand_qty")
        else:
            # Sparse mode (no dense daily index): add exact calendar-day lags via merge,
            # time-window rollups, and gap-based lifecycle features.
            df = _add_exact_daily_lags_by_merge(df, group_cols, "date", "demand_qty", self.config.lags)
            df = _add_sparse_time_window_rollups(df, group_cols, "date", "demand_qty", self.config.rolling_windows)

            df = df.sort_values(["place_id", "item_id", "date"])
            first_date = df.groupby(group_cols)["date"].transform("min")
            prev_date = df.groupby(group_cols)["date"].shift(1)
            df["days_since_first_sale"] = (df["date"] - first_date).dt.days.fillna(0).astype(int)
            gap = (df["date"] - prev_date).dt.days
            df["days_since_last_sale"] = gap.fillna(9999).astype(int)
            df["zero_sales_streak"] = (df["days_since_last_sale"] - 1).clip(lower=0).astype(int)

        # Popularity rank within place using 28-day rolling sum
        if "demand_qty_roll_sum_28" in df.columns:
            score = df["demand_qty_roll_sum_28"]
        else:
            score = df["demand_qty"].fillna(0)
        df["place_item_popularity_rank_pct"] = (
            df.assign(_score=score)
            .groupby(["place_id", "date"])["_score"]
            .rank(pct=True)
            .fillna(0)
            .astype(float)
        )
        df = df.drop(columns=[c for c in ["_score"] if c in df.columns], errors="ignore")

        # Final cleanup
        df = df.sort_values(["date", "place_id", "item_id"]).reset_index(drop=True)
        return df
