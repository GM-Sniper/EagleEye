#!/usr/bin/env python3
"""
EagleEye - Comprehensive Data Analysis Script
Generates a detailed findings report for the inventory management hackathon.
"""

import sys
from pathlib import Path
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🦅 EAGLEEYE - COMPREHENSIVE DATA ANALYSIS")
print("=" * 70)

# Load data
DATA_DIR = '../Data'
print("\n📥 Loading datasets...")

datasets = {}
for f in sorted(os.listdir(DATA_DIR)):
    if f.endswith('.csv'):
        name = f.replace('.csv', '')
        datasets[name] = pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False)
        print(f"  ✓ {name}: {datasets[name].shape[0]:,} rows")

# =============================================================================
# 1. TEMPORAL ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("📅 TEMPORAL ANALYSIS")
print("=" * 70)

orders = datasets['fct_orders'].copy()
orders['created_dt'] = pd.to_datetime(orders['created'], unit='s')
orders['date'] = orders['created_dt'].dt.date
orders['year'] = orders['created_dt'].dt.year
orders['month'] = orders['created_dt'].dt.month
orders['day_of_week'] = orders['created_dt'].dt.day_name()
orders['hour'] = orders['created_dt'].dt.hour
orders['week'] = orders['created_dt'].dt.isocalendar().week

print(f"\n📆 Date Range: {orders['created_dt'].min().date()} to {orders['created_dt'].max().date()}")
print(f"📊 Total Orders: {len(orders):,}")
print(f"📍 Unique Locations: {orders['place_id'].nunique()}")

# Orders by year
print("\n📈 Orders by Year:")
yearly = orders.groupby('year').size()
for year, count in yearly.items():
    print(f"   {year}: {count:,} orders")

# Day of week patterns
print("\n📅 Orders by Day of Week:")
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow = orders.groupby('day_of_week').size().reindex(dow_order)
total = dow.sum()
for day, count in dow.items():
    pct = count / total * 100
    bar = "█" * int(pct / 2)
    print(f"   {day:10s}: {count:>7,} ({pct:5.1f}%) {bar}")

# Peak hours
print("\n⏰ Peak Hours (Top 5):")
hourly = orders.groupby('hour').size().sort_values(ascending=False)
for hour, count in hourly.head(5).items():
    print(f"   {hour:02d}:00 - {count:,} orders")

# =============================================================================
# 2. REVENUE ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("💰 REVENUE ANALYSIS")
print("=" * 70)

orders['total_amount'] = pd.to_numeric(orders['total_amount'], errors='coerce')
total_revenue = orders['total_amount'].sum()
avg_order_value = orders['total_amount'].mean()

print(f"\n💵 Total Revenue: {total_revenue:,.2f} DKK")
print(f"📊 Average Order Value: {avg_order_value:.2f} DKK")
print(f"🔸 Median Order Value: {orders['total_amount'].median():.2f} DKK")
print(f"🔺 Max Order Value: {orders['total_amount'].max():,.2f} DKK")

# Revenue by year
print("\n📈 Revenue by Year:")
yearly_rev = orders.groupby('year')['total_amount'].sum()
for year, rev in yearly_rev.items():
    print(f"   {year}: {rev:,.2f} DKK")

# Top places by revenue
print("\n🏪 Top 10 Places by Revenue:")
place_rev = orders.groupby('place_id')['total_amount'].sum().sort_values(ascending=False)
for i, (place, rev) in enumerate(place_rev.head(10).items()):
    print(f"   {i+1}. Place {int(place)}: {rev:,.2f} DKK")

# =============================================================================
# 3. ORDER ITEMS ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("🍽️ ORDER ITEMS ANALYSIS")
print("=" * 70)

order_items = datasets['fct_order_items']
print(f"\n📦 Total Order Items: {len(order_items):,}")
print(f"🆔 Unique Items Ordered: {order_items['item_id'].nunique()}")

# Most ordered items (from pre-computed table)
if 'most_ordered' in datasets:
    most_ordered = datasets['most_ordered']
    print("\n🏆 Top 20 Most Ordered Items:")
    top_items = most_ordered.nlargest(20, 'order_count')
    for i, row in top_items.iterrows():
        name = row['item_name'] if pd.notna(row['item_name']) else 'Unknown'
        count = row['order_count']
        print(f"   {name[:40]:40s}: {count:,} orders")

# =============================================================================
# 4. DEMAND VARIABILITY (For Forecasting)
# =============================================================================
print("\n" + "=" * 70)
print("📊 DEMAND VARIABILITY ANALYSIS")
print("=" * 70)

# Daily demand patterns
daily_orders = orders.groupby('date').size()
print(f"\n📅 Daily Order Statistics:")
print(f"   Mean: {daily_orders.mean():.1f} orders/day")
print(f"   Std Dev: {daily_orders.std():.1f}")
print(f"   Min: {daily_orders.min()} orders")
print(f"   Max: {daily_orders.max()} orders")
print(f"   Coefficient of Variation: {(daily_orders.std() / daily_orders.mean() * 100):.1f}%")

# Item-level demand variability
item_daily = order_items.copy()
item_daily['date'] = pd.to_datetime(item_daily['created'], unit='s').dt.date
item_demand = item_daily.groupby(['item_id', 'date'])['quantity'].sum().reset_index()
item_cv = item_demand.groupby('item_id')['quantity'].agg(['mean', 'std', 'count'])
item_cv['cv'] = item_cv['std'] / item_cv['mean']
item_cv = item_cv[item_cv['count'] >= 30]  # At least 30 days of data

print("\n📈 Item Demand Variability (items with 30+ days of data):")
print(f"   Items analyzed: {len(item_cv):,}")
print(f"   Avg CV: {item_cv['cv'].mean():.2f}")
print(f"   High variability items (CV > 1.0): {(item_cv['cv'] > 1.0).sum()}")
print(f"   Low variability items (CV < 0.5): {(item_cv['cv'] < 0.5).sum()}")

# =============================================================================
# 5. PLACES ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("🏪 PLACES ANALYSIS")
print("=" * 70)

places = datasets['dim_places']
print(f"\n📍 Total Places: {len(places)}")

if 'country' in places.columns:
    print("\n🌍 Places by Country:")
    country_counts = places['country'].value_counts().head(10)
    for country, count in country_counts.items():
        print(f"   {country}: {count}")

# Orders per place
orders_per_place = orders.groupby('place_id').size()
print(f"\n📊 Orders per Place Statistics:")
print(f"   Mean: {orders_per_place.mean():.1f}")
print(f"   Median: {orders_per_place.median():.1f}")
print(f"   Top 10 places account for: {orders_per_place.nlargest(10).sum() / orders_per_place.sum() * 100:.1f}% of orders")

# =============================================================================
# 6. ORDER TYPES & PAYMENT ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("🛒 ORDER TYPES & PAYMENT ANALYSIS")
print("=" * 70)

if 'type' in orders.columns:
    print("\n📦 Order Types:")
    type_counts = orders['type'].value_counts()
    for otype, count in type_counts.items():
        pct = count / len(orders) * 100
        if pd.notna(otype):
            print(f"   {otype}: {count:,} ({pct:.1f}%)")

if 'payment_method' in orders.columns:
    print("\n💳 Payment Methods:")
    payment_counts = orders['payment_method'].value_counts()
    for method, count in payment_counts.head(10).items():
        pct = count / len(orders) * 100
        if pd.notna(method):
            print(f"   {method}: {count:,} ({pct:.1f}%)")

if 'status' in orders.columns:
    print("\n📋 Order Status:")
    status_counts = orders['status'].value_counts()
    for status, count in status_counts.items():
        pct = count / len(orders) * 100
        print(f"   {status}: {count:,} ({pct:.1f}%)")

# =============================================================================
# 7. WEEKDAY VS WEEKEND
# =============================================================================
print("\n" + "=" * 70)
print("📅 WEEKDAY VS WEEKEND ANALYSIS")
print("=" * 70)

orders['is_weekend'] = orders['day_of_week'].isin(['Saturday', 'Sunday'])
weekend_stats = orders.groupby('is_weekend').agg({
    'id': 'count',
    'total_amount': ['mean', 'sum']
}).round(2)

weekday_orders = orders[~orders['is_weekend']]
weekend_orders = orders[orders['is_weekend']]

print(f"\n📅 Weekday Orders: {len(weekday_orders):,} ({len(weekday_orders)/len(orders)*100:.1f}%)")
print(f"   Avg Order Value: {weekday_orders['total_amount'].mean():.2f} DKK")

print(f"\n🌴 Weekend Orders: {len(weekend_orders):,} ({len(weekend_orders)/len(orders)*100:.1f}%)")
print(f"   Avg Order Value: {weekend_orders['total_amount'].mean():.2f} DKK")

# =============================================================================
# 8. MONTHLY TRENDS
# =============================================================================
print("\n" + "=" * 70)
print("📆 MONTHLY TRENDS")
print("=" * 70)

orders['year_month'] = orders['created_dt'].dt.to_period('M')
monthly = orders.groupby('year_month').agg({
    'id': 'count',
    'total_amount': 'sum'
}).rename(columns={'id': 'orders', 'total_amount': 'revenue'})

print("\n📈 Monthly Order Volume (last 12 months shown):")
for period, row in monthly.tail(12).iterrows():
    print(f"   {period}: {int(row['orders']):,} orders, {row['revenue']:,.0f} DKK")

# =============================================================================
# 9. KEY FINDINGS SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("📋 KEY FINDINGS SUMMARY")
print("=" * 70)

findings = {
    "data_overview": {
        "total_orders": int(len(orders)),
        "total_order_items": int(len(order_items)),
        "unique_places": int(orders['place_id'].nunique()),
        "unique_items": int(order_items['item_id'].nunique()),
        "date_range": f"{orders['created_dt'].min().date()} to {orders['created_dt'].max().date()}"
    },
    "revenue": {
        "total_revenue_dkk": float(total_revenue),
        "avg_order_value_dkk": float(avg_order_value)
    },
    "temporal_patterns": {
        "busiest_day": dow.idxmax(),
        "slowest_day": dow.idxmin(),
        "peak_hour": int(hourly.idxmax()),
        "daily_avg_orders": float(daily_orders.mean()),
        "daily_cv": float(daily_orders.std() / daily_orders.mean())
    },
    "demand_insights": {
        "high_variability_items": int((item_cv['cv'] > 1.0).sum()),
        "low_variability_items": int((item_cv['cv'] < 0.5).sum())
    }
}

print("\n✅ Analysis complete! Key insights:")
print(f"   • {findings['data_overview']['total_orders']:,} orders worth {findings['revenue']['total_revenue_dkk']:,.0f} DKK")
print(f"   • Busiest day: {findings['temporal_patterns']['busiest_day']}")
print(f"   • Peak hour: {findings['temporal_patterns']['peak_hour']}:00")
print(f"   • Daily variability (CV): {findings['temporal_patterns']['daily_cv']:.2f}")
print(f"   • {findings['demand_insights']['high_variability_items']} items with high demand variability")

# Save findings to JSON
with open('findings.json', 'w') as f:
    json.dump(findings, f, indent=2, default=str)
print("\n💾 Findings saved to src/findings.json")

print("\n" + "=" * 70)
print("✅ ANALYSIS COMPLETE")
print("=" * 70)