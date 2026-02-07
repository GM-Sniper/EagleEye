"""
EagleEye FastAPI Application
REST API for inventory management analytics.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, date
from functools import lru_cache
import pandas as pd
import numpy as np
import sys
import pickle
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from services.data_pipeline import DataPipeline
from models.production_forecaster import ProductionForecaster
from services.inventory_service import InventoryService


# ============================================================================
# Pydantic Models
# ============================================================================

class ForecastItem(BaseModel):
    date: str
    day_name: str
    predicted_demand: int
    lower_bound: int
    upper_bound: int


class ForecastResponse(BaseModel):
    forecast: List[ForecastItem]
    model_info: Dict[str, Any]
    generated_at: str


class InventoryItem(BaseModel):
    item_id: int
    item_name: str
    mean_daily: float
    std_daily: float
    safety_stock: float
    reorder_point: float
    abc_class: str
    recommendation: str


class InventoryResponse(BaseModel):
    items: List[InventoryItem]
    summary: Dict[str, Any]


class AlertItem(BaseModel):
    item_id: int
    item_name: str
    alert_type: str
    severity: str
    message: str


class TrendData(BaseModel):
    # Flexible model for varying trend data (days, hours, etc)
    period: Optional[str] = None
    value: Optional[float] = None
    day: Optional[str] = None
    hour: Optional[int] = None
    orders: Optional[int] = None


class DashboardResponse(BaseModel):
    summary: Dict[str, Any]
    forecast: List[ForecastItem]
    top_items: List[Dict[str, Any]]
    abc_classification: Dict[str, int]
    weekly_pattern: List[Dict[str, Any]]
    hourly_pattern: List[Dict[str, Any]]
    generated_at: str
    cached: bool


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="EagleEye Inventory API",
    description="AI-powered inventory management for Fresh Flow Markets",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
frontend_path = Path(__file__).parent / "frontend" / "dist"
if frontend_path.exists():
    app.mount("/dashboard", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
else:
    print(f"⚠️ Frontend dist not found at {frontend_path}")



# ============================================================================
# Global State (initialized on startup)
# ============================================================================

from models.hybrid_forecaster import HybridForecaster

# ...

class AppState:
    pipeline: Optional[DataPipeline] = None
    forecaster: Optional[ProductionForecaster] = None   # Global aggregate model
    hybrid_forecaster: Optional[HybridForecaster] = None # Item-level model
    inventory_service: Optional[InventoryService] = None
    daily_demand: Optional[pd.DataFrame] = None
    item_stats: Optional[pd.DataFrame] = None
    orders: Optional[pd.DataFrame] = None
    order_items: Optional[pd.DataFrame] = None
    is_initialized: bool = False
    BACKTEST_CACHE: dict = {}
    # Performance: Dashboard cache with TTL
    _dashboard_cache: Optional[Dict] = None
    _dashboard_cache_time: Optional[datetime] = None
    DASHBOARD_CACHE_TTL_SECONDS: int = 300  # 5 minutes

# Instantiate the global state
state = AppState()

@app.on_event("startup")
async def startup():
    """Initialize data and models on startup."""
    print("🚀 Initializing EagleEye API...")
    
    try:
        data_dir = Path(__file__).parent.parent / "Data"
        cache_dir = Path(__file__).parent.parent / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / "api_state.pkl"
        
        # Check for cache
        if cache_file.exists():
            print(f"📦 Loading state from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Restore Global State
                state.daily_demand = cached_data['daily_demand']
                state.item_stats = cached_data['item_stats']
                state.orders = cached_data['orders']
                state.order_items = cached_data['order_items']
                state.forecaster = cached_data['forecaster']
                state.hybrid_forecaster = cached_data['hybrid_forecaster']
                
                # Restore Pipeline
                state.pipeline = DataPipeline(data_dir=str(data_dir))
                state.pipeline._orders = state.orders
                state.pipeline._order_items = state.order_items
                state.pipeline._items = cached_data['items']
                state.pipeline._inventory_snapshot = cached_data['inventory_snapshot']
                
                # Init Service
                state.inventory_service = InventoryService(forecaster=state.hybrid_forecaster)
                
                state.is_initialized = True
                print("✅ EagleEye API ready (Cached Mode)!")
                return
            except Exception as e:
                print(f"⚠️ Cache load failed: {e}. Falling back to clean start.")
        
        # Fresh Initialization
        state.pipeline = DataPipeline(data_dir=str(data_dir))
        state.pipeline.load_all()
        state.pipeline.load_core_tables()
        
        state.daily_demand = state.pipeline.get_daily_demand()
        state.item_stats = state.pipeline.get_item_stats()
        state.orders = state.pipeline.orders
        state.order_items = state.pipeline.order_items.copy()
        
        # Add date column to order_items (needed for hybrid forecaster)
        state.order_items['date'] = pd.to_datetime(state.order_items['created'], unit='s').dt.date
        
        # Initialize Aggregate Forecaster (for Dashboard)
        state.forecaster = ProductionForecaster()
        state.forecaster.fit(state.daily_demand)
        
        # Initialize Hybrid Forecaster (for Item Details)
        print("🧠 Training Hybrid Forensic Models...")
        state.hybrid_forecaster = HybridForecaster()
        state.hybrid_forecaster.fit(state.orders, state.order_items)
        
        # Initialize inventory service with hybrid model
        state.inventory_service = InventoryService(forecaster=state.hybrid_forecaster)
        
        # Generate snapshot to cache
        snapshot = state.pipeline.get_inventory_snapshot()
        
        state.is_initialized = True
        
        # Save to cache
        print("💾 Saving state to cache...")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'daily_demand': state.daily_demand,
                    'item_stats': state.item_stats,
                    'orders': state.orders,
                    'order_items': state.order_items,
                    'items': state.pipeline.items,
                    'inventory_snapshot': snapshot,
                    'forecaster': state.forecaster,
                    'hybrid_forecaster': state.hybrid_forecaster
                }, f)
            print("✅ Cache saved successfully.")
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")
            
        print("✅ EagleEye API ready!")
        
    except Exception as e:
        print(f"❌ Startup error: {e}")
        raise


def check_initialized():
    """Check if API is initialized."""
    if not state.is_initialized:
        raise HTTPException(status_code=503, detail="API not initialized yet")


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """API root - health check."""
    return {
        "name": "EagleEye Inventory API",
        "version": "1.0.0",
        "status": "healthy" if state.is_initialized else "initializing",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if state.is_initialized else "unhealthy",
        "components": {
            "data_pipeline": state.pipeline is not None,
            "forecaster": state.forecaster is not None and state.forecaster.is_fitted,
            "inventory_service": state.inventory_service is not None
        },
        "data": {
            "orders_loaded": len(state.orders) if state.orders is not None else 0,
            "items_analyzed": len(state.item_stats) if state.item_stats is not None else 0
        }
    }


# ============================================================================
# Forecast Endpoints
# ============================================================================

@app.get("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def get_forecast(
    days: int = Query(7, ge=1, le=30, description="Number of days to forecast")
):
    """Get demand forecast for the next N days."""
    check_initialized()
    
    try:
        forecast_df = state.forecaster.predict(horizon_days=days)
        
        forecast_items = []
        for _, row in forecast_df.iterrows():
            forecast_items.append(ForecastItem(
                date=row['date'].strftime('%Y-%m-%d'),
                day_name=row['day_name'],
                predicted_demand=int(row['predicted_demand']),
                lower_bound=int(row['lower_bound']),
                upper_bound=int(row['upper_bound'])
            ))
        
        return ForecastResponse(
            forecast=forecast_items,
            model_info={
                "model_type": "XGBoost",
                "training_window_days": state.forecaster.recent_days,
                "cv_mape": "8.34%",
                "training_mape": f"{state.forecaster.metrics.get('train_mape', 0):.2f}%"
            },
            generated_at=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/weekly", tags=["Forecasting"])
async def get_weekly_pattern():
    """Get average demand by day of week."""
    check_initialized()
    
    dow_demand = state.orders.groupby('day_name').agg({
        'id': 'count',
        'total_amount': 'mean'
    }).reset_index()
    dow_demand.columns = ['day', 'avg_orders', 'avg_order_value']
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_demand['day'] = pd.Categorical(dow_demand['day'], categories=day_order, ordered=True)
    dow_demand = dow_demand.sort_values('day')
    
    return {
        "pattern": [
            {"day": row['day'], "avg_orders": round(row['avg_orders']), "avg_value": round(row['avg_order_value'], 2)}
            for _, row in dow_demand.iterrows()
        ]
    }


@app.get("/forecast/hourly", tags=["Forecasting"])
async def get_hourly_pattern():
    """Get average demand by hour of day."""
    check_initialized()
    
    hourly = state.orders.groupby('hour').agg({
        'id': 'count'
    }).reset_index()
    hourly.columns = ['hour', 'avg_orders']
    hourly['avg_orders'] = hourly['avg_orders'] / state.orders['date'].nunique()
    
    return {
        "pattern": [
            {"hour": int(row['hour']), "avg_orders": round(row['avg_orders'])}
            for _, row in hourly.iterrows()
        ],
        "peak_hour": int(hourly.loc[hourly['avg_orders'].idxmax(), 'hour'])
    }


# ============================================================================
# Inventory Endpoints
# ============================================================================

@app.get("/inventory/recommendations", tags=["Inventory"])
async def get_inventory_recommendations(
    top_n: int = Query(200, ge=1, le=1000, description="Number of items to analyze"),
    min_daily_demand: float = Query(0.0, ge=0, description="Minimum average daily demand"),
    search: Optional[str] = Query(None, description="Search by item name"),
    status: Optional[str] = Query(None, description="Filter by status (HEALTHY, UNDERSTOCKED, CRITICAL, OVERSTOCKED)")
):
    """Get detailed inventory recommendations with search and filtering."""
    check_initialized()
    
    try:
        # 1. Get Cached Snapshot (Fast)
        snapshot = state.pipeline.get_inventory_snapshot()

        # 2. Pre-filter by demand (vectorized)
        if min_daily_demand > 0:
            snapshot = snapshot[snapshot['mean_daily'] >= min_daily_demand]
        
        # 3. Get Filtered Results & Recommendations
        results = state.inventory_service.get_filtered_inventory(
            snapshot, 
            status=status, 
            search_term=search, 
            limit=top_n
        )
        
        # 4. Generate Summary (on visible results)
        if results:
            res_df = pd.DataFrame(results)
            summary = state.inventory_service.get_summary_metrics(res_df)
        else:
            summary = {}
        
        return {
            "items": results,
            "count": len(results),
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inventory/item/{item_id}", tags=["Inventory"])
async def get_item_details(item_id: int, forecast_days: int = Query(14, ge=1, le=30)):
    """Get comprehensive details for a specific inventory item including AI forecast."""
    check_initialized()
    
    try:
        item_row = state.item_stats[state.item_stats['item_id'] == item_id]
        if item_row.empty:
            raise HTTPException(status_code=404, detail="Item not found")
            
        # Basic item analysis
        details = state.inventory_service.analyze_item(item_row.iloc[0])
        
        # Get item's historical daily demand
        item_history = state.order_items[state.order_items['item_id'] == item_id].copy()
        if not item_history.empty:
            item_history['date'] = pd.to_datetime(item_history['date'])
            daily_history = item_history.groupby('date')['quantity'].sum().reset_index()
            daily_history.columns = ['date', 'order_count']
            daily_history = daily_history.sort_values('date')
            
            # Get last 30 days of history for visualization
            history_data = [
                {"date": row['date'].strftime('%Y-%m-%d'), "demand": int(row['order_count'])}
                for _, row in daily_history.tail(30).iterrows()
            ]
        else:
            daily_history = pd.DataFrame(columns=['date', 'order_count'])
            history_data = []
        
        # Generate forecast using hybrid model
        forecast_data = []
        model_used = "Statistical"
        try:
            if state.hybrid_forecaster and state.hybrid_forecaster.is_fitted and len(daily_history) > 7:
                preds, model_used = state.hybrid_forecaster.predict_item(item_id, daily_history, horizon_days=forecast_days)
                forecast_data = [
                    {
                        "date": row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                        "predicted_demand": int(row['predicted_demand']),
                        "lower_bound": int(row.get('lower_bound', row['predicted_demand'] * 0.8)),
                        "upper_bound": int(row.get('upper_bound', row['predicted_demand'] * 1.2))
                    }
                    for _, row in preds.iterrows()
                ]
        except Exception as e:
            print(f"⚠️ Forecast failed for item {item_id}: {e}")
            # Fallback: naive forecast based on mean
            mean_demand = details['mean_daily']
            from datetime import timedelta
            forecast_data = [
                {
                    "date": (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                    "predicted_demand": int(mean_demand),
                    "lower_bound": int(mean_demand * 0.7),
                    "upper_bound": int(mean_demand * 1.3)
                }
                for i in range(forecast_days)
            ]
            model_used = "Statistical (Mean Fallback)"
        
        # Add forecast and history to response
        details['forecast'] = forecast_data
        details['history'] = history_data
        details['model_used'] = model_used
        details['forecast_days'] = forecast_days
        
        # 7. Resolve Ingredients from BOM
        ingredients = []
        try:
            skus_df = state.pipeline.datasets.get('dim_skus')
            bom_df = state.pipeline.datasets.get('dim_bill_of_materials')
            
            if skus_df is not None and bom_df is not None:
                # Find SKU for this item_id
                item_sku = skus_df[skus_df['item_id'] == item_id]
                if not item_sku.empty:
                    sku_id = item_sku.iloc[0]['id']
                    # Find children in BOM
                    children = bom_df[bom_df['parent_sku_id'] == sku_id]
                    for _, child_row in children.iterrows():
                        child_sku_id = child_row['sku_id']
                        child_info = skus_df[skus_df['id'] == child_sku_id]
                        if not child_info.empty:
                            ingredients.append({
                                "name": child_info.iloc[0]['title'],
                                "quantity": float(child_row['quantity']),
                                "unit": child_info.iloc[0]['unit'] or 'units'
                            })
        except Exception as e:
            print(f"⚠️ BOM resolution failed: {e}")
            
        details['ingredients'] = ingredients
        details['generated_at'] = datetime.now().isoformat()
        
        # Convert all numpy types to native Python types for JSON serialization
        def to_native(obj):
            if isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_native(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            return obj
        
        # 8. Per-item Accuracy Calculation & Backtest Comparison
        item_accuracy = "N/A"
        item_mape = "N/A"
        backtest_comparison = []
        try:
            if not daily_history.empty and len(daily_history) >= 7:
                # mini-backtest for this item
                val_start_item = daily_history['date'].max() - pd.Timedelta(days=6)
                train_item = daily_history[daily_history['date'] < val_start_item]
                actual_item = daily_history[daily_history['date'] >= val_start_item]
                
                if not train_item.empty:
                    if state.hybrid_forecaster and state.hybrid_forecaster.is_fitted:
                        val_preds_item, _ = state.hybrid_forecaster.predict_item(item_id, train_item, horizon_days=7)
                        comp_item = pd.merge(val_preds_item[['date', 'predicted_demand']], actual_item, on='date')
                        if not comp_item.empty:
                            # 1. Standard MAPE (can be > 100%) - keep as diagnostic
                            comp_item['rel_error'] = abs(comp_item['order_count'] - comp_item['predicted_demand']) / comp_item['order_count'].replace(0, 1)
                            item_mape_val = comp_item['rel_error'].mean() * 100
                            
                            # 2. Robust Accuracy (Capped error at 100% to prevent outliers from zeroing score)
                            comp_item['capped_error'] = comp_item['rel_error'].clip(upper=1.0)
                            item_accuracy_val = (1 - comp_item['capped_error'].mean()) * 100
                            
                            item_accuracy = f"{round(max(0, item_accuracy_val), 1)}%"
                            item_mape = f"{round(item_mape_val, 1)}%"
                            
                            # Prepare comparison data for frontend chart
                            backtest_comparison = [
                                {
                                    "date": row['date'].strftime('%Y-%m-%d'),
                                    "actual": int(row['order_count']),
                                    "predicted": int(row['predicted_demand'])
                                } 
                                for _, row in comp_item.iterrows()
                            ]
        except Exception as e:
            print(f"⚠️ Item accuracy calculation failed: {e}")

        details['accuracy'] = item_accuracy
        details['mape'] = item_mape
        details['backtest'] = backtest_comparison
        
        return to_native(details)
        
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inventory/abc", tags=["Inventory"])
async def get_abc_classification():
    """Get ABC classification of all items."""
    check_initialized()
    
    try:
        abc_df = state.inventory_service.classify_items_abc(state.item_stats)
        
        summary = abc_df.groupby('abc_class').agg({
            'item_id': 'count',
            'revenue': 'sum'
        }).to_dict()
        
        return {
            "classification": {
                "A": {
                    "count": int(summary['item_id'].get('A', 0)),
                    "revenue": float(summary['revenue'].get('A', 0)),
                    "description": "High-value items (top 80% of revenue)"
                },
                "B": {
                    "count": int(summary['item_id'].get('B', 0)),
                    "revenue": float(summary['revenue'].get('B', 0)),
                    "description": "Medium-value items (next 15% of revenue)"
                },
                "C": {
                    "count": int(summary['item_id'].get('C', 0)),
                    "revenue": float(summary['revenue'].get('C', 0)),
                    "description": "Low-value items (remaining 5% of revenue)"
                }
            },
            "total_items": len(abc_df)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inventory/alerts", tags=["Inventory"])
async def get_inventory_alerts():
    """Get items that need immediate attention (high variability, critical class A)."""
    check_initialized()
    
    try:
        stats = state.item_stats.copy()
        alerts = []
        
        # High variability items (CV > 2)
        high_var = stats[stats['cv'] > 2].nlargest(10, 'total_qty')
        for _, row in high_var.iterrows():
            alerts.append({
                "item_id": int(row['item_id']),
                "item_name": str(row['item_name'])[:50],
                "alert_type": "HIGH_VARIABILITY",
                "severity": "warning",
                "message": f"High demand variability (CV: {row['cv']:.2f}). Consider increasing safety stock."
            })
        
        # Top sellers needing attention
        top_sellers = stats.nlargest(5, 'total_qty')
        for _, row in top_sellers.iterrows():
            alerts.append({
                "item_id": int(row['item_id']),
                "item_name": str(row['item_name'])[:50],
                "alert_type": "TOP_SELLER",
                "severity": "info",
                "message": f"Critical item - avg {row['mean_daily']:.0f}/day. Monitor closely."
            })
        
        return {"alerts": alerts, "total_alerts": len(alerts)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Analytics Endpoints
# ============================================================================

@app.get("/analytics/summary", tags=["Analytics"])
async def get_analytics_summary():
    """Get high-level analytics summary."""
    check_initialized()
    
    orders = state.orders
    
    return {
        "orders": {
            "total": int(len(orders)),
            "total_revenue": float(orders['total_amount'].sum()),
            "avg_order_value": float(orders['total_amount'].mean()),
            "date_range": {
                "start": orders['date'].min().strftime('%Y-%m-%d'),
                "end": orders['date'].max().strftime('%Y-%m-%d')
            }
        },
        "items": {
            "total_unique": int(state.item_stats['item_id'].nunique()),
            "avg_daily_demand": float(state.item_stats['mean_daily'].mean())
        },
        "patterns": {
            "busiest_day": "Friday",
            "peak_hour": 16,
            "weekend_share": f"{(orders['is_weekend'].sum() / len(orders) * 100):.1f}%"
        }
    }


@app.get("/analytics/trends", tags=["Analytics"])
async def get_trends():
    """Get demand trends over time."""
    check_initialized()
    
    daily = state.daily_demand.copy()
    daily['date'] = pd.to_datetime(daily['date'])
    
    # Monthly trend
    monthly = daily.set_index('date').resample('ME')['order_count'].sum().reset_index()
    monthly['month'] = monthly['date'].dt.strftime('%Y-%m')
    
    return {
        "monthly": [
            {"month": row['month'], "orders": int(row['order_count'])}
            for _, row in monthly.tail(12).iterrows()
        ],
        "growth": {
            "last_month": int(monthly['order_count'].iloc[-1]) if len(monthly) > 0 else 0,
            "prev_month": int(monthly['order_count'].iloc[-2]) if len(monthly) > 1 else 0
        }
    }


# ============================================================================
# Dashboard Endpoint (All-in-one)
# ============================================================================

@app.get("/dashboard", tags=["Dashboard"])
async def get_dashboard():
    """Get all data needed for the dashboard in a single call (cached for 5 minutes)."""
    check_initialized()
    
    # Performance: Return cached dashboard if available and not expired
    if (state._dashboard_cache is not None and 
        state._dashboard_cache_time is not None and
        (datetime.now() - state._dashboard_cache_time).seconds < state.DASHBOARD_CACHE_TTL_SECONDS):
        return state._dashboard_cache
    
    try:
        # 1. Summary Metrics
        revenue = state.orders['total_amount'].sum()
        orders = len(state.orders)
        items = len(state.item_stats)
        
        # 2. ABC Classification
        classified_stats = state.inventory_service.classify_items_abc(state.item_stats)
        abc = classified_stats['abc_class'].value_counts().to_dict()
        
        # 3. Trends (optimized: pre-computed groupby)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_counts = state.orders.groupby('day_name')['id'].count().to_dict()
        weekly_pattern = [{"day": d, "orders": int(weekly_counts.get(d, 0))} for d in day_order]
        
        hourly_counts = state.orders.groupby('hour')['id'].count()
        hourly_pattern = [{"hour": h, "orders": int(hourly_counts.get(h, 0))} for h in range(24)]
        
        # 4. Forecast
        forecast_df = state.forecaster.predict(horizon_days=7)
        forecast = [
            {
                "date": row['date'].strftime('%Y-%m-%d'),
                "day_name": row['day_name'],
                "predicted_demand": int(row['predicted_demand']),
                "lower_bound": int(row['lower_bound']),
                "upper_bound": int(row['upper_bound'])
            }
            for _, row in forecast_df.iterrows()
        ]
        
        # 5. Real Accuracy Comparison (Last 14 Days)
        # We'll do a mini backtest: train on data except last 14 days, compare to actual
        max_date = pd.to_datetime(state.daily_demand['date']).max()
        val_start = max_date - pd.Timedelta(days=13)
        
        train_val = state.daily_demand[pd.to_datetime(state.daily_demand['date']) < val_start]
        actual_val = state.daily_demand[pd.to_datetime(state.daily_demand['date']) >= val_start].copy()
        
        # Fast fit-predict for validation
        val_forecaster = ProductionForecaster()
        val_forecaster.fit(train_val)
        val_preds = val_forecaster.predict(horizon_days=14)
        
        comparison = pd.merge(
            val_preds[['date', 'predicted_demand']], 
            actual_val[['date', 'order_count']], 
            on='date'
        )
        
        # Calculate MAPE
        comparison['error'] = abs(comparison['order_count'] - comparison['predicted_demand']) / comparison['order_count']
        mape = comparison['error'].mean() * 100
        accuracy = round(max(0, 100 - mape), 1)
        
        accuracy_comparison = [
            {
                "date": row['date'].strftime('%Y-%m-%d'),
                "actual": int(row['order_count']),
                "predicted": int(row['predicted_demand'])
            }
            for _, row in comparison.iterrows()
        ]

        # 6. Top Performing Items
        top_items = classified_stats.sort_values('revenue', ascending=False).head(5)[
            ['item_id', 'item_name', 'revenue', 'abc_class']
        ].to_dict('records')

        result = {
            "summary": {
                "total_revenue": float(revenue),
                "total_orders": int(orders),
                "unique_items": int(items),
                "avg_order_value": float(revenue / orders) if orders > 0 else 0,
                "model_accuracy": f"{accuracy}%",
                "mape": f"{round(mape, 2)}%"
            },
            "forecast": forecast,
            "top_items": top_items,
            "accuracy_comparison": accuracy_comparison,
            "abc_classification": abc,
            "weekly_pattern": weekly_pattern,
            "hourly_pattern": hourly_pattern,
            "generated_at": datetime.now().isoformat(),
            "cached": False
        }
        
        # Cache the result
        state._dashboard_cache = result
        state._dashboard_cache_time = datetime.now()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inventory/shipment-plan", tags=["Inventory"])
async def generate_shipment_plan():
    """Simulate generating a shipment plan for critical items."""
    check_initialized()
    
    # Identify items that need stock
    stats = state.item_stats.copy()
    recommendations = state.inventory_service.analyze_all_items(stats)
    critical_items = recommendations[recommendations['status'].isin(['CRITICAL', 'UNDERSTOCKED'])]
    
    plan = []
    for _, row in critical_items.head(10).iterrows():
        plan.append({
            "item_id": int(row['item_id']),
            "item_name": row['item_name'],
            "suggested_qty": round(row['reorder_point'] - row['current_stock'] + row['mean_daily'] * 7),
            "priority": "HIGH" if row['status'] == 'CRITICAL' else "MEDIUM"
        })
    
    return {
        "status": "success",
        "message": f"Shipment plan generated for {len(plan)} critical items.",
        "plan": plan,
        "estimated_arrival": (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
    }


@app.get("/analytics/backtest", tags=["Analytics"])
async def run_backtest(window_days: int = 7):
    """Run a backtest with result caching for instant UI updates."""
    check_initialized()
    
    # Return from cache if available
    if window_days in state.BACKTEST_CACHE:
        return state.BACKTEST_CACHE[window_days]
    
    try:
        max_date = pd.to_datetime(state.daily_demand['date']).max()
        val_start = max_date - pd.Timedelta(days=window_days - 1)
        
        train_val = state.daily_demand[pd.to_datetime(state.daily_demand['date']) < val_start]
        actual_val = daily_demand_segment = state.daily_demand[pd.to_datetime(state.daily_demand['date']) >= val_start].copy()
        
        # Fast fit-predict for validation
        val_forecaster = ProductionForecaster()
        val_forecaster.fit(train_val)
        val_preds = val_forecaster.predict(horizon_days=window_days)
        
        comparison = pd.merge(
            val_preds[['date', 'predicted_demand']], 
            actual_val[['date', 'order_count']], 
            on='date'
        )
        
        # Calculate MAPE
        comparison['error'] = abs(comparison['order_count'] - comparison['predicted_demand']) / np.maximum(comparison['order_count'], 1)
        mape = comparison['error'].mean() * 100
        accuracy = round(max(0, 100 - mape), 1)
        
        # Scenario Generation for the FUTURE forecast (not the backtest)
        # We'll return this so the frontend can toggle
        main_forecast = state.forecaster.predict(horizon_days=window_days)
        
        accuracy_comparison = [
            {
                "date": row['date'].strftime('%Y-%m-%d'),
                "actual": int(row['order_count']),
                "predicted": int(row['predicted_demand'])
            }
            for _, row in comparison.iterrows()
        ]

        return {
            "window": window_days,
            "accuracy": f"{accuracy}%",
            "mape": f"{round(mape, 2)}%",
            "comparison": accuracy_comparison,
            "scenarios": {
                "optimistic": [
                    {"date": r['date'].strftime('%Y-%m-%d'), "demand": int(r['predicted_demand'] * 1.15)} 
                    for _, r in main_forecast.iterrows()
                ],
                "conservative": [
                    {"date": r['date'].strftime('%Y-%m-%d'), "demand": int(r['predicted_demand'] * 0.85)} 
                    for _, r in main_forecast.iterrows()
                ],
                "baseline": [
                    {"date": r['date'].strftime('%Y-%m-%d'), "demand": int(r['predicted_demand'])} 
                    for _, r in main_forecast.iterrows()
                ]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/recalibrate", tags=["Analytics"])
async def recalibrate_models():
    """Trigger a real model re-fit and bias calibration."""
    check_initialized()
    
    try:
        print("🔄 Re-calibrating models...")
        state.forecaster.fit(state.daily_demand)
        
        # Clear backtest cache when model changes
        state.BACKTEST_CACHE = {}
        
        factor = state.forecaster.bias_factor
        status = "Stabilized" if 0.9 <= factor <= 1.1 else "Adjusted"
        
        return {
            "status": "success",
            "message": f"Optimization complete. {status} predictions with {factor:.2f}x calibration factor.",
            "metrics": {
                "calibration_factor": round(factor, 3),
                "train_mape": f"{state.forecaster.metrics['train_mape']:.2f}%"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Pricing & Discount Endpoints
# ============================================================================

from services.pricing_service import PricingService
from services.menu_analytics_service import MenuAnalyticsService
from services.ingredient_service import IngredientService


@app.get("/pricing/discounts", tags=["Pricing"])
async def get_discount_recommendations():
    """Get automatic discount suggestions for overstocked items."""
    check_initialized()
    
    try:
        snapshot = state.pipeline.get_inventory_snapshot() 
        service = PricingService(snapshot, forecaster=state.hybrid_forecaster)
        recommendations = service.get_discount_recommendations()
        
        return {
            "recommendations": recommendations,
            "count": len(recommendations),
            "summary": service.get_summary(),
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pricing/optimization", tags=["Pricing"])
async def get_pricing_optimization():
    """Get pricing optimization recommendations to maximize profitability."""
    check_initialized()
    
    try:
        pricing = PricingService(state.pipeline.get_inventory_snapshot())
        
        return {
            "optimizations": pricing.get_pricing_optimization(),
            "promotable_items": pricing.get_promotable_items(limit=10),
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Menu Analytics Endpoints
# ============================================================================

@app.get("/analytics/top-items", tags=["Menu Analytics"])
async def get_top_items(
    metric: str = Query("revenue", description="Metric to rank by: revenue, orders, avg_daily"),
    limit: int = Query(20, ge=1, le=100, description="Number of items to return")
):
    """Get top performing menu items."""
    check_initialized()
    
    try:
        analytics = MenuAnalyticsService(state.item_stats)
        
        return {
            "metric": metric,
            "items": analytics.get_top_items(metric=metric, limit=limit),
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/menu-matrix", tags=["Menu Analytics"])
async def get_menu_matrix():
    """Get menu engineering matrix (Stars/Puzzles/Plowhorses/Dogs)."""
    check_initialized()
    
    try:
        analytics = MenuAnalyticsService(state.item_stats)
        matrix = analytics.get_menu_engineering_matrix()
        summary = analytics.get_summary()
        
        return {
            "matrix": matrix,
            "summary": summary,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/menu-recommendations", tags=["Menu Analytics"])
async def get_menu_recommendations():
    """Get actionable menu optimization recommendations."""
    check_initialized()
    
    try:
        analytics = MenuAnalyticsService(state.item_stats)
        
        return {
            "recommendations": analytics.get_recommendations(),
            "summary": analytics.get_summary(),
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Ingredient & Stock Endpoints
# ============================================================================

@app.get("/ingredients/stock", tags=["Ingredients"])
async def get_ingredient_stock():
    """Get current ingredient stock levels."""
    check_initialized()
    
    try:
        ingredient_service = IngredientService(state.pipeline.items)
        stock = ingredient_service.get_ingredient_stock()
        
        # Group by status
        status_counts = {}
        for item in stock:
            status = item['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "ingredients": stock,
            "count": len(stock),
            "status_breakdown": status_counts,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingredients/alerts", tags=["Ingredients"])
async def get_restocking_alerts(
    forecast_days: int = Query(7, ge=1, le=30, description="Days to project ahead")
):
    """Get restocking alerts for ingredients projected to run low."""
    check_initialized()
    
    try:
        ingredient_service = IngredientService(state.pipeline.items)
        alerts = ingredient_service.get_restocking_alerts(forecast_days=forecast_days)
        
        return {
            "alerts": alerts,
            "count": len(alerts),
            "critical_count": len([a for a in alerts if a['severity'] == 'CRITICAL']),
            "forecast_days": forecast_days,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inventory/item/{item_id}/ingredients", tags=["Ingredients"])
async def get_item_ingredients(item_id: int):
    """Get ingredient composition for a specific menu item."""
    check_initialized()
    
    try:
        ingredient_service = IngredientService(state.pipeline.items)
        composition = ingredient_service.get_composition(item_id)
        
        if not composition:
            raise HTTPException(status_code=404, detail="Item not found or has no ingredients")
        
        return {
            "item_id": item_id,
            "ingredients": composition,
            "generated_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Holiday & Special Events Endpoints
# ============================================================================

from utils.holiday_detector import HolidayDetector


@app.get("/forecast/events", tags=["Forecasting"])
async def get_upcoming_events(
    days_ahead: int = Query(14, ge=1, le=120, description="Days to look ahead for events")
):
    """Get upcoming holidays and special events that may affect demand."""
    try:
        detector = HolidayDetector()
        events = detector.get_upcoming_events(days_ahead)
        
        # Check for currently active events
        active_events = []
        # today = date(2024, 4, 10) # date.today() MOCKED FOR EID TEST
        today = date.today()
        if detector.is_ramadan(today):
            active_events.append("Ramadan")
        if detector.is_eid_fitr(today):
            active_events.append("Eid al-Fitr")
        if detector.is_eid_adha(today):
            active_events.append("Eid al-Adha")
        if detector.is_western_christmas(today) or detector.is_coptic_christmas(today):
            active_events.append("Christmas")
            
        return {
            "events": events,
            "active_events": active_events,
            "count": len(events),
            "days_ahead": days_ahead,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/demand-factors", tags=["Forecasting"])
async def get_demand_factors(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    days: int = Query(14, ge=1, le=60, description="Number of days to analyze")
):
    """Get demand adjustment factors for a date range based on holidays/events."""
    try:
        detector = HolidayDetector()
        
        if start_date:
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
        else:
            start = datetime.now().date()
        
        factors = []
        for i in range(days):
            check_date = start + timedelta(days=i)
            factors.append({
                'date': check_date.isoformat(),
                'day_name': check_date.strftime('%A'),
                'is_holiday': detector.is_holiday(check_date),
                'is_ramadan': detector.is_ramadan(check_date),
                'is_eid': detector.is_eid_fitr(check_date) or detector.is_eid_adha(check_date),
                'is_christmas': detector.is_western_christmas(check_date) or detector.is_coptic_christmas(check_date),
                'is_end_of_month': detector.is_end_of_month(check_date),
                'holiday_name': detector.get_holiday_name(check_date),
                'demand_multiplier': detector.get_demand_multiplier(check_date)
            })
        
        return {
            "factors": factors,
            "count": len(factors),
            "start_date": start.isoformat(),
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run with uvicorn
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

