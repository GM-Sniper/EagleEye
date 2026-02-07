# Source Code Documentation

This directory contains the core application code for EagleEye.

## Directory Structure

### `src/` Root
- **`api.py`**: The main entry point for the FastAPI backend. Contains route definitions and application startup logic.

### `models/`
Machine Learning forecasting models.
- **`production_forecaster.py`**: The primary model used for generating live forecasts.
- **`hybrid_forecaster.py`**: An ensemble model combining global trends with item-specific patterns.
- **`global_forecaster.py`**: Captures store-wide demand trends.
- **`optimized_forecaster.py`**: Performance-tuned variant for high-throughput scenarios.

### `services/`
Business logic and data handling layers.
- **`data_pipeline.py`**: Handles data loading, cleaning, and preprocessing from the `Data/` directory.
- **`inventory_service.py`**: Core logic for inventory optimization, reorder points (ROP), and safety stock calculations.
- **`menu_analytics_service.py`**: Analyzes item performance and demand patterns.
- **`pricing_service.py`**: Logic for pricing strategies and elasticity (if applicable).
- **`demand_feature_builder.py`**: Feature engineering for the ML models.
- **`ingredient_service.py`**: Manages Bill of Materials (BOM) and ingredient-level demand.

### `utils/`
Shared utility functions and helpers.

### `frontend/`
The React-based web application. See `frontend/README.md` (if available) or `package.json` for frontend details.

## Related Directories
- **`scripts/`**: Contains standalone analysis, training, and benchmarking scripts that were previously in `src/`.
