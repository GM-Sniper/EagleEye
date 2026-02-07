# Testing Scripts

## Backtest script
This script validates the functionality and accuracy of the forecasting models. It runs a historical backtest to compare predicted demand against actual sales data, generating performance metrics to ensure model reliability.

## SmokeAPI script
This script verifies the health and accuracy of the API endpoints. It performs a series of request-response checks to ensure all backend services are running correctly and returning expected data structures.

## Smoke Script
A lightweight connectivity test for the global forecasting model. It runs a quick prediction cycle to confirm the model is loaded and responsive without performing a full-scale backtest.


## Analytics Engine
The central orchestration script that integrates data processing, forecasting, and inventory logic to generate dashboard data and full system reports.

## Training Scripts
- **train_final_forecaster_reports.py**: Retrains the production-grade forecasting models on the latest dataset.
- **train_xgb_item_demand_ensemble.py**: Trains the specialized XGBoost ensemble model for item-level demand prediction.

## Analysis & Evaluation
- **analyze_trends.py**: Generates a comprehensive report on temporal patterns, seasonal trends, and demand fluctuations.
- **analyze_top_items.py**: Performs a focused deep-dive analysis on the highest-performing inventory items ("Stars").
- **evaluate_advanced_forecaster_accuracy.py**: Runs detailed accuracy checks on advanced model variants to validate performance improvements.
- **model_evaluation.py**: A utility script for computing core performance metrics (MAPE, RMSE) across different models.

## Benchmarking & Verification
- **benchmark_global.py**: Benchmarks the performance and latency of the global forecasting model.
- **benchmark_hybrid.py**: Benchmarks the hybrid (item + store) forecasting model.
- **verify_predictions.py**: A validation utility to cross-check generated predictions against expected ranges and constraints.
- **leakage_audit.py**: Scans the data pipeline and training splits to detect potential data leakage issues.

## Inventory Planning
- **inventory_planning.py**: Runs standalone inventory optimization calculations to determine Reorder Points (ROP) and Safety Stock levels.
