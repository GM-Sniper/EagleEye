# 🦅 EagleEye - Intelligent Inventory Management

> **Deloitte x AUC Hackathon 2026** | Fresh Flow Markets Use Case

An AI-powered inventory management system that transforms reactive stock decisions into proactive demand intelligence—minimizing waste, preventing stockouts, and maximizing profitability.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Our Solution](#-our-solution)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Data Description](#-data-description)
- [Contributing](#-contributing)
- [Team Members](#-team-members)

> 📖 **New to the project?** See the [Contributing Guide](./CONTRIBUTING.md) for detailed setup instructions, git workflow, and best practices.

---

## 🎯 Problem Statement

Restaurant and grocery owners face a relentless balancing act:

| Problem | Impact |
|---------|--------|
| **Over-stocking** | Waste, expired inventory, reduced profits |
| **Under-stocking** | Stockouts, lost revenue, frustrated customers |
| **Poor demand forecasting** | Reactive decisions based on gut instinct |

**Fresh Flow Markets needs intelligent systems, not gut instinct.**

### Business Questions We Address

1. **Demand Prediction**: How do we accurately predict daily, weekly, and monthly demand?
2. **Prep Optimization**: What prep quantities should kitchens prepare to minimize waste?
3. **Expiration Management**: How can we prioritize inventory based on expiration dates?
4. **Smart Promotions**: What promotions or bundles can move near-expired items profitably?
5. **External Factors**: How do weather, holidays, and weekends impact sales?

---

## 💡 Our Solution

**EagleEye** is a data-driven inventory intelligence platform that:

- 📊 Analyzes historical sales data to forecast demand
- 🔮 Predicts optimal stock levels using ML models
- ⚠️ Alerts on expiring inventory with promotion suggestions
- 📈 Provides actionable dashboards for inventory decisions
- 🌦️ Incorporates external factors (weather, holidays, events)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Demand Forecasting** | ML-powered predictions for daily/weekly/monthly demand |
| **Smart Reorder Points** | Automated stock level recommendations |
| **Expiration Tracker** | Priority alerts for near-expiry items |
| **Promotion Engine** | Bundle suggestions for expiring inventory |
| **External Factor Analysis** | Weather and holiday impact on sales |
| **Interactive Dashboard** | Real-time inventory insights and KPIs |

*Screenshots will be added as features are implemented.*

---

## 🛠️ Technologies Used

### Data & Analytics
| Technology | Purpose |
|------------|---------|
| **Python 3.12+** | Core programming language |
| **Pandas / Polars** | Data processing and manipulation |
| **Jupyter Notebooks** | Exploratory data analysis |
| **scikit-learn** | Machine learning models |
| **Prophet / XGBoost** | Time-series forecasting |

### Backend
| Technology | Purpose |
|------------|---------|
| **FastAPI** | High-performance REST API |
| **DuckDB / SQLite** | Lightweight analytical database |
| **Pydantic** | Data validation |

### Frontend
| Technology | Purpose |
|------------|---------|
| **Next.js 15** | React framework with App Router |
| **TypeScript** | Type-safe JavaScript |
| **Tailwind CSS** | Utility-first styling |
| **Recharts / Plotly** | Data visualization |

### DevOps & Tools
| Technology | Purpose |
|------------|---------|
| **Git LFS** | Large file storage for datasets |
| **Docker** | Containerization (optional) |
| **pytest** | Testing framework |

---

## 🚀 Installation

### Prerequisites

- Python 3.12+
- Node.js 20+
- Git with LFS support

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ynakhla/DIH-X-AUC-Hackathon.git
cd DIH-X-AUC-Hackathon

# 2. Pull LFS data
git lfs pull

# 3. Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Install frontend dependencies (if applicable)
cd src/web
npm install
cd ../..
```

---

## 📖 Usage

### Running the Analysis

```bash
# Start Jupyter for data exploration
jupyter notebook src/analysis.ipynb
```

### Running the API

```bash
# Start FastAPI server
uvicorn src.api.main:app --reload
```

### Running the Frontend

```bash
cd src/web
npm run dev
```

---

## 🏗️ Architecture

```
DIH-X-AUC-Hackathon/
├── Data/                    # Dataset (via Git LFS)
│   ├── dim_*.csv            # Dimension tables (items, places, users)
│   ├── fct_*.csv            # Fact tables (orders, transactions)
│   └── most_ordered.csv     # Pre-computed analytics
├── src/
│   ├── analysis.ipynb       # Exploratory data analysis
│   ├── main.py              # Entry point
│   ├── api/                 # FastAPI endpoints
│   ├── models/              # ML models and data models
│   ├── services/            # Business logic
│   ├── utils/               # Helper functions
│   └── web/                 # Next.js frontend
├── tests/                   # Test suite
├── docs/                    # Documentation
└── config/                  # Configuration files
```

---

## 📊 Data Description

### Dataset Overview (~650 MB)

| Table | Size | Description |
|-------|------|-------------|
| `fct_orders.csv` | 62 MB | Order transactions with payments |
| `fct_order_items.csv` | 211 MB | Line items per order |
| `fct_cash_balances.csv` | 358 MB | Cash register sessions |
| `dim_items.csv` | 14 MB | Menu items with prices, VAT |
| `dim_users.csv` | 11 MB | Customer and staff accounts |
| `dim_places.csv` | 2 MB | Restaurant/venue locations |
| `dim_menu_items.csv` | 2 MB | Menu structure |
| + 12 more tables | | Campaigns, add-ons, inventory |

### Key Data Notes

- All timestamps are **UNIX integers**
- All monetary values are in **DKK (Danish Krone)**
- Data spans **2021-2025**

---

## 🤝 Contributing

We welcome contributions! Please see our **[Contributing Guide](./CONTRIBUTING.md)** for:

- 🚀 **Quick Start** - First-time setup instructions
- 🏃 **Running the App** - How to start backend and frontend
- 🌿 **Git Workflow** - Branching strategy and commit conventions
- ✅ **Best Practices** - Code quality guidelines
- ❓ **Troubleshooting** - Common issues and solutions

---

## 👥 Team Members

| Name | Role | Contributions |
|------|------|---------------|
| TBD | TBD | TBD |
| TBD | TBD | TBD |
| TBD | TBD | TBD |

---

## 📄 License

This project was created for the Deloitte x AUC Hackathon 2026.