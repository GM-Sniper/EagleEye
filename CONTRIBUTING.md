# 🦅 EagleEye - Contributing Guide

> **Quick Reference for New Contributors**

This document provides everything you need to set up, run, and contribute to the EagleEye Inventory Management project.

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Running the Application](#-running-the-application)
- [Git Workflow](#-git-workflow)
- [Best Practices](#-best-practices)
- [Common Issues](#-common-issues)

---

## 🚀 Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.12+ | Backend & ML models |
| **Node.js** | 20+ | Frontend development |
| **Git LFS** | Latest | Large file storage for datasets |

### First-Time Setup

```bash
# 1. Clone the repository
git clone https://github.com/GM-Sniper/EagleEye.git
cd DIH-X-AUC-Hackathon

# 2. Pull large data files (required!)
git lfs pull

# 3. Create and activate Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate   # Windows PowerShell

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Install frontend dependencies
cd src/frontend
npm install
cd ../..
```

---

## 📁 Project Structure

```
DIH-X-AUC-Hackathon/
├── Data/                       # Datasets (via Git LFS)
│   ├── dim_*.csv               # Dimension tables
│   └── fct_*.csv               # Fact tables
│
├── src/
│   ├── api.py                  # FastAPI REST API (main backend)
│   ├── models/                 # ML models
│   │   ├── production_forecaster.py   # Core forecasting logic
│   │   └── hybrid_forecaster.py       # Advanced hybrid model
│   ├── services/               # Business logic services
│   │   └── inventory_service.py       # Inventory calculations
│   ├── utils/                  # Helper utilities
│   └── frontend/               # React + Vite frontend
│       ├── src/                # React components & styles
│       ├── package.json        # Frontend dependencies
│       └── vite.config.js      # Vite configuration
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview
```

---

## 🏃 Running the Application

### Backend (FastAPI)

The backend serves the REST API for forecasting and inventory management.

```bash
# Make sure you're in the project root and venv is activated
cd DIH-X-AUC-Hackathon
source venv/bin/activate  # if not already activated

# Start the FastAPI server
uvicorn src.api:app --reload --port 8000

# Server will be available at:
# - API: http://localhost:8000
# - Docs: http://localhost:8000/docs (Swagger UI)
# - ReDoc: http://localhost:8000/redoc
```

**Backend Key Endpoints:**

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check |
| `GET /forecast` | Demand forecast (7-30 days) |
| `GET /inventory` | Inventory recommendations |
| `GET /inventory/alerts` | Critical inventory alerts |
| `GET /analytics/summary` | Analytics dashboard data |

### Frontend (React + Vite)

The frontend provides an interactive dashboard for inventory insights.

```bash
# Navigate to frontend directory
cd src/frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev

# Frontend will be available at:
# - http://localhost:5173
```

**Frontend Scripts:**

| Command | Purpose |
|---------|---------|
| `npm run dev` | Start development server with hot reload |
| `npm run build` | Build production bundle to `dist/` |
| `npm run preview` | Preview production build locally |
| `npm run lint` | Run ESLint for code quality |

### Running Both Together

Open two terminal windows:

**Terminal 1 - Backend:**
```bash
cd DIH-X-AUC-Hackathon
source venv/bin/activate
uvicorn src.api:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd DIH-X-AUC-Hackathon/src/frontend
npm run dev
```

---

## 🌿 Git Workflow

### Current Branches

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready code |
| `ai-sandbox-optimization` | AI model tuning & experiments |
| `feature/inventory-management` | Inventory features |
| `frontend-with-backend` | Frontend-backend integration |

### Working on a New Feature

```bash
# 1. Make sure you're on the latest main
git checkout main
git pull origin main

# 2. Create a new feature branch
git checkout -b feature/your-feature-name

# 3. Make your changes and commit frequently
git add .
git commit -m "feat: add new inventory alert system"

# 4. Push to remote
git push origin feature/your-feature-name

# 5. Create a Pull Request on GitHub
```

### Commit Message Convention

Use semantic commit messages:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feat:` | New feature | `feat: add weekly forecast chart` |
| `fix:` | Bug fix | `fix: correct demand calculation` |
| `docs:` | Documentation | `docs: update API endpoints` |
| `refactor:` | Code refactoring | `refactor: simplify forecast logic` |
| `style:` | Code style/formatting | `style: format with prettier` |
| `test:` | Adding tests | `test: add unit tests for forecaster` |
| `chore:` | Maintenance | `chore: update dependencies` |

### Syncing with Main

```bash
# While on your feature branch
git fetch origin
git merge origin/main

# Resolve any conflicts, then:
git add .
git commit -m "chore: merge main into feature branch"
git push
```

---

## ✅ Best Practices

### Code Quality

1. **Backend (Python)**
   - Use type hints for function parameters and returns
   - Follow PEP 8 style guidelines
   - Add docstrings to functions and classes
   - Keep API endpoint logic minimal; use services for business logic

2. **Frontend (React/TypeScript)**
   - Run `npm run lint` before committing
   - Use functional components with hooks
   - Keep components small and focused
   - Use Tailwind CSS utility classes consistently

### Data Handling

- **Never commit large data files** - Use Git LFS
- Always run `git lfs pull` after cloning
- Keep data transformations in `services/` or `utils/`

### Development Workflow

1. **Pull latest changes** before starting work
2. **Create feature branches** for new work
3. **Commit frequently** with meaningful messages
4. **Test locally** before pushing
5. **Create Pull Requests** for code review

### Environment Variables

Create a `.env` file in the project root if needed:

```bash
# Example .env file
API_HOST=localhost
API_PORT=8000
DEBUG=true
```

---

## ❓ Common Issues

### Backend Won't Start

```bash
# Check Python version
python --version  # Should be 3.12+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check for missing data files
git lfs pull
```

### Frontend Build Errors

```bash
# Clear node_modules and reinstall
cd src/frontend
rm -rf node_modules package-lock.json
npm install
```

### Data Files Missing

```bash
# Install Git LFS if not installed
# Ubuntu/Debian
sudo apt install git-lfs

# Then pull LFS files
git lfs install
git lfs pull
```

### CORS Issues

If the frontend can't reach the backend, check that CORS is properly configured in `src/api.py`. The API should allow requests from `http://localhost:5173`.

---

## 📞 Getting Help

- Check the [README.md](./README.md) for project overview
- Review the API documentation at `/docs` when backend is running
- Look at existing code patterns in `src/` for examples

---

*Happy coding! 🚀*
