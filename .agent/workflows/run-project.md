---
description: how to run the EagleEye project (backend and frontend)
---

# Running EagleEye

## Prerequisites
- Python 3.12+ with venv activated
- Node.js 20+
- Run `git lfs pull` to get data files

## Backend (FastAPI)

// turbo
1. Navigate to project root
```bash
cd /home/marwan/Antigravity/Hackathon/DIH-X-AUC-Hackathon
```

2. Activate virtual environment
```bash
source venv/bin/activate
```

// turbo
3. Start the API server
```bash
uvicorn src.api:app --reload --port 8000
```

Backend will be at: http://localhost:8000
API Docs at: http://localhost:8000/docs

## Frontend (React + Vite)

// turbo
1. Navigate to frontend directory
```bash
cd /home/marwan/Antigravity/Hackathon/DIH-X-AUC-Hackathon/src/frontend
```

// turbo
2. Install dependencies (if needed)
```bash
npm install
```

// turbo
3. Start development server
```bash
npm run dev
```

Frontend will be at: http://localhost:5173

## Git Workflow

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and commit: `git commit -m "feat: description"`
3. Push to remote: `git push origin feature/my-feature`
4. Create Pull Request on GitHub
