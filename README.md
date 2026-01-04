# Mission7 - Credit Scoring MLOps Platform

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-24.0+-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0+-purple.svg)](https://mlflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/Tests-16%20passed-brightgreen.svg)]()

Production-grade credit scoring API with full MLOps pipeline, SHAP explainability, and regulatory audit trail.

### ✨ Key Features
- **Human-readable SHAP**: Feature values displayed in intuitive formats (e.g., "918.5K$" for Credit Amount, "42.6 yrs" for Age)
- **PostgreSQL Audit Trail**: All predictions logged with SHAP values for regulatory compliance
- **Download Artifacts**: Export drift reports and metadata directly from the API
- **Dynamic Data**: All audit pages pull live data from the database

---

## 📁 Project Structure

```
mission7/
├── app/                    # 🚀 PRODUCTION API
│   ├── api/               # Flask routes & services
│   │   ├── routes.py      # Main API endpoints
│   │   ├── audit_routes.py # Governance endpoints
│   │   └── services/      # Business logic
│   ├── config/            # Settings, Swagger config
│   ├── templates/         # HTML templates
│   ├── static/            # CSS, JS assets
│   ├── utils/             # Logging, database utils
│   ├── main.py            # Flask app factory
│   └── wsgi.py            # Gunicorn entry point
│
├── notebooks/              # 📓 JUPYTER NOTEBOOKS
│   └── mission7.ipynb     # Main training notebook
│
├── src/                    # 🔬 ML/TRAINING CODE
│   ├── classes/           # ML model classes
│   ├── database/          # SQLAlchemy models
│   └── scripts/           # Training & export scripts
│
├── prod_models/            # 📦 PRODUCTION MODEL
│   ├── model.pkl          # Trained LightGBM model
│   ├── threshold.json     # Optimal threshold
│   ├── metadata.json      # Model metrics & info
│   ├── feature_names.txt  # 125 features list
│   ├── evidently_data_drift_report.html  # Evidently drift report
│   └── evidently_data_drift_report.json  # Drift report (JSON)
│
├── dataset/                # 💾 DATA FILES
│   └── home_credit.db     # Optional local SQLite dataset (not committed)
│
├── nginx/                  # 🌐 REVERSE PROXY
│   └── default.conf       # Nginx configuration
│
├── tests/                  # ✅ PYTEST TESTS
│   ├── test_api_endpoints.py
│   ├── test_model_validation.py
│   └── test_environment.py
│
├── mlruns/                 # 📊 MLFLOW EXPERIMENTS
│
├── docker-compose.yml      # Development stack
├── docker-compose.prod.yml # Production stack
├── Dockerfile              # Dev image (Jupyter)
├── Dockerfile.prod         # Production image
└── requirements.txt        # Python dependencies
```

---

## 🚀 Quick Start

### Development Mode (Jupyter + MLflow)
```bash
docker compose up -d
```

| Service | URL |
|---------|-----|
| Jupyter Lab | http://localhost:8888 |
| MLflow UI | http://localhost:5005 |

### Production Mode (API + MLflow + PostgreSQL)
```bash
docker compose -f docker-compose.prod.yml up -d --build
```

On first startup, the API container automatically seeds PostgreSQL from compressed CSV samples included in the repo:
- `dataset/application_train_sample.csv.gz` (100k rows, stratified sample)
- `dataset/application_test.csv.gz` (48k rows, full test set)

Notes:
- Seeding takes ~7 minutes for 148k total rows
- Seeding is idempotent: if tables already have rows, it skips
- All features work with the sampled data (predictions, SHAP, similar clients, bivariate plots)

| Service | URL |
|---------|-----|
| Dashboard | http://localhost |
| Predict (Client) | http://localhost/predict |
| Predict (Test) | http://localhost/predict-test |
| Simulator | http://localhost/simulator |
| History | http://localhost/history |
| Audit | http://localhost/audit |
| API Docs (Swagger) | http://localhost/api/docs |
| API Health | http://localhost/api/health |
| MLflow Registry | http://localhost:5002 |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Algorithm** | LightGBM |
| **AUC-ROC** | 0.751 |
| **F1-Score** | 0.275 |
| **Recall** | 0.640 |
| **Optimal Threshold** | 0.45 |
| **Business Cost** | 0.535 |
| **Features** | 125 |

---

## 🔌 API Endpoints

### Core Endpoints
```bash
# Health check
curl http://localhost/api/health

# Make prediction
curl -X POST http://localhost/predict \
  -H "Content-Type: application/json" \
  -d '{"client_id": 100002}'

# Get client data
curl http://localhost/api/client/100002

# Model info
curl http://localhost/api/model/info
```

### Audit & Governance
```bash
# Model governance documentation
curl http://localhost/api/audit/model-governance

# ML Model Card
curl http://localhost/api/audit/model-card

# Feature documentation (125 features)
curl http://localhost/api/audit/features

# Data drift report (JSON)
curl http://localhost/api/audit/drift-report

# Evidently HTML report (browser)
open http://localhost/api/audit/drift-report-html

# Download artifacts (for audit compliance)
curl -O http://localhost/api/audit/download/drift_report_html
curl -O http://localhost/api/audit/download/drift_report_json
curl -O http://localhost/api/audit/download/metadata

# Prediction audit log (from PostgreSQL)
curl http://localhost/api/audit/predictions
```

### Model Management
```bash
# List all models
curl http://localhost/api/models/list

# Current model details
curl http://localhost/api/models/current
```

### Prediction History & Analytics
```bash
# Search predictions with filters
curl "http://localhost/api/predictions/search?limit=50&decision=ACCEPTED"
curl "http://localhost/api/predictions/search?client_id=100002&min_score=0.3"

# Get prediction statistics
curl http://localhost/api/predictions/stats
```

---

## 🔄 CI/CD Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    LOCAL (Development)                   │
├─────────────────────────────────────────────────────────┤
│  1. Train model in notebook (notebooks/mission7.ipynb)  │
│  2. Register to MLflow (experiments tracked)            │
│  3. Export best model to prod_models/                   │
│  4. Git commit + push                                   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  GitHub Actions CI/CD                    │
├─────────────────────────────────────────────────────────┤
│  1. Run tests (pytest) ✅                               │
│  2. Build Docker image                                  │
│  3. Deploy to Lightsail:                                │
│     • git pull (includes prod_models/)                  │
│     • docker compose up --build                         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 PRODUCTION (Lightsail)                   │
├─────────────────────────────────────────────────────────┤
│  • API serves model from prod_models/                   │
│  • PostgreSQL for audit trail                           │
│  • SHAP values logged for each prediction               │
└─────────────────────────────────────────────────────────┘
```

---

## 🧪 Testing

```bash
# Run all tests
docker exec mission7_api_prod pytest tests/ -v

# Run specific test file
docker exec mission7_api_prod pytest tests/test_api_endpoints.py -v

# Run with coverage
docker exec mission7_api_prod pytest tests/ --cov=app
```

**Test Coverage:**
- ✅ Health endpoint
- ✅ Prediction endpoint
- ✅ Client data endpoint
- ✅ Model governance
- ✅ Audit features (125 features)
- ✅ Model validation
- ✅ WCAG accessibility

---

## 🏗️ Architecture

### Production Stack (docker-compose.prod.yml)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PRODUCTION ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    ┌─────────────────────────────────────────────────────────────────┐  │
│    │                        FRONTEND NETWORK                          │  │
│    │                                                                  │  │
│    │                      ┌──────────────┐                           │  │
│    │       :80 ──────────▶│    NGINX     │                           │  │
│    │      (public)        │   (alpine)   │                           │  │
│    │                      │              │                           │  │
│    │                      │ • Static CSS │                           │  │
│    │                      │ • Templates  │                           │  │
│    │                      └──────┬───────┘                           │  │
│    └─────────────────────────────│───────────────────────────────────┘  │
│                                  │                                       │
│    ┌─────────────────────────────│───────────────────────────────────┐  │
│    │                        BACKEND NETWORK                           │  │
│    │                                  │                               │  │
│    │                      ┌──────────▼───────────┐                   │  │
│    │                      │     Flask API        │                   │  │
│    │        :8000 ◀───────│     (Gunicorn)       │                   │  │
│    │       (internal)     │                      │                   │  │
│    │                      │ • Predictions        │                   │  │
│    │                      │ • SHAP values        │                   │  │
│    │                      │ • Audit endpoints    │                   │  │
│    │                      └──────────┬───────────┘                   │  │
│    └─────────────────────────────────│───────────────────────────────┘  │
│                                      │                                   │
│    ┌─────────────────────────────────│───────────────────────────────┐  │
│    │                      MIDDLEWARE NETWORK                          │  │
│    │                                 │                                │  │
│    │              ┌──────────────────┴──────────────────┐            │  │
│    │              │                                     │            │  │
│    │              ▼                                     ▼            │  │
│    │   ┌──────────────────┐              ┌──────────────────────┐   │  │
│    │   │     MLflow       │              │  MLflow Serving      │   │  │
│    │   │    Registry      │ :5002        │    (optional)        │   │  │
│    │   │                  │ (internal)   │                      │   │  │
│    │   │ • Experiments    │              │ :5003 (internal)     │   │  │
│    │   │ • Model versions │              │ • PyFunc model       │   │  │
│    │   └──────────────────┘              │ • SHAP inference     │   │  │
│    │                                     └──────────────────────┘   │  │
│    └────────────────────────────────────────────────────────────────┘  │
│                                      │                                   │
│    ┌─────────────────────────────────│───────────────────────────────┐  │
│    │                       DATABASE NETWORK                           │  │
│    │                                 │                                │  │
│    │                      ┌──────────▼───────────┐                   │  │
│    │                      │     PostgreSQL       │                   │  │
│    │        :5432 ◀───────│    (15-alpine)       │                   │  │
│    │       (internal)     │                      │                   │  │
│    │                      │ • Client data        │                   │  │
│    │                      │ • Prediction logs    │                   │  │
│    │                      │ • Audit trail        │                   │  │
│    │                      └──────────────────────┘                   │  │
│    └────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

Model Loading: prod_models/model.pkl (or MLflow Serving with --profile mlflow-serving)
```

### Development Stack (docker-compose.yml)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DEVELOPMENT ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│         ┌────────────────────┐         ┌────────────────────┐           │
│         │      Jupyter       │         │       MLflow       │           │
│   :8870 │      Notebook      │   :5075 │        Dev         │           │
│         │                    │         │                    │           │
│         │ • Training         │ ◀──────▶│ • Experiments      │           │
│         │ • Exploration      │         │ • Model tracking   │           │
│         └────────────────────┘         └────────────────────┘           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Quick Reference

| Environment | Command | Services |
|-------------|---------|----------|
| **Dev** | `docker compose up -d` | Jupyter (:8870), MLflow (:5075) |
| **Prod** | `docker compose -f docker-compose.prod.yml up -d` | Nginx (:80), API, MLflow, PostgreSQL |
| **Prod + MLflow Serving** | `docker compose -f docker-compose.prod.yml --profile mlflow-serving up -d` | + MLflow Serving (:5003) |

---

## 📋 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_POSTGRES` | `true` | Enable PostgreSQL for audit |
| `DB_URI` | `postgresql://...` | Database connection |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5002` | MLflow server |
| `PROD_MODEL_PATH` | `/app/prod_models/model.pkl` | Model location |

---

## 📜 License

**Proprietary License** - Copyright © 2025-2026 All Rights Reserved.

This software is proprietary and confidential. Unauthorized copying, modification, distribution, or use of this software, via any medium, is strictly prohibited.

**Commercial Use:** Any commercial use of this software requires a paid license agreement.

📧 **Contact for licensing:** [datascience-adventure.xyz/contact](https://datascience-adventure.xyz/contact)

See [LICENSE](LICENSE) for full terms.

---

## 👥 Author

Data Science Training - Mission 7 (2025)
