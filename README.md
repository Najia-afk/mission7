# Project: Credit Scoring Model Implementation
## Scoring System for "Prêt à dépenser" - End-to-End MLOps Approach

[![Docker](https://img.shields.io/badge/Docker-24.0+-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0+-uppercase.svg)](https://mlflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)

###  Project Context
This project was developed as part of a Data Scientist training program. The mission involves working for "Prêt à dépenser," a financial company that offers consumer loans to individuals with little to no credit history.

The objective is to develop a **classification algorithm** to predict the probability of a client defaulting on a loan, while integrating a complete **MLOps** workflow to manage the model's lifecycle.

###  Business & Technical Objectives
- **Business Cost Optimization**: Implementation of an asymmetric cost function where a **False Negative (Default)** is **10 times more expensive** than a False Positive (Missed Opportunity).
- **Transparency (XAI)**: Global and local feature importance analysis via **SHAP** to explain model decisions.
- **Industrialization (MLOps)**:
    - Experiment tracking and model registry with **MLflow**.
    - **Data Drift** analysis using the **Evidently** library.
    - Continuous Integration and Deployment (CI/CD) via **GitHub Actions**.
    - Unit testing with **Pytest**.

###  Technical Architecture
The project is built on a modular and containerized architecture:

1.  **Preprocessing Pipeline**: Class imbalance management and optimized imputation (SimpleImputer + Missing Indicators).
2.  **Advanced Modeling**: Utilization of **LightGBM** with hyperparameter optimization via HalvingGridSearchCV.
3.  **Zero-Leakage Architecture**: Strict data separation (Train/Val/Test) before any processing to ensure result integrity.
4.  **Champion vs Challenger**: Automated comparison against the current production model to ensure business cost reduction before registration.
5.  **Observability**: Statistical monitoring of feature distributions to detect production drift.

---

###  Quick Start (Docker)

The entire environment (Jupyter + MLflow) is orchestrated via Docker Compose.

#### 1. Prerequisites
- Docker Desktop
- Docker Compose V2

#### 2. Launch the System
`ash
docker-compose up --build
`

#### 3. Access the Services (Development - Unique Ports)
| Service | URL | Description |
|---------|-----|-------------|
| **Jupyter Notebook** | [http://localhost:8870](http://localhost:8870) | Open `mission7.ipynb` |
| **MLflow UI (Dev)** | [http://localhost:5075](http://localhost:5075) | Track runs and registered models |
| **API** | [http://localhost:8070](http://localhost:8070) | REST API for predictions |
| **Dashboard (nginx)** | [http://localhost:8071](http://localhost:8071) | Web interface |
| **Postgres** | `localhost:5470` | Database (dev mode uses SQLite) |

---

### 🚀 Production Deployment

#### Option 1: Docker Compose (Recommended)
```bash
docker-compose -f docker-compose.prod.yml up -d
```

#### Option 2: AWS Lightsail
```bash
./scripts/setup_lightsail.sh
```

#### CI/CD Pipeline
- Push to `main` branch triggers GitHub Actions
- Tests run automatically via pytest
- Docker image built and verified
- Deploy to Lightsail (requires secrets)

---

### 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Algorithm** | LightGBM |
| **Business Cost** | 0.5354 |
| **Optimal Threshold** | 0.45 |
| **AUC** | ~0.75 |
| **Features** | 125 |

---

### 🔧 API Endpoints

```bash
# Health check
curl http://localhost:8070/api/health

# Model info
curl http://localhost:8070/api/model/info

# Prediction
curl -X POST http://localhost:8070/predict -d "client_id=100002"
```

---

###  Project Structure
`	ext
 notebooks/           # Complete ML Pipeline (12-step workflow)
 src/
    classes/         # Business Logic (Training, Visualization, Scoring)
    scripts/         # Utilities (Split, Drift, Registration)
 dataset/             # Data Storage (SQLite & CSV)
 docker-compose.yml   # Multi-container orchestration
 Dockerfile           # Python environment
`

###  Key Performance Indicators (KPIs)
- **Model Stability**: Generalization gap < 0.004 (AUC).
- **Financial Impact**: Optimized threshold (0.45) reducing default costs by ~15% compared to the standard 0.5 threshold.
- **Performance**: Imputation pipeline 90% faster than traditional KNN methods.

---
*This project demonstrates the ability to design, train, and industrialize a Machine Learning model in a demanding financial context.*
