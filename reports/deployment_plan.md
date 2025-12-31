# Mission 7 - Deployment Plan

> **Target:** AWS Lightsail Ubuntu Server  
> **Architecture:** Docker Compose with Nginx + Flask API + PostgreSQL + MLflow  
> **Last Updated:** 2025-12-31

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LIGHTSAIL UBUNTU SERVER                           │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      Docker Compose Stack                         │   │
│  │                                                                   │   │
│  │   ┌─────────┐     ┌─────────────┐     ┌──────────────┐           │   │
│  │   │  nginx  │────▶│  Flask API  │────▶│  PostgreSQL  │           │   │
│  │   │  :80    │     │   :8000     │     │    :5432     │           │   │
│  │   └────┬────┘     └──────┬──────┘     └──────────────┘           │   │
│  │        │                 │                                        │   │
│  │        │                 ▼                                        │   │
│  │        │          ┌─────────────┐     ┌──────────────┐           │   │
│  │        └─────────▶│ MLflow Prod │     │ /prod_models │           │   │
│  │                   │   :5002     │◀────│  model.pkl   │           │   │
│  │                   └─────────────┘     └──────────────┘           │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  LOCAL DEV (not deployed):                                               │
│  ┌─────────────┐     ┌─────────────┐                                    │
│  │   Jupyter   │────▶│ MLflow Dev  │                                    │
│  │   :8888     │     │   :5005     │                                    │
│  └─────────────┘     └─────────────┘                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Services Description

| Service | Port | Purpose | Profile |
|---------|------|---------|---------|
| **nginx** | 80 | HTTP reverse proxy, landing page | prod |
| **api** | 8000 | Flask prediction API with gunicorn | prod |
| **postgres** | 5432 | Client data + prediction logs | prod |
| **mlflow-prod** | 5002 | Production model registry UI | prod |
| **jupyter** | 8888 | Notebook experimentation | dev |
| **mlflow-dev** | 5005 | Development experiments tracking | dev |

---

## Database Schema

### Tables with Indexes

```sql
-- Main client data (loaded from CSV)
CREATE TABLE application_train (
    SK_ID_CURR BIGINT PRIMARY KEY,  -- UNIQUE INDEX automatic
    TARGET INTEGER,
    -- ... 120+ columns
);
CREATE UNIQUE INDEX idx_app_train_sk_id ON application_train(SK_ID_CURR);

CREATE TABLE application_test (
    SK_ID_CURR BIGINT PRIMARY KEY,
    -- ... 120+ columns (no TARGET)
);
CREATE UNIQUE INDEX idx_app_test_sk_id ON application_test(SK_ID_CURR);

-- Secondary tables for multi-table features
CREATE TABLE bureau (SK_ID_CURR BIGINT, ...);
CREATE INDEX idx_bureau_sk_id ON bureau(SK_ID_CURR);

-- Prediction logging (for drift monitoring)
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    client_id BIGINT NOT NULL,
    probability FLOAT NOT NULL,
    threshold FLOAT NOT NULL,
    decision VARCHAR(20) NOT NULL,
    model_version VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_predictions_client ON predictions(client_id);
CREATE INDEX idx_predictions_date ON predictions(created_at);
```

---

## Model Promotion Workflow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Notebook Dev   │───▶│   MLflow Dev     │───▶│  /prod_models/  │
│  Train + Tune   │    │   Experiments    │    │   Candidate     │
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                        │
                                                   git commit
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Lightsail API  │◀───│   MLflow Prod    │◀───│   git pull      │
│  Load & Serve   │    │   Registry       │    │   Deploy        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### /prod_models/ Structure

```
prod_models/
├── model.pkl              # Serialized LightGBM model
├── threshold.json         # {"optimal_threshold": 0.45}
├── metadata.json          # {"run_id": "...", "auc": 0.76, "features": [...]}
├── feature_names.txt      # One feature per line
└── README.md              # Promotion process documentation
```

---

## Environment Variables

### Production (.env)

```bash
# PostgreSQL
POSTGRES_USER=mission7
POSTGRES_PASSWORD=<secure_password>
POSTGRES_DB=credit_scoring

# Flask
FLASK_SECRET_KEY=<secure_key>
FLASK_ENV=production

# MLflow
MLFLOW_TRACKING_URI=http://mlflow-prod:5002

# Paths
DB_URI=postgresql://mission7:<password>@postgres:5432/credit_scoring
PROD_MODEL_PATH=/app/prod_models/model.pkl
```

---

## Deployment Steps

### 1. Lightsail Setup (One-time)

```bash
# SSH into Lightsail instance
ssh -i lightsail-key.pem ubuntu@YOUR_IP

# Run setup script
curl -sSL https://raw.githubusercontent.com/YOUR_USER/mission7/main/scripts/setup_lightsail.sh | bash
```

### 2. Manual Deployment

```bash
# On Lightsail server
cd /opt/mission7
git pull origin main
docker-compose -f docker-compose.prod.yml up -d --build

# Load data into Postgres (first time only)
docker exec mission7_api python scripts/load_data.py
```

### 3. CI/CD Deployment (GitHub Actions)

On every push to `main`:
1. Run pytest tests
2. Build Docker images
3. SSH to Lightsail
4. Pull latest code
5. Restart containers

---

## Nginx Routes

| Path | Target | Description |
|------|--------|-------------|
| `/` | Static HTML | Landing page ([index.html](../src/api/templates/index.html)) |
| `/api/*` | `api:8000` | Flask API endpoints |
| `/mlflow/*` | `mlflow-prod:5002` | MLflow UI (optional, can be disabled) |
| `/static/*` | Static files | CSS, JS assets |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Landing page HTML |
| GET | `/api/client/<id>` | Get client data by SK_ID_CURR |
| POST | `/predict` | Get prediction for client |
| GET | `/api/health` | Health check |
| GET | `/api/model/info` | Current model metadata |

### Example Request

```bash
# Get prediction
curl -X POST http://YOUR_IP/predict \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=100001"

# Response
{
  "client_id": 100001,
  "probability": 0.234,
  "threshold": 0.45,
  "decision": "ACCEPTED",
  "shap_html": "<div>...</div>"
}
```

---

## Monitoring & Logs

```bash
# View all logs
docker-compose -f docker-compose.prod.yml logs -f

# View specific service
docker-compose -f docker-compose.prod.yml logs -f api

# Check API health
curl http://localhost/api/health

# Check Postgres connection
docker exec mission7_postgres psql -U mission7 -d credit_scoring -c "SELECT COUNT(*) FROM application_train;"
```

---

## Cleanup After Evaluation

```bash
# Stop and remove all containers + volumes
docker-compose -f docker-compose.prod.yml down -v

# Remove images
docker rmi $(docker images -q mission7*)

# Remove data
rm -rf /opt/mission7
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API returns 500 | Check `docker logs mission7_api` for traceback |
| Model not found | Ensure `/prod_models/model.pkl` exists |
| Slow queries | Verify indexes with `\di` in psql |
| Port 80 blocked | Check Lightsail firewall rules |
| Memory issues | Use `docker stats` to monitor, increase instance size |
