# Production Models Directory

This directory contains the **production-ready model artifacts** that have been promoted after passing all quality checks.

## Purpose

In the CI/CD workflow:
1. **Notebook trains models** → tracked in MLflow dev
2. **Best model selected** → passes tests (AUC, business cost, no overfitting)
3. **Model exported here** → committed to git
4. **Git push triggers deployment** → Lightsail pulls and serves this model

## Files

| File | Description |
|------|-------------|
| `model.pkl` | Serialized LightGBM classifier (pickle format) |
| `threshold.json` | Optimal business threshold for classification |
| `metadata.json` | Model metadata: run_id, metrics, feature names |
| `feature_names.txt` | List of features in training order |

## Usage

### Export model from notebook
```python
import pickle
import json

# After training and selecting best model
model = best_pipeline  # Your trained model

# Save model
with open('prod_models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save threshold
with open('prod_models/threshold.json', 'w') as f:
    json.dump({"optimal_threshold": 0.45}, f)

# Save metadata
with open('prod_models/metadata.json', 'w') as f:
    json.dump({
        "run_id": mlflow_run_id,
        "auc": 0.76,
        "business_cost": 1234,
        "features": feature_names
    }, f)
```

### Commit and deploy
```bash
git add prod_models/
git commit -m "Promote model v1.2 - AUC 0.76"
git push origin main

# On Lightsail, GitHub Actions will:
# 1. Pull the new code
# 2. API will automatically load new model on restart
```

## Promotion Criteria

Before adding a model here, ensure it passes:

- [ ] **AUC on test set**: 0.70 < AUC < 0.82 (no overfitting)
- [ ] **Business cost**: Lower than baseline
- [ ] **Cross-validation**: Consistent across folds
- [ ] **Feature importance**: Sensible top features
- [ ] **Unit tests**: All tests pass

## Model History

| Date | Version | AUC | Threshold | Notes |
|------|---------|-----|-----------|-------|
| 2025-12-31 | v1.0 | TBD | TBD | Initial deployment |

---

⚠️ **Note**: Do not commit models larger than 100MB. Use Git LFS if needed, or consider model compression.
