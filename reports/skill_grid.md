# Mission 7 - Skill Grid: Credit Scoring MLOps

> **Project:** Prêt à dépenser - Credit Scoring API  
> **Last Updated:** 2025-12-31  
> **Status:** 🚧 IN PROGRESS

---

## Overview

This document tracks all evaluation criteria (CE) for the two modules:
- **Module A:** Implémentez un modèle de scoring (80 heures)
- **Module B:** Réalisez un dashboard (40 heures)

---

## MODULE 1: Définir la stratégie d'élaboration d'un modèle d'apprentissage supervisé

| CE | Criterion | Status | Implementation | Evidence |
|---|---|---|---|---|
| CE1 | Variables catégorielles transformées (OneHotEncoder/TargetEncoder) | ✅ DONE | LabelEncoder + pandas get_dummies dans FeatureEngineering | [feature_engineering.py](../src/classes/feature_engineering.py) |
| CE2 | Création de nouvelles variables à partir d'existantes | ✅ DONE | Ratios calculés: CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, etc. | [mission7.ipynb](../mission7.ipynb) Section Feature Engineering |
| CE3 | Transformations mathématiques des distributions | ✅ DONE | Log transform sur variables skewed, normalization | [feature_engineering.py](../src/classes/feature_engineering.py) |
| CE4 | Normalisation des variables | ✅ DONE | StandardScaler dans pipeline preprocessing | [model_trainer.py](../src/classes/model_trainer.py) |
| CE5 | Stratégie d'élaboration définie pour besoin métier | ✅ DONE | Business cost function (FN=10x, FP=1x), threshold optimization | [business_scorer.py](../src/classes/business_scorer.py) |
| CE6 | Variable cible pertinente choisie | ✅ DONE | TARGET (0=remboursement OK, 1=défaut de paiement) | [mission7.ipynb](../mission7.ipynb) Section 1 |
| CE7 | Vérification absence de data leakage | ✅ DONE | Exclusion variables post-décision, analyse corrélation | [mission7.ipynb](../mission7.ipynb) Section EDA |
| CE8 | Test plusieurs algorithmes (linéaire + non-linéaire) | ✅ DONE | LogisticRegression, RandomForest, LightGBM comparés | [mission7.ipynb](../mission7.ipynb) Section Modeling |

**Module 1 Summary**: ✅ **100% COMPLETE** (8/8 criteria)

---

## MODULE 2: Évaluer les performances des modèles d'apprentissage supervisé

| CE | Criterion | Status | Implementation | Evidence |
|---|---|---|---|---|
| CE1 | Métrique adaptée + Score métier (FN/FP différenciés) | ✅ DONE | business_cost_score avec FN_COST=10, FP_COST=1 | [business_scorer.py](../src/classes/business_scorer.py) |
| CE2 | Autres indicateurs explorés (temps calcul, coefficients) | ✅ DONE | AUC, Accuracy, F1, temps d'entraînement logués | [mission7.ipynb](../mission7.ipynb) MLflow tracking |
| CE3 | Séparation train/test pour détecter overfitting | ✅ DONE | train_test_split 80/20 + validation set | [data_split.py](../src/scripts/data_split.py) |
| CE4 | Modèle de référence (DummyClassifier) | ✅ DONE | DummyClassifier stratified comme baseline | [mission7.ipynb](../mission7.ipynb) Section Baseline |
| CE5 | Déséquilibre des classes pris en compte | ✅ DONE | class_weight='balanced' + SMOTE testé | [model_trainer.py](../src/classes/model_trainer.py) |
| CE6 | Optimisation hyperparamètres | ✅ DONE | GridSearchCV sur n_estimators, max_depth, learning_rate | [mission7.ipynb](../mission7.ipynb) Section Tuning |
| CE7 | Cross-validation mise en place (AUC < 0.82) | ✅ DONE | StratifiedKFold 5-fold, AUC ~0.75-0.78 | [mission7.ipynb](../mission7.ipynb) MLflow UI |
| CE8 | Résultats présentés simple→complexe, choix justifié | ✅ DONE | Dummy→LogReg→RF→LGBM progression documentée | [mission7.ipynb](../mission7.ipynb) |
| CE9 | Feature importance globale et locale (SHAP) | ✅ DONE | SHAP values global + local waterfall plots | [model_visualizer.py](../src/classes/model_visualizer.py) |

**Module 2 Summary**: ✅ **100% COMPLETE** (9/9 criteria)

---

## MODULE 3: Pipeline d'entraînement des modèles

| CE | Criterion | Status | Implementation | Evidence |
|---|---|---|---|---|
| CE1 | Pipeline d'entraînement reproductible | ✅ DONE | MLflow experiments avec paramètres trackés | [mission7.ipynb](../mission7.ipynb) |
| CE2 | Modèles sérialisés dans registre centralisé | ✅ DONE | MLflow Model Registry avec stages | [model_registration.py](../src/scripts/model_registration.py) |
| CE3 | Mesures et résultats formalisés par expérimentation | ✅ DONE | MLflow tracking: metrics, params, artifacts | MLflow UI `http://localhost:5007` |

**Module 3 Summary**: ✅ **100% COMPLETE** (3/3 criteria)

---

## MODULE 4: Logiciel de version de code

| CE | Criterion | Status | Implementation | Evidence |
|---|---|---|---|---|
| CE1 | Dossier Git partagé sur Github | ✅ DONE | Repository mission7 sur GitHub | [GitHub Link](https://github.com/YOUR_USERNAME/mission7) |
| CE2 | Historique avec 3+ versions distinctes | ✅ DONE | Commits: initial, modeling, api, deployment | `git log --oneline` |
| CE3 | Liste packages avec versions | ✅ DONE | requirements.txt avec versions pinned | [requirements.txt](../requirements.txt) |
| CE4 | Fichier introductif (objectif + découpage) | ✅ DONE | README.md complet | [README.md](../README.md) |
| CE5 | Scripts et fonctions commentés | ✅ DONE | Docstrings dans tous les modules | [src/classes/](../src/classes/) |

**Module 4 Summary**: ✅ **100% COMPLETE** (5/5 criteria)

---

## MODULE 5: Déploiement continu d'un moteur d'inférence

| CE | Criterion | Status | Implementation | Evidence |
|---|---|---|---|---|
| CE1 | Pipeline de déploiement continu défini | ✅ DONE | GitHub Actions CI/CD workflow | [.github/workflows/ci.yml](../.github/workflows/ci.yml) |
| CE2 | API Flask déployée retournant prédictions | ✅ DONE | Flask API avec /predict endpoint | [src/api/app.py](../src/api/app.py) |
| CE3 | Pipeline déploiement sur plateforme Cloud | ✅ DONE | Lightsail deployment via docker-compose | [scripts/setup_lightsail.sh](../scripts/setup_lightsail.sh) |
| CE4 | Tests unitaires automatisés (pytest) | ✅ DONE | pytest dans tests/ exécuté par GitHub Actions | [tests/](../tests/) |
| CE5 | API indépendante de l'application cliente | ✅ DONE | API REST JSON, séparée du dashboard | [src/api/app.py](../src/api/app.py) |

**Module 5 Summary**: ✅ **100% COMPLETE** (5/5 criteria)

---

## MODULE 6: Suivi de la performance du modèle

| CE | Criterion | Status | Implementation | Evidence |
|---|---|---|---|---|
| CE1 | Stratégie de suivi définie (data drift) | ✅ DONE | Evidently drift analysis train vs test | [evidently_drift_report.py](../src/scripts/evidently_drift_report.py) |
| CE2 | Système de stockage événements + alertes | ✅ DONE | Predictions logged to Postgres + drift HTML | [evidently_drift_report.html](../evidently_drift_report.html) |
| CE3 | Analyse stabilité + actions d'amélioration | ✅ DONE | Drift report analysis, retraining strategy | [mission7.ipynb](../mission7.ipynb) Section Drift |

**Module 6 Summary**: ✅ **100% COMPLETE** (3/3 criteria)

---

## MODULE 7: Dashboard (40h Module)

| CE | Criterion | Status | Implementation | Evidence |
|---|---|---|---|---|
| CE1 | Parcours utilisateur simple conçu | ✅ DONE | 3 tabs: Lookup, What-if, Simulator | [index.html](../src/api/templates/index.html) |
| CE2 | 2+ graphiques interactifs | ✅ DONE | SHAP waterfall, feature distributions | [index.html](../src/api/templates/index.html) |
| CE3 | Graphiques lisibles (taille texte, définition) | ✅ DONE | Plotly responsive charts, 14px+ fonts | [index.html](../src/api/templates/index.html) |
| CE4 | Graphiques répondant à problématique métier | ✅ DONE | Score gauge, SHAP feature impact | [app.py](../src/api/app.py) |
| CE5 | Accessibilité WCAG (1.1.1, 1.4.1, 1.4.3, 1.4.4, 2.4.2) | ✅ DONE | Alt texts, color contrast, resize, title | [index.html](../src/api/templates/index.html) |
| CE6 | Dashboard déployé sur le web | ✅ DONE | Nginx serving on Lightsail port 80 | [nginx/default.conf](../nginx/default.conf) |

**Module 7 Summary**: ✅ **100% COMPLETE** (6/6 criteria)

---

## Overall Progress

| Module | CEs Done | Total CEs | % Complete | Status |
|---|---|---|---|---|
| 1. Stratégie modélisation | 8 | 8 | 100% | ✅ |
| 2. Évaluation performances | 9 | 9 | 100% | ✅ |
| 3. Pipeline entraînement | 3 | 3 | 100% | ✅ |
| 4. Version de code | 5 | 5 | 100% | ✅ |
| 5. Déploiement continu | 5 | 5 | 100% | ✅ |
| 6. Suivi performance | 3 | 3 | 100% | ✅ |
| 7. Dashboard | 6 | 6 | 100% | ✅ |
| **TOTAL** | **39** | **39** | **100%** | ✅ |

---

## Livrables Checklist

| # | Livrable | Status | Location |
|---|---|---|---|
| 1 | API de prédiction déployée (lien) | ✅ | `http://YOUR_LIGHTSAIL_IP/api/` |
| 2 | Notebook modélisation (MLflow intégré) | ✅ | [mission7.ipynb](../mission7.ipynb) |
| 3 | Dossier code versionné | ✅ | GitHub repository |
| 4 | Tableau HTML data drift (Evidently) | ✅ | [evidently_drift_report.html](../evidently_drift_report.html) |
| 5 | Notebook/App test API | ✅ | [index.html](../src/api/templates/index.html) |
| 6 | Support présentation (30 slides max) | ⏳ | À créer |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LIGHTSAIL (Ubuntu)                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌─────────────┐    ┌──────────────────────────┐ │
│  │  nginx  │───▶│   Flask API │───▶│      PostgreSQL          │ │
│  │  :80    │    │    :8000    │    │        :5432             │ │
│  └─────────┘    └─────────────┘    │  - application_train     │ │
│       │              │             │  - application_test      │ │
│       │              ▼             │  - predictions (log)     │ │
│       │         ┌─────────┐       └──────────────────────────┘ │
│       └────────▶│ MLflow  │                                     │
│                 │  :5002  │◀── /prod_models/model.pkl           │
│                 └─────────┘                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Notes for Soutenance

1. **MLflow UI Screenshot**: Capture experiments comparison view
2. **GitHub Commits**: Show `git log --oneline` with 3+ commits
3. **Tests Execution**: Show GitHub Actions green build
4. **API Demo**: Live call to `http://LIGHTSAIL_IP/predict` with client_id
5. **Drift Report**: Open evidently_drift_report.html in browser
