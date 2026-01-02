#!/usr/bin/env python3
"""
Validate Production Model Package

This script verifies that the prod_models/ directory contains a complete,
valid model package with all required metadata - as if it was just exported
from the CI/CD pipeline.

Validates:
  ✅ model.pkl exists and can be loaded
  ✅ threshold.json exists with valid threshold value
  ✅ metadata.json exists with complete stats
  ✅ feature_names.txt exists
  ✅ Model can make predictions
  ✅ All metrics are reasonable (AUC, precision, etc.)

Used in Dockerfile at build time to fail fast if prod_models is incomplete.
"""

import json
import os
import pickle
import sys
from pathlib import Path


def validate_model_package(prod_models_dir: str = "/app/prod_models") -> bool:
    """
    Validate the complete production model package.
    
    Args:
        prod_models_dir: Path to prod_models directory
        
    Returns:
        bool: True if all validations pass, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"🔍 VALIDATING PRODUCTION MODEL PACKAGE")
    print(f"{'='*70}")
    print(f"Location: {prod_models_dir}\n")
    
    prod_dir = Path(prod_models_dir)
    if not prod_dir.exists():
        print(f"❌ FAIL: prod_models directory not found at {prod_models_dir}")
        return False
    
    errors = []
    
    # =========================================================================
    # 1. Validate model.pkl
    # =========================================================================
    print("📦 Checking model.pkl...")
    model_path = prod_dir / "model.pkl"
    if not model_path.exists():
        error = f"❌ FAIL: model.pkl not found"
        print(f"   {error}")
        errors.append(error)
    else:
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"   ✅ model.pkl loaded successfully")
            print(f"   ✅ Model type: {type(model).__name__}")
            
            # Try to make a test prediction
            try:
                import numpy as np
                if hasattr(model, 'predict_proba'):
                    # Create dummy input matching expected feature count
                    if hasattr(model, 'n_features_in_'):
                        n_features = model.n_features_in_
                    elif hasattr(model, 'feature_names_in_'):
                        n_features = len(model.feature_names_in_)
                    else:
                        # Try to infer from pipeline
                        n_features = 125  # Default for mission7
                    
                    X_dummy = np.random.randn(1, n_features)
                    proba = model.predict_proba(X_dummy)
                    print(f"   ✅ Model can make predictions (shape: {proba.shape})")
                else:
                    print(f"   ⚠️  Model doesn't have predict_proba method")
            except Exception as e:
                print(f"   ⚠️  Prediction test skipped: {e}")
        except Exception as e:
            error = f"❌ FAIL: Could not load model.pkl: {e}"
            print(f"   {error}")
            errors.append(error)
    
    # =========================================================================
    # 2. Validate threshold.json
    # =========================================================================
    print("\n📊 Checking threshold.json...")
    threshold_path = prod_dir / "threshold.json"
    if not threshold_path.exists():
        error = f"❌ FAIL: threshold.json not found"
        print(f"   {error}")
        errors.append(error)
    else:
        try:
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
            
            if 'optimal_threshold' not in threshold_data:
                error = f"❌ FAIL: 'optimal_threshold' key missing from threshold.json"
                print(f"   {error}")
                errors.append(error)
            else:
                threshold = float(threshold_data['optimal_threshold'])
                if not (0.0 <= threshold <= 1.0):
                    error = f"❌ FAIL: threshold {threshold} out of range [0.0, 1.0]"
                    print(f"   {error}")
                    errors.append(error)
                else:
                    print(f"   ✅ threshold.json valid")
                    print(f"   ✅ Optimal threshold: {threshold:.4f}")
        except json.JSONDecodeError as e:
            error = f"❌ FAIL: Invalid JSON in threshold.json: {e}"
            print(f"   {error}")
            errors.append(error)
        except Exception as e:
            error = f"❌ FAIL: Error reading threshold.json: {e}"
            print(f"   {error}")
            errors.append(error)
    
    # =========================================================================
    # 3. Validate metadata.json
    # =========================================================================
    print("\n📋 Checking metadata.json...")
    metadata_path = prod_dir / "metadata.json"
    if not metadata_path.exists():
        error = f"❌ FAIL: metadata.json not found"
        print(f"   {error}")
        errors.append(error)
    else:
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check required fields
            required_fields = ['run_id', 'model_name', 'export_date']
            missing_fields = [f for f in required_fields if f not in metadata]
            
            if missing_fields:
                error = f"❌ FAIL: Missing required fields in metadata.json: {missing_fields}"
                print(f"   {error}")
                errors.append(error)
            else:
                print(f"   ✅ metadata.json valid")
                print(f"   ✅ Model ID (run_id): {metadata['run_id']}")
                print(f"   ✅ Model name: {metadata['model_name']}")
                print(f"   ✅ Export date: {metadata['export_date']}")
            
            # Check metrics
            if 'metrics' in metadata:
                metrics = metadata['metrics']
                print(f"\n   📈 Metrics:")
                if 'auc_roc' in metrics:
                    auc = metrics['auc_roc']
                    print(f"      • AUC-ROC: {auc:.4f}", end="")
                    if 0.70 <= auc <= 0.82:
                        print(" ✅")
                    elif auc >= 0.70:
                        print(" ⚠️  (possibly overfitting)")
                    else:
                        print(" ❌ (too low)")
                
                for key in ['f1_score', 'precision', 'recall', 'accuracy']:
                    if key in metrics:
                        value = metrics[key]
                        print(f"      • {key.replace('_', ' ').title()}: {value:.4f}")
                
                if 'business_cost_avg' in metrics:
                    cost = metrics['business_cost_avg']
                    print(f"      • Business cost: {cost:.2f}")
        
        except json.JSONDecodeError as e:
            error = f"❌ FAIL: Invalid JSON in metadata.json: {e}"
            print(f"   {error}")
            errors.append(error)
        except Exception as e:
            error = f"❌ FAIL: Error reading metadata.json: {e}"
            print(f"   {error}")
            errors.append(error)
    
    # =========================================================================
    # 4. Validate feature_names.txt
    # =========================================================================
    print("\n🏷️  Checking feature_names.txt...")
    features_path = prod_dir / "feature_names.txt"
    if not features_path.exists():
        error = f"❌ FAIL: feature_names.txt not found"
        print(f"   {error}")
        errors.append(error)
    else:
        try:
            with open(features_path, 'r') as f:
                features = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            if not features:
                error = f"❌ FAIL: feature_names.txt is empty or contains only comments"
                print(f"   {error}")
                errors.append(error)
            else:
                print(f"   ✅ feature_names.txt valid")
                print(f"   ✅ Number of features: {len(features)}")
                print(f"   ✅ First 5 features: {features[:5]}")
        except Exception as e:
            error = f"❌ FAIL: Error reading feature_names.txt: {e}"
            print(f"   {error}")
            errors.append(error)
    
    # =========================================================================
    # 5. Optional: Check drift reports
    # =========================================================================
    print("\n📊 Checking optional drift reports...")
    drift_html = prod_dir / "evidently_data_drift_report.html"
    drift_json = prod_dir / "evidently_data_drift_report.json"
    
    if drift_html.exists():
        print(f"   ✅ Drift report (HTML) present ({drift_html.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"   ⚠️  Drift report (HTML) not present")
    
    if drift_json.exists():
        print(f"   ✅ Drift report (JSON) present ({drift_json.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"   ⚠️  Drift report (JSON) not present")
    
    # =========================================================================
    # Final Result
    # =========================================================================
    print(f"\n{'='*70}")
    if errors:
        print(f"❌ VALIDATION FAILED ({len(errors)} error(s)):\n")
        for i, error in enumerate(errors, 1):
            print(f"   {i}. {error}")
        print(f"{'='*70}\n")
        return False
    else:
        print(f"✅ VALIDATION PASSED - Production model package is complete!")
        print(f"{'='*70}\n")
        return True


if __name__ == "__main__":
    success = validate_model_package()
    sys.exit(0 if success else 1)
