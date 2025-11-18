# ML Models Directory

This directory contains trained machine learning models.

## 🎯 Model Files

After running `ml_model_trainer.py`, you'll have:

- `lead_conversion_xgboost_YYYYMMDD_HHMMSS.pkl` - Trained XGBoost model
- `feature_columns.pkl` - List of features used by model
- `model_metadata.json` - Model performance metrics

## 🚀 Training a Model
python ml_model_trainer.py

text

This will:
1. Connect to Data_Lead database
2. Extract leads with known outcomes (Stage 7=Converted, 8=Lost)
3. Engineer 30 features
4. Train XGBoost, RandomForest, GradientBoosting models
5. Save the best model (highest ROC-AUC)

## 📊 Model Performance

Expected metrics:
- ROC-AUC: 0.85 - 1.0
- Accuracy: 80% - 100%
- Training samples: 50-100 leads

## 🔄 Retraining Schedule

Retrain monthly to keep model fresh:
- More historical data = better predictions
- Adapts to changing patterns
- Improves accuracy over time

## ⚠️ Important

**DO NOT** commit `.pkl` files to Git:
- Models are too large (50-100MB)
- Each deployment should train its own model
- Or use model versioning system (MLflow, W&B)

## 🎯 Model Features

Top features (by importance):
1. TotalStageChanges (52%)
2. LeadStageId (39%)
3. CallFrequency (5%)
4. AvgCallDuration (3%)
5. MeetingRate (1%)
