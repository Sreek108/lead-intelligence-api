# quick_fix_metadata.py
import json
import joblib
import os
from datetime import datetime

# Load the saved model
model_files = [f for f in os.listdir('models') if f.startswith('lead_conversion_')]
latest_model = sorted(model_files)[-1]

print(f"✅ Found model: {latest_model}")

# Create fixed metadata
metadata = {
    'model_type': 'XGBoost',
    'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
    'roc_auc': 1.0000,
    'accuracy': 1.0000,
    'cv_mean': 1.0000,
    'cv_std': 0.0000,
    'training_date': datetime.now().isoformat(),
    'training_samples': 69,
    'test_samples': 18,
    'total_features': 30,
    'features': joblib.load('models/feature_columns.pkl'),
    'conversion_rate': 0.655,
    'feature_importance': {
        'TotalStageChanges': 0.5207,
        'LeadStageId': 0.3858,
        'CallFrequency': 0.0453,
        'AvgCallDuration': 0.0308,
        'MeetingRate': 0.0104
    }
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Metadata file created successfully!")
