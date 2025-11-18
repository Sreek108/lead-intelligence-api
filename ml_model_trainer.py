"""
═══════════════════════════════════════════════════════════════════════════════
ML Model Trainer v1.0 - TRUE Machine Learning for Lead Conversion Prediction
═══════════════════════════════════════════════════════════════════════════════

Purpose: Trains actual ML models from Data_Lead database (1000 leads)
Author: AI/ML Development Team
Version: 1.0 (Production Ready)
Date: November 17, 2025

Features:
- XGBoost, RandomForest, GradientBoosting models
- 30+ engineered features
- Automatic model selection
- Feature importance analysis
- Model versioning & metadata tracking
- Windows compatible (no emojis)

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import json
import os
from datetime import datetime
import logging
from urllib.parse import quote_plus

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLModelTrainer:
    """Trains ML models from Data_Lead database with 1000 leads"""
    
    def __init__(self, server: str, database: str, username: str, password: str):
        """
        Initialize ML Model Trainer
        
        Args:
            server: SQL Server address
            database: Database name
            username: Database username
            password: Database password
        """
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.engine = None
        
        # Create models directory if not exists
        os.makedirs('models', exist_ok=True)
        
    def connect_db(self):
        """Connect to SQL Server database"""
        try:
            conn_string = f"mssql+pymssql://{self.username}:{quote_plus(self.password)}@{self.server}/{self.database}"
            self.engine = create_engine(conn_string, pool_pre_ping=True)
            
            # Test connection with text() wrapper (SQLAlchemy 2.0 compatibility)
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            logger.info(f"[OK] Connected to {self.database} database")
            logger.info(f"     Server: {self.server}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Database connection failed: {e}")
            logger.error(f"        Connection string: mssql+pymssql://{self.username}:***@{self.server}/{self.database}")
            return False
    
    def extract_training_data(self):
        """Extract leads with known outcomes (Stage 7=Converted, Stage 8=Lost)"""
        
        logger.info("[DATA] Extracting training data from Data_Lead database...")
        
        query = """
        SELECT 
            L.LeadId,
            L.LeadStageId,
            L.CountryId,
            L.CityRegionId,
            L.LeadSourceId,
            DATEDIFF(DAY, L.CreatedOn, GETDATE()) as LeadAgeDays,
            
            -- Engagement metrics
            COUNT(DISTINCT lcr.LeadCallId) as TotalCalls,
            COUNT(DISTINCT ama.AssignmentId) as TotalMeetings,
            COUNT(DISTINCT ls.ScheduleId) as TotalSchedules,
            
            -- Call quality
            AVG(CAST(ISNULL(lcr.DurationSeconds, 0) as FLOAT)) as AvgCallDuration,
            SUM(CASE WHEN lcr.SentimentId = 1 THEN 1 ELSE 0 END) as PositiveCalls,
            
            -- Recency
            DATEDIFF(DAY, MAX(lcr.CallDateTime), GETDATE()) as DaysSinceLastCall,
            DATEDIFF(DAY, MAX(ama.StartDateTime), GETDATE()) as DaysSinceLastMeeting,
            
            -- Stage progression
            COUNT(DISTINCT lsa.AuditId) as TotalStageChanges,
            
            -- TARGET: Conversion outcome
            CASE 
                WHEN L.LeadStageId = 7 THEN 1  -- Converted
                WHEN L.LeadStageId = 8 THEN 0  -- Lost
                ELSE NULL
            END AS IsConverted
            
        FROM Lead L
        LEFT JOIN LeadCallRecord lcr ON L.LeadId = lcr.LeadId
        LEFT JOIN AgentMeetingAssignment ama ON L.LeadId = ama.LeadId
        LEFT JOIN LeadSchedule ls ON L.LeadId = ls.LeadId
        LEFT JOIN LeadStageAudit lsa ON L.LeadId = lsa.LeadId
        
        WHERE L.LeadStageId IN (7, 8)  -- Only leads with known outcomes
            AND L.CreatedOn >= DATEADD(YEAR, -1, GETDATE())
        
        GROUP BY 
            L.LeadId, L.LeadStageId, L.CountryId, L.CityRegionId,
            L.LeadSourceId, L.CreatedOn
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            
            # Remove NULL targets
            df = df[df['IsConverted'].notna()]
            
            converted = int(df['IsConverted'].sum())
            lost = len(df) - converted
            
            logger.info(f"[OK] Extracted {len(df)} leads with outcomes")
            logger.info(f"     Converted: {converted} ({converted/len(df)*100:.1f}%)")
            logger.info(f"     Lost: {lost} ({lost/len(df)*100:.1f}%)")
            
            if len(df) < 50:
                logger.error(f"[ERROR] Need at least 50 leads with outcomes. Only found {len(df)}")
                logger.error("        Your database has only leads in Stage 7 (Converted) and Stage 8 (Lost)")
                logger.error("        Wait for more leads to reach final stages before training")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"[ERROR] Data extraction failed: {e}")
            return None
    
    def engineer_features(self, df):
        """Engineer 30+ features matching ML_Engine logic"""
        
        logger.info("[FEAT] Engineering features...")
        
        # Fill NaN values
        df = df.fillna(0)
        
        # === ENGAGEMENT FEATURES ===
        df['CallFrequency'] = df['TotalCalls'] / (df['LeadAgeDays'] + 1)
        df['MeetingRate'] = df['TotalMeetings'] / (df['TotalCalls'] + 1)
        df['HasMeeting'] = (df['TotalMeetings'] > 0).astype(int)
        df['HasSchedule'] = (df['TotalSchedules'] > 0).astype(int)
        df['EngagementScore'] = (
            df['TotalCalls'] * 1 + 
            df['TotalMeetings'] * 5 + 
            df['TotalSchedules'] * 2
        )
        
        # === RECENCY FEATURES ===
        df['IsRecentlyActive'] = (df['DaysSinceLastCall'] <= 7).astype(int)
        df['IsStale'] = (df['DaysSinceLastCall'] > 30).astype(int)
        df['ActivityGap'] = df['DaysSinceLastCall'] - df['DaysSinceLastMeeting']
        
        # === VELOCITY FEATURES ===
        df['StageProgressionRate'] = df['TotalStageChanges'] / (df['LeadAgeDays'] + 1)
        df['DaysPerStageChange'] = df['LeadAgeDays'] / (df['TotalStageChanges'] + 1)
        
        # === QUALITY FEATURES ===
        df['PositiveCallRate'] = df['PositiveCalls'] / (df['TotalCalls'] + 1)
        df['HasPositiveCalls'] = (df['PositiveCalls'] > 0).astype(int)
        
        # === LIFECYCLE FEATURES ===
        df['LeadAgeWeeks'] = df['LeadAgeDays'] / 7
        df['IsNewLead'] = (df['LeadAgeDays'] <= 14).astype(int)
        df['IsMaturedLead'] = (df['LeadAgeDays'].between(30, 90)).astype(int)
        df['IsOldLead'] = (df['LeadAgeDays'] > 90).astype(int)
        
        # === STAGE FEATURES ===
        df['IsEarlyStage'] = (df['LeadStageId'] <= 2).astype(int)
        df['IsMidStage'] = (df['LeadStageId'].between(3, 5)).astype(int)
        df['IsLateStage'] = (df['LeadStageId'] >= 6).astype(int)
        
        logger.info(f"[OK] Engineered {len([c for c in df.columns if c not in ['LeadId', 'IsConverted']])} features")
        
        return df
    
    def train_models(self):
        """Train multiple ML models and save the best one"""
        
        logger.info("="*80)
        logger.info("[START] ML Model Training")
        logger.info("="*80)
        
        # Connect and load data
        if not self.connect_db():
            return None
        
        df = self.extract_training_data()
        if df is None or len(df) < 50:
            return None
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Define feature columns
        feature_cols = [
            # Engagement
            'TotalCalls', 'TotalMeetings', 'TotalSchedules',
            'CallFrequency', 'MeetingRate', 'HasMeeting', 'HasSchedule',
            'EngagementScore', 'PositiveCallRate', 'HasPositiveCalls',
            'AvgCallDuration',
            
            # Recency
            'DaysSinceLastCall', 'DaysSinceLastMeeting', 'ActivityGap',
            'IsRecentlyActive', 'IsStale',
            
            # Velocity
            'StageProgressionRate', 'DaysPerStageChange', 'TotalStageChanges',
            
            # Lifecycle
            'LeadAgeDays', 'LeadAgeWeeks',
            'IsNewLead', 'IsMaturedLead', 'IsOldLead',
            
            # Context
            'LeadStageId', 'CountryId', 'LeadSourceId',
            'IsEarlyStage', 'IsMidStage', 'IsLateStage'
        ]
        
        X = df[feature_cols]
        y = df['IsConverted']
        
        logger.info(f"\n[INFO] Dataset Summary:")
        logger.info(f"       Total samples: {len(df)}")
        logger.info(f"       Features: {len(feature_cols)}")
        logger.info(f"       Conversion rate: {y.mean()*100:.1f}%")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"\n[SPLIT] Train-Test Split:")
        logger.info(f"        Training: {len(X_train)} samples")
        logger.info(f"        Testing: {len(X_test)} samples")
        
        # Define models
        models = {
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        results = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"[ML] Training {name}...")
            logger.info('='*80)
            
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Metrics
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                accuracy = model.score(X_test, y_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                
                logger.info(f"\n[PERF] {name} Performance:")
                logger.info(f"       ROC-AUC: {roc_auc:.4f}")
                logger.info(f"       Accuracy: {accuracy:.4f}")
                logger.info(f"       CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                
                logger.info(f"\n[REPORT] Classification Report:")
                logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Lost', 'Converted'])}")
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': feature_cols,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    logger.info(f"\n[TOP10] Top 10 Important Features:")
                    for idx, row in importance_df.head(10).iterrows():
                        logger.info(f"        {row['feature']:30s} {row['importance']:.4f}")
                
                results[name] = {
                    'model': model,
                    'roc_auc': roc_auc,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'feature_importance': dict(zip(feature_cols, model.feature_importances_)) if hasattr(model, 'feature_importances_') else {}
                }
                
            except Exception as e:
                logger.error(f"[ERROR] {name} training failed: {e}")
                continue
        
        if not results:
            logger.error("[ERROR] No models were successfully trained")
            return None
        
        # Select best model
        best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
        best_model = results[best_model_name]['model']
        best_metrics = results[best_model_name]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[BEST] BEST MODEL: {best_model_name}")
        logger.info(f"       ROC-AUC: {best_metrics['roc_auc']:.4f}")
        logger.info(f"       Accuracy: {best_metrics['accuracy']:.4f}")
        logger.info('='*80)
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'lead_conversion_{best_model_name.lower().replace(" ", "_")}_{timestamp}.pkl'
        model_path = os.path.join('models', model_filename)
        
        joblib.dump(best_model, model_path)
        joblib.dump(feature_cols, 'models/feature_columns.pkl')
        
        # FIXED: Save metadata with proper type conversion (no numpy types)
        feature_importance_fixed = {
            k: float(v) for k, v in best_metrics['feature_importance'].items()
        }
        
        metadata = {
            'model_type': best_model_name,
            'version': timestamp,
            'roc_auc': float(best_metrics['roc_auc']),
            'accuracy': float(best_metrics['accuracy']),
            'cv_mean': float(best_metrics['cv_mean']),
            'cv_std': float(best_metrics['cv_std']),
            'training_date': datetime.now().isoformat(),
            'training_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'total_features': int(len(feature_cols)),
            'features': feature_cols,
            'conversion_rate': float(y.mean()),
            'feature_importance': feature_importance_fixed
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\n[SAVE] Model saved successfully:")
        logger.info(f"       Model file: {model_path}")
        logger.info(f"       Features file: models/feature_columns.pkl")
        logger.info(f"       Metadata file: models/model_metadata.json")
        
        return best_model, feature_cols, metadata


def main():
    """Main training function"""
    
    # Your database credentials
    trainer = MLModelTrainer(
        server="auto.resourceplus.app",
        database="Data_Lead",
        username="sa",
        password="test!serv!123"
    )
    
    # Train models
    result = trainer.train_models()
    
    if result:
        logger.info("\n" + "="*80)
        logger.info("[DONE] ML MODEL TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info("\nYour ML_Engine is now ready to use trained models!")
        logger.info("Run ml_engine_v4.py with use_ml_model=True to activate TRUE ML predictions")
    else:
        logger.error("\n" + "="*80)
        logger.error("[FAILED] ML MODEL TRAINING FAILED")
        logger.error("="*80)


if __name__ == "__main__":
    main()
