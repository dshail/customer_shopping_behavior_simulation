"""
MLOps Pipeline for Customer Behavior Simulation

This module provides:
- Model training and validation
- Performance monitoring
- A/B testing framework
- Automated retraining
- Model deployment and versioning
"""

import json
import logging
import os
import pickle
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import mlflow
import mlflow.sklearn
from pathlib import Path
import yaml

from .models import PersonaConfig, Transaction, Customer


@dataclass
class ModelConfig:
    """Configuration for ML models"""
    model_type: str
    hyperparameters: Dict[str, Any]
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    performance_threshold: float = 0.8
    retrain_frequency_days: int = 7


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    mae: float
    r2_score: float
    training_time: float
    prediction_time: float
    model_size_mb: float


class CustomerBehaviorPredictor:
    """ML model for predicting customer behavior"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.logger = logging.getLogger(__name__)
        self.model_version = None
        
    def prepare_features(
        self, 
        transactions_df: pd.DataFrame, 
        customers_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare features for ML model training"""
        
        # Customer-level aggregations
        customer_features = transactions_df.groupby('customer_id').agg({
            'total_amount': ['sum', 'mean', 'std', 'count'],
            'num_items': ['sum', 'mean'],
            'date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names properly
        customer_features.columns = ['_'.join(col).strip() if col[1] else col[0] for col in customer_features.columns.values]
        
        # Rename columns to meaningful names
        column_mapping = {
            'customer_id': 'customer_id',  # This stays the same
            'total_amount_sum': 'total_spent',
            'total_amount_mean': 'avg_transaction', 
            'total_amount_std': 'spending_std',
            'total_amount_count': 'transaction_count',
            'num_items_sum': 'total_items',
            'num_items_mean': 'avg_items',
            'date_min': 'first_purchase',
            'date_max': 'last_purchase'
        }
        
        customer_features = customer_features.rename(columns=column_mapping)
        
        # Calculate customer lifetime and frequency
        customer_features['customer_lifetime_days'] = (
            pd.to_datetime(customer_features['last_purchase']) - 
            pd.to_datetime(customer_features['first_purchase'])
        ).dt.days + 1
        
        customer_features['purchase_frequency'] = (
            customer_features['transaction_count'] / customer_features['customer_lifetime_days']
        )
        
        # Merge with customer demographics
        features_df = customer_features.merge(customers_df, on='customer_id', how='left')
        
        # Add temporal features
        features_df['days_since_last_purchase'] = (
            datetime.now() - pd.to_datetime(features_df['last_purchase'])
        ).dt.days
        
        # Rename columns to match expected names
        if 'persona_type' in features_df.columns:
            features_df['persona'] = features_df['persona_type']
        
        # Add persona-based features
        persona_stats = transactions_df.groupby('persona_type')['total_amount'].agg(['mean', 'std']).reset_index()
        persona_stats.columns = ['persona_type', 'persona_avg_spend', 'persona_spend_std']
        features_df = features_df.merge(persona_stats, on='persona_type', how='left')
        
        # Calculate relative spending (compared to persona average)
        features_df['relative_spending'] = (
            features_df['avg_transaction'] / features_df['persona_avg_spend']
        )
        
        return features_df
    
    def train_churn_prediction_model(
        self, 
        features_df: pd.DataFrame,
        churn_threshold_days: int = 30
    ) -> ModelMetrics:
        """Train model to predict customer churn"""
        
        start_time = datetime.now()
        
        # Define churn target
        features_df['is_churned'] = (features_df['days_since_last_purchase'] > churn_threshold_days).astype(int)
        
        # Prepare features
        feature_cols = [
            'total_spent', 'avg_transaction', 'spending_std', 'transaction_count',
            'total_items', 'avg_items', 'customer_lifetime_days', 'purchase_frequency',
            'age', 'income', 'relative_spending'
        ]
        
        X = features_df[feature_cols].fillna(0)
        y = features_df['is_churned']
        
        # Encode categorical variables if any
        for col in X.select_dtypes(include=['object']).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.feature_columns = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config.validation_split, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(**self.config.hyperparameters)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        # Handle case where only one class is present
        y_pred_proba_raw = self.model.predict_proba(X_test)
        if y_pred_proba_raw.shape[1] > 1:
            y_pred_proba = y_pred_proba_raw[:, 1]  # Probability of positive class
        else:
            y_pred_proba = y_pred_proba_raw[:, 0]  # Only one class available
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Model size
        model_path = 'temp_model.pkl'
        joblib.dump(self.model, model_path)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        os.remove(model_path)
        
        metrics = ModelMetrics(
            accuracy=accuracy,
            precision=report['weighted avg']['precision'],
            recall=report['weighted avg']['recall'],
            f1_score=report['weighted avg']['f1-score'],
            mse=0,  # Not applicable for classification
            mae=0,  # Not applicable for classification
            r2_score=0,  # Not applicable for classification
            training_time=training_time,
            prediction_time=0,  # Will be measured during prediction
            model_size_mb=model_size_mb
        )
        
        self.logger.info(f"Churn prediction model trained with accuracy: {accuracy:.3f}")
        return metrics
    
    def train_spending_prediction_model(
        self, 
        features_df: pd.DataFrame
    ) -> ModelMetrics:
        """Train model to predict customer spending"""
        
        start_time = datetime.now()
        
        # Prepare features for spending prediction
        feature_cols = [
            'transaction_count', 'total_items', 'avg_items', 'customer_lifetime_days',
            'purchase_frequency', 'age', 'income', 'days_since_last_purchase'
        ]
        
        X = features_df[feature_cols].fillna(0)
        y = features_df['total_spent']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.feature_columns = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config.validation_split, random_state=42
        )
        
        # Train model
        self.model = RandomForestRegressor(**self.config.hyperparameters)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = self.model.score(X_test, y_test)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Model size
        model_path = 'temp_model.pkl'
        joblib.dump(self.model, model_path)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        os.remove(model_path)
        
        metrics = ModelMetrics(
            accuracy=0,  # Not applicable for regression
            precision=0,  # Not applicable for regression
            recall=0,  # Not applicable for regression
            f1_score=0,  # Not applicable for regression
            mse=mse,
            mae=mae,
            r2_score=r2,
            training_time=training_time,
            prediction_time=0,
            model_size_mb=model_size_mb
        )
        
        self.logger.info(f"Spending prediction model trained with R²: {r2:.3f}")
        return metrics
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model"""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_* method first.")
        
        start_time = datetime.now()
        
        # Prepare features
        X = features[self.feature_columns].fillna(0)
        
        # Apply label encoding if needed
        for col in X.select_dtypes(include=['object']).columns:
            if col in self.label_encoders:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        prediction_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Made {len(predictions)} predictions in {prediction_time:.3f}s")
        
        return predictions
    
    def save_model(self, model_path: str, version: str = None):
        """Save trained model and preprocessing components"""
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.model_version = version
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'config': self.config,
            'version': version,
            'created_at': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_data, model_path)
        
        self.logger.info(f"Model saved to {model_path} with version {version}")
    
    def load_model(self, model_path: str):
        """Load trained model and preprocessing components"""
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.config = model_data['config']
        self.model_version = model_data['version']
        
        self.logger.info(f"Model loaded from {model_path}, version {self.model_version}")


class MLOpsMonitor:
    """Monitor ML model performance and trigger retraining"""
    
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
        
    def monitor_model_performance(
        self, 
        model: CustomerBehaviorPredictor,
        new_data: pd.DataFrame,
        ground_truth: pd.Series
    ) -> Dict[str, Any]:
        """Monitor model performance on new data"""
        
        # Make predictions
        predictions = model.predict(new_data)
        
        # Calculate current performance
        if model.config.model_type == 'classification':
            current_accuracy = accuracy_score(ground_truth, predictions)
            performance_metric = current_accuracy
        else:
            current_mse = mean_squared_error(ground_truth, predictions)
            current_r2 = model.model.score(
                model.scaler.transform(new_data[model.feature_columns]), 
                ground_truth
            )
            performance_metric = current_r2
        
        # Check if performance degraded
        needs_retraining = performance_metric < self.config.performance_threshold
        
        # Log metrics
        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            'performance_metric': performance_metric,
            'needs_retraining': needs_retraining,
            'data_size': len(new_data)
        }
        
        self.metrics_history.append(metrics_entry)
        
        # Alert if retraining needed
        if needs_retraining:
            self.logger.warning(
                f"Model performance degraded: {performance_metric:.3f} < {self.config.performance_threshold}"
            )
        
        return metrics_entry
    
    def check_data_drift(
        self, 
        reference_data: pd.DataFrame,
        new_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Detect data drift in features"""
        
        drift_results = {}
        
        for column in reference_data.columns:
            if column in new_data.columns:
                # Statistical tests for drift detection
                ref_mean = reference_data[column].mean()
                new_mean = new_data[column].mean()
                ref_std = reference_data[column].std()
                new_std = new_data[column].std()
                
                # Simple drift detection (can be enhanced with KS test, etc.)
                mean_drift = abs(new_mean - ref_mean) / ref_std if ref_std > 0 else 0
                std_drift = abs(new_std - ref_std) / ref_std if ref_std > 0 else 0
                
                drift_results[column] = {
                    'mean_drift': mean_drift,
                    'std_drift': std_drift,
                    'has_drift': mean_drift > 2 or std_drift > 0.5  # Threshold-based detection
                }
        
        overall_drift = any(result['has_drift'] for result in drift_results.values())
        
        self.logger.info(f"Data drift analysis completed. Overall drift detected: {overall_drift}")
        
        return {
            'overall_drift': overall_drift,
            'feature_drift': drift_results,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def should_retrain_model(self, last_training_date: datetime) -> bool:
        """Determine if model should be retrained based on schedule and performance"""
        
        # Check time-based retraining
        days_since_training = (datetime.now() - last_training_date).days
        time_based_retrain = days_since_training >= self.config.retrain_frequency_days
        
        # Check performance-based retraining
        recent_metrics = self.metrics_history[-5:] if len(self.metrics_history) >= 5 else self.metrics_history
        performance_based_retrain = any(m['needs_retraining'] for m in recent_metrics)
        
        should_retrain = time_based_retrain or performance_based_retrain
        
        self.logger.info(
            f"Retrain decision: time_based={time_based_retrain}, "
            f"performance_based={performance_based_retrain}, should_retrain={should_retrain}"
        )
        
        return should_retrain


class ABTestingFramework:
    """A/B testing framework for model experiments"""
    
    def __init__(self):
        self.experiments = {}
        self.logger = logging.getLogger(__name__)
    
    def create_experiment(
        self, 
        experiment_name: str,
        control_model: CustomerBehaviorPredictor,
        treatment_model: CustomerBehaviorPredictor,
        traffic_split: float = 0.5
    ):
        """Create new A/B test experiment"""
        
        experiment = {
            'name': experiment_name,
            'control_model': control_model,
            'treatment_model': treatment_model,
            'traffic_split': traffic_split,
            'created_at': datetime.now(),
            'results': {
                'control': {'predictions': [], 'actuals': [], 'metrics': {}},
                'treatment': {'predictions': [], 'actuals': [], 'metrics': {}}
            }
        }
        
        self.experiments[experiment_name] = experiment
        self.logger.info(f"Created A/B test experiment: {experiment_name}")
    
    def run_prediction(
        self, 
        experiment_name: str,
        features: pd.DataFrame,
        customer_id: str = None
    ) -> Tuple[np.ndarray, str]:
        """Run prediction through A/B test"""
        
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        experiment = self.experiments[experiment_name]
        
        # Determine which model to use (simple hash-based assignment)
        if customer_id:
            assignment_hash = hash(customer_id) % 100
            use_treatment = assignment_hash < (experiment['traffic_split'] * 100)
        else:
            use_treatment = np.random.random() < experiment['traffic_split']
        
        if use_treatment:
            predictions = experiment['treatment_model'].predict(features)
            variant = 'treatment'
        else:
            predictions = experiment['control_model'].predict(features)
            variant = 'control'
        
        return predictions, variant
    
    def record_outcome(
        self, 
        experiment_name: str,
        variant: str,
        predictions: np.ndarray,
        actuals: np.ndarray
    ):
        """Record experiment outcomes"""
        
        if experiment_name not in self.experiments:
            return
        
        experiment = self.experiments[experiment_name]
        experiment['results'][variant]['predictions'].extend(predictions)
        experiment['results'][variant]['actuals'].extend(actuals)
    
    def analyze_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Analyze A/B test results"""
        
        if experiment_name not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_name]
        results = {}
        
        for variant in ['control', 'treatment']:
            variant_data = experiment['results'][variant]
            
            if len(variant_data['predictions']) > 0:
                predictions = np.array(variant_data['predictions'])
                actuals = np.array(variant_data['actuals'])
                
                # Calculate metrics based on model type
                if experiment['control_model'].config.model_type == 'classification':
                    accuracy = accuracy_score(actuals, predictions)
                    results[variant] = {
                        'accuracy': accuracy,
                        'sample_size': len(predictions)
                    }
                else:
                    mse = mean_squared_error(actuals, predictions)
                    mae = np.mean(np.abs(actuals - predictions))
                    results[variant] = {
                        'mse': mse,
                        'mae': mae,
                        'sample_size': len(predictions)
                    }
        
        # Statistical significance test (simplified)
        if 'control' in results and 'treatment' in results:
            control_metric = list(results['control'].values())[0]
            treatment_metric = list(results['treatment'].values())[0]
            
            improvement = ((treatment_metric - control_metric) / control_metric) * 100
            results['improvement_percent'] = improvement
            results['is_significant'] = abs(improvement) > 5  # Simplified significance test
        
        self.logger.info(f"A/B test analysis completed for {experiment_name}")
        return results


class ModelVersionManager:
    """Manage model versions and deployments"""
    
    def __init__(self, models_directory: str = "models"):
        self.models_dir = Path(models_directory)
        self.models_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def save_model_version(
        self, 
        model: CustomerBehaviorPredictor,
        model_name: str,
        version: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Save model version with metadata"""
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = self.models_dir / model_name / version
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_path / "model.pkl"
        model.save_model(str(model_file), version)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'model_name': model_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'model_type': model.config.model_type,
            'feature_columns': model.feature_columns
        })
        
        metadata_file = model_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved model version {model_name}:{version}")
        return version
    
    def load_model_version(
        self, 
        model_name: str,
        version: str = "latest"
    ) -> CustomerBehaviorPredictor:
        """Load specific model version"""
        
        if version == "latest":
            version = self.get_latest_version(model_name)
        
        model_path = self.models_dir / model_name / version / "model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name}:{version} not found")
        
        # Load model config from metadata
        metadata_path = self.models_dir / model_name / version / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create model instance and load
        model_config = ModelConfig(
            model_type=metadata['model_type'],
            hyperparameters={}  # Will be loaded from saved model
        )
        
        model = CustomerBehaviorPredictor(model_config)
        model.load_model(str(model_path))
        
        self.logger.info(f"Loaded model version {model_name}:{version}")
        return model
    
    def get_latest_version(self, model_name: str) -> str:
        """Get latest version of a model"""
        
        model_dir = self.models_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_name} not found")
        
        versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
        versions.sort(reverse=True)
        
        return versions[0] if versions else None
    
    def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a model"""
        
        model_dir = self.models_dir / model_name
        if not model_dir.exists():
            return []
        
        versions = []
        for version_dir in model_dir.iterdir():
            if version_dir.is_dir():
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    versions.append(metadata)
        
        return sorted(versions, key=lambda x: x['created_at'], reverse=True)


# Utility functions
def create_mlops_config():
    """Create sample MLOps configuration"""
    
    config = {
        'models': {
            'churn_prediction': {
                'model_type': 'classification',
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                },
                'performance_threshold': 0.8,
                'retrain_frequency_days': 7
            },
            'spending_prediction': {
                'model_type': 'regression',
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': 15,
                    'random_state': 42
                },
                'performance_threshold': 0.7,
                'retrain_frequency_days': 14
            }
        },
        'monitoring': {
            'enable_drift_detection': True,
            'drift_threshold': 0.1,
            'performance_alert_threshold': 0.05
        },
        'ab_testing': {
            'default_traffic_split': 0.1,
            'minimum_sample_size': 1000,
            'significance_threshold': 0.05
        }
    }
    
    os.makedirs('config', exist_ok=True)
    with open('config/mlops_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("MLOps configuration created at config/mlops_config.yaml")


def setup_mlflow_tracking():
    """Setup MLflow for experiment tracking"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Create experiment if it doesn't exist
    experiment_name = "customer_behavior_simulation"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow tracking setup complete. Experiment ID: {experiment_id}")
    return experiment_id