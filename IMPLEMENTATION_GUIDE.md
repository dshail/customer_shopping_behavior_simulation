# 🚀 LLM Integration & MLOps Pipeline Implementation Guide

This guide shows how to implement and use the advanced LLM Integration and MLOps Pipeline features in the Customer Behavior Simulation System.

## 📋 Table of Contents

1. [Quick Setup](#quick-setup)
2. [LLM Integration Features](#llm-integration-features)
3. [MLOps Pipeline Features](#mlops-pipeline-features)
4. [Usage Examples](#usage-examples)
5. [Advanced Configuration](#advanced-configuration)
6. [Troubleshooting](#troubleshooting)

## 🚀 Quick Setup

### 1. Install Enhanced Dependencies

```bash
# Install all dependencies including LLM and MLOps features
pip install -r requirements.txt

# Or install specific feature sets
pip install requests>=2.31.0  # LLM features via OpenRouter
pip install mlflow>=2.5.0 scikit-learn>=1.3.0  # MLOps features
```

### 2. Configure API Keys

```bash
# Set OpenRouter API key (required for LLM features)
export OPENROUTER_API_KEY="your-openrouter-api-key-here"

# Or update config/llm_config.yaml
```

### 3. Initialize MLflow Tracking

```bash
# MLflow will be automatically initialized
# View experiments at: http://localhost:5000
mlflow ui
```

## 🤖 LLM Integration Features

### AI-Powered Persona Generation

Generate customer personas from market research data using GPT-4:

```python
from src.llm_integration import LLMPersonaGenerator, load_llm_config

# Load configuration
llm_config = load_llm_config()
persona_generator = LLMPersonaGenerator(llm_config)

# Market data input
market_data = {
    "demographics": {
        "age_groups": {"18-25": 0.15, "26-35": 0.25, "36-50": 0.35, "51+": 0.25},
        "income_levels": {"low": 0.3, "medium": 0.5, "high": 0.2}
    },
    "shopping_trends": {
        "online_preference": 0.6,
        "sustainability_focus": 0.4,
        "premium_products": 0.3
    }
}

# Generate personas
personas = persona_generator.generate_personas_from_market_data(
    market_data, num_personas=5
)
```

### Intelligent Insights Generation

Generate business insights from simulation results:

```python
from src.llm_integration import LLMInsightsGenerator

insights_generator = LLMInsightsGenerator(llm_config)

# Generate executive summary
summary = insights_generator.generate_executive_summary(simulation_results)

# Analyze customer segments
segment_insights = insights_generator.analyze_customer_segments(
    transactions_df, customers_df
)

# Predict future trends
predictions = insights_generator.predict_future_trends(
    historical_data, external_factors
)
```

### Natural Language Reporting

Generate comprehensive business reports:

```python
from src.llm_integration import LLMReportGenerator

report_generator = LLMReportGenerator(llm_config)

# Generate different report types
comprehensive_report = report_generator.generate_narrative_report(
    simulation_data, "comprehensive"
)

executive_report = report_generator.generate_narrative_report(
    simulation_data, "executive"
)

marketing_report = report_generator.generate_narrative_report(
    simulation_data, "marketing"
)
```

## 🔧 MLOps Pipeline Features

### Model Training and Validation

Train ML models for customer behavior prediction:

```python
from src.mlops_pipeline import CustomerBehaviorPredictor, ModelConfig

# Configure model
config = ModelConfig(
    model_type="classification",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    performance_threshold=0.8
)

# Train churn prediction model
model = CustomerBehaviorPredictor(config)
features_df = model.prepare_features(transactions_df, customers_df)
metrics = model.train_churn_prediction_model(features_df)

print(f"Model accuracy: {metrics.accuracy:.3f}")
```

### Model Monitoring and Drift Detection

Monitor model performance and detect data drift:

```python
from src.mlops_pipeline import MLOpsMonitor

# Initialize monitor
monitor = MLOpsMonitor(config)

# Monitor performance on new data
performance_metrics = monitor.monitor_model_performance(
    model, new_data, ground_truth
)

# Check for data drift
drift_analysis = monitor.check_data_drift(reference_data, new_data)

if drift_analysis['overall_drift']:
    print("⚠️ Data drift detected - consider retraining model")
```

### A/B Testing Framework

Compare model performance with A/B testing:

```python
from src.mlops_pipeline import ABTestingFramework

# Initialize A/B testing
ab_framework = ABTestingFramework()

# Create experiment
ab_framework.create_experiment(
    "model_comparison_v1",
    control_model=old_model,
    treatment_model=new_model,
    traffic_split=0.1  # 10% traffic to new model
)

# Run predictions through A/B test
for customer_data in test_dataset:
    predictions, variant = ab_framework.run_prediction(
        "model_comparison_v1",
        customer_data,
        customer_id
    )

    # Record outcomes
    ab_framework.record_outcome(
        "model_comparison_v1",
        variant,
        predictions,
        actual_outcomes
    )

# Analyze results
results = ab_framework.analyze_experiment("model_comparison_v1")
print(f"Treatment model improvement: {results['improvement_percent']:.2f}%")
```

### Model Version Management

Manage model versions and deployments:

```python
from src.mlops_pipeline import ModelVersionManager

# Initialize version manager
version_manager = ModelVersionManager()

# Save model version
version = version_manager.save_model_version(
    model,
    "churn_prediction",
    metadata={
        "training_data_size": len(training_data),
        "performance_metrics": metrics.__dict__,
        "features": model.feature_columns
    }
)

# Load specific version
loaded_model = version_manager.load_model_version(
    "churn_prediction",
    version="20240823_143022"
)

# List all versions
versions = version_manager.list_model_versions("churn_prediction")
```

## 🎯 Usage Examples

### 1. Run Enhanced Simulation with All Features

```bash
# Generate AI personas from market data
python main_enhanced.py --generate-ai-personas --market-data market_research.json

# Run full simulation with ML training and AI insights
python main_enhanced.py --days 30 --customers 1000 --enable-ml-training --enable-ai-insights

# Run simulation with custom configuration
python main_enhanced.py --config config/ai_generated_personas.yaml --days 60
```

### 2. AI-Powered Persona Generation

```bash
# Create sample market data
cat > market_data.json << EOF
{
  "demographics": {
    "age_groups": {"18-25": 0.15, "26-35": 0.25, "36-50": 0.35, "51+": 0.25},
    "income_levels": {"low": 0.3, "medium": 0.5, "high": 0.2}
  },
  "shopping_trends": {
    "online_preference": 0.6,
    "sustainability_focus": 0.4,
    "premium_products": 0.3
  },
  "seasonal_patterns": {
    "holiday_boost": 1.4,
    "summer_decline": 0.9
  }
}
EOF

# Generate personas
python main_enhanced.py --generate-ai-personas --market-data market_data.json
```

### 3. Model Training and Monitoring

```python
# Complete MLOps workflow example
from main_enhanced import EnhancedSimulationSystem

# Initialize system
system = EnhancedSimulationSystem()

# Run simulation with ML training
results = system.run_enhanced_simulation(
    days=30,
    customers_per_persona=1000,
    enable_ml_training=True,
    enable_ai_insights=True
)

# Monitor model performance
for model_name in system.ml_models:
    monitoring_results = system.monitor_model_performance(
        model_name,
        new_customer_data
    )
    print(f"{model_name} monitoring: {monitoring_results}")

# Run A/B test
ab_results = system.run_ab_test(
    "churn_model_v2_test",
    "churn_prediction",
    "churn_prediction_v2",
    test_data
)
```

## ⚙️ Advanced Configuration

### LLM Configuration (`config/llm_config.yaml`)

```yaml
# OpenRouter API Configuration
api_key: "your-openrouter-api-key-here"
base_url: "https://openrouter.ai/api/v1"
model: "anthropic/claude-3-haiku" # Cost-effective default
temperature: 0.7
max_tokens: 2000
site_url: "https://github.com/your-repo"
site_name: "Customer Behavior Simulation"

# Feature toggles
features:
  persona_generation: true
  insights_generation: true
  report_generation: true
  trend_prediction: true

# Cost management (OpenRouter is typically 50-80% cheaper)
cost_management:
  max_monthly_spend: 25.0
  alert_threshold: 20.0
  track_usage: true

# Rate limiting
rate_limiting:
  requests_per_minute: 60
  retry_attempts: 3
```

### MLOps Configuration (`config/mlops_config.yaml`)

```yaml
# Model configurations
models:
  churn_prediction:
    model_type: "classification"
    hyperparameters:
      n_estimators: 100
      max_depth: 10
      random_state: 42
    performance_threshold: 0.8
    retrain_frequency_days: 7

# Monitoring settings
monitoring:
  enable_drift_detection: true
  drift_threshold: 0.1
  performance_alert_threshold: 0.05

# A/B testing
ab_testing:
  default_traffic_split: 0.1
  minimum_sample_size: 1000
  significance_threshold: 0.05

# Deployment settings
deployment:
  environment: "staging"
  auto_deployment: false
  rollback_enabled: true
```

## 🎨 Dashboard Integration

### Enhanced Streamlit Dashboard

```bash
# Run enhanced dashboard with ML and AI features
streamlit run dashboard_enhanced.py
```

The enhanced dashboard includes:

- **AI Insights Panel**: Display LLM-generated insights
- **Model Performance Monitoring**: Real-time ML model metrics
- **A/B Test Results**: Experiment tracking and analysis
- **Persona Comparison**: AI vs. traditional personas
- **Predictive Analytics**: Customer behavior predictions

## 📊 Output Files and Reports

The enhanced system generates comprehensive outputs:

```
data/output/enhanced/
├── transactions.csv              # Core transaction data
├── customers.csv                 # Customer profiles
├── analysis_results.json         # Statistical analysis
├── ml_models_summary.json        # ML model performance
├── ai_insights.json              # LLM-generated insights
├── comprehensive_report.txt      # Full business report
├── executive_report.txt          # Executive summary
├── dashboard_data.json           # Dashboard configuration
└── model_monitoring/             # Model performance logs
    ├── churn_prediction/
    └── spending_prediction/
```

## 🔍 Monitoring and Alerting

### Performance Monitoring

```python
# Set up automated monitoring
from src.mlops_pipeline import setup_mlflow_tracking

# Initialize tracking
experiment_id = setup_mlflow_tracking()

# Monitor in production
import mlflow

with mlflow.start_run():
    # Log metrics
    mlflow.log_metric("accuracy", model_accuracy)
    mlflow.log_metric("prediction_time", inference_time)

    # Log model
    mlflow.sklearn.log_model(model, "churn_prediction")
```

### Alerting System

```python
# Configure alerts for model degradation
def setup_alerts():
    if model_performance < threshold:
        send_alert("Model performance degraded")

    if data_drift_detected:
        send_alert("Data drift detected - retraining recommended")

    if prediction_latency > sla_threshold:
        send_alert("High prediction latency detected")
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. OpenRouter API Key Issues

```bash
# Error: "No API key provided"
export OPENROUTER_API_KEY="your-key-here"

# Or update config file
vim config/llm_config.yaml
```

#### 2. MLflow Tracking Issues

```bash
# Error: "No such file or directory: mlruns"
mkdir mlruns
mlflow ui --backend-store-uri file:./mlruns
```

#### 3. Model Training Memory Issues

```python
# Reduce batch size or use incremental learning
config.batch_size = 1000  # Reduce from default
model.partial_fit(X_batch, y_batch)  # Incremental training
```

#### 4. LLM Rate Limiting

```yaml
# Update rate limiting in config
rate_limiting:
  requests_per_minute: 20 # Reduce from 60
  retry_delay_seconds: 2 # Increase delay
```

## 🎯 Best Practices

### 1. LLM Integration

- **Cost Management**: Monitor API usage and set spending limits
- **Prompt Engineering**: Use specific, well-structured prompts
- **Error Handling**: Implement robust fallbacks for API failures
- **Caching**: Cache LLM responses to reduce costs

### 2. MLOps Pipeline

- **Model Versioning**: Always version models with metadata
- **Performance Monitoring**: Set up automated monitoring
- **Data Quality**: Validate input data before training
- **A/B Testing**: Use statistical significance testing

### 3. Production Deployment

- **Environment Separation**: Use staging before production
- **Gradual Rollout**: Start with small traffic percentages
- **Monitoring**: Implement comprehensive logging
- **Rollback Strategy**: Have quick rollback procedures

## 🚀 Next Steps

1. **Set up your OpenRouter API key** in `config/llm_config.yaml`
2. **Run the enhanced simulation** with `python main_enhanced.py`
3. **Explore AI-generated personas** and insights
4. **Train ML models** for customer behavior prediction
5. **Set up monitoring** and A/B testing
6. **Deploy to production** with proper monitoring

## 📚 Additional Resources

- [OpenRouter API Documentation](https://openrouter.ai/docs)
- [OpenRouter Models](https://openrouter.ai/models)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**🎉 You now have a production-ready system with AI and MLOps capabilities!**

The enhanced Customer Behavior Simulation System demonstrates enterprise-level features including AI-powered insights, automated ML pipelines, and comprehensive monitoring - perfect for real-world retail analytics applications.
