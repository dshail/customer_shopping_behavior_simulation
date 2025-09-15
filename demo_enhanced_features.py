#!/usr/bin/env python3
"""
Demo Script: Enhanced Features Showcase

This script demonstrates the LLM Integration and MLOps Pipeline features
of the Customer Behavior Simulation System.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Suppress warnings for demo
import warnings
warnings.filterwarnings('ignore')

def setup_demo_logging():
    """Setup logging for demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_sample_market_data():
    """Create sample market research data for AI persona generation"""
    
    market_data = {
        "market_overview": {
            "total_market_size": "₹2.5 trillion",
            "growth_rate": "8.5% annually",
            "key_segments": ["grocery", "household", "personal_care"]
        },
        "demographics": {
            "age_groups": {
                "18-25": 0.18,  # Gen Z - tech-savvy, budget-conscious
                "26-35": 0.28,  # Millennials - family-oriented, convenience-focused
                "36-50": 0.32,  # Gen X - premium buyers, brand loyal
                "51-65": 0.15,  # Baby Boomers - traditional shoppers
                "65+": 0.07     # Seniors - health-conscious, price-sensitive
            },
            "income_levels": {
                "low": {"range": "₹0-30k", "percentage": 0.35},
                "medium": {"range": "₹30k-80k", "percentage": 0.45},
                "high": {"range": "₹80k-200k", "percentage": 0.15},
                "premium": {"range": "₹200k+", "percentage": 0.05}
            },
            "location_types": {
                "urban": 0.45,
                "suburban": 0.35,
                "rural": 0.20
            }
        },
        "shopping_trends": {
            "digital_adoption": {
                "online_shopping": 0.65,
                "mobile_apps": 0.58,
                "social_commerce": 0.32
            },
            "sustainability_focus": {
                "eco_friendly_products": 0.42,
                "local_sourcing": 0.38,
                "minimal_packaging": 0.35
            },
            "health_consciousness": {
                "organic_products": 0.28,
                "sugar_free": 0.45,
                "protein_rich": 0.38
            },
            "convenience_factors": {
                "quick_delivery": 0.72,
                "one_click_ordering": 0.55,
                "subscription_services": 0.23
            }
        },
        "seasonal_patterns": {
            "festival_seasons": {
                "diwali_boost": 1.8,
                "christmas_boost": 1.4,
                "eid_boost": 1.3,
                "new_year_boost": 1.2
            },
            "weather_impact": {
                "summer_cooling_products": 1.5,
                "monsoon_essentials": 1.3,
                "winter_comfort_foods": 1.2
            },
            "monthly_patterns": {
                "salary_week_boost": 1.4,
                "month_end_decline": 0.8
            }
        },
        "competitive_landscape": {
            "market_leaders": ["BigBasket", "Grofers", "Amazon Fresh"],
            "key_differentiators": ["price", "delivery_speed", "product_variety"],
            "customer_loyalty_factors": ["quality", "service", "convenience"]
        },
        "emerging_trends": {
            "technology": ["AI recommendations", "voice ordering", "AR try-before-buy"],
            "payment": ["digital wallets", "BNPL", "cryptocurrency"],
            "delivery": ["drone delivery", "dark stores", "15-min delivery"]
        }
    }
    
    # Save to file
    os.makedirs('data/demo', exist_ok=True)
    with open('data/demo/market_research.json', 'w') as f:
        json.dump(market_data, f, indent=2)
    
    return market_data

def demo_llm_features(logger):
    """Demonstrate LLM Integration features"""
    
    logger.info("🤖 DEMONSTRATING LLM INTEGRATION FEATURES")
    logger.info("=" * 60)
    
    try:
        from src.llm_integration import (
            LLMPersonaGenerator, 
            LLMInsightsGenerator, 
            LLMReportGenerator,
            load_llm_config
        )
        
        # Load LLM configuration
        llm_config = load_llm_config()
        
        if not llm_config or llm_config.api_key == "your-openrouter-api-key-here":
            logger.warning("⚠️ OpenRouter API key not configured - skipping LLM demos")
            logger.info("To enable LLM features:")
            logger.info("1. Get an OpenRouter API key from https://openrouter.ai/")
            logger.info("2. Update config/llm_config.yaml with your API key")
            logger.info("3. Or set environment variable: export OPENROUTER_API_KEY='your-key'")
            return False
        
        # Create sample market data
        market_data = create_sample_market_data()
        logger.info("✅ Created sample market research data")
        
        # Demo 1: AI Persona Generation
        logger.info("\n🎭 Demo 1: AI-Powered Persona Generation")
        logger.info("-" * 40)
        
        persona_generator = LLMPersonaGenerator(llm_config)
        
        logger.info("Generating customer personas from market data...")
        personas = persona_generator.generate_personas_from_market_data(
            market_data, num_personas=3
        )
        
        if personas:
            logger.info(f"✅ Generated {len(personas)} AI-powered personas:")
            for i, persona in enumerate(personas, 1):
                logger.info(f"   {i}. {persona.name}")
                logger.info(f"      Frequency: {persona.frequency}")
                logger.info(f"      Age Range: {persona.demographics.age_range}")
                logger.info(f"      Income Range: {persona.demographics.income_range}")
        
        # Demo 2: Insights Generation
        logger.info("\n🧠 Demo 2: AI-Powered Insights Generation")
        logger.info("-" * 40)
        
        insights_generator = LLMInsightsGenerator(llm_config)
        
        # Sample simulation results
        sample_results = {
            "total_revenue": 295384376.70,
            "total_transactions": 60102,
            "total_customers": 5000,
            "avg_transaction_value": 4914.72,
            "persona_performance": {
                "Premium Shopper": {"revenue": 221861408.10, "transactions": 25920},
                "Family Shopper": {"revenue": 37105237.55, "transactions": 6931},
                "Young Professional": {"revenue": 21573037.83, "transactions": 17518}
            }
        }
        
        logger.info("Generating executive summary...")
        executive_summary = insights_generator.generate_executive_summary(sample_results)
        
        if executive_summary:
            logger.info("✅ Generated AI executive summary")
            logger.info("Preview:")
            logger.info(executive_summary[:200] + "..." if len(executive_summary) > 200 else executive_summary)
        
        # Demo 3: Report Generation
        logger.info("\n📝 Demo 3: Natural Language Report Generation")
        logger.info("-" * 40)
        
        report_generator = LLMReportGenerator(llm_config)
        
        logger.info("Generating comprehensive business report...")
        comprehensive_report = report_generator.generate_narrative_report(
            sample_results, "executive"
        )
        
        if comprehensive_report:
            logger.info("✅ Generated AI business report")
            
            # Save reports
            os.makedirs('data/demo/reports', exist_ok=True)
            with open('data/demo/reports/ai_executive_summary.txt', 'w') as f:
                f.write(executive_summary)
            with open('data/demo/reports/ai_business_report.txt', 'w') as f:
                f.write(comprehensive_report)
            
            logger.info("📁 Reports saved to data/demo/reports/")
        
        logger.info("\n🎉 LLM Integration demo completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ LLM dependencies not installed: {e}")
        logger.info("Install with: pip install requests>=2.31.0")
        return False
    except Exception as e:
        logger.error(f"❌ LLM demo failed: {e}")
        return False

def demo_mlops_features(logger):
    """Demonstrate MLOps Pipeline features"""
    
    logger.info("\n🔧 DEMONSTRATING MLOPS PIPELINE FEATURES")
    logger.info("=" * 60)
    
    try:
        from src.mlops_pipeline import (
            CustomerBehaviorPredictor,
            MLOpsMonitor,
            ABTestingFramework,
            ModelVersionManager,
            ModelConfig,
            setup_mlflow_tracking
        )
        import pandas as pd
        import numpy as np
        
        # Setup MLflow tracking
        logger.info("Setting up MLflow experiment tracking...")
        try:
            experiment_id = setup_mlflow_tracking()
            logger.info(f"✅ MLflow tracking initialized (Experiment ID: {experiment_id})")
        except Exception as e:
            logger.warning(f"⚠️ MLflow setup failed: {e}")
        
        # Create sample data for demo
        logger.info("Creating sample customer data...")
        np.random.seed(42)
        
        # Generate synthetic customer features
        n_customers = 1000
        sample_data = pd.DataFrame({
            'customer_id': [f'CUST_{i:04d}' for i in range(n_customers)],
            'total_spent': np.random.lognormal(8, 1, n_customers),
            'avg_transaction': np.random.lognormal(7, 0.5, n_customers),
            'spending_std': np.random.exponential(500, n_customers),
            'transaction_count': np.random.poisson(15, n_customers),
            'total_items': np.random.poisson(50, n_customers),
            'avg_items': np.random.normal(3.5, 1, n_customers),
            'customer_lifetime_days': np.random.randint(30, 365, n_customers),
            'purchase_frequency': np.random.exponential(0.1, n_customers),
            'age': np.random.randint(18, 70, n_customers),
            'income': np.random.lognormal(10, 0.5, n_customers),
            'days_since_last_purchase': np.random.randint(0, 60, n_customers),
            'relative_spending': np.random.normal(1, 0.3, n_customers)
        })
        
        logger.info(f"✅ Created sample dataset with {len(sample_data)} customers")
        
        # Demo 1: Model Training
        logger.info("\n🤖 Demo 1: ML Model Training")
        logger.info("-" * 40)
        
        # Configure churn prediction model
        churn_config = ModelConfig(
            model_type="classification",
            hyperparameters={
                "n_estimators": 50,  # Reduced for demo speed
                "max_depth": 8,
                "random_state": 42
            },
            performance_threshold=0.75
        )
        
        logger.info("Training churn prediction model...")
        churn_model = CustomerBehaviorPredictor(churn_config)
        
        # Train model (using sample data directly)
        churn_metrics = churn_model.train_churn_prediction_model(sample_data)
        
        logger.info(f"✅ Churn model trained successfully!")
        logger.info(f"   Accuracy: {churn_metrics.accuracy:.3f}")
        logger.info(f"   Precision: {churn_metrics.precision:.3f}")
        logger.info(f"   Recall: {churn_metrics.recall:.3f}")
        logger.info(f"   Training time: {churn_metrics.training_time:.2f}s")
        
        # Train spending prediction model
        spending_config = ModelConfig(
            model_type="regression",
            hyperparameters={
                "n_estimators": 50,
                "max_depth": 10,
                "random_state": 42
            }
        )
        
        logger.info("Training spending prediction model...")
        spending_model = CustomerBehaviorPredictor(spending_config)
        spending_metrics = spending_model.train_spending_prediction_model(sample_data)
        
        logger.info(f"✅ Spending model trained successfully!")
        logger.info(f"   R² Score: {spending_metrics.r2_score:.3f}")
        logger.info(f"   MSE: {spending_metrics.mse:.2f}")
        logger.info(f"   MAE: {spending_metrics.mae:.2f}")
        
        # Demo 2: Model Version Management
        logger.info("\n📦 Demo 2: Model Version Management")
        logger.info("-" * 40)
        
        version_manager = ModelVersionManager("data/demo/models")
        
        # Save model versions
        churn_version = version_manager.save_model_version(
            churn_model, 
            "churn_prediction",
            metadata={
                "training_data_size": len(sample_data),
                "performance_metrics": churn_metrics.__dict__,
                "model_type": "RandomForestClassifier"
            }
        )
        
        spending_version = version_manager.save_model_version(
            spending_model,
            "spending_prediction", 
            metadata={
                "training_data_size": len(sample_data),
                "performance_metrics": spending_metrics.__dict__,
                "model_type": "RandomForestRegressor"
            }
        )
        
        logger.info(f"✅ Saved churn model version: {churn_version}")
        logger.info(f"✅ Saved spending model version: {spending_version}")
        
        # List model versions
        churn_versions = version_manager.list_model_versions("churn_prediction")
        logger.info(f"📋 Churn model has {len(churn_versions)} version(s)")
        
        # Demo 3: Model Monitoring
        logger.info("\n📊 Demo 3: Model Performance Monitoring")
        logger.info("-" * 40)
        
        monitor = MLOpsMonitor(churn_config)
        
        # Create new data for monitoring
        new_data = sample_data.sample(100).copy()
        new_data['total_spent'] *= np.random.normal(1, 0.1, 100)  # Add some drift
        
        # Generate synthetic ground truth
        ground_truth = np.random.choice([0, 1], size=len(new_data))
        
        logger.info("Monitoring model performance on new data...")
        performance_metrics = monitor.monitor_model_performance(
            churn_model, new_data, ground_truth
        )
        
        logger.info(f"✅ Performance monitoring completed")
        logger.info(f"   Current performance: {performance_metrics['performance_metric']:.3f}")
        logger.info(f"   Needs retraining: {performance_metrics['needs_retraining']}")
        
        # Check for data drift
        logger.info("Checking for data drift...")
        drift_analysis = monitor.check_data_drift(sample_data, new_data)
        
        logger.info(f"✅ Data drift analysis completed")
        logger.info(f"   Overall drift detected: {drift_analysis['overall_drift']}")
        
        drifted_features = [
            feature for feature, result in drift_analysis['feature_drift'].items() 
            if result['has_drift']
        ]
        if drifted_features:
            logger.info(f"   Features with drift: {', '.join(drifted_features[:3])}")
        
        # Demo 4: A/B Testing
        logger.info("\n🧪 Demo 4: A/B Testing Framework")
        logger.info("-" * 40)
        
        ab_framework = ABTestingFramework()
        
        # Create a second model for comparison (with different hyperparameters)
        treatment_config = ModelConfig(
            model_type="classification",
            hyperparameters={
                "n_estimators": 100,  # More trees
                "max_depth": 12,      # Deeper trees
                "random_state": 42
            }
        )
        
        treatment_model = CustomerBehaviorPredictor(treatment_config)
        treatment_model.train_churn_prediction_model(sample_data)
        
        # Create A/B test experiment
        logger.info("Creating A/B test experiment...")
        ab_framework.create_experiment(
            "churn_model_comparison",
            control_model=churn_model,
            treatment_model=treatment_model,
            traffic_split=0.5
        )
        
        # Run predictions through A/B test
        test_data = sample_data.sample(200)
        control_predictions = []
        treatment_predictions = []
        
        logger.info("Running A/B test predictions...")
        for _, row in test_data.iterrows():
            features = pd.DataFrame([row])
            predictions, variant = ab_framework.run_prediction(
                "churn_model_comparison",
                features,
                row['customer_id']
            )
            
            if variant == 'control':
                control_predictions.extend(predictions)
            else:
                treatment_predictions.extend(predictions)
        
        # Record outcomes (synthetic for demo)
        if control_predictions:
            ab_framework.record_outcome(
                "churn_model_comparison",
                "control",
                control_predictions,
                np.random.choice([0, 1], len(control_predictions))
            )
        
        if treatment_predictions:
            ab_framework.record_outcome(
                "churn_model_comparison", 
                "treatment",
                treatment_predictions,
                np.random.choice([0, 1], len(treatment_predictions))
            )
        
        # Analyze A/B test results
        logger.info("Analyzing A/B test results...")
        ab_results = ab_framework.analyze_experiment("churn_model_comparison")
        
        logger.info(f"✅ A/B test analysis completed")
        if 'control' in ab_results and 'treatment' in ab_results:
            logger.info(f"   Control accuracy: {ab_results['control'].get('accuracy', 'N/A')}")
            logger.info(f"   Treatment accuracy: {ab_results['treatment'].get('accuracy', 'N/A')}")
            if 'improvement_percent' in ab_results:
                logger.info(f"   Improvement: {ab_results['improvement_percent']:.2f}%")
        
        # Save demo results
        demo_results = {
            "models_trained": {
                "churn_prediction": {
                    "version": churn_version,
                    "accuracy": churn_metrics.accuracy,
                    "training_time": churn_metrics.training_time
                },
                "spending_prediction": {
                    "version": spending_version,
                    "r2_score": spending_metrics.r2_score,
                    "training_time": spending_metrics.training_time
                }
            },
            "monitoring_results": {
                "performance_metrics": performance_metrics,
                "drift_analysis": drift_analysis
            },
            "ab_test_results": ab_results,
            "demo_completed_at": datetime.now().isoformat()
        }
        
        os.makedirs('data/demo/mlops', exist_ok=True)
        with open('data/demo/mlops/demo_results.json', 'w') as f:
            json.dump(demo_results, f, indent=2, default=str)
        
        logger.info("\n🎉 MLOps Pipeline demo completed successfully!")
        logger.info("📁 Results saved to data/demo/mlops/demo_results.json")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ MLOps dependencies not installed: {e}")
        logger.info("Install with: pip install scikit-learn>=1.3.0 mlflow>=2.5.0")
        return False
    except Exception as e:
        logger.error(f"❌ MLOps demo failed: {e}")
        return False

def demo_integration_showcase(logger):
    """Demonstrate integrated LLM + MLOps workflow"""
    
    logger.info("\n🚀 INTEGRATION SHOWCASE: LLM + MLOPS WORKFLOW")
    logger.info("=" * 60)
    
    try:
        # This would demonstrate a complete workflow combining both features
        logger.info("🔄 Integrated Workflow Steps:")
        logger.info("1. 🤖 Generate AI personas from market data")
        logger.info("2. 🎯 Run simulation with AI personas")
        logger.info("3. 🔧 Train ML models on simulation data")
        logger.info("4. 📊 Monitor model performance")
        logger.info("5. 🧠 Generate AI insights from results")
        logger.info("6. 📝 Create comprehensive AI reports")
        logger.info("7. 🧪 A/B test model improvements")
        logger.info("8. 🔄 Continuous improvement loop")
        
        logger.info("\n💡 Business Value:")
        logger.info("• Automated persona generation from market research")
        logger.info("• Predictive customer behavior modeling")
        logger.info("• Real-time performance monitoring")
        logger.info("• AI-powered business insights")
        logger.info("• Continuous model improvement")
        logger.info("• Data-driven decision making")
        
        logger.info("\n🎯 Use Cases:")
        logger.info("• Customer segmentation and targeting")
        logger.info("• Churn prediction and prevention")
        logger.info("• Revenue optimization strategies")
        logger.info("• Inventory planning and demand forecasting")
        logger.info("• Marketing campaign optimization")
        logger.info("• Product recommendation systems")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration showcase failed: {e}")
        return False

def main():
    """Main demo function"""
    
    logger = setup_demo_logging()
    
    print("\n" + "="*80)
    print("🎯 ENHANCED FEATURES DEMONSTRATION")
    print("Customer Behavior Simulation System")
    print("LLM Integration + MLOps Pipeline")
    print("="*80)
    
    # Create demo directories
    Path("data/demo").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    demo_results = {
        "llm_demo": False,
        "mlops_demo": False,
        "integration_demo": False
    }
    
    # Run LLM Integration demo
    try:
        demo_results["llm_demo"] = demo_llm_features(logger)
    except Exception as e:
        logger.error(f"LLM demo failed: {e}")
    
    # Run MLOps Pipeline demo
    try:
        demo_results["mlops_demo"] = demo_mlops_features(logger)
    except Exception as e:
        logger.error(f"MLOps demo failed: {e}")
    
    # Run Integration showcase
    try:
        demo_results["integration_demo"] = demo_integration_showcase(logger)
    except Exception as e:
        logger.error(f"Integration demo failed: {e}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📋 DEMO SUMMARY")
    logger.info("="*60)
    
    successful_demos = sum(demo_results.values())
    total_demos = len(demo_results)
    
    logger.info(f"✅ Completed: {successful_demos}/{total_demos} demos")
    
    for demo_name, success in demo_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"   {demo_name.replace('_', ' ').title()}: {status}")
    
    if demo_results["llm_demo"]:
        logger.info("\n🤖 LLM Features Available:")
        logger.info("   • AI-powered persona generation")
        logger.info("   • Intelligent insights generation")
        logger.info("   • Natural language reporting")
    
    if demo_results["mlops_demo"]:
        logger.info("\n🔧 MLOps Features Available:")
        logger.info("   • ML model training and validation")
        logger.info("   • Model version management")
        logger.info("   • Performance monitoring")
        logger.info("   • A/B testing framework")
    
    logger.info("\n📁 Demo Files Generated:")
    logger.info("   • data/demo/market_research.json")
    logger.info("   • data/demo/reports/ (AI-generated reports)")
    logger.info("   • data/demo/mlops/ (ML model results)")
    logger.info("   • data/demo/models/ (Saved ML models)")
    
    logger.info("\n🚀 Next Steps:")
    logger.info("1. Configure your OpenRouter API key for LLM features")
    logger.info("2. Run: python main_enhanced.py --enable-ml-training --enable-ai-insights")
    logger.info("3. Explore the enhanced dashboard: streamlit run dashboard.py")
    logger.info("4. Check the Implementation Guide: IMPLEMENTATION_GUIDE.md")
    
    print("\n🎉 Enhanced Features Demo Completed!")
    print("The system now includes enterprise-level AI and MLOps capabilities!")

if __name__ == "__main__":
    main()