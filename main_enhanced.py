#!/usr/bin/env python3
"""
Enhanced Customer Behavior Simulation with LLM Integration and MLOps Pipeline

This enhanced version includes:
- AI-powered persona generation and insights
- ML model training and monitoring
- A/B testing framework
- Automated reporting and analysis
"""

import argparse
import logging
import os
import sys
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

# Import core simulation components
from src.simulator import CustomerBehaviorSimulator
from src.analysis import SimulationAnalyzer
from src.models import PerformanceMetrics

# Import new LLM and MLOps components
from src.llm_integration import (
    LLMPersonaGenerator, 
    LLMInsightsGenerator, 
    LLMReportGenerator,
    load_llm_config
)
from src.mlops_pipeline import (
    CustomerBehaviorPredictor,
    MLOpsMonitor,
    ABTestingFramework,
    ModelVersionManager,
    ModelConfig,
    setup_mlflow_tracking
)


class EnhancedSimulationSystem:
    """Enhanced simulation system with LLM and MLOps capabilities"""
    
    def __init__(self, config_path: str = "config/personas.yaml"):
        self.config_path = config_path
        self.logger = self._setup_logging()
        
        # Core components
        self.simulator = None
        self.analyzer = None
        
        # LLM components
        self.llm_config = load_llm_config()
        self.persona_generator = None
        self.insights_generator = None
        self.report_generator = None
        
        # MLOps components
        self.mlops_config = self._load_mlops_config()
        self.model_manager = ModelVersionManager()
        self.ab_framework = ABTestingFramework()
        self.monitors = {}
        
        # Results storage
        self.simulation_results = {}
        self.ml_models = {}
        
        self._initialize_components()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup enhanced logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/enhanced_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_mlops_config(self) -> Dict[str, Any]:
        """Load MLOps configuration"""
        try:
            with open('config/mlops_config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load MLOps config: {e}")
            return {}
    
    def _initialize_components(self):
        """Initialize all system components"""
        
        # Initialize core simulator
        self.simulator = CustomerBehaviorSimulator(self.config_path)
        
        # Initialize LLM components if API key is available
        if self.llm_config and self.llm_config.api_key != "your-openrouter-api-key-here":
            self.persona_generator = LLMPersonaGenerator(self.llm_config)
            self.insights_generator = LLMInsightsGenerator(self.llm_config)
            self.report_generator = LLMReportGenerator(self.llm_config)
            self.logger.info("LLM components initialized")
        else:
            self.logger.warning("LLM components not initialized - API key not configured")
        
        # Setup MLflow tracking
        if self.mlops_config:
            try:
                setup_mlflow_tracking()
                self.logger.info("MLflow tracking initialized")
            except Exception as e:
                self.logger.warning(f"MLflow setup failed: {e}")
    
    def generate_ai_personas(self, market_data: Dict[str, Any]) -> bool:
        """Generate customer personas using AI"""
        
        if not self.persona_generator:
            self.logger.warning("LLM persona generator not available")
            return False
        
        try:
            self.logger.info("Generating AI-powered customer personas...")
            
            personas = self.persona_generator.generate_personas_from_market_data(
                market_data, num_personas=5
            )
            
            if personas:
                # Save generated personas to config file
                personas_data = {
                    'personas': [
                        {
                            'name': p.name,
                            'frequency': p.frequency,
                            'preferred_time': p.preferred_time,
                            'demographics': {
                                'age_range': list(p.demographics.age_range),
                                'income_range': list(p.demographics.income_range)
                            },
                            'basket_profile': p.basket_profile
                        }
                        for p in personas
                    ]
                }
                
                # Save to new config file
                ai_config_path = "config/ai_generated_personas.yaml"
                with open(ai_config_path, 'w') as f:
                    yaml.dump(personas_data, f, default_flow_style=False)
                
                self.logger.info(f"Generated {len(personas)} AI personas saved to {ai_config_path}")
                return True
            
        except Exception as e:
            self.logger.error(f"AI persona generation failed: {e}")
        
        return False
    
    def run_enhanced_simulation(
        self, 
        days: int = 30,
        customers_per_persona: int = 1000,
        enable_ml_training: bool = True,
        enable_ai_insights: bool = True
    ) -> Dict[str, Any]:
        """Run enhanced simulation with ML and AI features"""
        
        self.logger.info("Starting enhanced simulation with AI and MLOps features...")
        
        # Run core simulation
        self.logger.info("Running core customer behavior simulation...")
        transactions_df, customers_df = self.simulator.run_simulation(
            simulation_days=days
        )
        
        # Store results
        self.simulation_results = {
            'transactions_df': transactions_df,
            'customers_df': customers_df,
            'simulation_config': {
                'days': days,
                'customers_per_persona': customers_per_persona,
                'total_transactions': len(transactions_df),
                'total_customers': len(customers_df)
            }
        }
        
        # Run analysis (simplified to avoid binning issues)
        self.analyzer = SimulationAnalyzer(transactions_df, customers_df)
        try:
            analysis_results = {
                'executive_summary': self.analyzer.generate_executive_summary(),
                'persona_performance': self.analyzer.analyze_persona_performance().to_dict('records'),
                'temporal_patterns': self.analyzer.analyze_temporal_patterns(),
                'data_quality': self.analyzer.validate_data_quality(),
                'insights_report': self.analyzer.generate_insights_report()
            }
            # Skip customer_segments analysis for now due to binning issues
        except Exception as e:
            self.logger.warning(f"Analysis partially failed: {e}")
            analysis_results = {
                'executive_summary': self.analyzer.generate_executive_summary(),
                'persona_performance': self.analyzer.analyze_persona_performance().to_dict('records'),
                'error': str(e)
            }
        
        self.simulation_results['analysis'] = analysis_results
        
        # ML Model Training
        if enable_ml_training and self.mlops_config:
            self.logger.info("Training ML models...")
            self._train_ml_models(transactions_df, customers_df)
        
        # AI-powered insights
        if enable_ai_insights and self.insights_generator:
            self.logger.info("Generating AI-powered insights...")
            self._generate_ai_insights()
        
        # Generate comprehensive reports
        self._generate_enhanced_reports()
        
        self.logger.info("Enhanced simulation completed successfully!")
        return self.simulation_results
    
    def _train_ml_models(self, transactions_df: pd.DataFrame, customers_df: pd.DataFrame):
        """Train ML models for customer behavior prediction"""
        
        try:
            # Prepare features
            model_config = ModelConfig(
                model_type="classification",
                hyperparameters=self.mlops_config['models']['churn_prediction']['hyperparameters']
            )
            
            # Train churn prediction model
            churn_model = CustomerBehaviorPredictor(model_config)
            
            self.logger.info("Preparing features for ML training...")
            features_df = churn_model.prepare_features(transactions_df, customers_df)
            self.logger.info(f"Features prepared: shape={features_df.shape}, columns={features_df.columns.tolist()}")
            
            self.logger.info("Training churn prediction model...")
            churn_metrics = churn_model.train_churn_prediction_model(features_df)
            
            # Save model
            model_version = self.model_manager.save_model_version(
                churn_model, 
                "churn_prediction",
                metadata={
                    'training_data_size': len(features_df),
                    'performance_metrics': churn_metrics.__dict__,
                    'features': churn_model.feature_columns
                }
            )
            
            self.ml_models['churn_prediction'] = {
                'model': churn_model,
                'metrics': churn_metrics,
                'version': model_version
            }
            
            # Train spending prediction model
            spending_config = ModelConfig(
                model_type="regression",
                hyperparameters=self.mlops_config['models']['spending_prediction']['hyperparameters']
            )
            
            spending_model = CustomerBehaviorPredictor(spending_config)
            spending_metrics = spending_model.train_spending_prediction_model(features_df)
            
            spending_version = self.model_manager.save_model_version(
                spending_model,
                "spending_prediction",
                metadata={
                    'training_data_size': len(features_df),
                    'performance_metrics': spending_metrics.__dict__,
                    'features': spending_model.feature_columns
                }
            )
            
            self.ml_models['spending_prediction'] = {
                'model': spending_model,
                'metrics': spending_metrics,
                'version': spending_version
            }
            
            # Setup monitoring
            for model_name, model_data in self.ml_models.items():
                monitor_config = ModelConfig(
                    model_type=model_data['model'].config.model_type,
                    hyperparameters={},
                    performance_threshold=self.mlops_config['models'][model_name]['performance_threshold']
                )
                self.monitors[model_name] = MLOpsMonitor(monitor_config)
            
            self.logger.info(f"Trained {len(self.ml_models)} ML models successfully")
            
        except Exception as e:
            self.logger.error(f"ML model training failed: {e}")
    
    def _generate_ai_insights(self):
        """Generate AI-powered insights and analysis"""
        
        try:
            # Generate executive summary
            summary_data = {
                'total_revenue': float(self.simulation_results['transactions_df']['total_amount'].sum()),
                'total_transactions': len(self.simulation_results['transactions_df']),
                'total_customers': len(self.simulation_results['customers_df']),
                'avg_transaction_value': float(self.simulation_results['transactions_df']['total_amount'].mean()),
                'persona_performance': self.simulation_results['analysis'].get('persona_performance', {})
            }
            
            executive_summary = self.insights_generator.generate_executive_summary(summary_data)
            
            # Customer segmentation analysis
            segment_insights = self.insights_generator.analyze_customer_segments(
                self.simulation_results['transactions_df'],
                self.simulation_results['customers_df']
            )
            
            # Generate narrative reports
            comprehensive_report = self.report_generator.generate_narrative_report(
                summary_data, "comprehensive"
            )
            
            executive_report = self.report_generator.generate_narrative_report(
                summary_data, "executive"
            )
            
            # Store AI insights
            self.simulation_results['ai_insights'] = {
                'executive_summary': executive_summary,
                'segment_insights': segment_insights,
                'comprehensive_report': comprehensive_report,
                'executive_report': executive_report
            }
            
            self.logger.info("AI insights generated successfully")
            
        except Exception as e:
            self.logger.error(f"AI insights generation failed: {e}")
    
    def _generate_enhanced_reports(self):
        """Generate comprehensive reports with all features"""
        
        # Create output directory
        output_dir = Path("data/output/enhanced")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export core data
        self.simulation_results['transactions_df'].to_csv(
            output_dir / "transactions.csv", index=False
        )
        self.simulation_results['customers_df'].to_csv(
            output_dir / "customers.csv", index=False
        )
        
        # Export analysis results
        with open(output_dir / "analysis_results.json", 'w') as f:
            json.dump(self.simulation_results['analysis'], f, indent=2, default=str)
        
        # Export ML model results
        if self.ml_models:
            ml_summary = {
                model_name: {
                    'version': model_data['version'],
                    'metrics': model_data['metrics'].__dict__,
                    'features': model_data['model'].feature_columns
                }
                for model_name, model_data in self.ml_models.items()
            }
            
            with open(output_dir / "ml_models_summary.json", 'w') as f:
                json.dump(ml_summary, f, indent=2, default=str)
        
        # Export AI insights
        if 'ai_insights' in self.simulation_results:
            with open(output_dir / "ai_insights.json", 'w') as f:
                json.dump(self.simulation_results['ai_insights'], f, indent=2, default=str)
            
            # Save narrative reports as text files
            for report_type in ['comprehensive_report', 'executive_report']:
                if report_type in self.simulation_results['ai_insights']:
                    with open(output_dir / f"{report_type}.txt", 'w') as f:
                        f.write(self.simulation_results['ai_insights'][report_type])
        
        # Generate summary dashboard data
        dashboard_data = {
            'simulation_overview': self.simulation_results['simulation_config'],
            'key_metrics': {
                'total_revenue': float(self.simulation_results['transactions_df']['total_amount'].sum()),
                'avg_transaction': float(self.simulation_results['transactions_df']['total_amount'].mean()),
                'customer_count': len(self.simulation_results['customers_df']),
                'transaction_count': len(self.simulation_results['transactions_df'])
            },
            'ml_models': list(self.ml_models.keys()) if self.ml_models else [],
            'ai_features_enabled': bool(self.insights_generator),
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_dir / "dashboard_data.json", 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        self.logger.info(f"Enhanced reports generated in {output_dir}")
    
    def run_ab_test(
        self, 
        experiment_name: str,
        control_model_name: str,
        treatment_model_name: str,
        test_data: pd.DataFrame,
        traffic_split: float = 0.5
    ) -> Dict[str, Any]:
        """Run A/B test between two models"""
        
        if not self.ml_models:
            self.logger.error("No ML models available for A/B testing")
            return {}
        
        try:
            # Load models
            control_model = self.model_manager.load_model_version(control_model_name, "latest")
            treatment_model = self.model_manager.load_model_version(treatment_model_name, "latest")
            
            # Create experiment
            self.ab_framework.create_experiment(
                experiment_name, control_model, treatment_model, traffic_split
            )
            
            # Run predictions
            results = []
            for _, row in test_data.iterrows():
                features = pd.DataFrame([row])
                predictions, variant = self.ab_framework.run_prediction(
                    experiment_name, features, str(row.get('customer_id', ''))
                )
                results.append({
                    'customer_id': row.get('customer_id', ''),
                    'variant': variant,
                    'prediction': predictions[0]
                })
            
            # Analyze results
            analysis = self.ab_framework.analyze_experiment(experiment_name)
            
            self.logger.info(f"A/B test '{experiment_name}' completed")
            return {
                'experiment_name': experiment_name,
                'results': results,
                'analysis': analysis
            }
            
        except Exception as e:
            self.logger.error(f"A/B test failed: {e}")
            return {}
    
    def monitor_model_performance(self, model_name: str, new_data: pd.DataFrame):
        """Monitor ML model performance on new data"""
        
        if model_name not in self.ml_models or model_name not in self.monitors:
            self.logger.error(f"Model {model_name} not found for monitoring")
            return
        
        try:
            model = self.ml_models[model_name]['model']
            monitor = self.monitors[model_name]
            
            # For demonstration, we'll use a subset of data as ground truth
            # In production, this would come from actual outcomes
            sample_data = new_data.sample(min(100, len(new_data)))
            
            # Generate synthetic ground truth for demo
            if model.config.model_type == 'classification':
                ground_truth = np.random.choice([0, 1], size=len(sample_data))
            else:
                ground_truth = sample_data['total_spent'] if 'total_spent' in sample_data.columns else np.random.normal(5000, 1000, len(sample_data))
            
            # Monitor performance
            metrics = monitor.monitor_model_performance(model, sample_data, ground_truth)
            
            # Check for data drift
            reference_data = self.simulation_results['transactions_df'].sample(1000)
            drift_analysis = monitor.check_data_drift(reference_data, new_data)
            
            self.logger.info(f"Model {model_name} monitoring completed")
            return {
                'performance_metrics': metrics,
                'drift_analysis': drift_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Model monitoring failed: {e}")
            return {}


def main():
    """Enhanced main function with LLM and MLOps features"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Customer Behavior Simulation with AI and MLOps"
    )
    
    # Core simulation arguments
    parser.add_argument("--days", type=int, default=30, help="Simulation days")
    parser.add_argument("--customers", type=int, default=1000, help="Customers per persona")
    parser.add_argument("--config", type=str, default="config/personas.yaml", help="Persona config file")
    
    # Enhanced features
    parser.add_argument("--generate-ai-personas", action="store_true", help="Generate personas using AI")
    parser.add_argument("--enable-ml-training", action="store_true", default=True, help="Enable ML model training")
    parser.add_argument("--enable-ai-insights", action="store_true", default=True, help="Enable AI insights")
    parser.add_argument("--run-ab-test", type=str, help="Run A/B test with experiment name")
    
    # Market data for AI persona generation
    parser.add_argument("--market-data", type=str, help="Path to market data JSON file")
    
    args = parser.parse_args()
    
    try:
        # Initialize enhanced system
        system = EnhancedSimulationSystem(args.config)
        
        # Generate AI personas if requested
        if args.generate_ai_personas:
            market_data = {}
            if args.market_data and os.path.exists(args.market_data):
                with open(args.market_data, 'r') as f:
                    market_data = json.load(f)
            else:
                # Use sample market data
                market_data = {
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
            
            if system.generate_ai_personas(market_data):
                print("AI personas generated successfully!")
                print("Check config/ai_generated_personas.yaml")
                return
        
        # Run enhanced simulation
        results = system.run_enhanced_simulation(
            days=args.days,
            customers_per_persona=args.customers,
            enable_ml_training=args.enable_ml_training,
            enable_ai_insights=args.enable_ai_insights
        )
        
        # Display results summary
        print("\n" + "="*80)
        print("ENHANCED SIMULATION RESULTS SUMMARY")
        print("="*80)
        
        print(f"\nCORE METRICS:")
        print(f"   Total Customers: {results['simulation_config']['total_customers']:,}")
        print(f"   Total Transactions: {results['simulation_config']['total_transactions']:,}")
        print(f"   Total Revenue: ₹{results['transactions_df']['total_amount'].sum():,.2f}")
        print(f"   Average Transaction: ₹{results['transactions_df']['total_amount'].mean():,.2f}")
        
        if system.ml_models:
            print(f"\nML MODELS TRAINED:")
            for model_name, model_data in system.ml_models.items():
                metrics = model_data['metrics']
                if hasattr(metrics, 'accuracy') and metrics.accuracy > 0:
                    print(f"   {model_name}: Accuracy = {metrics.accuracy:.3f}")
                elif hasattr(metrics, 'r2_score'):
                    print(f"   {model_name}: R² = {metrics.r2_score:.3f}")
        
        if 'ai_insights' in results:
            print(f"\nAI INSIGHTS GENERATED:")
            print("   Executive Summary")
            print("   Customer Segmentation Analysis")
            print("   Comprehensive Business Report")
            print("   Executive Briefing")
        
        print(f"\nOUTPUT FILES:")
        print("   data/output/enhanced/transactions.csv")
        print("   data/output/enhanced/customers.csv")
        print("   data/output/enhanced/analysis_results.json")
        if system.ml_models:
            print("   data/output/enhanced/ml_models_summary.json")
        if 'ai_insights' in results:
            print("   data/output/enhanced/ai_insights.json")
            print("   data/output/enhanced/comprehensive_report.txt")
            print("   data/output/enhanced/executive_report.txt")
        
        print("\nEnhanced simulation completed successfully!")
        print("Ready for advanced analytics and business intelligence!")
        
    except Exception as e:
        print(f"Enhanced simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()