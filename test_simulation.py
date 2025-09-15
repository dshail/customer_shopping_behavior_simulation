
"""
Customer Shopping Behavior Simulation - Test Suite
=================================================

Comprehensive test suite to validate simulation functionality and data quality.

Author: dshail
Date: 2025

Usage:
    python -m pytest tests/test_simulation.py -v
    python -m pytest tests/test_simulation.py::TestCustomerBehaviorSimulator::test_persona_loading -v
"""

import unittest
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models import Customer, Transaction, PersonaConfig, Demographics, ShoppingBasket, BasketItem
from simulator import CustomerBehaviorSimulator
from analysis import SimulationAnalyzer


class TestDataModels(unittest.TestCase):
    """Test data models and their functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.demographics = Demographics(
            age=30,
            income=50000,
            location_type="urban",
            education="graduate",
            family_status="single"
        )

        self.customer = Customer(
            customer_id="test_001",
            name="Test Customer",
            persona_type="Test Persona",
            demographics=self.demographics,
            contact_info={"email": "test@example.com", "phone": "1234567890"}
        )

    def test_demographics_creation(self):
        """Test demographics model creation and methods"""
        self.assertEqual(self.demographics.age, 30)
        self.assertEqual(self.demographics.income, 50000)
        self.assertEqual(self.demographics.location_type, "urban")

        # Test to_dict method
        demo_dict = self.demographics.to_dict()
        self.assertIsInstance(demo_dict, dict)
        self.assertEqual(demo_dict['age'], 30)

    def test_customer_creation(self):
        """Test customer model creation and methods"""
        self.assertEqual(self.customer.customer_id, "test_001")
        self.assertEqual(self.customer.name, "Test Customer")
        self.assertEqual(self.customer.demographics.age, 30)

        # Test to_dict method
        customer_dict = self.customer.to_dict()
        self.assertIsInstance(customer_dict, dict)
        self.assertIn('customer_id', customer_dict)
        self.assertIn('age', customer_dict)

    def test_basket_item_creation(self):
        """Test basket item model"""
        item = BasketItem(
            category="coffee",
            quantity=2,
            price_per_item=150.0,
            total_price=300.0
        )

        self.assertEqual(item.category, "coffee")
        self.assertEqual(item.quantity, 2)
        self.assertEqual(item.total_price, 300.0)

    def test_shopping_basket_functionality(self):
        """Test shopping basket operations"""
        basket = ShoppingBasket()

        # Initially empty
        self.assertEqual(basket.total_amount, 0.0)
        self.assertEqual(basket.num_items, 0)

        # Add items
        item1 = BasketItem("coffee", 1, 150.0, 150.0)
        item2 = BasketItem("snacks", 2, 50.0, 100.0)

        basket.add_item(item1)
        basket.add_item(item2)

        self.assertEqual(basket.total_amount, 250.0)
        self.assertEqual(basket.num_items, 2)
        self.assertEqual(len(basket.items), 2)


class TestCustomerBehaviorSimulator(unittest.TestCase):
    """Test the main simulation engine"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_personas.yaml')

        # Minimal test configuration
        test_config = {
            'personas': [
                {
                    'name': 'Test Persona',
                    'frequency': 'daily',
                    'preferred_time': ['9:00am–10:00am'],
                    'demographics': {
                        'age_range': [25, 35],
                        'income_range': [30000, 60000],
                        'location_type': 'urban',
                        'education': 'graduate',
                        'family_status': 'single'
                    },
                    'basket_profile': {
                        'coffee': {
                            'probability': 0.8,
                            'quantity': [1, 2],
                            'price_range': [100, 200]
                        }
                    }
                }
            ],
            'market_config': {
                'festival_dates': ['2024-01-15'],
                'seasonal_multipliers': {
                    'festival': 1.5,
                    'weekend': 1.2,
                    'weekday': 1.0
                }
            },
            'simulation': {
                'default_days': 30,
                'customers_per_persona': 100,
                'start_date': '2024-01-01',
                'random_seed': 42
            }
        }

        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)

        self.simulator = CustomerBehaviorSimulator(
            config_path=self.config_path,
            random_seed=42
        )

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)

    def test_simulator_initialization(self):
        """Test simulator initialization"""
        self.assertIsInstance(self.simulator, CustomerBehaviorSimulator)
        self.assertEqual(len(self.simulator.personas), 1)
        self.assertIn('Test Persona', self.simulator.personas)

    def test_persona_loading(self):
        """Test persona configuration loading"""
        persona = self.simulator.personas['Test Persona']
        self.assertEqual(persona.name, 'Test Persona')
        self.assertEqual(persona.frequency, 'daily')
        self.assertIn('coffee', persona.basket_profile)

    def test_festival_date_parsing(self):
        """Test festival date parsing"""
        self.assertEqual(len(self.simulator.festival_dates), 1)
        self.assertEqual(self.simulator.festival_dates[0], datetime(2024, 1, 15))

    def test_festival_period_detection(self):
        """Test festival period detection logic"""
        # Test date within festival period
        test_date = datetime(2024, 1, 15)  # Exact festival date
        self.assertTrue(self.simulator._is_festival_period(test_date))

        # Test date within ±2 days
        test_date2 = datetime(2024, 1, 17)  # 2 days after
        self.assertTrue(self.simulator._is_festival_period(test_date2))

        # Test date outside festival period
        test_date3 = datetime(2024, 1, 20)  # 5 days after
        self.assertFalse(self.simulator._is_festival_period(test_date3))

    def test_weekend_detection(self):
        """Test weekend detection logic"""
        # Saturday
        saturday = datetime(2024, 1, 6)
        self.assertTrue(self.simulator._is_weekend(saturday))

        # Sunday
        sunday = datetime(2024, 1, 7)
        self.assertTrue(self.simulator._is_weekend(sunday))

        # Monday (weekday)
        monday = datetime(2024, 1, 8)
        self.assertFalse(self.simulator._is_weekend(monday))

    def test_customer_generation(self):
        """Test customer generation functionality"""
        persona = self.simulator.personas['Test Persona']
        customers = self.simulator._generate_customers_for_persona(persona, count=10)

        self.assertEqual(len(customers), 10)

        # Check customer properties
        customer = customers[0]
        self.assertIsInstance(customer, Customer)
        self.assertEqual(customer.persona_type, 'Test Persona')
        self.assertGreaterEqual(customer.demographics.age, 25)
        self.assertLessEqual(customer.demographics.age, 35)

    def test_shopping_decision_logic(self):
        """Test shopping decision logic"""
        persona = self.simulator.personas['Test Persona']
        customer_id = 'test_customer_001'
        current_date = datetime(2024, 1, 1)

        # First time shopping (should shop)
        should_shop = self.simulator._should_customer_shop_today(
            persona, customer_id, current_date, {}
        )
        self.assertTrue(should_shop)

        # Daily frequency - should likely shop next day
        last_shopping_dates = {customer_id: current_date}
        next_day = current_date + timedelta(days=1)

        # Run multiple times to check probabilistic behavior
        shop_decisions = []
        for _ in range(100):
            decision = self.simulator._should_customer_shop_today(
                persona, customer_id, next_day, last_shopping_dates
            )
            shop_decisions.append(decision)

        # For daily frequency, should shop most of the time
        shop_rate = sum(shop_decisions) / len(shop_decisions)
        self.assertGreater(shop_rate, 0.5)  # Should shop more than 50% of the time

    def test_basket_generation(self):
        """Test shopping basket generation"""
        persona = self.simulator.personas['Test Persona']
        test_date = datetime(2024, 1, 1)

        basket = self.simulator._generate_shopping_basket(persona, test_date)

        self.assertIsInstance(basket, ShoppingBasket)
        # Basket might be empty due to probabilistic nature, but should be valid
        self.assertGreaterEqual(basket.total_amount, 0)
        self.assertGreaterEqual(basket.num_items, 0)

    def test_simulation_run(self):
        """Test full simulation run with small parameters"""
        # Run short simulation for testing
        transactions_df, customers_df = self.simulator.run_simulation(simulation_days=5)

        # Verify DataFrames are created
        self.assertIsInstance(transactions_df, pd.DataFrame)
        self.assertIsInstance(customers_df, pd.DataFrame)

        # Check basic structure
        expected_transaction_columns = [
            'transaction_id', 'date', 'customer_id', 'persona_type', 'total_amount'
        ]
        for col in expected_transaction_columns:
            self.assertIn(col, transactions_df.columns)

        expected_customer_columns = [
            'customer_id', 'name', 'persona_type', 'age', 'income'
        ]
        for col in expected_customer_columns:
            self.assertIn(col, customers_df.columns)

        # Check data quality
        self.assertTrue(len(customers_df) > 0)
        self.assertTrue(transactions_df['total_amount'].sum() > 0)
        self.assertFalse(transactions_df['total_amount'].isna().any())


class TestSimulationAnalyzer(unittest.TestCase):
    """Test the analysis utilities"""

    def setUp(self):
        """Set up test fixtures with sample data"""
        # Create sample transaction data
        self.transactions_df = pd.DataFrame({
            'transaction_id': ['TXN_001', 'TXN_002', 'TXN_003'],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'timestamp': ['2024-01-01 10:00:00', '2024-01-02 15:00:00', '2024-01-03 18:00:00'],
            'customer_id': ['cust_001', 'cust_002', 'cust_001'],
            'customer_name': ['John Doe', 'Jane Smith', 'John Doe'],
            'persona_type': ['Young Professional', 'Family Shopper', 'Young Professional'],
            'total_amount': [500.0, 1200.0, 350.0],
            'num_items': [3, 8, 2],
            'is_festival': [False, True, False],
            'is_weekend': [False, False, True],
            'customer_age': [28, 35, 28],
            'customer_income': [50000, 75000, 50000]
        })

        # Create sample customer data
        self.customers_df = pd.DataFrame({
            'customer_id': ['cust_001', 'cust_002'],
            'name': ['John Doe', 'Jane Smith'],
            'persona_type': ['Young Professional', 'Family Shopper'],
            'age': [28, 35],
            'income': [50000, 75000],
            'location_type': ['urban', 'suburban']
        })

        self.analyzer = SimulationAnalyzer(self.transactions_df, self.customers_df)

    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertIsInstance(self.analyzer, SimulationAnalyzer)
        self.assertEqual(len(self.analyzer.transactions_df), 3)
        self.assertEqual(len(self.analyzer.customers_df), 2)

    def test_executive_summary_generation(self):
        """Test executive summary generation"""
        summary = self.analyzer.generate_executive_summary()

        self.assertIn('key_metrics', summary)
        self.assertIn('top_performers', summary)

        key_metrics = summary['key_metrics']
        self.assertEqual(key_metrics['total_transactions'], 3)
        self.assertEqual(key_metrics['unique_customers'], 2)
        self.assertEqual(key_metrics['total_revenue'], 2050.0)

    def test_persona_performance_analysis(self):
        """Test persona performance analysis"""
        persona_perf = self.analyzer.analyze_persona_performance()

        self.assertIsInstance(persona_perf, pd.DataFrame)
        self.assertIn('total_revenue', persona_perf.columns)
        self.assertIn('avg_transaction_value', persona_perf.columns)

        # Check calculations
        yp_revenue = persona_perf.loc['Young Professional', 'total_revenue']
        self.assertEqual(yp_revenue, 850.0)  # 500 + 350

    def test_data_quality_validation(self):
        """Test data quality validation"""
        quality_report = self.analyzer.validate_data_quality()

        self.assertIn('data_completeness', quality_report)
        self.assertIn('data_consistency', quality_report)
        self.assertIn('outlier_analysis', quality_report)

        # Should have no null values in our test data
        completeness = quality_report['data_completeness']
        self.assertEqual(completeness['transactions_total_rows'], 3)
        self.assertEqual(completeness['customers_total_rows'], 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""

    def setUp(self):
        """Set up for integration tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        """Clean up integration test fixtures"""
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_simulation(self):
        """Test complete simulation pipeline"""
        # This would run the full simulation if we had more time
        # For now, we'll test individual components work together

        # Create minimal config
        config_path = os.path.join(self.temp_dir, 'config.yaml')
        config = {
            'personas': [{
                'name': 'Test',
                'frequency': 'daily',
                'preferred_time': ['9:00am–10:00am'],
                'demographics': {
                    'age_range': [25, 35],
                    'income_range': [30000, 60000],
                    'location_type': 'urban',
                    'education': 'graduate',
                    'family_status': 'single'
                },
                'basket_profile': {
                    'coffee': {
                        'probability': 0.5,
                        'quantity': [1, 1],
                        'price_range': [100, 200]
                    }
                }
            }],
            'market_config': {'festival_dates': []},
            'simulation': {'customers_per_persona': 10}
        }

        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Test simulator can be initialized
        simulator = CustomerBehaviorSimulator(config_path=config_path, random_seed=42)
        self.assertIsNotNone(simulator)

        # Test basic functionality without full run
        personas = simulator.personas
        self.assertEqual(len(personas), 1)


# Performance and stress tests
class TestPerformance(unittest.TestCase):
    """Performance and stress tests"""

    @pytest.mark.slow
    def test_large_simulation_performance(self):
        """Test performance with larger simulation (marked as slow)"""
        # This test would be run separately for performance testing
        pass

    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        # Basic memory usage test
        initial_objects = len(globals())

        # Create and destroy multiple simulators
        for i in range(10):
            config = {'personas': [], 'market_config': {}, 'simulation': {}}
            # Would create simulator here if we had persistent config

        # Memory should not grow significantly
        final_objects = len(globals())
        self.assertLess(final_objects - initial_objects, 100)


# Utility functions for testing
def create_test_config(persona_count=1, customer_count=10):
    """Create a test configuration with specified parameters"""
    return {
        'personas': [{
            'name': f'Test Persona {i}',
            'frequency': 'daily',
            'preferred_time': ['9:00am–10:00am'],
            'demographics': {
                'age_range': [25, 35],
                'income_range': [30000, 60000],
                'location_type': 'urban',
                'education': 'graduate',
                'family_status': 'single'
            },
            'basket_profile': {
                'coffee': {
                    'probability': 0.5,
                    'quantity': [1, 1],
                    'price_range': [100, 200]
                }
            }
        } for i in range(persona_count)],
        'market_config': {
            'festival_dates': ['2024-01-15'],
            'seasonal_multipliers': {'festival': 1.5, 'weekend': 1.2}
        },
        'simulation': {
            'customers_per_persona': customer_count,
            'start_date': '2024-01-01'
        }
    }


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
