
"""
Customer Shopping Behavior Simulation Engine
===========================================

Main simulation engine that orchestrates the entire customer behavior simulation process.

Author: dshail
Date: August 2025
"""

import yaml
import json
import pandas as pd
import numpy as np
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from faker import Faker

# Import our custom models
import sys
sys.path.append('src')
from models import (
    Customer, Transaction, PersonaConfig, SimulationConfig,
    ShoppingBasket, BasketItem, Demographics, PerformanceMetrics,
    generate_transaction_id, generate_customer_id, parse_time_slot
)


class CustomerBehaviorSimulator:
    """
    Main class for simulating customer shopping behavior based on personas.

    This class handles:
    - Loading and parsing persona configurations
    - Generating synthetic customers
    - Simulating shopping sessions and transactions
    - Managing temporal variations (festivals, weekends)
    - Exporting results to various formats
    """

    def __init__(self, config_path: str = 'config/personas.yaml', random_seed: int = 42):
        """
        Initialize the simulator with configuration.

        Args:
            config_path: Path to personas configuration file
            random_seed: Random seed for reproducible results
        """
        self.config_path = config_path
        self.random_seed = random_seed
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Initialize Faker for generating synthetic data
        self.fake = Faker('en_IN')  # Indian locale
        self.fake.seed_instance(random_seed)

        # Load configuration
        self.config = self._load_configuration()
        self.personas = self._load_personas()
        self.festival_dates = self._parse_festival_dates()

        # Initialize tracking
        self.metrics = PerformanceMetrics()
        self.customers_db: Dict[str, Customer] = {}
        self.transactions_log: List[Transaction] = []

        self.logger.info(f"CustomerBehaviorSimulator initialized with {len(self.personas)} personas")

    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML in configuration file: {e}")
            raise

    def _load_personas(self) -> Dict[str, PersonaConfig]:
        """Parse and load persona configurations"""
        personas = {}

        for persona_data in self.config.get('personas', []):
            persona = PersonaConfig(
                name=persona_data['name'],
                frequency=persona_data['frequency'],
                preferred_times=persona_data['preferred_time'],
                demographics=persona_data['demographics'],
                basket_profile=persona_data['basket_profile']
            )
            personas[persona.name] = persona

        self.logger.info(f"Loaded {len(personas)} persona configurations")
        return personas

    def _parse_festival_dates(self) -> List[datetime]:
        """Parse festival dates from configuration"""
        festival_dates = []

        for date_str in self.config.get('market_config', {}).get('festival_dates', []):
            try:
                festival_date = datetime.strptime(date_str, '%Y-%m-%d')
                festival_dates.append(festival_date)
            except ValueError:
                self.logger.warning(f"Invalid festival date format: {date_str}")

        self.logger.info(f"Loaded {len(festival_dates)} festival dates")
        return festival_dates

    def _is_festival_period(self, date: datetime) -> bool:
        """Check if date falls within festival period (±2 days)"""
        for festival_date in self.festival_dates:
            if abs((date - festival_date).days) <= 2:
                return True
        return False

    def _is_weekend(self, date: datetime) -> bool:
        """Check if date is weekend (Saturday=5, Sunday=6)"""
        return date.weekday() >= 5

    def _generate_customer_demographics(self, persona: PersonaConfig) -> Demographics:
        """Generate realistic demographics based on persona"""
        demo_config = persona.demographics

        age = random.randint(demo_config['age_range'][0], demo_config['age_range'][1])
        income = random.randint(demo_config['income_range'][0], demo_config['income_range'][1])

        return Demographics(
            age=age,
            income=income,
            location_type=demo_config['location_type'],
            education=demo_config['education'],
            family_status=demo_config['family_status']
        )

    def _generate_customers_for_persona(self, persona: PersonaConfig, count: int = 1000) -> List[Customer]:
        """Generate synthetic customers for a given persona"""
        customers = []

        for i in range(count):
            customer_id = generate_customer_id(persona.name, i + 1)
            demographics = self._generate_customer_demographics(persona)

            # Generate contact information
            contact_info = {
                'phone': self.fake.phone_number(),
                'email': self.fake.email(),
                'address': self.fake.address().replace('\n', ', ')
            }

            customer = Customer(
                customer_id=customer_id,
                name=self.fake.name(),
                persona_type=persona.name,
                demographics=demographics,
                contact_info=contact_info
            )

            customers.append(customer)
            self.customers_db[customer_id] = customer

        self.metrics.customers_generated += len(customers)
        self.logger.info(f"Generated {len(customers)} customers for persona: {persona.name}")
        return customers

    def _should_customer_shop_today(self, persona: PersonaConfig, customer_id: str, 
                                   current_date: datetime, last_shopping_dates: Dict[str, datetime]) -> bool:
        """Determine if customer should shop today based on persona frequency"""

        frequency = persona.frequency
        last_shop_date = last_shopping_dates.get(customer_id)

        if last_shop_date is None:
            return True  # First shopping day

        days_since_last_shop = (current_date - last_shop_date).days

        # Apply temporal multipliers
        multiplier = 1.0
        if self._is_festival_period(current_date):
            multiplier *= self.config.get('market_config', {}).get('seasonal_multipliers', {}).get('festival', 1.5)
        if self._is_weekend(current_date):
            multiplier *= self.config.get('market_config', {}).get('seasonal_multipliers', {}).get('weekend', 1.2)

        # Base probability based on frequency
        if frequency == "daily":
            base_prob = 0.8
        elif frequency == "alternate":
            base_prob = 0.5 if days_since_last_shop >= 1 else 0.1
        elif frequency == "weekly":
            base_prob = 0.7 if days_since_last_shop >= 6 else 0.1
        elif frequency == "monthly":
            base_prob = 0.8 if days_since_last_shop >= 28 else 0.05
        else:
            base_prob = 0.3

        adjusted_prob = min(base_prob * multiplier, 0.95)  # Cap at 95%
        return random.random() < adjusted_prob

    def _generate_shopping_time(self, persona: PersonaConfig, date: datetime) -> datetime:
        """Generate realistic shopping time within preferred time slots"""

        preferred_times = persona.preferred_times
        chosen_slot = random.choice(preferred_times)

        try:
            start_time, end_time = parse_time_slot(chosen_slot)

            # Convert to minutes for random generation
            start_minutes = start_time.hour * 60 + start_time.minute
            end_minutes = end_time.hour * 60 + end_time.minute

            # Handle time slots that cross midnight
            if end_minutes <= start_minutes:
                end_minutes += 24 * 60

            random_minutes = random.randint(start_minutes, end_minutes)

            # Handle day overflow
            if random_minutes >= 24 * 60:
                random_minutes -= 24 * 60
                date += timedelta(days=1)

            shopping_time = date.replace(
                hour=random_minutes // 60,
                minute=random_minutes % 60,
                second=random.randint(0, 59)
            )

            return shopping_time

        except Exception as e:
            self.logger.warning(f"Error generating shopping time for slot '{chosen_slot}': {e}")
            # Return default time
            return date.replace(hour=random.randint(9, 20), minute=random.randint(0, 59))

    def _generate_shopping_basket(self, persona: PersonaConfig, date: datetime) -> ShoppingBasket:
        """Generate shopping basket based on persona preferences and temporal factors"""

        basket = ShoppingBasket()

        # Apply temporal multipliers
        festival_multiplier = 1.5 if self._is_festival_period(date) else 1.0
        weekend_multiplier = 1.2 if self._is_weekend(date) else 1.0

        for item_category, item_config in persona.basket_profile.items():
            # Check if customer buys this item category
            base_probability = item_config['probability']
            adjusted_probability = min(base_probability * festival_multiplier * weekend_multiplier, 0.95)

            if random.random() < adjusted_probability:
                # Determine quantity
                quantity_range = item_config['quantity']
                quantity = random.randint(quantity_range[0], quantity_range[1])

                # Apply festival quantity boost
                if self._is_festival_period(date):
                    quantity = min(int(quantity * 1.3), quantity_range[1] * 2)

                # Determine price per item
                price_range = item_config['price_range']
                price_per_item = random.uniform(price_range[0], price_range[1])

                # Apply festival price variation
                if self._is_festival_period(date):
                    price_variation = self.config.get('market_config', {}).get('price_variations', {}).get('festival_markup', [1.05, 1.15])
                    price_per_item *= random.uniform(price_variation[0], price_variation[1])

                # Create basket item
                item = BasketItem(
                    category=item_category,
                    quantity=quantity,
                    price_per_item=round(price_per_item, 2),
                    total_price=round(quantity * price_per_item, 2)
                )

                basket.add_item(item)

        return basket

    def _create_transaction_context(self, date: datetime, customer: Customer) -> Dict[str, Any]:
        """Create context information for transaction"""
        return {
            'is_festival': self._is_festival_period(date),
            'is_weekend': self._is_weekend(date),
            'day_of_week': date.strftime('%A'),
            'month': date.strftime('%B'),
            'customer_age': customer.demographics.age,
            'customer_income': customer.demographics.income,
            'customer_location': customer.demographics.location_type,
            'customer_education': customer.demographics.education,
            'customer_family_status': customer.demographics.family_status
        }

    def run_simulation(self, simulation_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete customer behavior simulation.

        Args:
            simulation_days: Number of days to simulate

        Returns:
            Tuple of (transactions_df, customers_df)
        """

        self.logger.info("Starting customer behavior simulation...")
        self.metrics.start_timer()

        # Generate customers for each persona
        all_customers = []
        customers_per_persona = self.config.get('simulation', {}).get('customers_per_persona', 1000)

        for persona_name, persona_config in self.personas.items():
            customers = self._generate_customers_for_persona(persona_config, customers_per_persona)
            all_customers.extend(customers)

        # Track last shopping dates for frequency management
        last_shopping_dates: Dict[str, datetime] = {}

        # Generate transactions for each day
        start_date_str = self.config.get('simulation', {}).get('start_date', '2024-01-01')
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

        for day in range(simulation_days):
            current_date = start_date + timedelta(days=day)
            daily_transactions = 0

            # Process each customer
            for customer in all_customers:
                persona_config = self.personas[customer.persona_type]

                # Check if customer should shop today
                if self._should_customer_shop_today(
                    persona_config, customer.customer_id, current_date, last_shopping_dates
                ):
                    try:
                        # Generate shopping session
                        shopping_time = self._generate_shopping_time(persona_config, current_date)
                        basket = self._generate_shopping_basket(persona_config, current_date)

                        if basket.num_items > 0:  # Only create transaction if basket has items
                            context = self._create_transaction_context(current_date, customer)

                            transaction = Transaction(
                                transaction_id=generate_transaction_id(),
                                customer_id=customer.customer_id,
                                customer_name=customer.name,
                                persona_type=customer.persona_type,
                                timestamp=shopping_time,
                                basket=basket,
                                context=context
                            )

                            self.transactions_log.append(transaction)
                            last_shopping_dates[customer.customer_id] = current_date
                            daily_transactions += 1

                            self.metrics.add_transaction(basket.total_amount)

                    except Exception as e:
                        self.logger.error(f"Error generating transaction for customer {customer.customer_id}: {e}")
                        self.metrics.add_error()

            # Log progress every 5 days
            if (day + 1) % 5 == 0 or day == simulation_days - 1:
                festival_marker = "[Festival]" if self._is_festival_period(current_date) else ""
                weekend_marker = "[Weekend]" if self._is_weekend(current_date) else ""

                self.logger.info(
                    f"Day {day+1:2d} ({current_date.strftime('%Y-%m-%d')}): "
                    f"{daily_transactions:4d} transactions | "
                    f"Total: {len(self.transactions_log):5d} | {festival_marker} {weekend_marker}"
                )

        self.metrics.stop_timer()

        # Convert to DataFrames
        transactions_df = self._create_transactions_dataframe()
        customers_df = self._create_customers_dataframe()

        # Log final results
        self.logger.info(f"\nSimulation Complete!")
        self.logger.info(f"Generated {len(transactions_df)} transactions")
        self.logger.info(f"Across {len(customers_df)} customers")
        self.logger.info(f"Total revenue: Rs.{transactions_df['total_amount'].sum():,.2f}")
        self.logger.info(f"Execution time: {self.metrics.get_duration()}")

        return transactions_df, customers_df

    def _create_transactions_dataframe(self) -> pd.DataFrame:
        """Convert transactions to pandas DataFrame"""

        transaction_dicts = []
        for transaction in self.transactions_log:
            transaction_dicts.append(transaction.to_dict())

        return pd.DataFrame(transaction_dicts)

    def _create_customers_dataframe(self) -> pd.DataFrame:
        """Convert customers to pandas DataFrame"""

        customer_dicts = []
        for customer in self.customers_db.values():
            customer_dicts.append(customer.to_dict())

        return pd.DataFrame(customer_dicts)

    def export_data(self, transactions_df: pd.DataFrame, customers_df: pd.DataFrame, 
                   output_dir: str = 'data/output') -> None:
        """Export simulation data to files"""

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Export main datasets
        transactions_df.to_csv(f'{output_dir}/transactions.csv', index=False)
        customers_df.to_csv(f'{output_dir}/customers.csv', index=False)

        # Export performance metrics
        with open(f'{output_dir}/simulation_metrics.json', 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=2)

        # Export configuration
        with open(f'{output_dir}/simulation_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)

        self.logger.info(f"Data exported to {output_dir}/")

    def get_simulation_summary(self, transactions_df: pd.DataFrame, customers_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive simulation summary"""

        summary = {
            'overview': {
                'total_customers': len(customers_df),
                'total_transactions': len(transactions_df),
                'simulation_period': f"{transactions_df['date'].min()} to {transactions_df['date'].max()}",
                'total_revenue': round(transactions_df['total_amount'].sum(), 2),
                'avg_transaction_value': round(transactions_df['total_amount'].mean(), 2)
            },
            'persona_breakdown': {},
            'temporal_analysis': {},
            'performance_metrics': self.metrics.to_dict()
        }

        # Persona analysis
        persona_stats = transactions_df.groupby('persona_type').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'num_items': 'mean'
        }).round(2)

        for persona in persona_stats.index:
            summary['persona_breakdown'][persona] = {
                'total_revenue': persona_stats.loc[persona, ('total_amount', 'sum')],
                'avg_transaction': persona_stats.loc[persona, ('total_amount', 'mean')],
                'transaction_count': int(persona_stats.loc[persona, ('total_amount', 'count')]),
                'avg_items_per_basket': persona_stats.loc[persona, ('num_items', 'mean')]
            }

        # Temporal analysis
        festival_analysis = transactions_df.groupby('is_festival')['total_amount'].agg(['sum', 'mean', 'count'])
        weekend_analysis = transactions_df.groupby('is_weekend')['total_amount'].agg(['sum', 'mean', 'count'])

        summary['temporal_analysis'] = {
            'festival_vs_regular': {
                'festival_revenue': round(festival_analysis.loc[True, 'sum'] if True in festival_analysis.index else 0, 2),
                'regular_revenue': round(festival_analysis.loc[False, 'sum'] if False in festival_analysis.index else 0, 2),
                'festival_avg_transaction': round(festival_analysis.loc[True, 'mean'] if True in festival_analysis.index else 0, 2),
                'regular_avg_transaction': round(festival_analysis.loc[False, 'mean'] if False in festival_analysis.index else 0, 2)
            },
            'weekend_vs_weekday': {
                'weekend_revenue': round(weekend_analysis.loc[True, 'sum'] if True in weekend_analysis.index else 0, 2),
                'weekday_revenue': round(weekend_analysis.loc[False, 'sum'] if False in weekend_analysis.index else 0, 2),
                'weekend_avg_transaction': round(weekend_analysis.loc[True, 'mean'] if True in weekend_analysis.index else 0, 2),
                'weekday_avg_transaction': round(weekend_analysis.loc[False, 'mean'] if False in weekend_analysis.index else 0, 2)
            }
        }

        return summary
